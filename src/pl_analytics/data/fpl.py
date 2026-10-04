"""Offline FPL Decision Lab artifact builder from archived public snapshots."""

import json
import re
from pathlib import Path

import httpx
import pandas as pd

from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.features.fpl import (
    fixture_rows,
    latest_players,
    league_table,
    match_results,
    price_watch_backtest,
    prior_season_rates,
    project_players,
)
from pl_analytics.features.fpl_optimizer import backtest_summary, rolling_backtest
from pl_analytics.statistics.season import simulate_season

MAX_LEAGUE_TEAMS = 21

_ALIASES = {
    "spurs": "tottenham",
    "wolverhampton wanderers": "wolves",
    "nottingham forest": "nott'm forest",
}


def normalise_club(name: object) -> str:
    """Match FPL club names to Football-Data names (e.g. "Man Utd" → "Man United")."""
    value = str(name).lower().replace("&", "and").strip()
    value = re.sub(r"\butd\b", "united", value)
    value = _ALIASES.get(value, value)
    value = re.sub(r"^afc\s+|\s+(city|town|fc|afc)$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def model_goal_rates(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    model: object,
    clubs: pd.DataFrame,
) -> pd.DataFrame:
    """Score FPL fixtures with a fitted goal model, matching clubs by normalised name.

    ``model`` needs ``rates(home, away)`` and ``teams``; ``clubs`` maps its ``club_id``
    to ``name``. Fixtures with a club the model has not seen are left out so the
    caller's season-xG fallback applies instead of a neutral guess.
    """
    known = set(getattr(model, "teams", ()))
    lookup = {
        normalise_club(name): club
        for club, name in clubs[["club_id", "name"]].itertuples(index=False)
        if club in known
    }
    names = teams.set_index("code")["name"].to_dict()
    rows = []
    for match in matches.itertuples(index=False):
        home = lookup.get(normalise_club(names.get(match.home_team, "")))
        away = lookup.get(normalise_club(names.get(match.away_team, "")))
        if home is None or away is None:
            continue
        home_rate, away_rate = model.rates(home, away)
        rows.append(
            {"match_id": str(match.match_id), "home_rate": home_rate, "away_rate": away_rate}
        )
    return pd.DataFrame(rows, columns=["match_id", "home_rate", "away_rate"])


def _prior_frames(config: dict, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Player stats and identities of the latest earlier season with cumulative stats."""
    for season in reversed(config["seasons"][:-1]):
        files = season["files"]
        if {"playerstats", "players"} <= {entry["kind"] for entry in files}:
            return (
                _read_frame(raw_dir, _one(files, "playerstats"), source_kind="playerstats"),
                _read_frame(raw_dir, _one(files, "players"), source_kind="players"),
            )
    return None


def _prior_rates(config: dict, raw_dir: Path) -> pd.DataFrame | None:
    """Final rates of the most recent earlier season that publishes cumulative stats."""
    frames = _prior_frames(config, raw_dir)
    return None if frames is None else prior_season_rates(*frames)


def manager_summary(entry_id: int, entry: dict, event: int) -> dict:
    """The public team facts the dashboard keeps: no manager name, private leagues only."""
    classic = (entry.get("leagues") or {}).get("classic") or []
    return {
        "entry_id": entry_id,
        "team_name": entry.get("name"),
        "event": event,
        "bank": entry.get("last_deadline_bank"),
        "team_value": entry.get("last_deadline_value"),
        "overall_points": entry.get("summary_overall_points"),
        "overall_rank": entry.get("summary_overall_rank"),
        "leagues": [
            {
                "id": int(league["id"]),
                "name": str(league.get("name", "")),
                "entry_rank": league.get("entry_rank"),
                "rank_count": league.get("rank_count"),
            }
            for league in classic
            if league.get("league_type") == "x" and league.get("id") is not None
        ],
    }


def _get_json(client: httpx.Client, url: str) -> dict:
    try:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ValueError("The public FPL data could not be loaded") from error
    if not isinstance(payload, dict):
        raise ValueError("The public FPL response has an invalid format")
    return payload


def fetch_league_standings(
    *, league_id: int, page: int = 1, client: httpx.Client | None = None
) -> tuple[str, pd.DataFrame]:
    """One page (up to 50 teams) of a classic league table, without manager names."""
    if league_id < 1 or page < 1:
        raise ValueError("League and page identifiers must be positive")
    if client is None:
        with httpx.Client(timeout=15, follow_redirects=True) as owned:
            return fetch_league_standings(league_id=league_id, page=page, client=owned)
    payload = _get_json(
        client,
        f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
        f"?page_standings={page}",
    )
    rows = (payload.get("standings") or {}).get("results") or []
    standings = pd.DataFrame(rows)
    required = {"entry", "entry_name", "rank", "total"}
    if standings.empty or not required <= set(standings):
        raise ValueError("The league table is empty or unavailable")
    columns = [*required, *({"event_total"} & set(standings))]
    return str((payload.get("league") or {}).get("name", "")), standings[sorted(columns)]


def fetch_league_picks(
    *, entries: list[int], event: int, client: httpx.Client | None = None
) -> pd.DataFrame:
    """Gameweek picks for each listed team; a team that cannot be read is skipped."""
    if event < 1 or not entries or len(entries) > MAX_LEAGUE_TEAMS:
        raise ValueError(f"Between 1 and {MAX_LEAGUE_TEAMS} teams and a valid gameweek required")
    if client is None:
        with httpx.Client(timeout=15, follow_redirects=True) as owned:
            return fetch_league_picks(entries=entries, event=event, client=owned)
    frames = []
    for entry in entries:
        try:
            payload = _get_json(
                client,
                f"https://fantasy.premierleague.com/api/entry/{int(entry)}/event/{event}/picks/",
            )
        except ValueError:
            continue
        picks = pd.DataFrame(payload.get("picks", []))
        if {"element", "multiplier", "is_captain"} <= set(picks) and len(picks) == 15:
            frames.append(picks[["element", "multiplier", "is_captain"]].assign(entry=int(entry)))
    if not frames:
        raise ValueError("No squad in this league could be loaded")
    return pd.concat(frames, ignore_index=True)


def _read_frame(raw_dir: Path, entry: dict, *, source_kind: str) -> pd.DataFrame:
    sidecar = raw_dir / f"fpl-core.{source_kind}" / entry["sha256"] / "metadata.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing archived FPL {source_kind} snapshot")
    snapshot = Snapshot.read(sidecar)
    if snapshot.sha256 != entry["sha256"]:
        raise ValueError("FPL snapshot checksum mismatch")
    return pd.read_csv(snapshot.path)


def _one(files: list[dict], kind: str) -> dict:
    entries = [entry for entry in files if entry["kind"] == kind]
    if len(entries) != 1:
        raise ValueError(f"Expected one FPL {kind} snapshot")
    return entries[0]


def download_manager_team(
    store: SnapshotStore,
    *,
    entry_id: int,
    event: int,
    client: httpx.Client | None = None,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Archive and parse one public manager entry and its latest completed picks."""
    if client is None:
        with httpx.Client(timeout=30, follow_redirects=True) as owned:
            return download_manager_team(store, entry_id=entry_id, event=event, client=owned)
    note = "Official Fantasy Premier League public endpoint; subject to Premier League terms"
    sources = []

    def fetch(url: str, source: str) -> dict:
        snapshot = store.download(
            url,
            source_name=source,
            dataset_version=f"event-{event}",
            license_note=note,
            client=client,
        )
        sources.append(
            {"source": source, "url": url, "sha256": snapshot.sha256, "event": event}
        )
        value = json.loads(snapshot.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("FPL manager response is not an object")
        return value

    entry = fetch(
        f"https://fantasy.premierleague.com/api/entry/{entry_id}/", "fpl.entry"
    )
    picks_payload = fetch(
        f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event}/picks/",
        "fpl.picks",
    )
    picks = pd.DataFrame(picks_payload.get("picks", []))
    required = {"element", "position", "multiplier", "is_captain", "is_vice_captain"}
    if picks.empty or not required <= set(picks) or picks.element.duplicated().any():
        raise ValueError("FPL manager picks are missing or ambiguous")
    return entry, picks, sources


def fetch_public_team(
    *,
    entry_id: int,
    event: int,
    client: httpx.Client | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fetch a public FPL team for an explicit interactive request without persisting it."""
    if entry_id < 1 or event < 1:
        raise ValueError("FPL entry and event identifiers must be positive")
    if client is None:
        with httpx.Client(timeout=15, follow_redirects=True) as owned:
            return fetch_public_team(entry_id=entry_id, event=event, client=owned)
    try:
        entry_response = client.get(f"https://fantasy.premierleague.com/api/entry/{entry_id}/")
        entry_response.raise_for_status()
        picks_response = client.get(
            f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event}/picks/"
        )
        picks_response.raise_for_status()
        entry = entry_response.json()
        picks_payload = picks_response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ValueError("The public FPL team could not be loaded") from error
    if not isinstance(entry, dict) or not isinstance(picks_payload, dict):
        raise ValueError("The public FPL response has an invalid format")
    picks = pd.DataFrame(picks_payload.get("picks", []))
    required = {"element", "position", "multiplier", "is_captain", "is_vice_captain"}
    if len(picks) != 15 or not required <= set(picks) or picks.element.duplicated().any():
        raise ValueError("The public FPL squad is missing or ambiguous")
    return entry, picks


def map_public_squad(picks: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Map 15 public picks to prepared players without colliding position fields."""
    squad = picks.rename(columns={"position": "squad_position"}).merge(
        players,
        left_on="element",
        right_on="source_player_id",
        validate="one_to_one",
    )
    if len(squad) != 15:
        raise ValueError("Some squad players are absent from current artifacts")
    return squad


def build_fpl_artifacts(
    config: dict,
    raw_dir: Path,
    destination: Path,
    *,
    entry_id: int | None = None,
    goal_model: object | None = None,
    model_clubs: pd.DataFrame | None = None,
) -> dict:
    """Build read-only FPL planning tables from one resolved current source commit.

    ``goal_model`` and ``model_clubs`` (the live match model and its club names) make
    fixture difficulty opponent-aware; without them season xG strengths are used.
    """
    season = config["seasons"][-1]
    files = season["files"]
    players_source = _read_frame(raw_dir, _one(files, "players"), source_kind="players")
    teams_source = _read_frame(raw_dir, _one(files, "teams"), source_kind="teams")
    playerstats = _read_frame(raw_dir, _one(files, "playerstats"), source_kind="playerstats")
    appearance_frames = []
    for entry in (item for item in files if item["kind"] == "appearances"):
        found = re.search(r"/GW(\d+)/", entry["path"])
        if found is None:
            raise ValueError("Cannot determine the FPL appearance gameweek")
        frame = _read_frame(raw_dir, entry, source_kind="appearances")
        frame["gameweek"] = int(found.group(1))
        appearance_frames.append(frame)
    if not appearance_frames:
        raise ValueError("Current FPL appearance history is missing")
    appearances = pd.concat(appearance_frames, ignore_index=True)
    fixture_sources = [
        _read_frame(raw_dir, entry, source_kind="matches")
        for entry in season.get("fixture_files", [])
    ]
    if not fixture_sources:
        raise ValueError("Current FPL fixture snapshots are missing")
    matches = pd.concat(fixture_sources, ignore_index=True)
    matches = matches.loc[matches.tournament.eq(config["source_tournament_codes"][0])]
    if matches.match_id.duplicated().any():
        raise ValueError("Duplicate current FPL fixtures")
    players, history = latest_players(
        playerstats,
        players_source,
        teams_source,
        competition_id=config["competition_id"],
        season=season["season"],
        position_map=config["position_map"],
    )
    goal_rates = (
        model_goal_rates(matches, teams_source, goal_model, model_clubs)
        if goal_model is not None and model_clubs is not None
        else None
    )
    fixtures = fixture_rows(matches, teams_source, goal_rates)
    prior_frames = _prior_frames(config, raw_dir)
    priors = None if prior_frames is None else prior_season_rates(*prior_frames)
    # The earlier, complete season gives the price watch far more gameweeks to score.
    price_history = playerstats if prior_frames is None else prior_frames[0]
    standings = league_table(
        matches,
        teams_source,
        competition_id=config["competition_id"],
        season=season["season"],
    )
    results = match_results(matches, teams_source)
    home_rows = fixtures.loc[fixtures.venue.eq("H")]
    simulation = simulate_season(
        standings[["club_id", "points", "goals_for", "goals_against"]],
        pd.DataFrame(
            {
                "home_club_id": home_rows.club_id,
                "away_club_id": home_rows.opponent_id,
                "home_rate": home_rows.expected_goals_for,
                "away_rate": home_rows.expected_goals_against,
            }
        ),
        rho=float(getattr(goal_model, "rho", 0.0)),
    ).merge(
        standings[["club_id", "club_name", "club_short_name", "competition_id", "season"]],
        on="club_id",
        validate="one_to_one",
    )
    projections = project_players(
        players, history, fixtures, appearances=appearances, priors=priors
    )
    backtest = rolling_backtest(
        playerstats,
        players_source,
        teams_source,
        matches,
        appearances,
        position_map=config["position_map"],
        competition_id=config["competition_id"],
        season=season["season"],
        priors=priors,
    )
    backtest_report = backtest_summary(backtest)
    destination.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(destination / "players.parquet", index=False)
    fixtures.to_parquet(destination / "fixtures.parquet", index=False)
    standings.to_parquet(destination / "standings.parquet", index=False)
    results.to_parquet(destination / "results.parquet", index=False)
    simulation.to_parquet(destination / "season_simulation.parquet", index=False)
    backtest.to_parquet(destination / "backtest.parquet", index=False)
    (destination / "backtest.json").write_text(
        json.dumps(backtest_report, indent=2) + "\n", encoding="utf-8"
    )
    manager = None
    manager_sources: list[dict] = []
    squad_path = destination / "squad.parquet"
    if entry_id is not None:
        event = int(players.gw.max())
        entry, picks, manager_sources = download_manager_team(
            SnapshotStore(raw_dir), entry_id=entry_id, event=event
        )
        squad = map_public_squad(picks, projections)
        squad.to_parquet(squad_path, index=False)
        manager = manager_summary(entry_id, entry, event)
    elif squad_path.exists():
        squad_path.unlink()
    metadata = {
        "competition_id": config["competition_id"],
        "season": season["season"],
        "source_commit": config["commit"],
        "reference_date": config["reference_date"],
        "current_gameweek": int(players.gw.max()),
        "next_gameweek": int(fixtures.gameweek.min()),
        "players": len(projections),
        "future_fixtures": fixtures.match_id.nunique(),
        "manager": manager,
        "manager_sources": manager_sources,
        "method": "recency-events-goalmodel-v3",
        "fixture_rate_sources": fixtures.rate_source.value_counts().to_dict(),
        "prior_season_players": 0 if priors is None else int(len(priors)),
        "backtest": backtest_report,
        "price_watch": price_watch_backtest(price_history),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
