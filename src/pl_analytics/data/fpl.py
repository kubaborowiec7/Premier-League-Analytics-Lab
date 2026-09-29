"""Offline FPL Decision Lab artifact builder from archived public snapshots."""

import json
from pathlib import Path

import httpx
import pandas as pd

from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.features.fpl import fixture_rows, latest_players, project_players
from pl_analytics.features.fpl_optimizer import backtest_summary, rolling_backtest


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
) -> dict:
    """Build read-only FPL planning tables from one resolved current source commit."""
    season = config["seasons"][-1]
    files = season["files"]
    players_source = _read_frame(raw_dir, _one(files, "players"), source_kind="players")
    teams_source = _read_frame(raw_dir, _one(files, "teams"), source_kind="teams")
    playerstats = _read_frame(raw_dir, _one(files, "playerstats"), source_kind="playerstats")
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
    fixtures = fixture_rows(matches, teams_source)
    projections = project_players(players, history, fixtures)
    backtest = rolling_backtest(
        playerstats,
        players_source,
        position_map=config["position_map"],
        competition_id=config["competition_id"],
        season=season["season"],
    )
    backtest_report = backtest_summary(backtest)
    destination.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(destination / "players.parquet", index=False)
    fixtures.to_parquet(destination / "fixtures.parquet", index=False)
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
        manager = {
            "entry_id": entry_id,
            "team_name": entry.get("name"),
            "event": event,
            "bank": entry.get("last_deadline_bank"),
            "team_value": entry.get("last_deadline_value"),
            "overall_points": entry.get("summary_overall_points"),
            "overall_rank": entry.get("summary_overall_rank"),
        }
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
        "method": "transparent-rate-elo-v1",
        "backtest": backtest_report,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
