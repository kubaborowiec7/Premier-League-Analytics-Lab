"""Transparent Fantasy Premier League planning features and projection baseline."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

HORIZONS = (1, 2, 3, 4, 5)
GOAL_POINTS = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CLEAN_SHEET_POINTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
RECENCY_DECAY = 0.65
# A quarter-match position prior: stronger priors under-rated nailed starters in the backtest.
MINUTES_PRIOR_STRENGTH = 0.25
SPREAD_PRIOR_WEEKS = 10.0
SUBSTITUTE_MINUTES = 20.0
DEFAULT_HOME_GOALS = 1.5
DEFAULT_AWAY_GOALS = 1.2
STRENGTH_PRIOR_MATCHES = 6.0
# Ascending edges for the 1 (easiest) â€“ 5 (hardest) fixture bands.
OVERALL_DIFFICULTY_EDGES = (-0.75, -0.25, 0.25, 0.75)  # expected goals against â’ for
ATTACK_DIFFICULTY_EDGES = (-1.9, -1.55, -1.25, -1.0)  # â’expected goals for
DEFENCE_DIFFICULTY_EDGES = (0.9, 1.15, 1.45, 1.75)  # expected goals against
# Last season's minutes count for half, capped so one season cannot dominate current form.
PRIOR_SEASON_WEIGHT = 0.5
PRIOR_SEASON_MINUTES_CAP = 1800.0
PRIOR_RATE_COLUMNS = {
    "xg_per90": "expected_goals_per_90",
    "xa_per90": "expected_assists_per_90",
    "defensive_contribution_per90": "defensive_contribution_per_90",
    "saves_per_90": "saves_per_90",
}
TOTAL_RATE_TARGETS = {
    "penalties_saved": "projected_penalty_saves_per90",
    "bonus": "projected_bonus_per90",
    "yellow_cards": "projected_yellow_cards_per90",
    "red_cards": "projected_red_cards_per90",
    "own_goals": "projected_own_goals_per90",
    "penalties_missed": "projected_penalties_missed_per90",
}


def latest_players(
    playerstats: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    competition_id: str,
    season: str,
    position_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a validated current player snapshot and retain prior gameweek points."""
    required_stats = {
        "id",
        "gw",
        "now_cost",
        "total_points",
        "event_points",
        "minutes",
        "starts",
    }
    required_players = {
        "player_id",
        "player_code",
        "first_name",
        "second_name",
        "position",
        "team_code",
    }
    required_teams = {"code", "name", "short_name", "elo"}
    if not required_stats <= set(playerstats):
        raise ValueError("FPL playerstats fields are missing")
    if not required_players <= set(players) or not required_teams <= set(teams):
        raise ValueError("FPL identity fields are missing")
    stats = playerstats.copy()
    stats["gw"] = pd.to_numeric(stats.gw, errors="raise").astype(int)
    if stats[["id", "gw"]].duplicated().any() or stats[["id", "gw"]].isna().any().any():
        raise ValueError("Duplicate or missing FPL player-gameweek identity")
    history = stats[["id", "gw", "event_points"]].rename(
        columns={"id": "source_player_id", "event_points": "points"}
    )
    history["points"] = pd.to_numeric(history.points, errors="coerce")
    latest = stats.sort_values("gw").drop_duplicates("id", keep="last")
    identity = players[
        ["player_id", "player_code", "first_name", "second_name", "position", "team_code"]
    ]
    clubs = teams[["code", "name", "short_name", "elo"]].rename(
        columns={"code": "team_code", "name": "club_name", "short_name": "club_short_name"}
    )
    identity_columns = set(identity) - {"player_id"}
    latest = latest.drop(columns=list(identity_columns & set(latest.columns)))
    latest = latest.merge(identity, left_on="id", right_on="player_id", validate="one_to_one")
    latest = latest.drop(columns=list({"name", "short_name", "elo"} & set(latest.columns)))
    latest = latest.merge(clubs, on="team_code", validate="many_to_one")
    latest["source_player_id"] = latest["id"].astype(int)
    latest["player_id"] = "fpl:" + latest.player_code.astype(int).astype(str)
    latest["player_name"] = (
        latest.first_name.fillna("").str.strip() + " " + latest.second_name.fillna("").str.strip()
    ).str.strip()
    latest["club_id"] = "fpl:club:" + latest.team_code.astype(int).astype(str)
    latest["position_group"] = latest.position.map(position_map).fillna("UNKNOWN")
    latest["competition_id"] = competition_id
    latest["season"] = season
    numeric = {
        "now_cost": "price",
        "total_points": "total_points",
        "event_points": "last_gw_points",
        "points_per_game": "points_per_game",
        "form": "form",
        "selected_by_percent": "ownership_pct",
        "ep_next": "provider_ep_next",
        "expected_goals_per_90": "xg_per90",
        "expected_assists_per_90": "xa_per90",
        "expected_goal_involvements_per_90": "xgi_per90",
        "minutes": "minutes",
        "starts": "starts",
        "defensive_contribution_per_90": "defensive_contribution_per90",
        "chance_of_playing_next_round": "chance_of_playing",
        "transfers_in_event": "transfers_in_event",
        "transfers_out_event": "transfers_out_event",
        "elo": "club_elo",
    }
    for source, target in numeric.items():
        latest[target] = pd.to_numeric(
            latest.get(source, pd.Series(np.nan, index=latest.index)), errors="coerce"
        )
    if latest.price.dropna().median() > 20:
        latest["price"] /= 10
    latest["net_transfers_event"] = latest.transfers_in_event - latest.transfers_out_event
    status = latest.get("status", pd.Series("a", index=latest.index))
    latest["status"] = status.fillna("a").astype(str)
    news = latest.get("news", pd.Series("", index=latest.index))
    latest["news"] = news.fillna("").astype(str)
    return latest, history


def _finished(matches: pd.DataFrame) -> pd.Series:
    return matches.finished.astype(str).str.lower().isin({"true", "1"})


def season_goal_rates(matches: pd.DataFrame) -> pd.DataFrame:
    """Estimate each scheduled match's goal rates from this season's completed matches.

    Club attack and defence ratios use xG (goals where xG is absent), shrunk toward the
    league average with the weight of ``STRENGTH_PRIOR_MATCHES`` matches. Only completed
    rows inform the ratios, so the caller controls leakage by what it marks finished.
    """
    done = matches.loc[_finished(matches)] if "finished" in matches else matches.iloc[0:0]
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        columns = {}
        for role, source in (("for", side), ("against", other)):
            xg = pd.to_numeric(
                done.get(f"{source}_expected_goals_xg", pd.Series(np.nan, index=done.index)),
                errors="coerce",
            )
            goals = pd.to_numeric(
                done.get(f"{source}_score", pd.Series(np.nan, index=done.index)), errors="coerce"
            )
            columns[role] = xg.fillna(goals)
        sides.append(
            pd.DataFrame({"club": done.get(f"{side}_team"), "venue": side, **columns}).dropna()
        )
    long = pd.concat(sides, ignore_index=True)
    home_mean = float(long.loc[long.venue.eq("home"), "for"].mean()) if len(long) else np.nan
    away_mean = float(long.loc[long.venue.eq("away"), "for"].mean()) if len(long) else np.nan
    home_mean = home_mean if np.isfinite(home_mean) and home_mean > 0 else DEFAULT_HOME_GOALS
    away_mean = away_mean if np.isfinite(away_mean) and away_mean > 0 else DEFAULT_AWAY_GOALS
    mean = (home_mean + away_mean) / 2
    prior = STRENGTH_PRIOR_MATCHES * mean
    totals = long.groupby("club").agg(played=("for", "size"), scored=("for", "sum"),
                                      conceded=("against", "sum"))
    attack = ((totals.scored + prior) / (totals.played * mean + prior)).to_dict()
    defence = ((totals.conceded + prior) / (totals.played * mean + prior)).to_dict()
    return pd.DataFrame(
        {
            "match_id": matches.match_id.astype(str),
            "home_rate": [
                home_mean * attack.get(home, 1.0) * defence.get(away, 1.0)
                for home, away in zip(matches.home_team, matches.away_team, strict=True)
            ],
            "away_rate": [
                away_mean * attack.get(away, 1.0) * defence.get(home, 1.0)
                for home, away in zip(matches.home_team, matches.away_team, strict=True)
            ],
        }
    )


def fixture_rows(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    goal_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand scheduled matches to one leakage-safe planning row per club and fixture.

    ``goal_rates`` (``match_id``, ``home_rate``, ``away_rate``) normally comes from the
    live goal model. Matches it does not cover fall back to :func:`season_goal_rates`.
    """
    required = {"gameweek", "kickoff_time", "home_team", "away_team", "finished", "match_id"}
    if not required <= set(matches):
        raise ValueError("FPL fixture fields are missing")
    club_lookup = teams.set_index("code")
    rates = season_goal_rates(matches).set_index("match_id")
    rates["rate_source"] = "season-xg"
    if goal_rates is not None and not goal_rates.empty:
        supplied = goal_rates.assign(match_id=goal_rates.match_id.astype(str)).set_index(
            "match_id"
        )[["home_rate", "away_rate"]]
        supplied = supplied.loc[supplied.index.isin(rates.index)].dropna()
        rates.loc[supplied.index, ["home_rate", "away_rate"]] = supplied
        rates.loc[supplied.index, "rate_source"] = "goal-model"
    rows: list[dict] = []
    for match in matches.itertuples(index=False):
        if str(match.finished).lower() in {"true", "1"}:
            continue
        rate = rates.loc[str(match.match_id)]
        for venue, team_code, opponent_code, scored, conceded in (
            ("H", match.home_team, match.away_team, rate.home_rate, rate.away_rate),
            ("A", match.away_team, match.home_team, rate.away_rate, rate.home_rate),
        ):
            if team_code not in club_lookup.index or opponent_code not in club_lookup.index:
                raise ValueError("Fixture references an unknown FPL club")
            club = club_lookup.loc[team_code]
            opponent = club_lookup.loc[opponent_code]
            scored, conceded = float(scored), float(conceded)
            rows.append(
                {
                    "match_id": str(match.match_id),
                    "gameweek": int(float(match.gameweek)),
                    "kickoff_time": pd.to_datetime(match.kickoff_time, utc=True),
                    "club_id": f"fpl:club:{int(float(team_code))}",
                    "club_name": club["name"],
                    "club_short_name": club["short_name"],
                    "opponent_id": f"fpl:club:{int(float(opponent_code))}",
                    "opponent_name": opponent["name"],
                    "opponent_short_name": opponent["short_name"],
                    "venue": venue,
                    "expected_goals_for": scored,
                    "expected_goals_against": conceded,
                    "difficulty": _band(conceded - scored, OVERALL_DIFFICULTY_EDGES),
                    "attack_difficulty": _band(-scored, ATTACK_DIFFICULTY_EDGES),
                    "defence_difficulty": _band(conceded, DEFENCE_DIFFICULTY_EDGES),
                    "clean_sheet_probability": float(np.clip(np.exp(-conceded), 0.02, 0.75)),
                    "rate_source": rate.rate_source,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["match_id", "club_id"]).any():
        raise ValueError("FPL fixture schedule is empty or ambiguous")
    # Player per-90 rates already contain their club's typical attack, so a fixture scales
    # them by its expected goals relative to that club's average over the listed schedule.
    typical = result.groupby("club_id").expected_goals_for.transform("mean")
    result["attack_multiplier"] = (result.expected_goals_for / typical).clip(0.6, 1.6)
    return result.sort_values(["gameweek", "kickoff_time", "club_name"]).reset_index(drop=True)


def _band(value: float, edges: tuple[float, ...]) -> int:
    """Map a value to a 1 (easiest) â€“ 5 (hardest) band using ascending edges."""
    return int(np.digitize(value, edges) + 1)


def match_results(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Return completed matches, one row each, with scores, xG and club identities."""
    required = {
        "match_id",
        "gameweek",
        "kickoff_time",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "finished",
    }
    if not required <= set(matches) or not {"code", "name", "short_name"} <= set(teams):
        raise ValueError("Match result fields are missing")
    completed = matches.loc[matches.finished.astype(str).str.lower().isin({"true", "1"})]
    clubs = teams.set_index("code")
    rows = []
    for match in completed.itertuples(index=False):
        home, away = int(float(match.home_team)), int(float(match.away_team))
        if home not in clubs.index or away not in clubs.index:
            raise ValueError("Result references an unknown FPL club")
        rows.append(
            {
                "match_id": str(match.match_id),
                "gameweek": int(float(match.gameweek)),
                "kickoff_time": pd.to_datetime(match.kickoff_time, utc=True),
                "home_club_id": f"fpl:club:{home}",
                "home_name": clubs.loc[home, "name"],
                "home_short_name": clubs.loc[home, "short_name"],
                "away_club_id": f"fpl:club:{away}",
                "away_name": clubs.loc[away, "name"],
                "away_short_name": clubs.loc[away, "short_name"],
                "home_score": int(float(match.home_score)),
                "away_score": int(float(match.away_score)),
                "home_xg": pd.to_numeric(
                    getattr(match, "home_expected_goals_xg", np.nan), errors="coerce"
                ),
                "away_xg": pd.to_numeric(
                    getattr(match, "away_expected_goals_xg", np.nan), errors="coerce"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if result.match_id.duplicated().any():
        raise ValueError("Completed match results are duplicated")
    return result.sort_values(["kickoff_time", "home_name"]).reset_index(drop=True)


def league_table(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    competition_id: str,
    season: str,
) -> pd.DataFrame:
    """Build observed standings plus xG-derived expected points from completed matches."""
    required_matches = {
        "match_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_expected_goals_xg",
        "away_expected_goals_xg",
        "finished",
    }
    if not required_matches <= set(matches) or not {"code", "name", "short_name"} <= set(teams):
        raise ValueError("League-table source fields are missing")
    completed = matches.loc[matches.finished.astype(str).str.lower().isin({"true", "1"})].copy()
    if completed.empty or completed.match_id.duplicated().any():
        raise ValueError("Completed league matches are empty or duplicated")
    for column in (
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_expected_goals_xg",
        "away_expected_goals_xg",
    ):
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
    if completed[["home_team", "away_team", "home_score", "away_score"]].isna().any().any():
        raise ValueError("Completed league match results are incomplete")

    rows: list[dict[str, float | int]] = []
    score_counts = np.arange(13)
    for match in completed.itertuples(index=False):
        home_goals, away_goals = int(match.home_score), int(match.away_score)
        home_xg = float(match.home_expected_goals_xg)
        away_xg = float(match.away_expected_goals_xg)
        home_xpoints = away_xpoints = np.nan
        if np.isfinite(home_xg) and np.isfinite(away_xg):
            matrix = np.outer(
                poisson.pmf(score_counts, home_xg), poisson.pmf(score_counts, away_xg)
            )
            matrix /= matrix.sum()
            draw = float(np.trace(matrix))
            home_win = float(np.tril(matrix, -1).sum())
            away_win = float(np.triu(matrix, 1).sum())
            home_xpoints = 3 * home_win + draw
            away_xpoints = 3 * away_win + draw
        for venue, club, goals_for, goals_against, xg_for, xg_against, xpoints in (
            (
                "H",
                int(match.home_team),
                home_goals,
                away_goals,
                home_xg,
                away_xg,
                home_xpoints,
            ),
            (
                "A",
                int(match.away_team),
                away_goals,
                home_goals,
                away_xg,
                home_xg,
                away_xpoints,
            ),
        ):
            rows.append(
                {
                    "club_code": club,
                    "venue": venue,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "win": int(goals_for > goals_against),
                    "draw": int(goals_for == goals_against),
                    "loss": int(goals_for < goals_against),
                    "points": 3 * int(goals_for > goals_against) + int(goals_for == goals_against),
                    "expected_goals": xg_for,
                    "expected_goals_against": xg_against,
                    "expected_points": xpoints,
                }
            )
    long = pd.DataFrame(rows)
    table = long.groupby("club_code", as_index=False).agg(
        played=("club_code", "size"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        points=("points", "sum"),
        expected_goals=("expected_goals", "sum"),
        expected_goals_against=("expected_goals_against", "sum"),
        expected_points=("expected_points", lambda values: values.sum(min_count=1)),
        xg_matches=("expected_points", "count"),
    )
    clubs = teams[["code", "name", "short_name"]].rename(
        columns={"code": "club_code", "name": "club_name", "short_name": "club_short_name"}
    )
    table = table.merge(clubs, on="club_code", validate="one_to_one")
    table["goal_difference"] = table.goals_for - table.goals_against
    table["expected_goal_difference"] = table.expected_goals - table.expected_goals_against
    table["expected_points_difference"] = table.points - table.expected_points
    table["competition_id"] = competition_id
    table["season"] = season
    table["club_id"] = "fpl:club:" + table.club_code.astype(int).astype(str)
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "club_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "position", np.arange(1, len(table) + 1))
    return table


def _availability(row: pd.Series) -> float:
    chance = row.chance_of_playing
    if pd.notna(chance):
        return float(np.clip(chance / 100, 0, 1))
    return 1.0 if row.status == "a" else 0.5 if row.status == "d" else 0.0


def _posterior_probability(
    observations: pd.Series,
    weights: pd.Series,
    prior: float,
) -> float:
    successes = float((observations.astype(float) * weights).sum())
    return (successes + MINUTES_PRIOR_STRENGTH * prior) / (
        float(weights.sum()) + MINUTES_PRIOR_STRENGTH
    )


def expected_minutes_features(
    players: pd.DataFrame,
    appearances: pd.DataFrame | None,
    *,
    lookback: int = 6,
) -> pd.DataFrame:
    """Estimate start, appearance and 60-minute probabilities from recent matches.

    Recent fixtures receive exponentially greater weight. Position priors prevent one
    or two observations from producing certainty, while official availability remains
    an explicit final cap. The calculation uses only matches already completed.
    """
    required_players = {"source_player_id", "position_group", "minutes", "starts", "gw"}
    if not required_players <= set(players):
        raise ValueError("FPL minute-model player fields are missing")
    base = players.copy()
    base["availability_probability"] = base.apply(_availability, axis=1)
    if appearances is None or appearances.empty:
        games = base.gw.clip(lower=1)
        base["start_probability"] = (base.starts / games).clip(0, 1)
        base["appearance_probability"] = (base.minutes / (games * 60)).clip(0, 1)
        base["sixty_probability"] = (base.minutes / (games * 90)).clip(0, 1)
        base["expected_minutes"] = (base.minutes / games).clip(0, 90)
        for column in (
            "start_probability",
            "appearance_probability",
            "sixty_probability",
            "expected_minutes",
        ):
            base[column] *= base.availability_probability
        base["minutes_model_matches"] = games.astype(int)
        return base

    required = {"player_id", "match_id", "gameweek", "minutes_played", "start_min"}
    if not required <= set(appearances):
        raise ValueError("FPL appearance history fields are missing")
    history = appearances.copy()
    for column in ("player_id", "gameweek", "minutes_played", "start_min"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    if history[["player_id", "match_id", "gameweek"]].isna().any().any():
        raise ValueError("FPL appearance history has missing identities")
    if history.duplicated(["player_id", "match_id"]).any():
        raise ValueError("FPL appearance history has duplicate player-match rows")
    history = history.merge(
        base[["source_player_id", "position_group"]],
        left_on="player_id",
        right_on="source_player_id",
        how="inner",
        validate="many_to_one",
    )
    history["minutes_played"] = history.minutes_played.fillna(0).clip(0, 90)
    history["started"] = history.minutes_played.gt(0) & history.start_min.fillna(-1).eq(0)
    history["appeared"] = history.minutes_played.gt(0)
    history["sixty"] = history.minutes_played.ge(60)
    current_gw = int(base.gw.max())
    history = history.loc[history.gameweek.le(current_gw)].copy()
    history["age"] = (current_gw - history.gameweek).clip(lower=0)
    history = history.loc[history.age.lt(lookback)]
    history["weight"] = RECENCY_DECAY ** history.age

    priors: dict[str, dict[str, float]] = {}
    for position, frame in history.groupby("position_group"):
        started = frame.started.astype(float)
        appeared = frame.appeared.astype(float)
        sixty = frame.sixty.astype(float)
        start_rows = frame.loc[frame.started]
        sub_rows = frame.loc[frame.appeared & ~frame.started]
        priors[position] = {
            "start": float(started.mean()),
            "appear": float(appeared.mean()),
            "sixty": float(sixty.mean()),
            "start_minutes": (
                float(start_rows.minutes_played.mean()) if not start_rows.empty else 75
            ),
            "sub_minutes": float(sub_rows.minutes_played.mean()) if not sub_rows.empty else 15,
        }

    estimates: list[dict[str, float | int]] = []
    for player in base.itertuples(index=False):
        frame = history.loc[history.player_id.eq(player.source_player_id)].sort_values(
            "gameweek", ascending=False
        )
        prior = priors.get(
            player.position_group,
            {"start": 0.45, "appear": 0.60, "sixty": 0.40, "start_minutes": 75, "sub_minutes": 15},
        )
        weight = frame.weight
        start_probability = _posterior_probability(frame.started, weight, prior["start"])
        appearance_probability = max(
            start_probability,
            _posterior_probability(frame.appeared, weight, prior["appear"]),
        )
        sixty_probability = min(
            appearance_probability,
            _posterior_probability(frame.sixty, weight, prior["sixty"]),
        )
        starters = frame.loc[frame.started]
        substitutes = frame.loc[frame.appeared & ~frame.started]
        start_minutes = (
            float(
                (
                    (starters.minutes_played * starters.weight).sum()
                    + MINUTES_PRIOR_STRENGTH * prior["start_minutes"]
                )
                / (starters.weight.sum() + MINUTES_PRIOR_STRENGTH)
            )
            if not starters.empty
            else prior["start_minutes"]
        )
        sub_minutes = (
            float(
                (
                    (substitutes.minutes_played * substitutes.weight).sum()
                    + MINUTES_PRIOR_STRENGTH * prior["sub_minutes"]
                )
                / (substitutes.weight.sum() + MINUTES_PRIOR_STRENGTH)
            )
            if not substitutes.empty
            else prior["sub_minutes"]
        )
        expected = start_probability * start_minutes + max(
            0.0, appearance_probability - start_probability
        ) * sub_minutes
        availability = float(player.availability_probability)
        estimates.append(
            {
                "source_player_id": int(player.source_player_id),
                "start_probability": np.clip(start_probability * availability, 0, 1),
                "appearance_probability": np.clip(appearance_probability * availability, 0, 1),
                "sixty_probability": np.clip(sixty_probability * availability, 0, 1),
                "expected_minutes": np.clip(expected * availability, 0, 90),
                "minutes_model_matches": int(len(frame)),
            }
        )
    return base.drop(
        columns=[
            "start_probability",
            "appearance_probability",
            "sixty_probability",
            "expected_minutes",
        ],
        errors="ignore",
    ).merge(pd.DataFrame(estimates), on="source_player_id", validate="one_to_one")


def prior_season_rates(playerstats: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Return a completed earlier season's final per-90 rates keyed by stable player id.

    The result supplies player-specific priors, so an established player is not pulled
    toward the position median just because the new season is only a few matches old.
    """
    if not {"id", "gw", "minutes"} <= set(playerstats) or not {
        "player_id",
        "player_code",
    } <= set(players):
        raise ValueError("Prior-season FPL fields are missing")
    stats = playerstats.copy()
    stats["gw"] = pd.to_numeric(stats.gw, errors="coerce")
    final = stats.sort_values("gw").drop_duplicates("id", keep="last")
    final = final.merge(
        players[["player_id", "player_code"]], left_on="id", right_on="player_id", how="inner"
    )
    minutes = pd.to_numeric(final.minutes, errors="coerce").fillna(0)
    result = pd.DataFrame(
        {
            "player_id": "fpl:" + final.player_code.astype(int).astype(str),
            "prior_minutes": minutes,
        }
    )
    for target, source in PRIOR_RATE_COLUMNS.items():
        result[f"prior_{target}"] = pd.to_numeric(
            final.get(source, pd.Series(np.nan, index=final.index)), errors="coerce"
        ).to_numpy()
    for total, target in TOTAL_RATE_TARGETS.items():
        values = pd.to_numeric(
            final.get(total, pd.Series(np.nan, index=final.index)), errors="coerce"
        )
        result[f"prior_raw_{target}"] = (values * 90 / minutes.replace(0, np.nan)).to_numpy()
    result = result.loc[result.prior_minutes.gt(0)]
    return result.drop_duplicates("player_id", keep="last").reset_index(drop=True)


def _shrunk_per90(frame: pd.DataFrame, column: str, *, prior_minutes: float = 450) -> pd.Series:
    """Blend a per-90 rate with the player's previous season and the position median.

    Each source is weighted by minutes: this season's, a discounted and capped share of
    last season's (when ``prior_<column>`` is present), and ``prior_minutes`` of the
    position's minutes-weighted mean.
    """
    values = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
    # The anchor is the minutes-weighted position mean: a median would sit near zero for
    # skewed events (goals, bonus) and bias every low-minutes player downwards.
    weighted = (values.fillna(0) * frame.minutes).groupby(frame.position_group).transform("sum")
    exposure = frame.minutes.where(values.notna(), 0).groupby(frame.position_group).transform("sum")
    median = (weighted / exposure.replace(0, np.nan)).fillna(0)
    previous = pd.to_numeric(
        frame.get(f"prior_{column}", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    previous_minutes = pd.to_numeric(
        frame.get("prior_minutes", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0)
    previous_weight = (
        PRIOR_SEASON_WEIGHT * previous_minutes.clip(upper=PRIOR_SEASON_MINUTES_CAP)
    ).where(previous.notna(), 0.0)
    blended = (
        frame.minutes * values.fillna(median)
        + previous_weight * previous.fillna(0)
        + prior_minutes * median
    ) / (frame.minutes + previous_weight + prior_minutes)
    return blended.clip(lower=0)


def _expected_integer_awards(rate: float, divisor: int) -> float:
    """Return E[floor(N/divisor)] for a Poisson count N."""
    if rate <= 0:
        return 0.0
    upper = max(20, int(np.ceil(rate + 8 * np.sqrt(rate))))
    counts = np.arange(upper + 1)
    return float(np.sum((counts // divisor) * poisson.pmf(counts, rate)))


def project_players(
    players: pd.DataFrame,
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    appearances: pd.DataFrame | None = None,
    priors: pd.DataFrame | None = None,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Project simple expected FPL points from current rates and future fixture context.

    This deliberately transparent baseline uses only information available at the
    refresh origin. ``priors`` (see :func:`prior_season_rates`) adds player-specific
    previous-season rates. It is a planning estimate, not a claim of calibrated certainty.
    """
    result = expected_minutes_features(players, appearances)
    if priors is not None and not priors.empty:
        result = result.merge(priors, on="player_id", how="left", validate="many_to_one")
    reliability = result.minutes / (result.minutes + 450)
    result["reliability"] = reliability.clip(0, 1)
    observed_rate = result.total_points * 90 / result.minutes.replace(0, np.nan)
    active_prior = (
        observed_rate.where(result.minutes.gt(0))
        .groupby(result.position_group)
        .transform("median")
    )
    result["shrunk_points_per90"] = (
        reliability * observed_rate.fillna(active_prior)
        + (1 - reliability) * active_prior.fillna(2.0)
    )
    for source, target in (
        ("xg_per90", "projected_xg_per90"),
        ("xa_per90", "projected_xa_per90"),
        ("defensive_contribution_per90", "projected_defensive_contribution_per90"),
        ("saves_per_90", "projected_saves_per90"),
    ):
        result[target] = _shrunk_per90(result, source)
    for total, target in TOTAL_RATE_TARGETS.items():
        values = pd.to_numeric(result.get(total, pd.Series(0, index=result.index)), errors="coerce")
        result[f"raw_{target}"] = values.fillna(0) * 90 / result.minutes.replace(0, np.nan)
        result[target] = _shrunk_per90(result, f"raw_{target}", prior_minutes=900)
    spread = history.groupby("source_player_id").points.agg(["std", "count"])
    spread.columns = ["historical_points_sd", "points_sd_weeks"]
    result = result.merge(spread, on="source_player_id", how="left", validate="one_to_one")
    position_sd = (
        result.groupby("position_group")["historical_points_sd"].transform("median").fillna(2.0)
    )
    # A few gameweeks give a noisy spread (one haul inflates it), so pool the variance with
    # the position's typical spread, weighted as SPREAD_PRIOR_WEEKS gameweeks.
    weeks = result.points_sd_weeks.fillna(0)
    own = result.historical_points_sd.fillna(position_sd)
    result["historical_points_sd"] = np.sqrt(
        (weeks * own**2 + SPREAD_PRIOR_WEEKS * position_sd**2) / (weeks + SPREAD_PRIOR_WEEKS)
    )
    result = result.drop(columns="points_sd_weeks")

    future_gameweeks = sorted(fixtures.gameweek.unique())
    horizon_values = sorted(set(int(value) for value in horizons))
    for horizon in horizon_values:
        gameweeks = future_gameweeks[:horizon]
        selected_fixtures = fixtures.loc[fixtures.gameweek.isin(gameweeks)]
        by_club = {club: frame for club, frame in selected_fixtures.groupby("club_id")}
        no_fixture = selected_fixtures.iloc[0:0]
        points, variances, labels = [], [], []
        component_rows: list[dict[str, float]] = []
        for player in result.itertuples(index=False):
            schedule = by_club.get(player.club_id, no_fixture)
            projected = 0.0
            variance = 0.0
            descriptions = []
            components = {
                "appearance": 0.0,
                "attack": 0.0,
                "clean_sheet": 0.0,
                "saves": 0.0,
                "defensive": 0.0,
                "bonus": 0.0,
                "discipline": 0.0,
                "goals_conceded": 0.0,
            }
            for fixture in schedule.itertuples(index=False):
                minutes_factor = player.expected_minutes / 90
                appearance = player.appearance_probability + player.sixty_probability
                goal_points = GOAL_POINTS.get(player.position_group, 4.0)
                clean_points = CLEAN_SHEET_POINTS.get(player.position_group, 0.0)
                attack = minutes_factor * fixture.attack_multiplier * (
                    player.projected_xg_per90 * goal_points
                    + player.projected_xa_per90 * 3
                )
                clean_sheet = (
                    player.sixty_probability
                    * fixture.clean_sheet_probability
                    * clean_points
                )
                threshold = 10 if player.position_group == "DEF" else 12
                defensive = 0.0
                if player.position_group != "GK":
                    actions = player.projected_defensive_contribution_per90
                    defensive = 2 * poisson.sf(threshold - 1, actions * minutes_factor)
                saves = 0.0
                penalty_saves = 0.0
                if player.position_group == "GK":
                    saves = _expected_integer_awards(
                        player.projected_saves_per90 * minutes_factor, 3
                    )
                    penalty_saves = 5 * player.projected_penalty_saves_per90 * minutes_factor
                bonus = player.projected_bonus_per90 * minutes_factor
                discipline = -minutes_factor * (
                    player.projected_yellow_cards_per90
                    + 3 * player.projected_red_cards_per90
                    + 2 * player.projected_own_goals_per90
                    + 2 * player.projected_penalties_missed_per90
                )
                goals_conceded = 0.0
                if player.position_group in {"GK", "DEF"}:
                    conceded_rate = max(
                        0.01,
                        getattr(
                            fixture,
                            "expected_goals_against",
                            -np.log(fixture.clean_sheet_probability),
                        ),
                    )
                    goals_conceded = -_expected_integer_awards(conceded_rate * minutes_factor, 2)
                fixture_points = (
                    appearance
                    + attack
                    + clean_sheet
                    + saves
                    + penalty_saves
                    + defensive
                    + bonus
                    + discipline
                    + goals_conceded
                )
                projected += float(fixture_points)
                variance += float((player.historical_points_sd * minutes_factor) ** 2)
                descriptions.append(
                    f"GW{fixture.gameweek} {fixture.opponent_short_name} ({fixture.venue})"
                )
                components["appearance"] += float(appearance)
                components["attack"] += float(attack)
                components["clean_sheet"] += float(clean_sheet)
                components["saves"] += float(saves + penalty_saves)
                components["defensive"] += float(defensive)
                components["bonus"] += float(bonus)
                components["discipline"] += float(discipline)
                components["goals_conceded"] += float(goals_conceded)
            points.append(projected)
            variances.append(variance)
            labels.append(" Â· ".join(descriptions) if descriptions else "Blank")
            component_rows.append(components)
        result[f"projected_points_{horizon}"] = points
        spread = 1.28 * np.sqrt(variances)
        result[f"projection_low_{horizon}"] = np.maximum(0, np.asarray(points) - spread)
        result[f"projection_high_{horizon}"] = np.asarray(points) + spread
        result[f"fixtures_{horizon}"] = labels
        if horizon == 1:
            for component in component_rows[0] if component_rows else ():
                result[f"xpts_{component}_1"] = [row[component] for row in component_rows]
    previous = np.zeros(len(result))
    for horizon, gameweek in enumerate(future_gameweeks[: max(horizon_values)], start=1):
        cumulative = result[f"projected_points_{horizon}"].to_numpy(float)
        result[f"projected_points_gw_{int(gameweek)}"] = np.maximum(0, cumulative - previous)
        previous = cumulative
    return result


def price_watch(players: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Players under the most buying and selling pressure this gameweek.

    Ranks by net transfers in the current gameweek. It is a watch list, not a model of
    the game's price thresholds: it says who is most likely to move, not when.
    """
    if not {"net_transfers_event", "player_name", "price"} <= set(players):
        raise ValueError("Price-watch fields are missing")
    known = players.loc[players.net_transfers_event.notna()]
    risers = known.loc[known.net_transfers_event.gt(0)].nlargest(top_n, "net_transfers_event")
    fallers = known.loc[known.net_transfers_event.lt(0)].nsmallest(top_n, "net_transfers_event")
    return pd.concat(
        [risers.assign(direction="rise"), fallers.assign(direction="fall")], ignore_index=True
    )


def price_watch_backtest(playerstats: pd.DataFrame, top_n: int = 10) -> dict:
    """How often the watch list was right: did the price move by the next gameweek?

    For each gameweek snapshot the ``top_n`` players by net transfers in and out are
    compared with the next snapshot's price. The first snapshot is skipped because prices
    reset at the season start.
    """
    required = {"id", "gw", "now_cost", "transfers_in_event", "transfers_out_event"}
    if not required <= set(playerstats):
        raise ValueError("Price-watch backtest fields are missing")
    stats = playerstats[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    stats = stats.sort_values(["id", "gw"])
    stats["next_cost"] = stats.groupby("id").now_cost.shift(-1)
    stats["net"] = stats.transfers_in_event - stats.transfers_out_event
    scored = stats.loc[stats.next_cost.notna() & stats.gw.gt(stats.gw.min())]
    if scored.empty:
        return {}
    rose, fell = scored.next_cost.gt(scored.now_cost), scored.next_cost.lt(scored.now_cost)
    top = scored.sort_values("net", ascending=False).groupby("gw").head(top_n)
    bottom = scored.sort_values("net").groupby("gw").head(top_n)
    return {
        "top_n": top_n,
        "gameweeks": int(scored.gw.nunique()),
        "rise_hit_rate": float(top.next_cost.gt(top.now_cost).mean()),
        "rise_base_rate": float(rose.mean()),
        "fall_hit_rate": float(bottom.next_cost.lt(bottom.now_cost).mean()),
        "fall_base_rate": float(fell.mean()),
    }


def league_ownership(picks: pd.DataFrame, my_entry: int) -> pd.DataFrame:
    """Per player: how many rivals own, start and captain them, and whether I own them.

    ``picks`` holds one row per manager and player with ``entry``, ``element``,
    ``multiplier`` and ``is_captain``. Shares are over the rivals only, so a player I own
    that no rival owns has a share of zero.
    """
    required = {"entry", "element", "multiplier", "is_captain"}
    if not required <= set(picks) or picks.duplicated(["entry", "element"]).any():
        raise ValueError("League picks are incomplete or duplicated")
    rivals = picks.loc[picks.entry.ne(my_entry)]
    managers = int(rivals.entry.nunique())
    if managers == 0:
        raise ValueError("No rival squads to compare with")
    counts = rivals.groupby("element").agg(
        owners=("entry", "nunique"),
        starters=("multiplier", lambda values: int((values > 0).sum())),
        captains=("is_captain", lambda values: int(values.astype(bool).sum())),
    )
    mine = picks.loc[picks.entry.eq(my_entry)].set_index("element")
    result = counts.reindex(counts.index.union(mine.index)).fillna(0)
    result["owned_share"] = result.owners / managers
    result["started_share"] = result.starters / managers
    result["captain_share"] = result.captains / managers
    result["mine"] = result.index.isin(mine.index)
    result["my_captain"] = result.index.isin(mine.index[mine.is_captain.astype(bool)])
    result["rivals"] = managers
    return result.rename_axis("source_player_id").reset_index()


def scorer_markets(
    players: pd.DataFrame, fixtures: pd.DataFrame, gameweek: int | None = None
) -> pd.DataFrame:
    """Per player and fixture: probability of scoring and of assisting at least once.

    Goals and assists are Poisson with the projected per-90 rate scaled by the fixture's
    attack multiplier. Minutes are a three-way mixture (start, substitute, not used)
    rather than one average, because the chance of scoring is not linear in minutes.
    ``players`` is the output of :func:`project_players`.
    """
    required = {
        "club_id",
        "start_probability",
        "appearance_probability",
        "expected_minutes",
        "projected_xg_per90",
        "projected_xa_per90",
    }
    if not required <= set(players) or not {"club_id", "gameweek", "attack_multiplier"} <= set(
        fixtures
    ):
        raise ValueError("Scorer-market inputs are incomplete")
    week = int(fixtures.gameweek.min()) if gameweek is None else int(gameweek)
    schedule = fixtures.loc[fixtures.gameweek.eq(week)]
    keep = [
        column
        for column in (
            "match_id",
            "club_id",
            "gameweek",
            "opponent_short_name",
            "venue",
            "attack_multiplier",
            "clean_sheet_probability",
        )
        if column in schedule
    ]
    frame = players.merge(schedule[keep], on="club_id", how="inner")
    start = frame.start_probability.clip(0, 1)
    substitute = (frame.appearance_probability - frame.start_probability).clip(0, 1)
    substitute_minutes = np.minimum(SUBSTITUTE_MINUTES, frame.expected_minutes)
    # Minutes when starting are whatever is left of the expectation after substitute cameos.
    start_minutes = (
        (frame.expected_minutes - substitute * substitute_minutes) / start.replace(0, np.nan)
    ).clip(0, 90).fillna(0)
    for name, rate in (("scorer", "projected_xg_per90"), ("assist", "projected_xa_per90")):
        per_minute = frame[rate].clip(lower=0) * frame.attack_multiplier / 90
        frame[f"{name}_probability"] = start * (
            1 - np.exp(-per_minute * start_minutes)
        ) + substitute * (1 - np.exp(-per_minute * substitute_minutes))
        frame[f"expected_{name}_events"] = per_minute * frame.expected_minutes
    return frame
