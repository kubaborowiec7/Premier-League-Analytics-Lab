"""Transparent Fantasy Premier League planning features and projection baseline."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

HORIZONS = (1, 2, 3, 4, 5)
GOAL_POINTS = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CLEAN_SHEET_POINTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}


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
    return latest, history


def fixture_rows(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Expand scheduled matches to one leakage-safe planning row per club and fixture."""
    required = {"gameweek", "kickoff_time", "home_team", "away_team", "finished", "match_id"}
    if not required <= set(matches):
        raise ValueError("FPL fixture fields are missing")
    club_lookup = teams.set_index("code")
    rows: list[dict] = []
    for match in matches.itertuples(index=False):
        if str(match.finished).lower() in {"true", "1"}:
            continue
        for venue, team_code, opponent_code in (
            ("H", match.home_team, match.away_team),
            ("A", match.away_team, match.home_team),
        ):
            if team_code not in club_lookup.index or opponent_code not in club_lookup.index:
                raise ValueError("Fixture references an unknown FPL club")
            club = club_lookup.loc[team_code]
            opponent = club_lookup.loc[opponent_code]
            team_elo = pd.to_numeric(club.elo, errors="coerce")
            opponent_elo = pd.to_numeric(opponent.elo, errors="coerce")
            if pd.isna(team_elo) or pd.isna(opponent_elo):
                advantage = 80.0 if venue == "H" else -80.0
            else:
                advantage = float(team_elo - opponent_elo + (80 if venue == "H" else -80))
            difficulty = int(np.digitize(-advantage, [-150, -50, 50, 150]) + 1)
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
                    "difficulty": difficulty,
                    "attack_multiplier": float(np.clip(np.exp(advantage / 1000), 0.72, 1.35)),
                    "clean_sheet_probability": float(np.clip(0.29 + advantage / 1600, 0.08, 0.58)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["match_id", "club_id"]).any():
        raise ValueError("FPL fixture schedule is empty or ambiguous")
    return result.sort_values(["gameweek", "kickoff_time", "club_name"]).reset_index(drop=True)


def _availability(row: pd.Series) -> float:
    chance = row.chance_of_playing
    if pd.notna(chance):
        return float(np.clip(chance / 100, 0, 1))
    return 1.0 if row.status == "a" else 0.5 if row.status == "d" else 0.0


def project_players(
    players: pd.DataFrame,
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Project simple expected FPL points from current rates and future fixture context.

    This deliberately transparent baseline uses only information available at the
    refresh origin. It is a planning estimate, not a claim of calibrated certainty.
    """
    result = players.copy()
    current_gw = int(result.gw.max())
    result["availability_probability"] = result.apply(_availability, axis=1)
    result["expected_minutes"] = (
        (result.minutes / max(current_gw, 1)).clip(0, 90) * result.availability_probability
    )
    position_fallback = result.groupby("position_group")["points_per_game"].transform("median")
    observed_rate = (result.total_points * 90 / result.minutes.replace(0, np.nan)).fillna(
        position_fallback
    )
    reliability = result.minutes / (result.minutes + 450)
    position_rate = result.groupby("position_group")["points_per_game"].transform("median")
    result["reliability"] = reliability.clip(0, 1)
    result["shrunk_points_per90"] = (
        reliability * observed_rate + (1 - reliability) * position_rate.fillna(2.0)
    )
    point_sd = history.groupby("source_player_id").points.std().rename("historical_points_sd")
    result = result.merge(point_sd, on="source_player_id", how="left", validate="one_to_one")
    position_sd = result.groupby("position_group")["historical_points_sd"].transform("median")
    result["historical_points_sd"] = result.historical_points_sd.fillna(position_sd).fillna(2.0)

    future_gameweeks = sorted(fixtures.gameweek.unique())
    horizon_values = sorted(set(int(value) for value in horizons))
    for horizon in horizon_values:
        gameweeks = future_gameweeks[:horizon]
        selected_fixtures = fixtures.loc[fixtures.gameweek.isin(gameweeks)]
        points, variances, labels = [], [], []
        for player in result.itertuples(index=False):
            schedule = selected_fixtures.loc[selected_fixtures.club_id.eq(player.club_id)]
            projected = 0.0
            variance = 0.0
            descriptions = []
            for fixture in schedule.itertuples(index=False):
                minutes_factor = player.expected_minutes / 90
                appearance = min(2.0, player.expected_minutes / 30)
                goal_points = GOAL_POINTS.get(player.position_group, 4.0)
                clean_points = CLEAN_SHEET_POINTS.get(player.position_group, 0.0)
                attack = minutes_factor * fixture.attack_multiplier * (
                    np.nan_to_num(player.xg_per90) * goal_points
                    + np.nan_to_num(player.xa_per90) * 3
                )
                clean_sheet = minutes_factor * fixture.clean_sheet_probability * clean_points
                threshold = 10 if player.position_group == "DEF" else 12
                actions = max(0.0, np.nan_to_num(player.defensive_contribution_per90))
                defensive = 2 * poisson.sf(threshold - 1, actions * minutes_factor)
                rate_baseline = player.shrunk_points_per90 * minutes_factor
                modelled = appearance + attack + clean_sheet + defensive
                fixture_points = 0.5 * rate_baseline * fixture.attack_multiplier + 0.5 * modelled
                projected += float(fixture_points)
                variance += float((player.historical_points_sd * minutes_factor) ** 2)
                descriptions.append(
                    f"GW{fixture.gameweek} {fixture.opponent_short_name} ({fixture.venue})"
                )
            points.append(projected)
            variances.append(variance)
            labels.append(" · ".join(descriptions) if descriptions else "Blank")
        result[f"projected_points_{horizon}"] = points
        spread = 1.28 * np.sqrt(variances)
        result[f"projection_low_{horizon}"] = np.maximum(0, np.asarray(points) - spread)
        result[f"projection_high_{horizon}"] = np.asarray(points) + spread
        result[f"fixtures_{horizon}"] = labels
    previous = np.zeros(len(result))
    for horizon, gameweek in enumerate(future_gameweeks[: max(horizon_values)], start=1):
        cumulative = result[f"projected_points_{horizon}"].to_numpy(float)
        result[f"projected_points_gw_{int(gameweek)}"] = np.maximum(0, cumulative - previous)
        previous = cumulative
    return result
