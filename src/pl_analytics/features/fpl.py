"""Transparent Fantasy Premier League planning features and projection baseline."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

HORIZONS = (1, 2, 3, 4, 5)
GOAL_POINTS = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CLEAN_SHEET_POINTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
RECENCY_DECAY = 0.65
MINUTES_PRIOR_STRENGTH = 1.0


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


def _shrunk_per90(frame: pd.DataFrame, column: str, *, prior_minutes: float = 450) -> pd.Series:
    """Shrink a per-90 rate toward an active position median."""
    values = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
    active = values.where(frame.minutes.gt(0))
    prior = active.groupby(frame.position_group).transform("median").fillna(0)
    reliability = frame.minutes / (frame.minutes + prior_minutes)
    return (reliability * values.fillna(prior) + (1 - reliability) * prior).clip(lower=0)


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
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Project simple expected FPL points from current rates and future fixture context.

    This deliberately transparent baseline uses only information available at the
    refresh origin. It is a planning estimate, not a claim of calibrated certainty.
    """
    result = expected_minutes_features(players, appearances)
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
    for total, target in (
        ("penalties_saved", "projected_penalty_saves_per90"),
        ("bonus", "projected_bonus_per90"),
        ("yellow_cards", "projected_yellow_cards_per90"),
        ("red_cards", "projected_red_cards_per90"),
        ("own_goals", "projected_own_goals_per90"),
        ("penalties_missed", "projected_penalties_missed_per90"),
    ):
        values = pd.to_numeric(result.get(total, pd.Series(0, index=result.index)), errors="coerce")
        result[f"raw_{target}"] = values.fillna(0) * 90 / result.minutes.replace(0, np.nan)
        result[target] = _shrunk_per90(result, f"raw_{target}", prior_minutes=900)
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
        component_rows: list[dict[str, float]] = []
        for player in result.itertuples(index=False):
            schedule = selected_fixtures.loc[selected_fixtures.club_id.eq(player.club_id)]
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
                    conceded_rate = max(0.01, -np.log(fixture.clean_sheet_probability))
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
            labels.append(" · ".join(descriptions) if descriptions else "Blank")
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
