"""FPL squad optimization, legal line-up selection and chronological backtesting."""

from collections.abc import Iterable
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SQUAD_POSITIONS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def select_starting_xi(squad: pd.DataFrame, projection: str) -> pd.DataFrame:
    """Return the highest-projected legal XI and captain from a 15-player squad."""
    required = {"source_player_id", "position_group", projection}
    if len(squad) != 15 or not required <= set(squad):
        raise ValueError("A complete mapped 15-player squad is required")
    best_ids: set[int] | None = None
    best_points = -np.inf
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            counts = {"GK": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
            selected = []
            for position, count in counts.items():
                candidates = squad.loc[squad.position_group.eq(position)].nlargest(
                    count, projection
                )
                if len(candidates) != count:
                    selected = []
                    break
                selected.extend(candidates.source_player_id.astype(int))
            if not selected:
                continue
            points = squad.loc[squad.source_player_id.isin(selected), projection].sum()
            if points > best_points:
                best_points, best_ids = float(points), set(selected)
    if best_ids is None:
        raise ValueError("The squad cannot form a legal FPL starting XI")
    result = squad.copy()
    result["is_starter"] = result.source_player_id.isin(best_ids)
    starters = result.loc[result.is_starter]
    captain_id = int(starters.nlargest(1, projection).source_player_id.iloc[0])
    result["is_recommended_captain"] = result.source_player_id.eq(captain_id)
    return result


def optimize_squad(
    players: pd.DataFrame,
    *,
    budget: float,
    projection: str,
    current_ids: Iterable[int] = (),
    max_transfers: int = 15,
) -> pd.DataFrame:
    """Maximize starter, captain and small bench value under official squad constraints."""
    required = {
        "source_player_id",
        "position_group",
        "club_id",
        "price",
        projection,
    }
    pool = players.loc[
        players.position_group.isin(SQUAD_POSITIONS)
        & players.price.notna()
        & players[projection].notna()
    ].copy()
    if not required <= set(pool) or pool.source_player_id.duplicated().any():
        raise ValueError("The optimization player pool is invalid")
    pool = pool.reset_index(drop=True)
    n = len(pool)
    if n < 15 or budget <= 0 or not 0 <= max_transfers <= 15:
        raise ValueError("Optimization settings are invalid")

    # x=squad, y=starter, z=captain. Bench receives 5% weight; captain adds one copy.
    points = pool[projection].to_numpy(float)
    objective = -np.concatenate([0.05 * points, 0.95 * points, points])
    rows: list[tuple[dict[int, float], float, float]] = []

    def add(coefficients: dict[int, float], lower: float, upper: float) -> None:
        rows.append((coefficients, lower, upper))

    add({index: 1 for index in range(n)}, 15, 15)
    for position, count in SQUAD_POSITIONS.items():
        indices = pool.index[pool.position_group.eq(position)]
        add({int(index): 1 for index in indices}, count, count)
    for indices in pool.groupby("club_id").groups.values():
        add({int(index): 1 for index in indices}, 0, 3)
    add({index: float(pool.price.iloc[index]) for index in range(n)}, 0, budget)
    add({n + index: 1 for index in range(n)}, 11, 11)
    for position, (lower, upper) in STARTER_LIMITS.items():
        indices = pool.index[pool.position_group.eq(position)]
        add({n + int(index): 1 for index in indices}, lower, upper)
    add({2 * n + index: 1 for index in range(n)}, 1, 1)
    for index in range(n):
        add({n + index: 1, index: -1}, -np.inf, 0)
        add({2 * n + index: 1, n + index: -1}, -np.inf, 0)
    current = set(int(value) for value in current_ids)
    if current:
        indices = pool.index[pool.source_player_id.astype(int).isin(current)]
        add({int(index): 1 for index in indices}, 15 - max_transfers, np.inf)

    matrix = lil_matrix((len(rows), 3 * n), dtype=float)
    lower, upper = np.empty(len(rows)), np.empty(len(rows))
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index], upper[row_index] = row_lower, row_upper
    result = milp(
        objective,
        integrality=np.ones(3 * n),
        bounds=Bounds(np.zeros(3 * n), np.ones(3 * n)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 10},
    )
    if not result.success or result.x is None:
        raise ValueError("No legal FPL squad satisfies the selected budget and transfer limit")
    solution = pool.loc[result.x[:n] > 0.5].copy()
    solution["is_starter"] = result.x[n : 2 * n][result.x[:n] > 0.5] > 0.5
    solution["is_recommended_captain"] = result.x[2 * n :][result.x[:n] > 0.5] > 0.5
    solution["is_current"] = solution.source_player_id.astype(int).isin(current)
    if len(solution) != 15 or solution.is_starter.sum() != 11:
        raise ValueError("Optimizer returned an invalid squad")
    return solution.sort_values(
        ["is_starter", "position_group", projection], ascending=[False, True, False]
    )


def chip_advice(
    current: pd.DataFrame,
    free_hit: pd.DataFrame,
    *,
    projection: str = "projected_points_1",
) -> pd.DataFrame:
    """Return transparent chip signals, without claiming an optimal season schedule."""
    lineup = select_starting_xi(current, projection)
    starters = lineup.loc[lineup.is_starter]
    bench = lineup.loc[~lineup.is_starter]
    captain = starters.loc[starters.is_recommended_captain, projection].iloc[0]
    current_total = starters[projection].sum() + captain
    optimized_total = (
        free_hit.loc[free_hit.is_starter, projection].sum()
        + free_hit.loc[free_hit.is_recommended_captain, projection].sum()
    )
    values = [
        ("Triple captain", captain, 7.5, f"Captain projects {captain:.1f} points."),
        (
            "Bench boost",
            bench[projection].sum(),
            14.0,
            f"Bench projects {bench[projection].sum():.1f} points.",
        ),
        (
            "Free hit",
            optimized_total - current_total,
            12.0,
            f"One-week optimized gain is {optimized_total - current_total:.1f} points.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "chip": chip,
                "signal": "Consider" if value >= threshold else "Hold",
                "score": value,
                "threshold": threshold,
                "reason": reason,
            }
            for chip, value, threshold, reason in values
        ]
    )


def optimize_chip_schedule(
    current: pd.DataFrame,
    players: pd.DataFrame,
    *,
    budget: float,
    gameweeks: Iterable[int],
) -> pd.DataFrame:
    """Allocate TC, BB and FH once across distinct gameweeks in the prepared horizon."""
    weeks = tuple(int(value) for value in gameweeks)
    if len(weeks) < 3 or len(set(weeks)) != len(weeks):
        raise ValueError("At least three distinct projected gameweeks are required")
    gains: dict[str, dict[int, float]] = {
        "Triple captain": {},
        "Bench boost": {},
        "Free hit": {},
    }
    details: dict[tuple[str, int], str] = {}
    for gameweek in weeks:
        projection = f"projected_points_gw_{gameweek}"
        if projection not in current or projection not in players:
            raise ValueError("Per-gameweek projections are incomplete")
        lineup = select_starting_xi(current, projection)
        starters = lineup.loc[lineup.is_starter]
        bench = lineup.loc[~lineup.is_starter]
        captain = starters.loc[starters.is_recommended_captain].iloc[0]
        current_total = starters[projection].sum() + captain[projection]
        free_hit = optimize_squad(players, budget=budget, projection=projection)
        free_total = (
            free_hit.loc[free_hit.is_starter, projection].sum()
            + free_hit.loc[free_hit.is_recommended_captain, projection].sum()
        )
        gains["Triple captain"][gameweek] = float(captain[projection])
        gains["Bench boost"][gameweek] = float(bench[projection].sum())
        gains["Free hit"][gameweek] = float(free_total - current_total)
        details[("Triple captain", gameweek)] = f"Extra captain copy: {captain.player_name}"
        details[("Bench boost", gameweek)] = "Adds all four projected bench scores"
        details[("Free hit", gameweek)] = "Best legal one-week squad versus current XI"
    chips = tuple(gains)
    best = max(
        permutations(weeks, len(chips)),
        key=lambda assignment: sum(
            gains[chip][gameweek] for chip, gameweek in zip(chips, assignment, strict=True)
        ),
    )
    return pd.DataFrame(
        [
            {
                "chip": chip,
                "recommended_gameweek": gameweek,
                "incremental_points": gains[chip][gameweek],
                "signal": "Consider" if gains[chip][gameweek] >= threshold else "Hold",
                "reason": details[(chip, gameweek)],
            }
            for chip, gameweek, threshold in zip(chips, best, (7.5, 14.0, 12.0), strict=True)
        ]
    )


def rolling_backtest(
    playerstats: pd.DataFrame,
    players: pd.DataFrame,
    *,
    position_map: dict[str, str],
    competition_id: str,
    season: str,
) -> pd.DataFrame:
    """Evaluate a strictly prior-gameweek expected-points baseline against next-GW points."""
    stats = playerstats.copy()
    required = {"id", "gw", "event_points", "points_per_game", "minutes"}
    if not required <= set(stats) or not {"player_id", "position"} <= set(players):
        raise ValueError("Backtest inputs are incomplete")
    for column in ("gw", "event_points", "points_per_game", "minutes"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    identity = players[["player_id", "position"]].copy()
    identity["position_group"] = identity.position.map(position_map).fillna("UNKNOWN")
    rows = []
    gameweeks = sorted(int(value) for value in stats.gw.dropna().unique())
    for target_gw in gameweeks[1:]:
        prior = (
            stats.loc[stats.gw.lt(target_gw)]
            .sort_values("gw")
            .drop_duplicates("id", keep="last")
        )
        actual = stats.loc[stats.gw.eq(target_gw), ["id", "event_points"]].rename(
            columns={"event_points": "actual_points"}
        )
        frame = prior.merge(identity, left_on="id", right_on="player_id", validate="many_to_one")
        frame = frame.merge(actual, on="id", validate="one_to_one")
        frame["reliability"] = frame.minutes / (frame.minutes + 450)
        median = frame.groupby("position_group").points_per_game.transform("median").fillna(2.0)
        frame["predicted_points"] = (
            frame.reliability * frame.points_per_game.fillna(median)
            + (1 - frame.reliability) * median
        )
        chance = pd.to_numeric(
            frame.get("chance_of_playing_next_round", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        status = frame.get("status", pd.Series("a", index=frame.index)).fillna("a")
        availability = chance.div(100).where(
            chance.notna(), status.map({"a": 1, "d": 0.5}).fillna(0)
        )
        frame["predicted_points"] *= availability.clip(0, 1)
        frame["baseline_points"] = median
        frame["origin_gw"] = target_gw - 1
        frame["target_gw"] = target_gw
        frame["source_player_id"] = frame.id.astype(int)
        frame["competition_id"] = competition_id
        frame["season"] = season
        rows.append(
            frame[
                [
                    "competition_id",
                    "season",
                    "source_player_id",
                    "position_group",
                    "origin_gw",
                    "target_gw",
                    "predicted_points",
                    "baseline_points",
                    "actual_points",
                ]
            ]
        )
    if not rows:
        raise ValueError("At least two gameweeks are required for the FPL backtest")
    return pd.concat(rows, ignore_index=True)


def backtest_summary(backtest: pd.DataFrame) -> dict:
    """Summarize expected-points error and rank association without hiding sample size."""
    error = backtest.predicted_points - backtest.actual_points
    baseline_error = backtest.baseline_points - backtest.actual_points
    correlations = [
        frame.predicted_points.corr(frame.actual_points, method="spearman")
        for _, frame in backtest.groupby("target_gw")
    ]
    return {
        "observations": int(len(backtest)),
        "gameweeks": sorted(int(value) for value in backtest.target_gw.unique()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "baseline_mae": float(baseline_error.abs().mean()),
        "mean_spearman": float(np.nanmean(correlations)),
    }
