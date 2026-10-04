"""FPL squad optimization, legal line-up selection and chronological backtesting."""

from collections.abc import Iterable
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from pl_analytics.features.fpl import (
    fixture_rows,
    latest_players,
    project_players,
    scorer_markets,
)

SQUAD_POSITIONS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SCORER_BINS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 1.0)
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


def replacement_options(
    players: pd.DataFrame,
    outgoing: pd.Series,
    *,
    projection: str,
    bank: float = 0.0,
    squad: pd.DataFrame | None = None,
    limit: int = 10,
    min_minutes: float = 30.0,
) -> pd.DataFrame:
    """Rank same-position replacements affordable from the sale price plus the bank.

    With ``squad`` the options exclude current squad members and respect the
    three-per-club limit after the sale. Prices are current prices, not selling prices.
    """
    pool = players.loc[
        players.position_group.eq(outgoing.position_group)
        & players.price.le(float(outgoing.price) + float(bank) + 1e-9)
        & players.source_player_id.ne(outgoing.source_player_id)
    ]
    if "expected_minutes" in pool:
        pool = pool.loc[pool.expected_minutes.ge(min_minutes)]
    if squad is not None:
        remaining = squad.loc[squad.source_player_id.ne(outgoing.source_player_id)]
        club_counts = remaining.club_id.value_counts()
        pool = pool.loc[
            ~pool.source_player_id.isin(squad.source_player_id)
            & pool.club_id.map(club_counts).fillna(0).lt(3)
        ]
    pool = pool.assign(
        projected_gain=pool[projection] - float(outgoing[projection]),
        price_change=pool.price - float(outgoing.price),
    )
    if squad is None or len(squad) != 15:
        return pool.sort_values(["projected_gain", "price"], ascending=[False, True]).head(limit)
    # A swap only matters through the best XI it allows: upgrading a bench player who
    # stays on the bench is worth nothing this horizon.
    kept = squad.loc[squad.source_player_id.ne(outgoing.source_player_id)]
    positions = [*kept.position_group, outgoing.position_group]
    kept_points = kept[projection].to_numpy(float)
    current = lineup_points(squad.position_group, squad[projection])
    pool = pool.assign(
        lineup_gain=[
            lineup_points(positions, np.append(kept_points, points)) - current
            for points in pool[projection]
        ]
    )
    return pool.sort_values(
        ["lineup_gain", "projected_gain", "price"], ascending=[False, False, True]
    ).head(limit)


def lineup_points(positions: Iterable[str], points: Iterable[float]) -> float:
    """Expected points of the best legal XI from a 15-player squad, captain counted twice."""
    by_position: dict[str, list[float]] = {position: [] for position in STARTER_LIMITS}
    for position, value in zip(positions, points, strict=True):
        by_position.setdefault(position, []).append(float(value))
    ranked = {position: sorted(values, reverse=True) for position, values in by_position.items()}
    if not ranked["GK"]:
        raise ValueError("The squad cannot form a legal FPL starting XI")
    best = -np.inf
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            if (
                len(ranked["DEF"]) < defenders
                or len(ranked["MID"]) < midfielders
                or len(ranked["FWD"]) < forwards
            ):
                continue
            best = max(
                best,
                sum(ranked["DEF"][:defenders])
                + sum(ranked["MID"][:midfielders])
                + sum(ranked["FWD"][:forwards]),
            )
    if not np.isfinite(best):
        raise ValueError("The squad cannot form a legal FPL starting XI")
    captain = max(ranked[position][0] for position in STARTER_LIMITS if ranked[position])
    return float(ranked["GK"][0] + best + captain)


def best_single_transfers(
    squad: pd.DataFrame,
    players: pd.DataFrame,
    *,
    projection: str,
    bank: float = 0.0,
) -> pd.DataFrame:
    """Return each squad player's best legal one-for-one swap, largest XI gain first."""
    if len(squad) != 15:
        raise ValueError("A complete mapped 15-player squad is required")
    rows = []
    for _, outgoing in squad.iterrows():
        options = replacement_options(
            players, outgoing, projection=projection, bank=bank, squad=squad, limit=1
        )
        if options.empty:
            continue
        incoming = options.iloc[0]
        rows.append(
            {
                "out_id": int(outgoing.source_player_id),
                "in_id": int(incoming.source_player_id),
                "lineup_gain": float(incoming.lineup_gain),
                "projected_gain": float(incoming.projected_gain),
                "price_change": float(incoming.price_change),
            }
        )
    result = pd.DataFrame(
        rows, columns=["out_id", "in_id", "lineup_gain", "projected_gain", "price_change"]
    )
    return result.sort_values(
        ["lineup_gain", "projected_gain"], ascending=False
    ).reset_index(drop=True)


def plan_transfers(
    squad: pd.DataFrame,
    players: pd.DataFrame,
    *,
    gameweeks: Iterable[int],
    bank: float = 0.0,
    free_transfers: int = 1,
    max_free_transfers: int = 5,
    minimum_gain: float = 1.0,
    allow_hits: bool = False,
    hit_cost: float = 4.0,
    max_per_week: int = 3,
) -> dict:
    """Greedy week-by-week transfer plan over the prepared gameweek projections.

    Each week the best single swap is taken while it adds at least ``minimum_gain``
    points to the best XI over the remaining weeks (plus ``hit_cost`` once free transfers
    run out and hits are allowed). A swap that does not help this week's XI is deferred,
    which banks the free transfer, unless the bank of free transfers is already full.
    This is a heuristic, not an optimal plan: it never looks ahead to combine moves.
    """
    weeks = tuple(int(value) for value in gameweeks)
    columns = [f"projected_points_gw_{week}" for week in weeks]
    if not weeks or not set(columns) <= set(players):
        raise ValueError("Per-gameweek projections are incomplete")
    if len(squad) != 15 or free_transfers < 0 or minimum_gain < 0:
        raise ValueError("A 15-player squad and non-negative settings are required")
    pool = players.copy()
    current = pool.loc[pool.source_player_id.isin(squad.source_player_id)].copy()
    if len(current) != 15:
        raise ValueError("Some squad players are absent from the projections")
    original = current.copy()
    available, moves, summaries = int(free_transfers), [], []
    for index, (week, column) in enumerate(zip(weeks, columns, strict=True)):
        remaining = columns[index:]
        pool["plan_value"] = pool[remaining].sum(axis=1)
        current["plan_value"] = current[remaining].sum(axis=1)
        made = hits = 0
        while made < max_per_week:
            paid = available <= 0
            if paid and not allow_hits:
                break
            ranked = best_single_transfers(current, pool, projection="plan_value", bank=bank)
            if ranked.empty:
                break
            best = ranked.iloc[0]
            if best.lineup_gain < minimum_gain + (hit_cost if paid else 0.0):
                break
            incoming = pool.loc[pool.source_player_id.eq(int(best.in_id))]
            after = pd.concat(
                [current.loc[current.source_player_id.ne(int(best.out_id))], incoming]
            )
            this_week = lineup_points(after.position_group, after[column]) - lineup_points(
                current.position_group, current[column]
            )
            last_week = index == len(weeks) - 1
            if this_week <= 0 and not last_week and available < max_free_transfers:
                break
            outgoing = current.loc[current.source_player_id.eq(int(best.out_id))].iloc[0]
            bank += float(outgoing.price) - float(incoming.price.iloc[0])
            current = after
            moves.append(
                {
                    "gameweek": week,
                    "out_id": int(best.out_id),
                    "in_id": int(best.in_id),
                    "gain": float(best.lineup_gain),
                    "hit": bool(paid),
                }
            )
            made += 1
            hits += int(paid)
            available -= 0 if paid else 1
        summaries.append(
            {
                "gameweek": week,
                "transfers": made,
                "hit_points": hits * hit_cost,
                "plan_points": lineup_points(current.position_group, current[column])
                - hits * hit_cost,
                "hold_points": lineup_points(original.position_group, original[column]),
                "free_transfers_left": available,
                "bank": bank,
            }
        )
        available = min(available + 1, max_free_transfers)
    summary = pd.DataFrame(summaries)
    return {
        "moves": pd.DataFrame(moves, columns=["gameweek", "out_id", "in_id", "gain", "hit"]),
        "weeks": summary,
        "total_gain": float((summary.plan_points - summary.hold_points).sum()),
        "final_squad": current.drop(columns="plan_value"),
    }


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
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    appearances: pd.DataFrame | None = None,
    *,
    position_map: dict[str, str],
    competition_id: str,
    season: str,
    priors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score the production event model on each gameweek from strictly earlier data.

    For every target gameweek the player snapshot, appearance history and the
    season-xG fixture strengths are rebuilt from earlier gameweeks only, then
    :func:`project_players` predicts the target. Three references are kept beside it:
    the position median, a points-per-90 rate model and the provider's own next-gameweek
    estimate published before the target.
    """
    required = {"id", "gw", "event_points", "minutes"}
    if not required <= set(playerstats) or not {"gameweek", "finished"} <= set(matches):
        raise ValueError("Backtest inputs are incomplete")
    stats = playerstats.copy()
    stats["gw"] = pd.to_numeric(stats.gw, errors="raise").astype(int)
    stats["minutes"] = pd.to_numeric(stats.minutes, errors="coerce").fillna(0)
    stats = stats.sort_values(["id", "gw"])
    stats["fixture_minutes"] = stats.groupby("id").minutes.diff().fillna(stats.minutes).clip(0)
    goals = pd.to_numeric(
        stats.get("goals_scored", pd.Series(np.nan, index=stats.index)), errors="coerce"
    )
    stats["fixture_goals"] = goals.groupby(stats.id).diff().fillna(goals)
    match_gameweek = pd.to_numeric(matches.gameweek, errors="coerce")
    was_finished = matches.finished.astype(str).str.lower().isin({"true", "1"})
    appearance_gameweek = (
        pd.to_numeric(appearances.gameweek, errors="coerce") if appearances is not None else None
    )
    rows = []
    for target_gw in sorted(stats.gw.unique())[1:]:
        if not match_gameweek.eq(target_gw).any():
            continue
        # Earlier results stay finished; the target and later weeks become unplayed fixtures.
        schedule = matches.assign(finished=was_finished & match_gameweek.lt(target_gw))
        schedule = schedule.loc[schedule.finished | match_gameweek.ge(target_gw)]
        snapshot, history = latest_players(
            stats.loc[stats.gw.lt(target_gw)].drop(columns=["fixture_minutes", "fixture_goals"]),
            players,
            teams,
            competition_id=competition_id,
            season=season,
            position_map=position_map,
        )
        fixtures = fixture_rows(schedule, teams)
        projected = project_players(
            snapshot,
            history,
            fixtures,
            appearances=(
                appearances.loc[appearance_gameweek.lt(target_gw)]
                if appearances is not None
                else None
            ),
            priors=priors,
            horizons=(1,),
        )
        # A double gameweek gives several fixtures: the chance of scoring in any of them.
        markets = scorer_markets(projected, fixtures, target_gw)
        projected["scorer_probability"] = projected.source_player_id.map(
            1 - (1 - markets.scorer_probability).groupby(markets.source_player_id).prod()
        ).fillna(0.0)
        actual = stats.loc[
            stats.gw.eq(target_gw), ["id", "event_points", "fixture_minutes", "fixture_goals"]
        ].rename(
            columns={
                "id": "source_player_id",
                "event_points": "actual_points",
                "fixture_minutes": "actual_minutes",
            }
        )
        gameweek_goals = actual.pop("fixture_goals")
        # 1.0 / 0.0, missing when the source has no goal counts.
        actual["scored"] = gameweek_goals.gt(0).astype(float).where(gameweek_goals.notna())
        frame = projected.loc[projected.minutes.gt(0)].merge(
            actual, on="source_player_id", validate="one_to_one"
        )
        if frame.empty:
            continue
        frame["actual_points"] = pd.to_numeric(frame.actual_points, errors="coerce")
        frame = frame.dropna(subset=["actual_points"])
        frame["predicted_points"] = frame.projected_points_1
        frame["rate_model_points"] = frame.shrunk_points_per90 * frame.expected_minutes / 90
        frame["provider_points"] = frame.provider_ep_next
        active_ppg = frame.points_per_game.where(frame.minutes.gt(0))
        frame["baseline_points"] = (
            active_ppg.groupby(frame.position_group).transform("median").fillna(2)
        )
        frame["origin_gw"] = target_gw - 1
        frame["target_gw"] = target_gw
        rows.append(
            frame[
                [
                    "competition_id",
                    "season",
                    "source_player_id",
                    "position_group",
                    "origin_gw",
                    "target_gw",
                    "expected_minutes",
                    "actual_minutes",
                    "start_probability",
                    "appearance_probability",
                    "sixty_probability",
                    "predicted_points",
                    "rate_model_points",
                    "provider_points",
                    "baseline_points",
                    "actual_points",
                    "scorer_probability",
                    "scored",
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
    minute_error = backtest.expected_minutes - backtest.actual_minutes.clip(upper=90)

    def spearman(column: str) -> float:
        values = [
            frame[column].corr(frame.actual_points, method="spearman")
            for _, frame in backtest.groupby("target_gw")
        ]
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    summary = {
        "observations": int(len(backtest)),
        "gameweeks": sorted(int(value) for value in backtest.target_gw.unique()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "baseline_mae": float(baseline_error.abs().mean()),
        "minutes_mae": float(minute_error.abs().mean()),
        "mean_predicted_points": float(backtest.predicted_points.mean()),
        "mean_actual_points": float(backtest.actual_points.mean()),
        "zero_point_rate": float(backtest.actual_points.eq(0).mean()),
        "mean_spearman": spearman("predicted_points"),
    }
    scored = backtest.get("scored")
    if scored is not None and scored.notna().any():
        rows = backtest.loc[scored.notna()]
        outcome = rows.scored.astype(float)
        # Reference: each position's own scoring rate in this sample, a forecast that
        # already knows the answer on average and so is hard to beat.
        reference = outcome.groupby(rows.position_group).transform("mean")
        bins = pd.cut(rows.scorer_probability, SCORER_BINS, include_lowest=True)
        summary["scorer"] = {
            "observations": int(len(rows)),
            "brier": float(np.square(rows.scorer_probability - outcome).mean()),
            "position_rate_brier": float(np.square(reference - outcome).mean()),
            "mean_probability": float(rows.scorer_probability.mean()),
            "scoring_rate": float(outcome.mean()),
            "calibration": [
                {
                    "bin": str(interval),
                    "n": int(len(group)),
                    "predicted": float(group.scorer_probability.mean()),
                    "observed": float(group.scored.astype(float).mean()),
                }
                for interval, group in rows.groupby(bins, observed=True)
            ],
        }
    for name in ("rate_model", "provider"):
        column = f"{name}_points"
        if column in backtest and backtest[column].notna().any():
            compared = backtest.dropna(subset=[column])
            summary[f"{name}_mae"] = float(
                (compared[column] - compared.actual_points).abs().mean()
            )
            summary[f"{name}_spearman"] = spearman(column)
    return summary
