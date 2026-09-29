"""Metric definitions and descriptive, competition-season-position player comparisons."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    label: str
    group: str
    kind: str = "count"
    lower_better: bool = False
    aggregation: str = "sum"


METRICS = {
    "goals": Metric("Goals", "Shooting"),
    "xg": Metric("Expected goals (xG)", "Shooting"),
    "total_shots": Metric("Shots", "Shooting"),
    "shots_on_target": Metric("Shots on target", "Shooting"),
    "goal_conversion": Metric("Goal conversion %", "Shooting", "ratio"),
    "big_chances_missed": Metric("Big chances missed", "Shooting", lower_better=True),
    "headed_goals": Metric("Headed goals", "Shooting"),
    "penalties_scored": Metric("Penalty goals", "Shooting"),
    "assists": Metric("Assists", "Passing"),
    "xa": Metric("Expected assists (xA)", "Passing"),
    "accurate_passes": Metric("Accurate passes", "Passing"),
    "accurate_passes_percent": Metric("Pass accuracy · match avg %", "Passing", "match_pct"),
    "chances_created": Metric("Chances created / key passes", "Passing"),
    "big_chances_created": Metric("Big chances created", "Passing"),
    "accurate_long_balls": Metric("Accurate long balls", "Passing"),
    "accurate_long_balls_percent": Metric(
        "Long-ball accuracy · match avg %", "Passing", "match_pct"
    ),
    "accurate_crosses": Metric("Accurate crosses", "Passing"),
    "accurate_crosses_percent": Metric("Cross accuracy · match avg %", "Passing", "match_pct"),
    "final_third_passes": Metric("Final-third passes", "Passing"),
    "progressive_passes": Metric("Progressive passes", "Passing"),
    "forward_pass_accuracy": Metric("Forward-pass accuracy %", "Passing", "ratio"),
    "successful_dribbles": Metric("Successful dribbles", "Possession"),
    "successful_dribbles_percent": Metric(
        "Dribble success · match avg %", "Possession", "match_pct"
    ),
    "duels_won": Metric("Duels won", "Possession"),
    "duels_won_percent": Metric("Duels won %", "Possession", "ratio"),
    "aerial_duels_won": Metric("Aerial duels won", "Possession"),
    "aerial_duels_won_percent": Metric("Aerial success · match avg %", "Possession", "match_pct"),
    "ground_duels_won": Metric("Ground duels won", "Possession"),
    "touches": Metric("Touches", "Possession"),
    "touches_opposition_box": Metric("Touches in opposition box", "Possession"),
    "dispossessed": Metric("Dispossessed", "Possession", lower_better=True),
    "progressive_carries": Metric("Progressive carries", "Possession"),
    "tackles": Metric("Tackles", "Defending"),
    "tackles_won": Metric("Tackles won", "Defending"),
    "interceptions": Metric("Interceptions", "Defending"),
    "clearances": Metric("Clearances", "Defending"),
    "recoveries": Metric("Recoveries", "Defending"),
    "blocks": Metric("Blocks", "Defending"),
    "defensive_contribution": Metric(
        "Defensive contributions", "Defending", aggregation="season_total"
    ),
    "headed_clearances": Metric("Headed clearances", "Defending"),
    "dribbled_past": Metric("Dribbled past", "Defending", lower_better=True),
    "fouls_committed": Metric("Fouls committed", "Discipline", lower_better=True),
    "was_fouled": Metric("Fouls drawn", "Discipline"),
    "yellow_cards": Metric(
        "Yellow cards", "Discipline", lower_better=True, aggregation="season_total"
    ),
    "red_cards": Metric(
        "Red cards", "Discipline", lower_better=True, aggregation="season_total"
    ),
    "offsides": Metric("Offsides", "Discipline", lower_better=True),
    "saves": Metric("Saves", "Goalkeeping"),
    "goals_prevented": Metric("Goals prevented (provider)", "Goalkeeping"),
    "high_claim": Metric("High claims", "Goalkeeping"),
    "sweeper_actions": Metric("Sweeper actions", "Goalkeeping"),
}

PROFILES = {
    "DEF": (
        "defensive_contribution",
        "tackles_won",
        "interceptions",
        "clearances",
        "recoveries",
        "blocks",
        "aerial_duels_won",
        "accurate_passes",
        "accurate_long_balls",
    ),
    "MID": (
        "accurate_passes",
        "chances_created",
        "xa",
        "final_third_passes",
        "recoveries",
        "interceptions",
        "successful_dribbles",
        "duels_won",
    ),
    "FWD": (
        "goals",
        "xg",
        "total_shots",
        "shots_on_target",
        "xa",
        "assists",
        "touches_opposition_box",
        "successful_dribbles",
    ),
    "GK": (
        "saves",
        "goals_prevented",
        "high_claim",
        "sweeper_actions",
        "accurate_passes",
        "accurate_long_balls",
    ),
}


def summarize_players(appearances: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Aggregate strictly prior observations; per-90 denominators use observed metric minutes.

    Missing counts remain missing. Partial totals are accompanied by coverage, and
    percentages without attempt counts are explicitly minutes-weighted match averages.
    These retrospective summaries must never be joined onto earlier model observations.
    """
    data = appearances.loc[
        pd.to_datetime(appearances.match_date, utc=True).lt(pd.Timestamp(reference_date, tz="UTC"))
        & appearances.minutes.gt(0)
    ].copy()
    grain = ["competition_id", "season", "player_id"]
    if data.duplicated([*grain, "match_id"]).any():
        raise ValueError("Duplicate player-match observations")
    rows = []
    for key, group in data.groupby(grain, sort=True):
        row = dict(zip(grain, key, strict=True))
        row.update(
            player_name=group.player_name.iloc[0],
            position_group=group.position_group.iloc[0],
            club_id=group.club_id.iloc[0] if "club_id" in group else pd.NA,
            club_name=group.club_name.iloc[0] if "club_name" in group else pd.NA,
            club_short_name=(
                group.club_short_name.iloc[0] if "club_short_name" in group else pd.NA
            ),
            club_context="Season player snapshot; transfers are not split by club",
            position_context="FPL season snapshot; broad role, not verified match position",
            minutes=float(group.minutes.sum()),
            appearances=len(group),
            reference_date=reference_date,
            last_match_date=str(group.match_date.max()),
            source="FPL-Core-Insights",
        )
        for name, metric in METRICS.items():
            if metric.aggregation == "season_total":
                values = group.get(name, pd.Series(np.nan, index=group.index))
                known = values.dropna().unique()
                if len(known) > 1:
                    raise ValueError(f"Conflicting season total for {name}")
                value = float(known[0]) if len(known) else np.nan
                valid = pd.Series(pd.notna(value), index=group.index)
            elif metric.kind == "ratio":
                numerator, denominator = (
                    ("goals", "total_shots")
                    if name == "goal_conversion"
                    else ("duels_won", "duels_total")
                )
                if name == "forward_pass_accuracy":
                    values = pd.Series(np.nan, index=group.index)
                    valid = values.notna()
                    value = np.nan
                else:
                    a = group.get(numerator, pd.Series(np.nan, index=group.index))
                    b = (
                        group.duels_won + group.duels_lost
                        if denominator == "duels_total"
                        else group.get(denominator, pd.Series(np.nan, index=group.index))
                    )
                    valid = a.notna() & b.notna()
                    value = 100 * a[valid].sum() / b[valid].sum() if b[valid].sum() > 0 else np.nan
            else:
                values = group.get(name, pd.Series(np.nan, index=group.index))
                valid = values.notna()
                value = (
                    np.average(values[valid], weights=group.loc[valid, "minutes"])
                    if metric.kind == "match_pct" and valid.any()
                    else values.sum(min_count=1)
                )
            observed_minutes = float(group.loc[valid, "minutes"].sum())
            row[name] = float(value)
            row[f"{name}_coverage"] = observed_minutes / row["minutes"]
            row[f"{name}_per90"] = (
                value * 90 / observed_minutes
                if metric.kind == "count" and observed_minutes
                else value
            )
        rows.append(row)
    return pd.DataFrame(rows)


def peer_percentiles(
    frame: pd.DataFrame, *, min_minutes: int = 450, min_coverage: float = 0.9, min_peers: int = 5
) -> pd.DataFrame:
    """Midrank percentiles within competition/season/position; adverse metrics reversed."""
    extra = {
        f"{name}_{suffix}": np.nan if suffix == "percentile" else 0
        for name in METRICS
        for suffix in ("percentile", "peers")
    }
    result = pd.concat(
        [frame.drop(columns=list(extra), errors="ignore"), pd.DataFrame(extra, index=frame.index)],
        axis=1,
    ).copy()
    for name, metric in METRICS.items():
        for _, peers in result.groupby(["competition_id", "season", "position_group"]):
            valid = (
                peers.minutes.ge(min_minutes)
                & peers[f"{name}_coverage"].ge(min_coverage)
                & peers[f"{name}_per90"].notna()
            )
            indices = peers.index[valid]
            result.loc[peers.index, f"{name}_peers"] = len(indices)
            if len(indices) < min_peers or peers.position_group.iloc[0] == "UNKNOWN":
                continue
            values = peers.loc[indices, f"{name}_per90"]
            ranks = (values.rank(method="average") - 0.5) / len(values) * 100
            result.loc[indices, f"{name}_percentile"] = (
                100 - ranks if metric.lower_better else ranks
            )
    result = result.copy()
    result["normalization_context"] = (
        result.competition_id + "|" + result.season + "|" + result.position_group
    )
    return result


def cohort_percentiles(
    frame: pd.DataFrame,
    *,
    metrics: tuple[str, ...] | None = None,
    min_minutes: int = 1,
    min_coverage: float = 0.9,
    min_peers: int = 2,
) -> pd.DataFrame:
    """Rank a caller-defined comparison cohort without changing its membership.

    The dashboard uses this for a selected squad after its season, club and optional
    position filters have been applied. League role percentiles remain the canonical
    cross-club comparison produced by :func:`peer_percentiles`.
    """
    selected = tuple(METRICS) if metrics is None else metrics
    unknown = set(selected) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    result = frame.copy()
    context_parts = []
    for column in ("competition_id", "season", "club_id", "position_group"):
        if column in result:
            values = sorted(result[column].dropna().astype(str).unique())
            context_parts.append(f"{column}={','.join(values)}")
    result["cohort_normalization_context"] = "|".join(context_parts)
    for name in selected:
        metric = METRICS[name]
        valid = (
            result.minutes.ge(min_minutes)
            & result[f"{name}_coverage"].ge(min_coverage)
            & result[f"{name}_per90"].notna()
        )
        indices = result.index[valid]
        result[f"{name}_cohort_peers"] = len(indices)
        result[f"{name}_cohort_percentile"] = np.nan
        if len(indices) < min_peers:
            continue
        values = result.loc[indices, f"{name}_per90"]
        ranks = (values.rank(method="average") - 0.5) / len(values) * 100
        result.loc[indices, f"{name}_cohort_percentile"] = (
            100 - ranks if metric.lower_better else ranks
        )
    return result
