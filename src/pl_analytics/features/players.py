"""As-of player counts and exposure-aware descriptive peer normalization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pl_analytics.data.appearances import COUNTS

GRAIN = ["player_id", "competition_id", "season"]
PEERS = ["competition_id", "season", "position_group", "position_context"]


def aggregate_players(appearances: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Aggregate strictly before the reference date; incomplete totals remain missing."""
    frame = appearances.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    cutoff = pd.to_datetime(reference_date, utc=True)
    if pd.isna(cutoff) or frame["match_date"].isna().any():
        raise ValueError("Valid match dates and a reference date are required")
    frame = frame.loc[frame["match_date"] < cutoff].copy()
    if frame.empty:
        raise ValueError("No appearances precede the reference date")
    if frame.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Duplicate appearance observations would double-count performance")
    for column in ["minutes", *COUNTS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        observed = frame[column].dropna()
        if (observed < 0).any() or not np.isfinite(observed).all():
            raise ValueError(f"{column} must contain finite nonnegative observations")
    if frame[GRAIN + ["position_group", "position_context"]].isna().any().any():
        raise ValueError("Player, competition, season and position context are required")
    records = []
    for keys, group in frame.groupby(GRAIN, sort=True):
        if group[["position_group", "position_context"]].drop_duplicates().shape[0] != 1:
            raise ValueError("Conflicting position contexts require an explicit reconciliation")
        record = dict(zip(GRAIN, keys, strict=True))
        record.update(
            player_name=group.iloc[0]["player_name"],
            position_group=group.iloc[0]["position_group"],
            position_context=group.iloc[0]["position_context"],
            reference_date=cutoff.isoformat(),
            last_match_date=group.match_date.max().isoformat(),
            appearances=len(group),
            recorded_clubs=int(group.club_id.nunique()),
        )
        for column in ["minutes", *COUNTS]:
            record[column] = group[column].sum() if group[column].notna().all() else np.nan
        for metric in COUNTS:
            record[f"{metric}_per90"] = (
                90 * record[metric] / record["minutes"] if record["minutes"] > 0 else np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


@dataclass
class PeerReference:
    """Explicit fitted cohort; transform never recomputes statistics from candidates."""

    cohort: pd.DataFrame
    min_minutes: float = 450
    prior_minutes: float = 900
    min_peers: int = 5

    def transform(self, players: pd.DataFrame) -> pd.DataFrame:
        """Attach reference percentiles, population z-scores and exposure shrinkage.

        Tied values use mid-distribution ranks (all-equal percentile 50). A constant
        reference has undefined z-scores. Small groups and low exposure stay unranked.
        """
        # Concatenated frames may reuse index labels; updates must stay row-local.
        result = players.reset_index(drop=True).copy()
        result["reliability_weight"] = result.minutes / (result.minutes + self.prior_minutes)
        result["normalization_context"] = "competition-season-position-context"
        result["ranking_eligible"] = result.minutes.ge(self.min_minutes)
        for metric in COUNTS:
            for suffix in ("percentile", "zscore", "shrunk_per90", "peer_count"):
                result[f"{metric}_{suffix}"] = np.nan
        for index, player in result.iterrows():
            peers = self.cohort
            for key in PEERS:
                peers = peers.loc[peers[key].eq(player[key])]
            if peers.empty or player.position_group == "UNKNOWN":
                continue
            if pd.to_datetime(peers.reference_date, utc=True).max() > pd.to_datetime(
                player.reference_date, utc=True
            ):
                raise ValueError("Peer reference uses information after the target reference date")
            for metric in COUNTS:
                observed = peers.loc[peers[f"{metric}_per90"].notna()]
                n = len(observed)
                result.loc[index, f"{metric}_peer_count"] = n
                value = player[f"{metric}_per90"]
                if n < self.min_peers or pd.isna(value):
                    continue
                reference = observed[f"{metric}_per90"].to_numpy(dtype=float)
                prior = 90 * observed[metric].sum() / observed.minutes.sum()
                weight = player.reliability_weight
                result.loc[index, f"{metric}_shrunk_per90"] = weight * value + (1 - weight) * prior
                if not player.ranking_eligible:
                    continue
                result.loc[index, f"{metric}_percentile"] = (
                    100 * ((reference < value).sum() + 0.5 * (reference == value).sum()) / n
                )
                sd = reference.std(ddof=0)
                if sd > 0:
                    result.loc[index, f"{metric}_zscore"] = (value - reference.mean()) / sd
        result.index = players.index
        return result


def fit_peers(
    training: pd.DataFrame,
    *,
    min_minutes: float = 450,
    prior_minutes: float = 900,
    min_peers: int = 5,
) -> PeerReference:
    """Fit on an explicitly supplied cohort; thresholds are documented heuristics."""
    if min_minutes <= 0 or prior_minutes <= 0 or min_peers < 2:
        raise ValueError("Positive exposure thresholds and at least two peers are required")
    if training.duplicated(GRAIN).any():
        raise ValueError("Peer cohort must have one row per player/competition/season")
    return PeerReference(
        training.loc[training.minutes.ge(min_minutes)].copy(), min_minutes, prior_minutes, min_peers
    )
