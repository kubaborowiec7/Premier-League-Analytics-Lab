"""Match-resampled uncertainty, explicit-cohort PCA and scoped player similarity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pl_analytics.data.appearances import COUNTS
from pl_analytics.features.players import GRAIN


def bootstrap_per90(
    appearances: pd.DataFrame,
    metric: str,
    *,
    reference_date: str,
    seed: int = 42,
    repetitions: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Resample complete appearance rows and recompute the ratio of totals.

    One player/competition/season only. Zero-minute rows cannot provide exposure.
    Missing counts/minutes invalidate the interval rather than silently biasing it.
    This describes empirical match variation, not future performance coverage.
    """
    if metric not in COUNTS or repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("Use an allowed count, at least 100 draws and confidence in (0, 1)")
    rows = appearances.loc[
        pd.to_datetime(appearances.match_date, utc=True) < pd.to_datetime(reference_date, utc=True)
    ]
    if rows[GRAIN].drop_duplicates().shape[0] > 1:
        raise ValueError("Bootstrap requires one player/competition/season")
    if rows.duplicated(["player_id", "match_id"]).any():
        raise ValueError("Duplicate appearances cannot be resampled")
    data = rows[[metric, "minutes"]].apply(pd.to_numeric, errors="raise")
    result = {
        "n_appearances": len(rows),
        "estimate": None,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "repetitions": repetitions,
        "seed": seed,
    }
    if data.isna().any().any():
        return result
    if (data < 0).any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Bootstrap observations must be finite and nonnegative")
    if ((data.minutes == 0) & (data[metric] > 0)).any():
        return result
    data = data.loc[data.minutes > 0].to_numpy(dtype=float)
    result["n_appearances"] = len(data)
    if not len(data):
        return result
    result["estimate"] = float(90 * data[:, 0].sum() / data[:, 1].sum())
    if len(data) < 2:
        return result
    rng = np.random.default_rng(seed)
    # Chunk draws to keep memory bounded for long appearance histories.
    rates = []
    for start in range(0, repetitions, 200):
        selected = data[rng.integers(0, len(data), size=(min(200, repetitions - start), len(data)))]
        totals = selected.sum(axis=1)
        rates.extend(90 * totals[:, 0] / totals[:, 1])
    alpha = (1 - confidence) / 2
    result["lower"], result["upper"] = map(float, np.quantile(rates, [alpha, 1 - alpha]))
    return result


@dataclass
class PlayerSpace:
    """Fitted scaler/PCA and reference date; only declared performance inputs enter."""

    features: tuple[str, ...]
    scaler: StandardScaler
    pca: PCA
    reference_date: pd.Timestamp
    position_group: str
    position_context: str

    def transform(self, players: pd.DataFrame) -> pd.DataFrame:
        """Project new observations without refitting the reference preprocessing."""
        if not players.position_group.eq(self.position_group).all() or not (
            players.position_context.eq(self.position_context).all()
        ):
            raise ValueError("Projection requires the fitted position/context")
        if (pd.to_datetime(players.reference_date, utc=True) < self.reference_date).any():
            raise ValueError("PCA reference is later than the projected observations")
        matrix = players.loc[:, self.features].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Projection features must be complete and finite")
        result = players[GRAIN + ["reference_date", "position_group", "position_context"]].copy()
        scores = self.pca.transform(self.scaler.transform(matrix))
        for column in range(scores.shape[1]):
            result[f"PC{column + 1}"] = scores[:, column]
        return result


def fit_player_space(
    training: pd.DataFrame,
    *,
    features: tuple[str, ...],
    components: int = 2,
) -> PlayerSpace:
    """Fit a descriptive PCA within one position/context; caller supplies training rows."""
    allowed = {f"{metric}_zscore" for metric in COUNTS}
    if not features or len(set(features)) != len(features) or not set(features) <= allowed:
        raise ValueError("PCA requires unique declared performance z-score features")
    if training.duplicated(GRAIN).any():
        raise ValueError("Duplicate players would distort PCA")
    if training[["position_group", "position_context"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Fit PCA within one position/context")
    matrix = training.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or len(training) < 3:
        raise ValueError("PCA requires at least three complete finite observations")
    if not 1 <= components <= min(len(features), len(training) - 1):
        raise ValueError("Invalid PCA component count")
    if (matrix.std(axis=0) == 0).any():
        raise ValueError("Remove constant features explicitly before PCA")
    scaler = StandardScaler().fit(matrix)
    pca = PCA(n_components=components, svd_solver="full").fit(scaler.transform(matrix))
    return PlayerSpace(
        features,
        scaler,
        pca,
        pd.to_datetime(training.reference_date, utc=True).max(),
        training.iloc[0].position_group,
        training.iloc[0].position_context,
    )


def similar_players(
    reference: pd.Series,
    candidates: pd.DataFrame,
    *,
    candidate_competitions: tuple[str, ...],
    features: tuple[str, ...],
    min_minutes: float = 450,
    limit: int = 5,
) -> pd.DataFrame:
    """Rank by Euclidean distance in declared peer-standardized performance space.

    Same season, reference date and position context only; exclude the reference
    player in every league. A distance is a metric resemblance, never player quality.
    """
    if not candidate_competitions or limit < 1 or min_minutes <= 0:
        raise ValueError("Explicit candidate competitions, positive minutes and limit required")
    allowed = {f"{metric}_zscore" for metric in COUNTS}
    if not features or len(set(features)) != len(features) or not set(features) <= allowed:
        raise ValueError("Similarity accepts only unique performance z-score features")
    vector = reference.loc[list(features)].to_numpy(dtype=float)
    if not np.isfinite(vector).all() or reference.minutes < min_minutes:
        raise ValueError("Reference must have sufficient minutes and complete features")
    selected = candidates.loc[
        candidates.competition_id.isin(candidate_competitions)
        & candidates.season.eq(reference.season)
        & candidates.position_group.eq(reference.position_group)
        & candidates.position_context.eq(reference.position_context)
        & pd.to_datetime(candidates.reference_date, utc=True).eq(
            pd.to_datetime(reference.reference_date, utc=True)
        )
        & candidates.minutes.ge(min_minutes)
        & candidates.player_id.ne(reference.player_id)
    ].copy()
    if selected.duplicated(GRAIN).any():
        raise ValueError("Candidate observations must be unique")
    selected = selected.loc[
        np.isfinite(selected.loc[:, features].to_numpy(dtype=float)).all(axis=1)
    ]
    selected["distance"] = np.linalg.norm(
        selected.loc[:, features].to_numpy(dtype=float) - vector, axis=1
    )
    return selected.sort_values(["distance", "player_id", "competition_id"]).head(limit)
