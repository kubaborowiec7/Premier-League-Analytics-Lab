"""Validated outcome probabilities and normalized, tail-audited score matrices."""

import numpy as np
from scipy.stats import poisson

OUTCOMES = ("H", "D", "A")


def tau(
    home: np.ndarray, away: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Dixon–Coles correction for the four low-scoring cells."""
    return np.select(
        [
            (home == 0) & (away == 0),
            (home == 0) & (away == 1),
            (home == 1) & (away == 0),
            (home == 1) & (away == 1),
        ],
        [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho],
        default=1.0,
    )


def score_matrix(
    lam: float, mu: float, rho: float = 0.0, *, max_goals: int = 30
) -> tuple[np.ndarray, float]:
    """Rows are home goals; columns away goals. Reject invalid dependence/truncation."""
    if not np.isfinite([lam, mu, rho]).all() or min(lam, mu) <= 0 or max_goals < 1:
        raise ValueError("Finite positive goal rates and a positive score limit are required")
    home, away = np.indices((max_goals + 1, max_goals + 1))
    correction = tau(home, away, lam, mu, rho)
    if (correction <= 0).any():
        raise ValueError("Rho produces nonpositive score probabilities")
    matrix = poisson.pmf(home, lam) * poisson.pmf(away, mu) * correction
    mass = float(matrix.sum())
    if mass < 1 - 1e-6:
        raise ValueError("Score grid discards too much probability; increase max_goals")
    return matrix / mass, max(0.0, 1 - mass)


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    """H/D/A probabilities from the home-row / away-column score convention."""
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


def probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict:
    """Unscaled multiclass Brier; RPS averages the first two cumulative H/D/A errors."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(actual)
    if p.shape != (len(y), 3) or not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Expected nonempty labels 0/1/2 and three probabilities per row")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Probabilities must be finite, nonnegative and normalized")
    observed = np.eye(3)[y.astype(int)]
    y = y.astype(int)
    return {
        "n": len(y),
        "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
        "brier": float(np.square(p - observed).sum(axis=1).mean()),
        "rps": float(np.square(np.cumsum(p - observed, axis=1)[:, :2]).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def calibration_bins(actual: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    """Fixed-width class-wise reliability bins, preserving empty bins as null rates."""
    probability_metrics(actual, probabilities)
    result = []
    for column, label in enumerate(OUTCOMES):
        assigned = np.minimum((probabilities[:, column] * bins).astype(int), bins - 1)
        for index in range(bins):
            mask = assigned == index
            result.append(
                {
                    "outcome": label,
                    "bin": index,
                    "n": int(mask.sum()),
                    "predicted": float(probabilities[mask, column].mean()) if mask.any() else None,
                    "observed": float((actual[mask] == column).mean()) if mask.any() else None,
                }
            )
    return result
