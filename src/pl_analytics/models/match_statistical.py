"""Past-only Elo and penalized Poisson/Dixon–Coles team-strength models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from pl_analytics.statistics.matches import tau


def validate_history(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Require one competition and strictly earlier local-calendar-day labels."""
    frame = history.copy()
    frame["match_day"] = pd.to_datetime(frame.match_day, utc=True)
    if frame.empty or frame.competition_id.nunique() != 1 or frame.match_id.duplicated().any():
        raise ValueError("Training needs unique matches from exactly one competition")
    if frame.match_day.isna().any() or (frame.match_day >= cutoff).any():
        raise ValueError("Training must strictly precede forecast cutoff")
    if frame[["home_club_id", "away_club_id", "season"]].isna().any().any():
        raise ValueError("Club and season identities are required")
    if frame.home_club_id.eq(frame.away_club_id).any():
        raise ValueError("A club cannot play itself")
    goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=float)
    if not np.isfinite(goals).all() or (goals < 0).any() or (goals != np.floor(goals)).any():
        raise ValueError("Training goals must be nonnegative integers")
    return frame.sort_values(["match_day", "match_id"])


@dataclass
class EloModel:
    """Elo expected-score updates; Davidson draw extension for H/D/A prediction."""

    competition_id: str
    ratings: dict[str, float]
    last_season: str
    draw_rate: float
    home_advantage: float
    season_retention: float

    def predict(self, home: str, away: str, season: str) -> tuple[np.ndarray, float, float]:
        """A new season regresses frozen ratings; an unseen club starts at 1500."""
        retention = self.season_retention if season != self.last_season else 1.0
        rh = 1500 + retention * (self.ratings.get(home, 1500) - 1500)
        ra = 1500 + retention * (self.ratings.get(away, 1500) - 1500)
        log_q = np.log(10) * (rh + self.home_advantage - ra) / 400
        nu = 2 * self.draw_rate / (1 - self.draw_rate)
        logits = np.array([log_q / 2, np.log(nu), -log_q / 2])
        weights = np.exp(logits - logits.max())
        return weights / weights.sum(), rh, ra


def fit_elo(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    k: float = 20,
    home_advantage: float = 60,
    season_retention: float = 0.75,
) -> EloModel:
    """Replay in date batches; all matches on one day use the same pre-day state."""
    frame = validate_history(history, cutoff)
    if (
        not np.isfinite([k, home_advantage, season_retention]).all()
        or k <= 0
        or not 0 <= season_retention <= 1
    ):
        raise ValueError("Invalid Elo parameters")
    ratings, last = {}, None
    for _, day in frame.groupby("match_day", sort=True):
        if day.season.nunique() != 1:
            raise ValueError("Overlapping seasons on one date are unsupported")
        season = day.iloc[0].season
        if season != last:
            ratings = {
                club: 1500 + season_retention * (value - 1500) for club, value in ratings.items()
            }
        delta = {}
        for row in day.itertuples():
            home, away = row.home_club_id, row.away_club_id
            expected = expit(
                np.log(10)
                * (ratings.get(home, 1500) + home_advantage - ratings.get(away, 1500))
                / 400
            )
            observed = (
                1
                if row.home_goals > row.away_goals
                else 0
                if row.home_goals < row.away_goals
                else 0.5
            )
            update = k * (observed - expected)
            delta[home] = delta.get(home, 0) + update
            delta[away] = delta.get(away, 0) - update
        for club, change in delta.items():
            ratings[club] = ratings.get(club, 1500) + change
        last = season
    draw_rate = (frame.home_goals.eq(frame.away_goals).sum() + 1) / (len(frame) + 3)
    return EloModel(
        frame.iloc[0].competition_id,
        ratings,
        last,
        float(draw_rate),
        home_advantage,
        season_retention,
    )


def _objective(
    theta: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n: int,
    penalty: float,
    max_rate: float,
) -> tuple[float, np.ndarray]:
    """Weighted joint likelihood and analytic gradient, including low-score correction."""
    attack = theta[:n] - theta[:n].mean()
    defense = theta[n : 2 * n] - theta[n : 2 * n].mean()
    eta_h = theta[-3] + theta[-2] + attack[h] + defense[a]
    eta_a = theta[-3] + attack[a] + defense[h]
    lower, upper = -6.0, np.log(max_rate)
    lam, mu = np.exp(np.clip(eta_h, lower, upper)), np.exp(np.clip(eta_a, lower, upper))
    rho = theta[-1]
    correction = tau(x, y, lam, mu, rho)
    logp = (
        x * np.log(lam)
        - lam
        - gammaln(x + 1)
        + y * np.log(mu)
        - mu
        - gammaln(y + 1)
        + np.log(correction)
    )
    c00, c01, c10, c11 = (
        (x == 0) & (y == 0),
        (x == 0) & (y == 1),
        (x == 1) & (y == 0),
        (x == 1) & (y == 1),
    )
    dl = np.select([c00, c01], [-mu * rho, np.full(len(x), rho)], default=0.0)
    dm = np.select([c00, c10], [-lam * rho, np.full(len(x), rho)], default=0.0)
    dr = np.select([c00, c01, c10, c11], [-lam * mu, lam, mu, -np.ones(len(x))], default=0.0)
    gh = weights * (lam - x - lam * dl / correction) * ((eta_h > lower) & (eta_h < upper))
    ga = weights * (mu - y - mu * dm / correction) * ((eta_a > lower) & (eta_a < upper))
    attack_gradient = np.bincount(h, weights=gh, minlength=n) + np.bincount(
        a, weights=ga, minlength=n
    )
    defense_gradient = np.bincount(a, weights=gh, minlength=n) + np.bincount(
        h, weights=ga, minlength=n
    )
    gradient = np.r_[
        attack_gradient - attack_gradient.mean(),
        defense_gradient - defense_gradient.mean(),
        (gh + ga).sum(),
        gh.sum(),
        -(weights * dr / correction).sum(),
    ]
    gradient[: 2 * n] += penalty * theta[: 2 * n]
    return float(-np.dot(weights, logp) + 0.5 * penalty * np.square(theta[: 2 * n]).sum()), gradient


@dataclass
class GoalModel:
    """Centered attack/defence effects; positive defence effects mean more goals conceded."""

    competition_id: str
    teams: tuple[str, ...]
    theta: np.ndarray
    max_rate: float
    diagnostics: dict

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Unknown clubs receive neutral centered effects, without borrowing another league."""
        n = len(self.teams)
        attack, defense = (
            self.theta[:n] - self.theta[:n].mean(),
            self.theta[n : 2 * n] - self.theta[n : 2 * n].mean(),
        )
        effects = {club: (attack[i], defense[i]) for i, club in enumerate(self.teams)}
        ah, dh = effects.get(home, (0.0, 0.0))
        aa, da = effects.get(away, (0.0, 0.0))
        rates = np.exp(
            np.clip(
                [self.theta[-3] + self.theta[-2] + ah + da, self.theta[-3] + aa + dh],
                -6.0,
                np.log(self.max_rate),
            )
        )
        return float(rates[0]), float(rates[1])

    @property
    def rho(self) -> float:
        return float(self.theta[-1])


def fit_goals(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    dixon_coles: bool = False,
    half_life_days: float | None = None,
    penalty: float = 0.01,
    max_rate: float = 8,
    rho_bounds: tuple[float, float] = (-0.12, 0.015),
) -> GoalModel:
    """Joint penalized maximum likelihood with optional exponential decay in calendar days.

    Rate and rho bounds ensure positive tau for every possible fixture, including
    unseen teams. Fail on optimizer failure instead of publishing a partial fit.
    """
    frame = validate_history(history, cutoff)
    lo, hi = rho_bounds
    if (
        penalty < 0
        or not np.isfinite([penalty, max_rate, lo, hi]).all()
        or max_rate <= 0
        or not -1 / max_rate < lo <= 0 <= hi < min(1.0, 1 / max_rate**2)
    ):
        raise ValueError("Rate/rho bounds must guarantee positive low-score probabilities")
    if half_life_days is not None and (not np.isfinite(half_life_days) or half_life_days <= 0):
        raise ValueError("Half-life must be finite and positive")
    teams = tuple(sorted(set(frame.home_club_id) | set(frame.away_club_id)))
    lookup = {club: i for i, club in enumerate(teams)}
    n = len(teams)
    h, a = frame.home_club_id.map(lookup).to_numpy(), frame.away_club_id.map(lookup).to_numpy()
    x, y = frame.home_goals.to_numpy(dtype=float), frame.away_goals.to_numpy(dtype=float)
    ages = (cutoff - frame.match_day).dt.total_seconds().to_numpy() / 86400
    # Subtract minimum age to avoid underflow; normalized relative weights are unchanged.
    weights = (
        np.ones(len(frame))
        if half_life_days is None
        else np.exp(-np.log(2) * (ages - ages.min()) / half_life_days)
    )
    weights /= weights.sum()
    initial = np.zeros(2 * n + 3)
    initial[-3] = np.log(max(y.mean(), 0.1))
    initial[-2] = np.log(max(x.mean(), 0.1) / max(y.mean(), 0.1))
    fitted = minimize(
        _objective,
        initial,
        args=(h, a, x, y, weights, n, penalty, max_rate),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-2, 2)] * (2 * n) + [(-2, 2), (-1, 1), (lo, hi) if dixon_coles else (0.0, 0.0)],
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Goal-model optimization failed: {fitted.message}")
    return GoalModel(
        frame.iloc[0].competition_id,
        teams,
        fitted.x,
        max_rate,
        {
            "converged": True,
            "iterations": int(fitted.nit),
            "objective": float(fitted.fun),
            "half_life_days": half_life_days,
            "training_matches": len(frame),
            "rho_at_bound": bool(
                dixon_coles and (abs(fitted.x[-1] - lo) < 1e-5 or abs(fitted.x[-1] - hi) < 1e-5)
            ),
        },
    )
