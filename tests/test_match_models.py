"""Offline probability, gradient, temporal-isolation and rolling-policy checks."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import poisson

from pl_analytics.models.match_experiment import (
    forecast_origin,
    rolling_backtest,
    run_match_experiment,
)
from pl_analytics.models.match_statistical import _objective, fit_elo, fit_goals
from pl_analytics.statistics.matches import (
    calibration_bins,
    outcome_probabilities,
    probability_metrics,
    score_matrix,
)


@pytest.fixture
def config():
    return dict(
        competition_id="TEST",
        elo_k=[20],
        elo_home_advantage=60,
        elo_season_retention=0.75,
        half_lives=[None, 365],
        goal_penalty=0.01,
        max_rate=8,
        rho_bounds=[-0.12, 0.015],
        max_goals=30,
        lookback_days=1095,
        min_train_matches=10,
        validation_start="2022-01-01",
        test_start="2023-01-01",
        test_end="2023-12-31",
        primary_metric="log_loss",
        seed=42,
    )


@pytest.fixture
def history():
    rng = np.random.default_rng(42)
    rows = []
    teams = ["a", "b", "c", "d"]
    for day in pd.date_range("2020-01-01", "2023-12-01", freq="7D", tz="UTC"):
        home, away = rng.choice(teams, 2, replace=False)
        rows.append(
            dict(
                match_id=str(len(rows)),
                competition_id="TEST",
                season=str(day.year),
                match_day=day,
                home_club_id=home,
                away_club_id=away,
                home_goals=int(rng.poisson(1.7)),
                away_goals=int(rng.poisson(1.1)),
            )
        )
    return pd.DataFrame(rows)


def test_poisson_grid_and_outcome_orientation():
    matrix, tail = score_matrix(2.0, 1.0)
    assert matrix.sum() == pytest.approx(1, abs=1e-12)
    assert matrix[2, 1] == pytest.approx(poisson.pmf(2, 2) * poisson.pmf(1, 1))
    p = outcome_probabilities(matrix)
    assert p.sum() == pytest.approx(1)
    assert p[0] > p[2]
    assert tail < 1e-10


@pytest.mark.parametrize("rho", [-0.12, 0.015])
def test_dixon_coles_mass_marginals_and_extreme_rates(rho):
    for lam, mu in [(1.5, 1.0), (8.0, 8.0), (0.01, 8.0)]:
        matrix, tail = score_matrix(lam, mu, rho)
        independent, _ = score_matrix(lam, mu)
        assert (matrix >= 0).all()
        assert matrix.sum() == pytest.approx(1)
        np.testing.assert_allclose(matrix.sum(axis=0), independent.sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(matrix.sum(axis=1), independent.sum(axis=1), atol=1e-12)
        assert tail < 1e-6
    negative, _ = score_matrix(1.5, 1.0, -0.1)
    independent, _ = score_matrix(1.5, 1.0)
    assert negative[0, 0] > independent[0, 0]
    assert negative[1, 1] > independent[1, 1]


@pytest.mark.parametrize("args", [(1.0, 1.0, 2.0), (-1.0, 1.0, 0.0), (np.nan, 1.0, 0.0)])
def test_invalid_score_distribution_rejected(args):
    with pytest.raises(ValueError):
        score_matrix(*args)


def test_truncated_tail_cannot_silently_disappear():
    with pytest.raises(ValueError, match="grid"):
        score_matrix(8, 8, max_goals=3)


def test_known_probability_scores_and_calibration():
    y = np.array([0, 1, 2])
    perfect = probability_metrics(y, np.eye(3))
    assert perfect == dict(n=3, log_loss=0.0, brier=0.0, rps=0.0, accuracy=1.0)
    uniform = probability_metrics(y, np.full((3, 3), 1 / 3))
    assert uniform["log_loss"] == pytest.approx(np.log(3))
    assert uniform["brier"] == pytest.approx(2 / 3)
    assert uniform["rps"] == pytest.approx(2 / 9)
    bins = calibration_bins(y, np.eye(3))
    assert sum(row["n"] for row in bins) == 9
    assert any(row["observed"] is None for row in bins)
    with pytest.raises(ValueError, match="normalized"):
        probability_metrics(y, np.ones((3, 3)))


def test_likelihood_analytic_gradient():
    n = 3
    h, a = np.array([0, 1, 2, 0, 1, 2]), np.array([1, 2, 0, 2, 0, 1])
    x, y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0, 2.0])
    theta = np.array([0.1, -0.2, 0.1, 0.05, 0.1, -0.15, 0.2, 0.15, -0.05])
    args = (h, a, x, y, np.arange(1, 7) / 21, n, 0.01, 8.0)
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], epsilon=1e-7)
    _, analytic = _objective(theta, *args)
    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=1e-4)


def test_elo_known_update_and_same_day_batch():
    frame = pd.DataFrame(
        [
            dict(
                match_id="1",
                competition_id="T",
                season="S",
                match_day=pd.Timestamp("2020-01-01", tz="UTC"),
                home_club_id="a",
                away_club_id="b",
                home_goals=1,
                away_goals=0,
            )
        ]
    )
    cutoff = pd.Timestamp("2020-01-02", tz="UTC")
    model = fit_elo(frame, cutoff, k=20, home_advantage=0)
    assert model.ratings == {"a": 1510, "b": 1490}
    reversed_row = frame.assign(match_id="2", home_club_id="b", away_club_id="a")
    tied = fit_elo(pd.concat([frame, reversed_row]), cutoff, k=20, home_advantage=0)
    assert tied.ratings == {"a": 1500, "b": 1500}
    probabilities, rh, _ = model.predict("new", "a", "S2")
    assert rh == 1500
    assert probabilities.sum() == pytest.approx(1)


@pytest.mark.parametrize("fit", [fit_elo, fit_goals])
def test_fit_rejects_future_or_mixed_competitions(history, fit):
    cutoff = history.match_day.max()
    with pytest.raises(ValueError, match="precede"):
        fit(history, cutoff)
    history.loc[0, "competition_id"] = "OTHER"
    with pytest.raises(ValueError, match="competition"):
        fit(history, cutoff + pd.Timedelta(days=1))


def test_goal_fitting_is_finite_and_unknown_teams_are_supported(history):
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    for dc in [False, True]:
        model = fit_goals(history, cutoff, dixon_coles=dc, half_life_days=365)
        lam, mu = model.rates("unknown", "a")
        assert 0 < lam <= 8 and 0 < mu <= 8
        assert model.diagnostics["converged"]
        assert model.rho == 0 if not dc else -0.12 <= model.rho <= 0.015
        score_matrix(lam, mu, model.rho)


def test_forecast_never_reads_fixture_outcomes(history, config):
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    train = history.loc[history.match_day < cutoff]
    fixtures = history.loc[history.match_day >= cutoff].iloc[:2]
    first, _, _ = forecast_origin(train, fixtures, cutoff, config)
    changed, _, _ = forecast_origin(
        train, fixtures.assign(home_goals=999, away_goals=999), cutoff, config
    )
    pd.testing.assert_frame_equal(first, changed)
    assert (first.training_end_day < first.origin).all()
    assert (first.origin <= first.match_day).all()
    with pytest.raises(ValueError, match="competition"):
        forecast_origin(train, fixtures.assign(competition_id="OTHER"), cutoff, config)


def test_rolling_backtest_future_invariance(history, config, tmp_path):
    config["half_lives"] = [None]
    first = rolling_backtest(
        history, config, start="2023-01-01", end="2023-02-28", output=tmp_path / "a"
    )
    history.loc[history.match_day >= "2023-02-01", "home_goals"] = 99
    changed = rolling_backtest(
        history, config, start="2023-01-01", end="2023-02-28", output=tmp_path / "b"
    )
    cols = ["match_id", "model", "p_home", "p_draw", "p_away"]
    pd.testing.assert_frame_equal(first[cols], changed[cols])
    assert first.groupby("match_id").size().eq(4).all()


def test_final_guard_precedes_any_data_access(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_match_experiment(config, tmp_path, tmp_path, stage="final")
    (tmp_path / "selection.json").write_text(json.dumps({"config_sha256": "bad"}))
    with pytest.raises(ValueError, match="policy"):
        run_match_experiment(config, tmp_path, tmp_path, stage="final")


def test_selection_never_reads_final_season(monkeypatch, config, tmp_path):
    def read(config, root, *, end_date, download):
        assert end_date == "2022-12-31"
        raise RuntimeError("cutoff verified")

    monkeypatch.setattr("pl_analytics.models.match_experiment.read_match_history", read)
    with pytest.raises(RuntimeError, match="cutoff verified"):
        run_match_experiment(config, tmp_path, tmp_path, stage="select")
