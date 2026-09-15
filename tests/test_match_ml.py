"""No-network tests for monthly features, training transforms and calibration isolation."""

import numpy as np
import pandas as pd
import pytest

from pl_analytics.features.matches import (
    MATCH_FEATURES,
    build_match_features,
    match_features_at_origin,
)
from pl_analytics.models.match_ml import fit_classifiers, fit_temperature, scale_temperature
from pl_analytics.models.match_ml_experiment import run_match_ml
from pl_analytics.statistics.matches import probability_metrics


@pytest.fixture
def experiment():
    rng = np.random.default_rng(91)
    rows = []
    for day in pd.date_range("2019-01-01", "2023-06-01", freq="7D", tz="UTC"):
        home, away = rng.choice(["a", "b", "c", "d"], 2, replace=False)
        rows.append(
            dict(
                match_id=str(len(rows)),
                competition_id="T",
                season=str(day.year),
                match_day=day,
                home_club_id=home,
                away_club_id=away,
                home_goals=int(rng.poisson(1.5)),
                away_goals=int(rng.poisson(1.1)),
            )
        )
    config = dict(
        competition_id="T",
        train_start="2020-01-01",
        validation_start="2021-01-01",
        calibration_start="2022-01-01",
        test_start="2023-01-01",
        test_end="2023-02-28",
        feature_elo_k=20,
        elo_k=[20],
        elo_home_advantage=60,
        elo_season_retention=0.75,
        rolling_matches=5,
        lookback_days=1095,
        min_feature_matches=1,
        min_train_matches=1,
        logistic_c=[0.1],
        xgb_depths=[2],
        xgb_estimators=20,
        xgb_learning_rate=0.03,
        temperature_bounds=[0.5, 5.0],
        seed=42,
        primary_metric="log_loss",
        half_lives=[None],
        goal_penalty=0.01,
        max_rate=8,
        rho_bounds=[-0.12, 0.015],
        max_goals=30,
    )
    return pd.DataFrame(rows), config


def test_monthly_features_are_shifted_and_future_invariant(experiment):
    frame, config = experiment
    features = build_match_features(frame, config)
    frame.loc[frame.match_day >= "2021-01-01", "home_goals"] = 99
    changed = build_match_features(frame, config)
    mask = features.match_day < "2021-02-01"
    pd.testing.assert_frame_equal(
        features.loc[mask, list(MATCH_FEATURES)], changed.loc[mask, list(MATCH_FEATURES)]
    )
    assert (features.feature_history_end < features.origin).all()
    assert features.home_recent_matches.between(0, 5).all()
    assert features.home_points_recent.between(0, 3).all()


def test_unknown_team_features_remain_missing(experiment):
    frame, config = experiment
    origin = pd.Timestamp("2023-01-01", tz="UTC")
    history = frame.loc[frame.match_day < origin]
    fixtures = frame.loc[frame.match_day >= origin].iloc[:1].assign(home_club_id="unseen")
    features = match_features_at_origin(history, fixtures, origin, config)
    assert features.iloc[0].home_recent_matches == 0
    assert pd.isna(features.iloc[0].home_points_recent)
    with pytest.raises(ValueError, match="precede"):
        match_features_at_origin(frame, fixtures, origin, config)


def test_temperature_normalization_and_separate_fit():
    p = np.tile([0.99, 0.005, 0.005], (60, 1))
    labels = np.tile([0, 1, 2], 20)
    temperature = fit_temperature(p, labels)
    calibrated = scale_temperature(p, temperature)
    assert temperature > 1
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    assert (
        probability_metrics(labels, calibrated)["log_loss"]
        < probability_metrics(labels, p)["log_loss"]
    )
    np.testing.assert_array_equal(calibrated.argmax(axis=1), p.argmax(axis=1))
    np.testing.assert_allclose(scale_temperature(p, 1), p)
    with pytest.raises(ValueError):
        fit_temperature(p[:2], labels[:2])


def test_classifier_transforms_remain_training_only(experiment):
    frame, config = experiment
    features = build_match_features(frame, config)
    training = features.loc[features.match_day < config["validation_start"]]
    for model in fit_classifiers(training, config):
        before = model.pipeline[0].statistics_.copy()
        probabilities = model.predict(features.assign(elo_difference=100000))
        np.testing.assert_array_equal(before, model.pipeline[0].statistics_)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        assert probabilities.shape == (len(features), 3)


def test_frozen_classifier_experiment_round_trip(experiment, monkeypatch, tmp_path):
    frame, config = experiment
    read_ends = []

    def read(config, raw_dir, *, end_date, download):
        read_ends.append(end_date)
        return frame.loc[frame.match_day <= pd.Timestamp(end_date, tz="UTC")].copy(), []

    monkeypatch.setattr("pl_analytics.models.match_ml_experiment.read_match_history", read)
    selected = run_match_ml(config, tmp_path, tmp_path, stage="select")
    before = (tmp_path / "classifiers.joblib").read_bytes()
    final = run_match_ml(config, tmp_path, tmp_path, stage="final")
    assert read_ends == ["2022-12-31", "2023-02-28"]
    assert final["chosen"] == selected["chosen"] + "_calibrated"
    assert len({m["n"] for m in final["metrics"].values()}) == 1
    assert "poisson_flat" in final["metrics"]
    assert before == (tmp_path / "classifiers.joblib").read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        run_match_ml(config, tmp_path, tmp_path, stage="select")
    config["rolling_matches"] = 9
    with pytest.raises(ValueError, match="changed"):
        run_match_ml(config, tmp_path, tmp_path, stage="final")
