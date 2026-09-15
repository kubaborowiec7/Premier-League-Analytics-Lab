"""Offline tests for point-in-time features, fitted transforms and frozen evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from pl_analytics.features.value import FEATURES, build_value_features, chronological_parts
from pl_analytics.models.value import (
    ValueModel,
    calibrate_interval,
    candidate_models,
    evaluate_model,
    regression_metrics,
)
from pl_analytics.models.value_experiment import run_experiment


@pytest.fixture
def config():
    return dict(
        competition_id="TEST",
        season_start_month=7,
        lookback_days=365,
        activity_days=90,
        train_start="2020-01-01",
        validation_start="2021-01-01",
        calibration_start="2022-01-01",
        test_start="2023-01-01",
        test_end="2023-12-31",
        primary_metric="mae_eur",
        seed=42,
        interval_alpha=0.1,
        interval_scale_floor_eur=1000000,
    )


@pytest.fixture
def history():
    apps = pd.DataFrame(
        {
            "player_id": ["p"] * 3,
            "game_id": ["a", "b", "c"],
            "competition_id": ["TEST"] * 3,
            "date": ["2019-12-20", "2020-01-10", "2020-02-01"],
            "minutes": [90, 90, 45],
            "goals": [1, 2, 1],
            "assists": [0, 1, 0],
        }
    )
    values = pd.DataFrame(
        {
            "player_id": ["p"] * 3,
            "date": ["2019-12-01", "2020-01-10", "2020-02-01"],
            "market_value_eur": [100, 200, 300],
        }
    )
    people = pd.DataFrame(
        {
            "player_id": ["p"],
            "name": ["Player"],
            "date_of_birth": ["2000-01-01"],
            "position": ["unknown"],
        }
    )
    return apps, values, people


def test_same_day_and_future_facts_are_excluded(history, config):
    apps, values, people = history
    first = build_value_features(apps, values, people, config)
    assert first.iloc[0].minutes_365 == 90
    assert first.iloc[0].previous_value_eur == 100
    apps.loc[apps.date >= "2020-01-10", "goals"] = 999
    values.loc[values.date >= "2020-01-10", "market_value_eur"] = 999999
    people["position"] = "changed future snapshot role"
    changed = build_value_features(apps, values, people, config)
    pd.testing.assert_series_equal(first.iloc[0][list(FEATURES)], changed.iloc[0][list(FEATURES)])
    assert (first.last_appearance_date < first.valuation_date).all()
    assert (first.previous_valuation_date < first.valuation_date).all()
    assert set(first.competition_id) == {"TEST"}
    assert set(first.season) == {"2019/20"}


def test_first_value_and_incomplete_exposure(history, config):
    apps, values, people = history
    apps.loc[0, "minutes"] = np.nan
    result = build_value_features(apps, values.iloc[1:], people, config)
    assert np.isnan(result.iloc[0].previous_value_eur)
    assert np.isnan(result.iloc[0].goals_per90_365)


@pytest.mark.parametrize("kind", ["scope", "duplicate", "negative"])
def test_bad_appearance_history_rejected(history, config, kind):
    apps, values, people = history
    if kind == "scope":
        apps.loc[0, "competition_id"] = "OTHER"
    elif kind == "duplicate":
        apps = pd.concat([apps, apps.iloc[:1]])
    else:
        apps.loc[0, "minutes"] = -1
    with pytest.raises(ValueError):
        build_value_features(apps, values, people, config)


def test_chronological_boundaries(config):
    dates = [
        config[key]
        for key in (
            "train_start",
            "validation_start",
            "calibration_start",
            "test_start",
            "test_end",
        )
    ]
    parts = chronological_parts(pd.DataFrame({"valuation_date": dates}), config)
    assert [len(part) for part in parts.values()] == [1, 1, 1, 2]
    assert sum(len(part) for part in parts.values()) == len(dates)
    config["test_start"] = config["calibration_start"]
    with pytest.raises(ValueError, match="increasing"):
        chronological_parts(pd.DataFrame({"valuation_date": dates}), config)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(1, 10, (80, len(FEATURES))), columns=FEATURES)
    frame["previous_value_eur"] *= 1000000
    frame["market_value_eur"] = frame.previous_value_eur * 0.9 + frame.minutes_365 * 10000
    frame.loc[0, "age_years"] = np.nan
    return frame, candidate_models(frame)


def test_candidates_preprocessing_and_shap_additivity(fitted):
    frame, models = fitted
    assert len(models) == 12
    for model in models:
        contributions, base = model.explain(frame.iloc[:5])
        if model.pipeline is not None:
            expected = model.pipeline.predict(frame.iloc[:5].loc[:, FEATURES])
            imputer = model.pipeline[0]
            original = imputer.statistics_.copy()
            model.predict(frame.assign(age_years=100000))
            np.testing.assert_array_equal(original, imputer.statistics_)
        else:
            expected = model.predict(frame.iloc[:5])
        np.testing.assert_allclose(contributions.sum(axis=1) + base, expected, rtol=1e-5, atol=1)
        assert (model.predict(frame) >= 0).all()


def test_interval_uses_calibration_order_statistic():
    frame = pd.DataFrame(
        {"previous_value_eur": np.full(20, 100), "market_value_eur": np.arange(20) + 100}
    )
    model = ValueModel("persistence", "eur", None, 100)
    assert calibrate_interval(model, frame, alpha=0.1, floor=100) == 0.18
    with pytest.raises(ValueError, match="calibration"):
        calibrate_interval(model, frame.iloc[:1], alpha=0.1, floor=100)


def test_metrics_and_residual_direction():
    metrics = regression_metrics(np.array([1.0, 4.0]), np.array([2.0, 2.0]))
    assert metrics["mae_eur"] == 1.5
    assert metrics["rmse_eur"] == pytest.approx(np.sqrt(2.5))
    frame = pd.DataFrame(
        dict(
            player_id=["p"],
            player_name=["Player"],
            competition_id=["TEST"],
            season=["2020/21"],
            valuation_date=["2020-09-01"],
            cohort_context=["recent_prior_competition_participant"],
            age_years=[20],
            market_value_eur=[50],
            previous_value_eur=[100],
        )
    )
    result = evaluate_model(ValueModel("persistence", "eur", None, 100), frame, 0.1, 100)
    assert result.iloc[0].model_undervaluation_eur == 50
    assert result.iloc[0].assessment == "below_model_interval"


def test_final_requires_unchanged_selection_before_reading_data(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        run_experiment(config, tmp_path, tmp_path, stage="final")
    (tmp_path / "selection.json").write_text(json.dumps({"config_sha256": "wrong"}))
    with pytest.raises(ValueError, match="Configuration"):
        run_experiment(config, tmp_path, tmp_path, stage="final")
    with pytest.raises(ValueError, match="already frozen"):
        run_experiment(config, tmp_path, tmp_path, stage="select")


def test_selection_reads_only_pretest_targets(monkeypatch, tmp_path, config):
    def stop_before_training(config, raw_dir, *, end_date, download):
        assert end_date == "2022-12-31"
        raise RuntimeError("stage boundary checked")

    monkeypatch.setattr(
        "pl_analytics.models.value_experiment.read_value_history", stop_before_training
    )
    with pytest.raises(RuntimeError, match="stage boundary checked"):
        run_experiment(config, tmp_path, tmp_path, stage="select")
