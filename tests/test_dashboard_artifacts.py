"""Trusted model loading fails closed and populated research pages retain caveats."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import ArtifactError, prepared_models

APP = Path(__file__).resolve().parents[1] / "app/Home.py"


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ACTIVE_COMPETITIONS", "TEST")
    settings = Settings()
    root = settings.artifact_dir
    for folder in ("dashboard", "m7/statistical_benchmarks", "m5", "m6"):
        (root / folder).mkdir(parents=True)
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    pd.DataFrame(
        {
            "club_id": ["a", "b"],
            "competition_id": ["TEST"] * 2,
            "season": ["2023/24"] * 2,
            "elo": [1500, 1500],
            "origin": [origin] * 2,
        }
    ).to_parquet(root / "dashboard/team_states.parquet")
    paths = {
        "statistics": root / "m7/statistical_benchmarks/latest_origin_models.joblib",
        "ml": root / "m7/classifiers.joblib",
    }
    joblib.dump(
        {
            "competition_id": "TEST",
            "origin": origin,
            "models": {"base_rate": np.array([0.4, 0.3, 0.3])},
        },
        paths["statistics"],
    )
    joblib.dump({"models": [], "chosen": "none"}, paths["ml"])
    metadata = {
        "origin": origin.isoformat(),
        "models": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
    }
    (root / "dashboard/catalog.json").write_text(json.dumps(metadata))
    return settings


def test_model_digest_checked_even_after_cached_load(artifacts):
    statistics, _, states, _ = prepared_models(artifacts)
    assert statistics["competition_id"] == "TEST" and len(states) == 2
    path = artifacts.artifact_dir / "m7/classifiers.joblib"
    path.write_bytes(b"tampered model")
    with pytest.raises(ArtifactError, match="missing or incompatible"):
        prepared_models(artifacts)


def test_mixed_forecast_origins_rejected(artifacts):
    path = artifacts.artifact_dir / "dashboard/team_states.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "origin"] += pd.Timedelta(days=1)
    frame.to_parquet(path)
    with pytest.raises(ArtifactError, match="missing or incompatible"):
        prepared_models(artifacts)


def test_value_filters_intervals_and_all_experiment_reports(artifacts):
    root = artifacts.artifact_dir
    pd.DataFrame(
        [
            dict(
                player_id="p",
                player_name="Example Player",
                competition_id="TEST",
                season="2023/24",
                valuation_date="2024-01-01",
                minutes_365=900,
                low_exposure=False,
                missing_previous_value=False,
                market_value_eur=1e6,
                predicted_value_eur=1.2e6,
                lower_eur=0.2e6,
                upper_eur=2e6,
                model_undervaluation_eur=0.2e6,
                uncertainty_scaled_gap=0.1,
                assessment="uncertain",
            )
        ]
    ).to_parquet(root / "m5/value_ranking.parquet")
    for experiment in ("m5", "m6", "m7"):
        (root / experiment / "evaluation.json").write_text(
            json.dumps(
                {
                    "chosen": "baseline",
                    "config": {"competition_id": "TEST"},
                    "metrics": {"baseline": {"n": 40, "log_loss": 1.0}},
                }
            )
        )
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page("pages/3_Market_Value.py").run()
    assert not app.error and not app.exception and len(app.metric) == 3
    assert any("Nominal 90%" in item.value for item in app.markdown)
    app.checkbox[0].check().run()
    assert not app.error
    app.switch_page("pages/5_Model_Lab.py").run()
    for experiment in ("m5", "m6", "m7"):
        app.selectbox[0].select(experiment).run()
        assert not app.error and not app.exception and len(app.dataframe) == 1


def test_missing_model_is_explained_when_other_data_exist(artifacts):
    root = artifacts.artifact_dir
    pd.DataFrame(
        [
            dict(
                competition_id="TEST",
                season="2023/24",
                match_id="m",
                model="base_rate",
                match_day="2024-01-01",
                home_club_id="a",
                away_club_id="b",
                p_home=0.4,
                p_draw=0.3,
                p_away=0.3,
            )
        ]
    ).to_parquet(root / "m7/predictions.parquet")
    (root / "m7/classifiers.joblib").unlink()
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page("pages/4_Match_Predictor.py").run()
    assert not app.exception and not app.error
    assert "missing or incompatible" in app.info[0].value


def test_match_button_uses_prepared_base_rate(artifacts):
    root = artifacts.artifact_dir
    pd.DataFrame(
        [
            dict(
                competition_id="TEST",
                season="2023/24",
                match_id="m",
                model="base_rate",
                match_day="2024-01-01",
                home_club_id="a",
                away_club_id="b",
                p_home=0.4,
                p_draw=0.3,
                p_away=0.3,
            )
        ]
    ).to_parquet(root / "m7/predictions.parquet")
    pd.DataFrame(
        {"club_id": ["a", "b"], "competition_id": ["TEST"] * 2, "name": ["Alpha", "Beta"]}
    ).to_parquet(root / "dashboard/clubs.parquet")
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page("pages/4_Match_Predictor.py").run()
    app.button[0].click().run()
    assert not app.error and not app.exception
    assert [metric.value for metric in app.metric] == ["40.0%", "30.0%", "30.0%"]
    assert any("does not produce scorelines" in item.value for item in app.info)
