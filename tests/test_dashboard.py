"""Offline UI navigation, artifact isolation and inference safety checks."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import ArtifactError, catalog, report, scenario, table
from pl_analytics.features.players import aggregate_players, fit_peers
from pl_analytics.models.match_statistical import GoalModel

APP = Path(__file__).resolve().parents[1] / "app/Home.py"
PAGES = [
    "1_Player_Explorer",
    "2_Scouting_Finder",
    "3_Market_Value",
    "4_Match_Predictor",
    "5_Model_Lab",
]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ACTIVE_COMPETITIONS", "TEST,SECOND")
    for target in (
        "sqlalchemy.create_engine",
        "httpx.Client.send",
        "pl_analytics.models.match_ml.fit_classifiers",
        "pl_analytics.models.match_statistical.fit_goals",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError("Unexpected external work")))
    return Settings()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_without_artifacts(isolated, page):
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page(f"pages/{page}.py").run()
    assert not app.exception and not app.error
    assert "artifacts are not built yet" in app.info[0].value


def test_real_player_navigation_with_synthetic_competition(isolated):
    appearances = pd.DataFrame(
        [
            dict(
                player_id=f"p{p}",
                player_name=f"Player {p}",
                competition_id="SECOND",
                season="2023/24",
                match_id=f"m{m}",
                club_id="club",
                match_date=f"2024-01-{m + 1:02d}",
                position_group="ST",
                position_context="synthetic",
                minutes=90,
                goals=p if m == 0 else 0,
                assists=p % 2,
                yellow_cards=0,
                red_cards=0,
            )
            for p in range(6)
            for m in range(6)
        ]
    )
    raw = aggregate_players(appearances, "2024-02-01")
    players = fit_peers(raw, min_minutes=450, min_peers=3).transform(raw)
    destination = isolated.data_dir / "processed/m4/player_features.parquet"
    destination.parent.mkdir(parents=True)
    players.to_parquet(destination)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert app.sidebar.selectbox[0].options == ["SECOND"]
    app.switch_page("pages/1_Player_Explorer.py").run()
    assert not app.exception and not app.error
    assert len(app.metric) == 4
    app.switch_page("pages/2_Scouting_Finder.py").run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 5
    app.multiselect[1].set_value([]).run()
    assert "Select at least one" in app.info[0].value


def test_corrupt_artifact_does_not_hide_independent_modules(isolated):
    target = isolated.data_dir / "processed/m4/player_features.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("private-password")
    frames, errors = catalog(isolated)
    assert all(frame.empty for frame in frames.values())
    assert len(errors) == 1 and "private-password" not in errors[0]
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception and app.warning


def test_cache_invalidates_and_report_rejects_nonobject(tmp_path):
    path = tmp_path / "data.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path)
    assert table(path).value.iloc[0] == 1
    pd.DataFrame({"value": [2, 3]}).to_parquet(path)
    assert table(path).value.tolist() == [2, 3]
    target = tmp_path / "report.json"
    target.write_text(json.dumps(["not an object"]))
    with pytest.raises(ArtifactError, match="cannot be read"):
        report(target)


def test_score_inference_has_no_training_and_validates_context():
    model = GoalModel("TEST", ("a", "b"), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3, -0.05]), 8, {})
    statistics = {"competition_id": "TEST", "origin": "2024-01-01", "models": {"dc": model}}
    states = pd.DataFrame(
        {"club_id": ["a", "b"], "competition_id": ["TEST"] * 2, "season": ["2023/24"] * 2}
    )
    result = scenario("a", "b", "TEST", "dc", statistics, {}, states)
    assert result["matrix"].sum() == pytest.approx(1)
    assert result["probabilities"].sum() == pytest.approx(1)
    assert result["probabilities"][0] > result["probabilities"][2]
    for home, away, competition in [
        ("a", "a", "TEST"),
        ("a", "b", "SECOND"),
        ("a", "missing", "TEST"),
    ]:
        with pytest.raises(ArtifactError):
            scenario(home, away, competition, "dc", statistics, {}, states)
