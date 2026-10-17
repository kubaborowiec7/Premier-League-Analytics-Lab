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
from pl_analytics.dashboard.formatting import top_percent_label
from pl_analytics.dashboard.match_centre import club_matches
from pl_analytics.dashboard.refresh import (
    RefreshAlreadyRunningError,
    refresh_lock,
    run_current_refresh,
)
from pl_analytics.features.fpl import (
    fixture_rows,
    latest_players,
    league_table,
    match_results,
    project_players,
)
from pl_analytics.features.fpl_optimizer import backtest_summary, rolling_backtest
from pl_analytics.features.players import aggregate_players, fit_peers
from pl_analytics.models.match_statistical import GoalModel
from pl_analytics.statistics.season import simulate_season

APP = Path(__file__).resolve().parents[1] / "app/Home.py"
PAGES = [
    "1_Player_Explorer",
    "2_Scouting_Finder",
    "3_Teams",
    "3_League_Table",
    "4_FPL_Decision_Lab",
    "3_Market_Value",
    "4_Match_Predictor",
    "5_Model_Lab",
    "6_Data_Coverage",
    "7_Betting_Lab",
    "8_Match_Centre",
]


def test_percentiles_are_presented_as_top_share():
    assert top_percent_label(90) == "Top 10%"
    assert top_percent_label(99.9) == "Top 1%"
    assert top_percent_label(50) == "Top 50%"
    assert top_percent_label(float("nan")) == "—"


def test_data_coverage_exposes_explicit_refresh_action(isolated):
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page("pages/6_Data_Coverage.py").run(timeout=20)
    assert any(button.label == "Refresh data now" for button in app.button)


def test_refresh_runner_serializes_processes(tmp_path, monkeypatch):
    root = tmp_path / "project"
    script = root / "scripts/refresh_current_data.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n")
    artifacts = tmp_path / "artifacts"
    calls = []

    def completed(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "done", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", completed)
    result = run_current_refresh(artifacts, project_root=root)
    assert result.succeeded and result.output == "done"
    assert calls[0][0][1] == str(script)
    assert calls[0][1]["cwd"] == root
    assert not (artifacts / ".current-data-refresh.lock").exists()

    lock = artifacts / ".current-data-refresh.lock"
    artifacts.mkdir()
    lock.write_text("active")
    with pytest.raises(RefreshAlreadyRunningError):
        run_current_refresh(artifacts, project_root=root)

    lock.unlink()
    with refresh_lock(artifacts):
        assert lock.exists()
        with pytest.raises(RefreshAlreadyRunningError):
            run_current_refresh(artifacts, project_root=root)
    assert not lock.exists()


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


def _write_player_features(settings: Settings, competition: str) -> None:
    appearances = pd.DataFrame(
        [
            dict(
                player_id=f"p{p}",
                player_name=f"Player {p}",
                competition_id=competition,
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
    destination = settings.data_dir / "processed/m4/player_features.parquet"
    destination.parent.mkdir(parents=True)
    players.to_parquet(destination)


def test_real_player_navigation_with_synthetic_competition(isolated):
    _write_player_features(isolated, "SECOND")
    app = AppTest.from_file(str(APP)).run(timeout=20)
    # A single available competition is applied without a redundant selector.
    assert not app.sidebar.selectbox
    app.switch_page("pages/1_Player_Explorer.py").run()
    assert not app.exception and not app.error
    assert len(app.metric) == 4
    app.switch_page("pages/2_Scouting_Finder.py").run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 5
    app.multiselect[1].set_value([]).run()
    assert "Select at least one" in app.info[0].value


def _write_fpl_artifacts(root: Path) -> None:
    """Build a tiny two-club FPL artifact set through the production feature functions."""
    positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    identities = pd.DataFrame(
        [
            dict(
                player_id=index,
                player_code=100 + index,
                first_name="Test",
                second_name=f"Player{index}",
                position=positions[index % 4],
                team_code=1 + index % 2,
            )
            for index in range(1, 17)
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "id": index,
                "gw": gw,
                "now_cost": 45 + index,
                "total_points": 3 * gw + index,
                "event_points": 2 + index % 3,
                "points_per_game": 3.0,
                "form": 3.0,
                "selected_by_percent": 5.0 + index,
                "ep_next": 3.0,
                "minutes": 90 * gw,
                "starts": gw,
                "expected_goals_per_90": 0.05 * index,
                "expected_assists_per_90": 0.1,
                "defensive_contribution_per_90": 6.0,
                "saves_per_90": 3.0,
                "status": "d" if index == 3 else "a",
                "chance_of_playing_next_round": 50 if index == 3 else None,
                "news": "Knock" if index == 3 else "",
                "web_name": f"Player{index}",
            }
            for index in range(1, 17)
            for gw in (1, 2)
        ]
    )
    teams = pd.DataFrame(
        [
            dict(code=1, name="Alpha", short_name="ALP", elo=None),
            dict(code=2, name="Beta", short_name="BET", elo=None),
        ]
    )
    matches = pd.DataFrame(
        [
            dict(
                match_id=f"m{gameweek}",
                gameweek=gameweek,
                kickoff_time=f"2026-09-{gameweek:02d}T14:00:00Z",
                home_team=1 + gameweek % 2,
                away_team=2 - gameweek % 2,
                home_score=2,
                away_score=1,
                home_expected_goals_xg=1.8,
                away_expected_goals_xg=0.9,
                finished=gameweek <= 2,
            )
            for gameweek in range(1, 8)
        ]
    )
    kwargs = dict(
        competition_id="TEST",
        season="2026/27",
        position_map={
            "Goalkeeper": "GK",
            "Defender": "DEF",
            "Midfielder": "MID",
            "Forward": "FWD",
        },
    )
    players, history = latest_players(stats, identities, teams, **kwargs)
    fixtures = fixture_rows(matches, teams)
    backtest = rolling_backtest(stats, identities, teams, matches, **kwargs)
    root.mkdir(parents=True)
    project_players(players, history, fixtures).to_parquet(root / "players.parquet")
    fixtures.to_parquet(root / "fixtures.parquet")
    standings = league_table(matches, teams, competition_id="TEST", season="2026/27")
    standings.to_parquet(root / "standings.parquet")
    match_results(matches, teams).to_parquet(root / "results.parquet")
    home = fixtures.loc[fixtures.venue.eq("H")]
    simulate_season(
        standings[["club_id", "points", "goals_for", "goals_against"]],
        pd.DataFrame(
            {
                "home_club_id": home.club_id,
                "away_club_id": home.opponent_id,
                "home_rate": home.expected_goals_for,
                "away_rate": home.expected_goals_against,
            }
        ),
        simulations=200,
        top_places=1,
        relegation_places=1,
    ).merge(
        standings[["club_id", "club_name", "club_short_name", "competition_id", "season"]],
        on="club_id",
    ).to_parquet(root / "season_simulation.parquet")
    backtest.to_parquet(root / "backtest.parquet")
    (root / "backtest.json").write_text(json.dumps(backtest_summary(backtest)))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "season": "2026/27",
                "current_gameweek": 2,
                "next_gameweek": 3,
                "source_commit": "0123456789abcdef",
                "manager": None,
            }
        )
    )


def test_fpl_lab_and_backtest_render_from_synthetic_artifacts(isolated):
    _write_player_features(isolated, "TEST")
    _write_fpl_artifacts(isolated.data_dir / "processed/fpl")
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.switch_page("pages/4_FPL_Decision_Lab.py").run(timeout=20)
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == [
        "This gameweek",
        "Players",
        "Fixtures",
        "My team & transfers",
        "Injury news",
        "Template",
        "Mini-league",
    ]
    rendered = " ".join(item.value for item in app.markdown)
    assert "Captain picks" in rendered and "Fixture ticker" in rendered
    assert "Transfer planner" in rendered and "Knock" in rendered
    app.switch_page("pages/3_League_Table.py").run(timeout=20)
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == [
        "Table",
        "Season outlook",
        "Points vs expected",
        "Team statistics",
        "Compare teams",
    ]
    assert sum("not prepared for this season" in item.value for item in app.info) == 2
    rendered = " ".join(item.value for item in app.markdown)
    assert "200 simulated seasons" in rendered and "Where each club finishes" in rendered
    app.switch_page("pages/8_Match_Centre.py").run(timeout=20)
    assert not app.exception and not app.error
    rendered = " ".join(item.value for item in app.markdown)
    # No match model is prepared here, so the page explains that and still shows the rest.
    assert any("no forecast for this fixture" in item.value for item in app.info)
    assert all(text in rendered for text in ("Last matches", "chance to score", "Projected points"))
    history = club_matches(table(isolated.data_dir / "processed/fpl/results.parquet"), "ALP")
    assert history.gameweek.tolist() == [1, 2] and set(history.venue) == {"H", "A"}
    assert history.result.isin(["W", "D", "L"]).all()
    app.switch_page("pages/5_Model_Lab.py").run(timeout=20)
    app.selectbox[0].set_value("fpl").run(timeout=20)
    assert not app.exception and not app.error
    assert any("Event model MAE" in item.value for item in app.markdown)


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
