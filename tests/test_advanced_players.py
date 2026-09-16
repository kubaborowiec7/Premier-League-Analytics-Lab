"""Offline coverage, temporal boundaries, position cohorts and visual comparison contracts."""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pl_analytics.dashboard.advanced_players import radar_chart
from pl_analytics.data.advanced_players import validate_season
from pl_analytics.features.advanced_players import (
    METRICS,
    PROFILES,
    peer_percentiles,
    summarize_players,
)


@pytest.fixture
def observations():
    rows = []
    for season, year in (("2024/25", 2025), ("2025/26", 2026)):
        for role in ("DEF", "FWD", "MID", "GK"):
            for player in range(6):
                for match in range(5):
                    row = dict(
                        player_id=f"{role}{player}",
                        player_name=f"{role} Player {player}",
                        position_group=role,
                        season=season,
                        competition_id="TEST",
                        minutes=90.0,
                        match_id=f"{season}-{match}",
                        match_date=f"{year}-01-{match + 1:02d}",
                        duels_lost=3.0,
                    )
                    row.update(
                        {
                            name: float(player + 1) if metric.kind == "count" else 50.0 + player
                            for name, metric in METRICS.items()
                            if metric.kind != "ratio"
                        }
                    )
                    row["total_shots"] = 2.0 * (player + 1)
                    row["progressive_passes"] = np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def test_coverage_missingness_and_percentage_aggregation(observations):
    chosen = observations.loc[
        observations.player_id.eq("DEF0") & observations.season.eq("2024/25")
    ].copy()
    chosen.loc[chosen.index[0], "interceptions"] = np.nan
    chosen.loc[chosen.index[0], "minutes"] = 30
    chosen.loc[chosen.index[0], "accurate_passes_percent"] = 100
    result = summarize_players(chosen, "2025-07-01").iloc[0]
    assert result.minutes == 390
    assert result.interceptions == 4
    assert result.interceptions_coverage == pytest.approx(360 / 390)
    assert result.interceptions_per90 == 1
    assert pd.isna(result.progressive_passes) and result.progressive_passes_coverage == 0
    assert result.accurate_passes_percent == pytest.approx((30 * 100 + 360 * 50) / 390)
    assert result.goal_conversion == 50
    assert result.duels_won_percent == 25


def test_future_and_zero_minutes_do_not_change_profile(observations):
    original = summarize_players(observations, "2025-07-01")
    changed = observations.copy()
    changed.loc[changed.season.eq("2025/26"), "goals"] = 100000
    bench = changed.iloc[:1].copy()
    bench["minutes"] = 0
    bench["match_id"] = "bench"
    bench["goals"] = 999
    pd.testing.assert_frame_equal(
        original, summarize_players(pd.concat([changed, bench]), "2025-07-01")
    )
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_players(pd.concat([observations, observations.iloc[:1]]), "2027-01-01")


def test_peer_groups_and_direction_do_not_mix_seasons_or_roles(observations):
    profiles = summarize_players(observations, "2027-01-01")
    ranked = peer_percentiles(profiles)
    target = ranked.loc[ranked.player_id.eq("DEF0") & ranked.season.eq("2024/25")].iloc[0]
    assert target.tackles_peers == 6
    assert target.tackles_percentile == pytest.approx(100 / 12)
    assert target.fouls_committed_percentile == pytest.approx(100 - 100 / 12)
    changed = profiles.copy()
    changed.loc[
        ~changed.position_group.eq("DEF") | ~changed.season.eq("2024/25"), "tackles_per90"
    ] *= 100
    rechecked = peer_percentiles(changed)
    assert rechecked.loc[target.name, "tackles_percentile"] == target.tackles_percentile
    assert pd.isna(peer_percentiles(profiles, min_minutes=451).iloc[0].tackles_percentile)
    assert "goals" not in PROFILES["DEF"] and "assists" not in PROFILES["DEF"]


def test_radar_has_distinct_colors_common_axes_and_missing_gaps(observations):
    ranked = peer_percentiles(summarize_players(observations, "2027-01-01"))
    metrics = ["tackles", "interceptions", "clearances"]
    figure = radar_chart(ranked.iloc[:3], metrics)
    assert len({trace.line.color for trace in figure.data}) == 3
    assert figure.layout.polar.radialaxis.range == (0, 100)
    assert all(tuple(trace.theta) == tuple(figure.data[0].theta) for trace in figure.data)
    ranked.loc[0, "tackles_percentile"] = np.nan
    trace = radar_chart(ranked.iloc[:1], metrics).data[0]
    assert trace.r[0] is None and trace.fill == "none" and not trace.connectgaps


def test_source_adapter_rejects_ambiguous_or_invalid_data():
    players = pd.DataFrame(
        [dict(player_id=1, player_code=42, first_name="A", second_name="B", position="Defender")]
    )
    matches = pd.DataFrame(
        [dict(match_id="m", kickoff_time="2025-01-01", finished=True, tournament="prem")]
    )
    appearances = pd.DataFrame(
        [dict(player_id=1, match_id="m", minutes_played=90, duels_lost=1, tackles=2)]
    )
    config = {
        "competition_id": "TEST",
        "source_tournament_codes": ["prem"],
        "position_map": {"Defender": "DEF"},
    }
    season = {"season": "2024/25", "start_date": "2024-07-01", "end_date": "2025-06-30"}
    result = validate_season(players, matches, appearances, config=config, season=season)
    assert result.player_id.iloc[0] == "fpl:42" and result.position_group.iloc[0] == "DEF"
    assert pd.isna(result.xg.iloc[0])
    for column, value in [
        ("minutes_played", -1),
        ("accurate_passes_percent", 101),
        ("tackles", -1),
        ("player_id", 999),
    ]:
        changed = appearances.copy()
        changed[column] = value
        with pytest.raises(ValueError):
            validate_season(players, matches, changed, config=config, season=season)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_season(
            players, matches, pd.concat([appearances, appearances]), config=config, season=season
        )
    matches["tournament"] = "cup"
    with pytest.raises(ValueError, match="competition"):
        validate_season(players, matches, appearances, config=config, season=season)


def test_advanced_ui_overlay_role_filters_and_scouting(observations, tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ACTIVE_COMPETITIONS", "TEST")
    destination = tmp_path / "data/processed/advanced/player_profiles.parquet"
    destination.parent.mkdir(parents=True)
    summarize_players(observations, "2027-01-01").to_parquet(destination)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/Home.py")).run(
        timeout=30
    )
    app.switch_page("pages/1_Player_Explorer.py").run(timeout=30)
    assert not app.error and not app.exception
    assert next(s for s in app.selectbox if s.label == "Season").options == ["2025/26", "2024/25"]
    app.multiselect[0].set_value(["TEST|DEF1|2025/26", "TEST|DEF2|2024/25"]).run(timeout=30)
    assert app.metric[2].value == "3" and not app.error
    app.radio[0].set_value("Per 90").run(timeout=30)
    assert not app.exception
    for role in ("FWD", "MID", "GK", "DEF"):
        next(s for s in app.selectbox if s.label == "Position group").select(role).run(timeout=30)
        assert not app.error and not app.exception
    app.switch_page("pages/2_Scouting_Finder.py").run(timeout=30)
    assert not app.error and not app.exception
    assert any("Closest profiles" in heading.value for heading in app.subheader)
    app.number_input[0].set_value(1000).run(timeout=30)
    assert not app.error and not app.exception
    assert any("three shared eligible metrics" in item.value for item in app.info)
