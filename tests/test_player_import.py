"""Session player import schema and unit-safety tests."""

import hashlib

import joblib
import pandas as pd
import pytest

from pl_analytics.dashboard.player_import import _trusted_value_bundle
from pl_analytics.features.player_import import performance_rows, value_feature_rows


def test_performance_import_accepts_total_and_per90_without_fabricating_missing_metrics():
    raw = pd.DataFrame(
        [
            {
                "player_name": "Example Defender",
                "position_group": "defender",
                "minutes": 900,
                "appearances": 10,
                "tackles_per90": 2.5,
                "aerial_duels_won": 20,
                "aerial_duels_won_coverage": 0.5,
                "accurate_passes_percent": 88,
            }
        ]
    )
    result = performance_rows(
        raw, competition_id="EPL", season="2026/27", reference_date="2026-09-30"
    ).iloc[0]
    assert result.position_group == "DEF"
    assert result.tackles == pytest.approx(25)
    assert result.tackles_per90 == pytest.approx(2.5)
    assert result.aerial_duels_won_per90 == pytest.approx(4)
    assert result.accurate_passes_percent == 88
    assert pd.isna(result.goals) and result.goals_coverage == 0
    assert result.source == "User import (session)"
    assert result.comparison_only


def test_performance_import_rejects_ambiguous_or_invalid_values():
    identity = {
        "player_name": "Example",
        "position_group": "MID",
        "minutes": 900,
    }
    with pytest.raises(ValueError, match="either tackles or tackles_per90"):
        performance_rows(
            pd.DataFrame([{**identity, "tackles": 10, "tackles_per90": 1}]),
            competition_id="EPL",
            season="2026/27",
            reference_date="2026-09-30",
        )
    with pytest.raises(ValueError, match="at most 100"):
        performance_rows(
            pd.DataFrame([{**identity, "accurate_passes_percent": 101}]),
            competition_id="EPL",
            season="2026/27",
            reference_date="2026-09-30",
        )


def test_value_import_requires_frozen_model_features_and_preserves_scope():
    raw = pd.DataFrame(
        [
            {
                "player_name": "Example Forward",
                "market_value_eur": 20_000_000,
                "age_years": 24,
                "previous_value_eur": 18_000_000,
                "days_since_valuation": 180,
                "days_since_appearance": 7,
                "appearances_365": 30,
                "minutes_365": 2200,
                "goals_per90_365": 0.5,
                "assists_per90_365": 0.2,
            }
        ]
    )
    result = value_feature_rows(
        raw, competition_id="EPL", default_season="2026/27"
    ).iloc[0]
    assert result.competition_id == "EPL" and result.season == "2026/27"
    assert result.market_value_eur == 20_000_000
    assert result.cohort_context == "user_supplied_scenario"


def test_value_import_rejects_implausible_exposure():
    raw = pd.DataFrame(
        [
            {
                "player_name": "Example",
                "market_value_eur": 1,
                "age_years": 24,
                "previous_value_eur": 1,
                "days_since_valuation": 10,
                "days_since_appearance": 2,
                "appearances_365": 30,
                "minutes_365": 4000,
                "goals_per90_365": 0,
                "assists_per90_365": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="at most 3420"):
        value_feature_rows(raw, competition_id="EPL", default_season="2026/27")


def test_value_import_loads_only_checksum_verified_local_bundle(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"models": [], "chosen": "ridge", "quantile": 1.0}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _trusted_value_bundle(str(path), digest)["chosen"] == "ridge"
    with pytest.raises(ValueError, match="checksum"):
        _trusted_value_bundle(str(path), "0" * 64)
