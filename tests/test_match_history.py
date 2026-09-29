"""The multi-season adapter retains canonical context and skips unrequested archives."""

from pathlib import Path

import pandas as pd
import pytest

from pl_analytics.data.football_data import SOURCE
from pl_analytics.data.match_history import read_match_history
from pl_analytics.data.snapshots import SnapshotStore


def test_pinned_history_and_future_archive_exclusion(tmp_path):
    raw = tmp_path / "raw"
    snapshot = SnapshotStore(raw).import_file(
        Path(__file__).parent / "fixtures/matches.csv",
        source_name=SOURCE,
        source_url="https://example.test/matches.csv",
        dataset_version="synthetic",
        license_note="Generated fixture",
    )
    season = dict(
        season="2023/24",
        start_date="2023-07-01",
        end_date="2024-06-30",
        source_season="2324",
        sha256=snapshot.sha256,
        expected_matches=2,
    )
    config = dict(
        competition_id="CUSTOM",
        competition_name="Custom",
        country="England",
        source_code="E0",
        timezone="Europe/London",
        seasons=[season, {"start_date": "2025-07-01"}],
    )
    frame, sources = read_match_history(config, raw, end_date="2024-06-30")
    assert len(frame) == 2 and len(sources) == 1
    assert set(frame.competition_id) == {"CUSTOM"}
    assert set(frame.season) == {"2023/24"}
    assert frame.home_shots.isna().all()
    assert str(frame.match_day.dt.tz) == "UTC"
    season["expected_matches"] = 380
    with pytest.raises(ValueError, match="coverage"):
        read_match_history(config, raw, end_date="2024-06-30")


def test_mutable_current_archive_must_reproduce_frozen_cutoff(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    snapshot = SnapshotStore(raw).import_file(
        Path(__file__).parent / "fixtures/matches.csv",
        source_name=SOURCE,
        source_url="https://example.test/current.csv",
        dataset_version="current",
        license_note="Generated fixture",
    )
    monkeypatch.setattr(
        "pl_analytics.data.match_history.download_season", lambda *args, **kwargs: snapshot
    )
    season = dict(
        season="2023/24",
        start_date="2023-07-01",
        end_date="2024-06-30",
        source_season="2324",
        sha256="0" * 64,
        expected_matches=1,
        mutable_current=True,
    )
    config = dict(
        competition_id="CUSTOM",
        competition_name="Custom",
        country="England",
        source_code="E0",
        timezone="Europe/London",
        seasons=[season],
    )
    frame, sources = read_match_history(
        config, raw, end_date="2023-08-12", download=True
    )
    assert len(frame) == 1
    assert sources[0]["sha256"] == snapshot.sha256
    assert pd.Timestamp(frame.match_day.max()) == pd.Timestamp("2023-08-12", tz="UTC")
    season["expected_canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Canonical match facts changed"):
        read_match_history(config, raw, end_date="2023-08-12", download=True)
