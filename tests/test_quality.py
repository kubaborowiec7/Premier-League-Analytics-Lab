"""Offline EDA contracts and clean-kernel notebook execution on synthetic archives."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from pl_analytics.data.contracts import DataValidationError, Scope
from pl_analytics.data.quality import (
    build_quality_report,
    latest_player_values,
    load_quality_sample,
    missingness,
    numeric_summary,
    temporal_coverage,
)
from pl_analytics.data.snapshots import SnapshotStore, sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def quality_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small manifest pointing only at version-controlled synthetic fixtures."""
    config = json.loads((ROOT / "data/manifests/m1_sources.json").read_text())
    store = SnapshotStore(tmp_path / "data/raw")
    for filename, source in (
        ("matches", "football-data.co.uk"),
        ("players", "transfermarkt-datasets.players"),
        ("player_valuations", "transfermarkt-datasets.player_valuations"),
        ("clubs", "transfermarkt-datasets.clubs"),
    ):
        snapshot = store.import_file(
            ROOT / f"tests/fixtures/{filename}.csv",
            source_name=source,
            source_url="https://example.test/synthetic.csv",
            dataset_version="synthetic-v1",
            license_note="Synthetic fixture",
        )
        if filename == "matches":
            config["match_sha256"] = snapshot.sha256
        else:
            config["player_sha256"][f"{filename}.csv.gz"] = snapshot.sha256
    config["expected_counts"] = {"matches": 2, "players": 2, "player_market_values": 3}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, store.raw_dir


def test_sample_report_and_immutability(quality_inputs: tuple[Path, Path]) -> None:
    manifest, raw = quality_inputs
    before = {p: sha256_file(p) for p in raw.rglob("*") if p.is_file()}
    sample = load_quality_sample(manifest, raw)
    report = build_quality_report(sample)
    assert report == build_quality_report(sample)
    json.dumps(report, allow_nan=False)
    assert report["match_targets"]["outcomes"]["H"] == {"count": 1, "fraction": 0.5}
    assert report["leakage_and_coverage"]["unknown_kickoffs"] == 1
    assert report["leakage_and_coverage"]["unverified_valuation_context"] == 3
    assert report["leakage_and_coverage"]["profile_forbidden_columns_present"] == []
    assert report["valuation_targets"]["all_observations_eur"]["n"] == 3
    assert report["valuation_targets"]["latest_per_player_source_eur"]["n"] == 2
    assert before == {p: sha256_file(p) for p in raw.rglob("*") if p.is_file()}


def test_missing_and_tampered_archives_fail(quality_inputs: tuple[Path, Path]) -> None:
    manifest, raw = quality_inputs
    with pytest.raises(FileNotFoundError, match="verify_ingestion"):
        load_quality_sample(manifest, raw / "nonexistent")
    payload = next(raw.rglob("payload.csv"))
    payload.write_bytes(b"changed")
    with pytest.raises(DataValidationError, match="checksum"):
        load_quality_sample(manifest, raw)


def test_bad_pin_rejected_before_filesystem_lookup(quality_inputs: tuple[Path, Path]) -> None:
    manifest, raw = quality_inputs
    config = json.loads(manifest.read_text())
    config["match_sha256"] = "../invalid"
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SHA-256"):
        load_quality_sample(manifest, raw)


def test_scope_is_generic_and_mixtures_rejected(quality_inputs: tuple[Path, Path]) -> None:
    sample = load_quality_sample(*quality_inputs)
    sample.scope = replace(sample.scope, competition_id="OTHER")
    sample.matches["competition_id"] = "OTHER"
    sample.values["competition_id"] = "OTHER"
    assert build_quality_report(sample)["scope"]["competition_id"] == "OTHER"
    sample.values.loc[0, "season"] = "2025/26"
    with pytest.raises(ValueError, match="scope"):
        build_quality_report(sample)


def test_missing_outcomes_are_not_classified_as_away_wins(
    quality_inputs: tuple[Path, Path],
) -> None:
    sample = load_quality_sample(*quality_inputs)
    sample.matches.loc[0, "home_goals"] = None
    report = build_quality_report(sample)["match_targets"]
    assert report["excluded_rows"] == 1
    assert report["outcomes"]["D"]["fraction"] == 1
    assert report["outcomes"]["A"]["count"] == 0


def test_empty_sample_keeps_unknown_statistics(quality_inputs: tuple[Path, Path]) -> None:
    sample = load_quality_sample(*quality_inputs)
    sample.matches = sample.matches.iloc[:0]
    sample.players = sample.players.iloc[:0]
    sample.values = sample.values.iloc[:0]
    report = build_quality_report(sample)
    assert report["match_targets"]["outcomes"]["H"]["fraction"] is None
    assert report["valuation_targets"]["all_observations_eur"]["median"] is None
    json.dumps(report, allow_nan=False)


def test_latest_values_preserve_provider_and_date_order() -> None:
    values = pd.DataFrame(
        [
            ["p", "A", "s", "one", "2024-02-01", 5],
            ["p", "A", "s", "one", "2024-01-01", 100],
            ["p", "A", "s", "two", "2024-01-01", 20],
            ["p", "B", "s", "one", "2024-01-01", 30],
        ],
        columns=[
            "player_id",
            "competition_id",
            "season",
            "source",
            "valuation_date",
            "market_value_eur",
        ],
    )
    latest = latest_player_values(values)
    assert sorted(latest.market_value_eur) == [5, 20, 30]
    with pytest.raises(ValueError, match="duplicate"):
        latest_player_values(pd.concat([values, values.iloc[:1]]))


def test_missingness_denominators_and_undefined_skew() -> None:
    assert missingness(pd.DataFrame({"x": [0, None, 2]})) == [
        {"column": "x", "rows": 3, "missing": 1, "missing_fraction": 1 / 3}
    ]
    assert missingness(pd.DataFrame(columns=["x"]))[0]["missing_fraction"] is None
    for values in ([], [None], [4], [4, 4, 4]):
        assert numeric_summary(pd.Series(values))["sample_skew"] is None
    assert numeric_summary(pd.Series([0, 1, 2, 100]))["sample_skew"] > 0
    with pytest.raises(ValueError, match="Non-finite"):
        numeric_summary(pd.Series([np.inf]))


def test_temporal_zero_months_missing_and_outside(scope: Scope) -> None:
    dates = pd.Series(["2023-08-01", None, "2025-01-01"])
    report = temporal_coverage(dates, scope)
    assert report["monthly_counts_utc"]["2023-07"] == 0
    assert report["monthly_counts_utc"]["2023-08"] == 1
    assert report["missing_dates"] == 1
    assert report["outside_requested_dates"] == 1


def test_notebook_runs_clean_kernel_on_synthetic_data(
    quality_inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, raw = quality_inputs
    monkeypatch.setenv("DATA_DIR", str(raw.parent))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PL_ANALYTICS_QUALITY_MANIFEST", str(manifest))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    destination = tmp_path / "executed.ipynb"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/execute_quality_notebook.py"),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    notebook = nbformat.read(destination, as_version=4)
    assert all(cell.execution_count for cell in notebook.cells if cell.cell_type == "code")
    assert len(list((tmp_path / "artifacts/m3").glob("*.png"))) == 3
    report = json.loads((tmp_path / "artifacts/m3-quality.json").read_text())
    assert report["counts"]["matches"] == 2
