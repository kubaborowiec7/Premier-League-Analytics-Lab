"""Synthetic source contracts, immutability, replay and failure-mode coverage."""

import csv
import gzip
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from pl_analytics.data.cli import main
from pl_analytics.data.contracts import DataValidationError, Scope, csv_rows
from pl_analytics.data.football_data import download_season, parse_results
from pl_analytics.data.repository import batch_id
from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.data.transfermarkt import parse_player_values

FIXTURES = Path(__file__).parent / "fixtures"


def archive(tmp_path: Path, filename: str, content: str | None = None) -> Snapshot:
    """Archive only generated data or tiny synthetic source fixtures."""
    path = FIXTURES / filename
    if content is not None:
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
    return SnapshotStore(tmp_path / "raw").import_file(
        path,
        source_name=filename,
        source_url="https://example.test/dataset",
        dataset_version="synthetic-v1",
        license_note="Synthetic fixture, generated for tests",
    )


def player_snapshots(tmp_path: Path) -> list[Snapshot]:
    return [
        archive(tmp_path, name) for name in ("players.csv", "player_valuations.csv", "clubs.csv")
    ]


def changed_csv(filename: str, changes: dict[str, str]) -> str:
    reader = csv.DictReader(io.StringIO((FIXTURES / filename).read_text()))
    rows = list(reader)
    rows[0].update(changes)
    output = io.StringIO()
    writer = csv.DictWriter(output, reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_snapshot_is_immutable_and_reusable(tmp_path: Path) -> None:
    first = archive(tmp_path, "matches.csv")
    manifest = first.path.parent / "metadata.json"
    original = manifest.read_bytes()
    second = archive(tmp_path, "matches.csv")
    assert first == second == Snapshot.read(manifest)
    assert manifest.read_bytes() == original
    assert first.path.read_bytes() == (FIXTURES / "matches.csv").read_bytes()
    assert datetime.fromisoformat(first.retrieved_at).tzinfo is not None
    first.path.write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="checksum"):
        Snapshot.read(manifest)


def test_new_version_preserves_original(tmp_path: Path) -> None:
    first = archive(tmp_path, "matches.csv")
    second = archive(tmp_path, "matches.csv", changed_csv("matches.csv", {"B365H": "2.0"}))
    assert first.sha256 != second.sha256
    first.verify()
    second.verify()


def test_checksum_pin_rejects_wrong_content(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "raw")
    with pytest.raises(DataValidationError, match="pinned"):
        store.import_file(
            FIXTURES / "matches.csv",
            source_name="test",
            source_url="https://example.test",
            dataset_version="v1",
            license_note="synthetic",
            expected_sha256="0" * 64,
        )
    assert not list(tmp_path.rglob("metadata.json"))


@pytest.mark.parametrize("first_status", [200, 429, 503, 404])
def test_download_retries_only_transient_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_status: int,
) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = first_status if len(calls) == 1 else 200
        return httpx.Response(status, content=b"column\nvalue\n")

    monkeypatch.setattr("pl_analytics.data.snapshots.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:

        def download() -> Snapshot:
            return SnapshotStore(tmp_path / "raw").download(
                "https://example.test/data.csv",
                source_name="test",
                dataset_version="v1",
                license_note="synthetic",
                client=client,
            )

        if first_status == 404:
            with pytest.raises(httpx.HTTPStatusError):
                download()
        else:
            download().verify()
    assert len(calls) == (2 if first_status in (429, 503) else 1)
    assert not list((tmp_path / "raw" / ".incoming").iterdir())


def test_download_size_limit(tmp_path: Path) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"1234"))
    ) as client:
        with pytest.raises(DataValidationError, match="size limit"):
            SnapshotStore(tmp_path / "raw").download(
                "https://example.test/data.csv",
                source_name="test",
                dataset_version="v1",
                license_note="synthetic",
                client=client,
                max_bytes=3,
            )
    assert not list(tmp_path.rglob("metadata.json"))


def test_gzip_archive(tmp_path: Path) -> None:
    path = tmp_path / "matches.csv.gz"
    path.write_bytes(gzip.compress((FIXTURES / "matches.csv").read_bytes()))
    snapshot = SnapshotStore(tmp_path / "raw").import_file(
        path,
        source_name="test",
        source_url="https://example.test",
        dataset_version="v1",
        license_note="synthetic",
    )
    assert len(list(csv_rows(snapshot.path, {"Date"}))) == 2


def test_results_normalization_and_replay(tmp_path: Path, scope: Scope) -> None:
    snapshot = archive(tmp_path, "matches.csv")
    batch = parse_results(snapshot, scope, timezone="Europe/London")
    assert batch.row_counts == {"clubs": 2, "matches": 2}
    first, second = batch.tables["matches"]
    assert first["match_date"] == datetime(2023, 8, 12, 14, tzinfo=UTC)
    assert first["match_time_known"] is True
    assert second["match_time_known"] is False
    assert second["home_shots"] is None
    assert first["competition_id"] == "EPL" and first["season"] == "2023/24"
    assert not any("B365" in key for row in batch.tables["matches"] for key in row)
    assert batch_id(batch, [snapshot]) == batch_id(
        parse_results(snapshot, scope, timezone="Europe/London"),
        [snapshot],
    )


@pytest.mark.parametrize(
    "change",
    [
        {"Div": "D1"},
        {"Date": "01/01/2020"},
        {"FTHG": "-1"},
        {"FTHG": "1.5"},
        {"FTR": "A"},
        {"AwayTeam": "Example North"},
        {"HomeTeam": ""},
        {"HST": "20"},
    ],
)
def test_bad_results_fail_before_loading(
    tmp_path: Path, scope: Scope, change: dict[str, str]
) -> None:
    snapshot = archive(tmp_path, "matches.csv", changed_csv("matches.csv", change))
    with pytest.raises(DataValidationError):
        parse_results(snapshot, scope, timezone="Europe/London")


def test_source_mapping_is_generic(tmp_path: Path, scope: Scope) -> None:
    text = (FIXTURES / "matches.csv").read_text().replace("E0,", "X1,")
    snapshot = archive(tmp_path, "matches.csv", text)
    batch = parse_results(
        snapshot,
        replace(scope, competition_id="CUSTOM", source_code="X1"),
        timezone="Europe/London",
    )
    assert all(row["competition_id"] == "CUSTOM" for row in batch.tables["matches"])


def test_duplicate_results_conflicts(tmp_path: Path, scope: Scope) -> None:
    text = (FIXTURES / "matches.csv").read_text()
    line = text.splitlines()[1]
    snapshot = archive(tmp_path, "matches.csv", text + line + "\n")
    assert parse_results(snapshot, scope, timezone="Europe/London").row_counts["matches"] == 2
    changed = line.replace(",2,1,H,", ",3,1,H,")
    snapshot = archive(tmp_path, "matches.csv", text + changed + "\n")
    with pytest.raises(DataValidationError, match="Conflicting"):
        parse_results(snapshot, scope, timezone="Europe/London")


def test_download_source_path_validation(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        download_season(SnapshotStore(tmp_path), source_code="../E0", source_season="2324")


def test_player_values_do_not_use_current_value_or_club(tmp_path: Path, scope: Scope) -> None:
    snapshots = player_snapshots(tmp_path)
    batch = parse_player_values(*snapshots, scope=replace(scope, source_code="GB1"))
    assert batch.row_counts == {"clubs": 2, "players": 2, "player_market_values": 3}
    assert batch.skipped_rows == 2
    for row in batch.tables["player_market_values"]:
        assert row["club_id"] is None
        assert row["competition_context"] == "source_reported_unverified"
        assert row["competition_id"] == "EPL" and row["season"] == "2023/24"
        assert row["valuation_date"] <= scope.end_date
    assert all("market_value_in_eur" not in row for row in batch.tables["players"])
    assert len(batch.warnings) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"market_value_in_eur": "NaN"},
        {"market_value_in_eur": "-1"},
        {"market_value_in_eur": "Infinity"},
        {"market_value_in_eur": "0.001"},
        {"player_id": "999"},
        {"current_club_id": "999"},
        {"date": "bad-date"},
    ],
)
def test_bad_player_values(tmp_path: Path, scope: Scope, change: dict[str, str]) -> None:
    snapshots = player_snapshots(tmp_path)
    snapshots[1] = archive(
        tmp_path, "player_valuations.csv", changed_csv("player_valuations.csv", change)
    )
    with pytest.raises(DataValidationError):
        parse_player_values(*snapshots, scope=replace(scope, source_code="GB1"))


def test_cli_validate_only_never_connects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = archive(tmp_path, "matches.csv")
    engine = Mock(side_effect=AssertionError("Unexpected database connection"))
    monkeypatch.setattr("pl_analytics.data.cli.create_engine", engine)
    arguments = [
        "matches",
        "--competition",
        "EPL",
        "--competition-name",
        "Premier League",
        "--country",
        "England",
        "--source-code",
        "E0",
        "--season",
        "2023/24",
        "--start-date",
        "2023-07-01",
        "--end-date",
        "2024-06-30",
        "--timezone",
        "Europe/London",
        "--manifest",
        str(snapshot.path.parent / "metadata.json"),
    ]
    assert main(arguments) == 0
    engine.assert_not_called()
    arguments[arguments.index("EPL")] = "DISABLED"
    assert main(arguments) == 1
    engine.assert_not_called()


def test_sidecar_rejects_path_traversal(tmp_path: Path) -> None:
    snapshot = archive(tmp_path, "matches.csv")
    manifest = snapshot.path.parent / "metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata["file_name"] = "../payload.csv"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(DataValidationError, match="filename"):
        Snapshot.read(manifest)


@pytest.mark.parametrize("invalid_hash", [None, 123, "invalid"])
def test_requested_checksum_pins_cannot_be_bypassed(tmp_path: Path, invalid_hash: object) -> None:
    checksums = tmp_path / "checksums.json"
    checksums.write_text(
        json.dumps(
            {name: invalid_hash for name in ("players.csv", "player_valuations.csv", "clubs.csv")}
        )
    )
    assert (
        main(
            [
                "player-values",
                "--competition",
                "EPL",
                "--competition-name",
                "Premier League",
                "--country",
                "England",
                "--source-code",
                "GB1",
                "--season",
                "2023/24",
                "--start-date",
                "2023-07-01",
                "--end-date",
                "2024-06-30",
                "--snapshot-dir",
                str(FIXTURES),
                "--dataset-version",
                "synthetic-v1",
                "--checksums",
                str(checksums),
            ]
        )
        == 1
    )
    assert not (tmp_path / "data/raw").exists()
