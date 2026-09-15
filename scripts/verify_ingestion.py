"""Reproduce M1's pinned public-data checks; optionally load and replay in PostgreSQL."""

import argparse
import json
import logging
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from pl_analytics.config import get_settings
from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import download_season, parse_results
from pl_analytics.data.repository import load_batch
from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.data.transfermarkt import PUBLISHED_BASE_URL, SOURCE, TERMS, parse_player_values


def main() -> None:
    """Use archived files when present; a missing file is downloaded with its pinned hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m1_sources.json"))
    parser.add_argument("--load", action="store_true", help="Load, replay and query PostgreSQL")
    parser.add_argument(
        "--initialize-schema",
        action="store_true",
        help="Explicitly apply the additive schema and migration before loading",
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/m1-verification.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("Manifest competition is not enabled in ACTIVE_COMPETITIONS")
    scope = Scope(
        config["competition_id"],
        config["competition_name"],
        config["country"],
        config["match_source_code"],
        config["season"],
        date.fromisoformat(config["start_date"]),
        date.fromisoformat(config["end_date"]),
    )
    store = SnapshotStore(settings.data_dir / "raw")
    match_manifest = (
        store.raw_dir / "football-data.co.uk" / config["match_sha256"] / "metadata.json"
    )
    match_snapshot = (
        Snapshot.read(match_manifest)
        if match_manifest.exists()
        else download_season(
            store,
            source_code=scope.source_code,
            source_season=config["match_source_season"],
            expected_sha256=config["match_sha256"],
        )
    )
    match_batch = parse_results(match_snapshot, scope, timezone=config["match_timezone"])
    player_snapshots = []
    for table in ("players", "player_valuations", "clubs"):
        filename = f"{table}.csv.gz"
        digest = config["player_sha256"][filename]
        manifest = store.raw_dir / f"{SOURCE}.{table}" / digest / "metadata.json"
        snapshot = (
            Snapshot.read(manifest)
            if manifest.exists()
            else store.download(
                f"{PUBLISHED_BASE_URL}/{filename}",
                source_name=f"{SOURCE}.{table}",
                dataset_version=config["player_dataset_version"],
                license_note=TERMS,
                expected_sha256=digest,
            )
        )
        player_snapshots.append(snapshot)
    player_batch = parse_player_values(
        *player_snapshots,
        scope=replace(scope, source_code=config["player_source_code"]),
    )
    counts = {**match_batch.row_counts, **player_batch.row_counts}
    for table, expected in config["expected_counts"].items():
        if counts[table] != expected:
            raise ValueError(f"Unexpected {table} count: {counts[table]} != {expected}")
    report = {
        "scope": asdict(scope),
        "match_counts": match_batch.row_counts,
        "player_counts": player_batch.row_counts,
        "warnings": player_batch.warnings,
        "snapshots": [
            {
                "source": item.source_name,
                "sha256": item.sha256,
                "retrieved_at": item.retrieved_at,
                "source_url": item.source_url,
            }
            for item in [match_snapshot, *player_snapshots]
        ],
    }
    if args.load:
        engine = create_engine(
            settings.database_url,
            poolclass=NullPool,
            connect_args={"connect_timeout": settings.database_connect_timeout},
        )
        try:
            if args.initialize_schema:
                root = Path(__file__).resolve().parents[1]
                with engine.begin() as connection:
                    connection.exec_driver_sql((root / "sql/schema.sql").read_text())
                    connection.exec_driver_sql(
                        (root / "sql/migrations/001_ingestion.sql").read_text()
                    )
            loads = []
            for batch, snapshots in (
                (match_batch, [match_snapshot]),
                (player_batch, player_snapshots),
            ):
                first = load_batch(engine, batch, snapshots)
                repeat = load_batch(engine, batch, snapshots)
                if not repeat.already_loaded or repeat.batch_id != first.batch_id:
                    raise AssertionError("Batch replay was not idempotent")
                loads.append(asdict(repeat))
            with engine.connect() as connection:
                for table, expected in config["expected_counts"].items():
                    # Names come from this fixed allowlist, never from interpolated user input.
                    if table not in {"matches", "players", "player_market_values"}:
                        raise ValueError("Unexpected verification table")
                    if table == "players":
                        query = (
                            "SELECT count(DISTINCT player_id) FROM player_market_values "
                            "WHERE competition_id=:competition AND season=:season"
                        )
                    else:
                        query = (
                            f"SELECT count(*) FROM {table} "
                            "WHERE competition_id=:competition AND season=:season"
                        )
                    actual = connection.scalar(
                        text(query), {"competition": scope.competition_id, "season": scope.season}
                    )
                    if actual != expected:
                        raise AssertionError(f"Database {table}: {actual} != {expected}")
                lineage = connection.scalar(text("SELECT count(*) FROM ingestion_batch_snapshots"))
                if lineage < 4:
                    raise AssertionError("Missing source lineage")
            report["postgresql"] = {"loads": loads, "idempotence_verified": True}
        finally:
            engine.dispose()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    logging.info("Verified M1 source counts; report: %s", args.report)


if __name__ == "__main__":
    main()
