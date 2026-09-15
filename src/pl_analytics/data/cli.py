"""Explicit download/validate/load commands; importing this module does no ingestion."""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from pl_analytics.config import get_settings
from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import download_season, parse_results
from pl_analytics.data.repository import load_batch
from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.data.transfermarkt import (
    PUBLISHED_BASE_URL,
    SOURCE,
    TERMS,
    parse_player_values,
)

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """All competition mappings and season boundaries are explicit arguments."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("matches", "player-values"):
        sub = commands.add_parser(command)
        sub.add_argument("--competition", required=True)
        sub.add_argument("--competition-name", required=True)
        sub.add_argument("--country", required=True)
        sub.add_argument("--source-code", required=True)
        sub.add_argument("--season", required=True)
        sub.add_argument("--start-date", required=True, type=date.fromisoformat)
        sub.add_argument("--end-date", required=True, type=date.fromisoformat)
        sub.add_argument(
            "--load", action="store_true", help="Commit to PostgreSQL after validation"
        )
        if command == "matches":
            choice = sub.add_mutually_exclusive_group(required=True)
            choice.add_argument("--source-season", help="Four-digit source season, e.g. 2324")
            choice.add_argument(
                "--manifest", type=Path, help="Replay an archived metadata.json offline"
            )
            sub.add_argument("--sha256", help="Required content hash for a pinned download")
            sub.add_argument(
                "--timezone", required=True, help="Source kickoff timezone, e.g. Europe/London"
            )
        else:
            choice = sub.add_mutually_exclusive_group(required=True)
            choice.add_argument(
                "--snapshot-dir", type=Path, help="Local published CSV/CSV.GZ files"
            )
            choice.add_argument("--download-published", action="store_true")
            choice.add_argument(
                "--manifests", type=Path, nargs=3, metavar=("PLAYERS", "VALUATIONS", "CLUBS")
            )
            sub.add_argument(
                "--dataset-version", help="Published version/date; required for acquisition"
            )
            sub.add_argument(
                "--source-url", default="https://github.com/dcaribou/transfermarkt-datasets"
            )
            sub.add_argument(
                "--checksums", type=Path, help="JSON mapping filename to pinned SHA-256"
            )
    return root


def _player_snapshots(args: argparse.Namespace, store: SnapshotStore) -> list[Snapshot]:
    if args.manifests:
        return [Snapshot.read(path) for path in args.manifests]
    if not args.dataset_version:
        raise ValueError("--dataset-version is required for player snapshot acquisition")
    checksums = json.loads(args.checksums.read_text()) if args.checksums else {}
    result = []
    for table in ("players", "player_valuations", "clubs"):
        filename = f"{table}.csv.gz"
        if args.checksums and not any(key in checksums for key in (filename, f"{table}.csv")):
            raise ValueError(f"Checksum file is missing the required {table} hash")
        common = dict(
            source_name=f"{SOURCE}.{table}",
            dataset_version=args.dataset_version,
            license_note=TERMS,
        )
        if args.download_published:
            if args.checksums and filename not in checksums:
                raise ValueError(f"Checksum file must contain {filename} for published downloads")
            result.append(
                store.download(
                    f"{PUBLISHED_BASE_URL}/{filename}",
                    expected_sha256=checksums.get(filename),
                    **common,
                )
            )
        else:
            path = args.snapshot_dir / filename
            if not path.is_file():
                path = args.snapshot_dir / f"{table}.csv"
            if args.checksums and path.name not in checksums:
                raise ValueError(f"Checksum file must contain {path.name}")
            result.append(
                store.import_file(
                    path,
                    source_url=args.source_url,
                    expected_sha256=checksums.get(path.name),
                    **common,
                )
            )
    return result


def main(argv: list[str] | None = None) -> int:
    """Validate first; database writes require --load and use one atomic transaction."""
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = get_settings()
        logger.setLevel(settings.log_level)
        if args.competition not in settings.active_competitions:
            raise ValueError("Requested competition is not enabled in ACTIVE_COMPETITIONS")
        scope = Scope(
            args.competition,
            args.competition_name,
            args.country,
            args.source_code,
            args.season,
            args.start_date,
            args.end_date,
        )
        store = SnapshotStore(settings.data_dir / "raw")
        if args.command == "matches":
            snapshot = (
                Snapshot.read(args.manifest)
                if args.manifest
                else download_season(
                    store,
                    source_code=args.source_code,
                    source_season=args.source_season,
                    expected_sha256=args.sha256,
                )
            )
            snapshots = [snapshot]
            batch = parse_results(snapshot, scope, timezone=args.timezone)
        else:
            snapshots = _player_snapshots(args, store)
            batch = parse_player_values(*snapshots, scope=scope)
        output = {
            "source": batch.source,
            "competition_id": scope.competition_id,
            "season": scope.season,
            "row_counts": batch.row_counts,
            "skipped_rows": batch.skipped_rows,
            "warnings": batch.warnings,
            "manifests": [str(item.path.parent / "metadata.json") for item in snapshots],
            "checksums": {item.source_name: item.sha256 for item in snapshots},
        }
        if args.load:
            engine = create_engine(
                settings.database_url,
                poolclass=NullPool,
                connect_args={"connect_timeout": settings.database_connect_timeout},
            )
            try:
                output["database"] = asdict(load_batch(engine, batch, snapshots))
            finally:
                engine.dispose()
        logger.info("%s", json.dumps(output, indent=2, default=str))
    except (ValidationError, SQLAlchemyError, httpx.HTTPError):
        logger.error(
            "Ingestion failed. Check configuration, source availability and database schema."
        )
        return 1
    except (ValueError, OSError) as error:
        logger.error("Ingestion failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
