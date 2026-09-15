"""Transactional, insert-only canonical ingestion with batch/snapshot lineage."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import MetaData, Table, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from pl_analytics.data.contracts import Batch, DataValidationError
from pl_analytics.data.snapshots import Snapshot

TABLE_ORDER = ("clubs", "players", "matches", "player_market_values")
TABLE_NAMES = (
    "competitions",
    "source_competition_mappings",
    *TABLE_ORDER,
    "source_snapshots",
    "ingestion_batches",
    "ingestion_batch_snapshots",
)


@dataclass(frozen=True)
class LoadResult:
    """A deterministic batch ID and whether its transaction was already committed."""

    batch_id: str
    already_loaded: bool
    row_counts: dict[str, int]


def batch_id(batch: Batch, snapshots: list[Snapshot]) -> str:
    """Include bytes, normalized records, scope and loader version in batch identity."""
    payload = {
        "source": batch.source,
        "scope": asdict(batch.scope),
        "tables": batch.tables,
        "loader_version": batch.loader_version,
        "snapshots": sorted(snapshot.snapshot_id for snapshot in snapshots),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return str(uuid5(NAMESPACE_URL, digest))


def _insert_compatible(connection: Connection, table: Table, rows: list[dict[str, Any]]) -> None:
    """Insert in chunks and reject changed existing records instead of silently overwriting."""
    keys = [column.name for column in table.primary_key.columns]
    for offset in range(0, len(rows), 500):
        chunk = rows[offset : offset + 500]
        connection.execute(insert(table).values(chunk).on_conflict_do_nothing())
        requested = {tuple(row[key] for key in keys): row for row in chunk}
        query = select(table).where(tuple_(*(table.c[key] for key in keys)).in_(list(requested)))
        for existing in connection.execute(query).mappings():
            record = requested[tuple(existing[key] for key in keys)]
            if any(existing[column] != value for column, value in record.items()):
                raise DataValidationError(
                    f"Conflicting existing {table.name} record; explicit reconciliation is required"
                )


def load_batch(engine: Engine, batch: Batch, snapshots: list[Snapshot]) -> LoadResult:
    """Commit canonical records, provenance and scope together; rollback on any conflict."""
    if engine.dialect.name != "postgresql":
        raise ValueError("Canonical ingestion requires PostgreSQL")
    if not snapshots or set(batch.tables) - set(TABLE_ORDER):
        raise DataValidationError("Batch needs snapshots and supported canonical tables")
    for snapshot in snapshots:
        snapshot.verify()
    identity = batch_id(batch, snapshots)
    scope = batch.scope
    with engine.begin() as connection:
        metadata = MetaData()
        metadata.reflect(connection, only=TABLE_NAMES)
        tables = metadata.tables
        # This unique insert also serializes simultaneous attempts at the same batch.
        _insert_compatible(
            connection,
            tables["competitions"],
            [
                {
                    "competition_id": scope.competition_id,
                    "name": scope.competition_name,
                    "country": scope.country,
                    "source": "configuration",
                }
            ],
        )
        batches = tables["ingestion_batches"]
        inserted = connection.execute(
            insert(batches)
            .values(
                batch_id=UUID(identity),
                source=batch.source,
                competition_id=scope.competition_id,
                season=scope.season,
                coverage_start=scope.start_date,
                coverage_end=scope.end_date,
                loader_version=batch.loader_version,
                row_counts=batch.row_counts,
                skipped_rows=batch.skipped_rows,
                warnings=batch.warnings,
            )
            .on_conflict_do_nothing()
            .returning(batches.c.batch_id)
        ).scalar_one_or_none()
        if inserted is None:
            return LoadResult(identity, True, batch.row_counts)
        _insert_compatible(
            connection,
            tables["source_competition_mappings"],
            [
                {
                    "source": batch.source,
                    "source_competition_code": scope.source_code,
                    "competition_id": scope.competition_id,
                }
            ],
        )
        for table_name in TABLE_ORDER:
            _insert_compatible(connection, tables[table_name], batch.tables.get(table_name, []))
        snapshot_table = tables["source_snapshots"]
        for snapshot in snapshots:
            connection.execute(
                insert(snapshot_table)
                .values(
                    snapshot_id=UUID(snapshot.snapshot_id),
                    source_name=snapshot.source_name,
                    source_url=snapshot.source_url,
                    retrieved_at=datetime.fromisoformat(snapshot.retrieved_at),
                    dataset_version=snapshot.dataset_version,
                    license_or_terms_note=snapshot.license_or_terms_note,
                    sha256=snapshot.sha256,
                    local_path=snapshot.path.as_posix(),
                )
                .on_conflict_do_nothing()
            )
            stored_id = connection.execute(
                select(snapshot_table.c.snapshot_id).where(
                    snapshot_table.c.source_name == snapshot.source_name,
                    snapshot_table.c.sha256 == snapshot.sha256,
                )
            ).scalar_one()
            connection.execute(
                insert(tables["ingestion_batch_snapshots"])
                .values(
                    batch_id=UUID(identity),
                    snapshot_id=stored_id,
                )
                .on_conflict_do_nothing()
            )
    return LoadResult(identity, False, batch.row_counts)
