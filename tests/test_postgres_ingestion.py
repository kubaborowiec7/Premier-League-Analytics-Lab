"""Real PostgreSQL tests; opt in with TEST_DATABASE_URL (isolated temporary schemas)."""

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool
from test_ingestion import archive, player_snapshots

from pl_analytics.data.contracts import DataValidationError, Scope
from pl_analytics.data.football_data import parse_results
from pl_analytics.data.repository import load_batch
from pl_analytics.data.transfermarkt import parse_player_values

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = "ingestion_test_" + uuid4().hex
    admin = create_engine(url, poolclass=NullPool)
    engine = create_engine(
        url, poolclass=NullPool, connect_args={"options": f"-csearch_path={schema}"}
    )
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        with engine.begin() as connection:
            connection.exec_driver_sql((ROOT / "sql" / "schema.sql").read_text())
            for _ in range(2):
                connection.exec_driver_sql((ROOT / "sql/migrations/001_ingestion.sql").read_text())
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


def test_postgres_idempotence_and_lineage(pg_engine: Engine, tmp_path: Path, scope: Scope) -> None:
    snapshot = archive(tmp_path, "matches.csv")
    batch = parse_results(snapshot, scope, timezone="Europe/London")
    first = load_batch(pg_engine, batch, [snapshot])
    second = load_batch(pg_engine, batch, [snapshot])
    assert not first.already_loaded and second.already_loaded
    assert first.batch_id == second.batch_id
    with pg_engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM matches")) == 2
        assert connection.scalar(text("SELECT count(*) FROM ingestion_batches")) == 1
        assert connection.scalar(text("SELECT count(*) FROM ingestion_batch_snapshots")) == 1
        assert connection.scalar(text("SELECT sha256 FROM source_snapshots")) == snapshot.sha256


def test_postgres_conflict_rolls_back(pg_engine: Engine, tmp_path: Path, scope: Scope) -> None:
    snapshot = archive(tmp_path, "matches.csv")
    batch = parse_results(snapshot, scope, timezone="Europe/London")
    load_batch(pg_engine, batch, [snapshot])
    batch.tables["matches"][0]["home_goals"] = 42
    with pytest.raises(DataValidationError, match="Conflicting"):
        load_batch(pg_engine, batch, [snapshot])
    with pg_engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM ingestion_batches")) == 1
        assert connection.scalar(text("SELECT max(home_goals) FROM matches")) == 2


def test_postgres_player_values(pg_engine: Engine, tmp_path: Path, scope: Scope) -> None:
    snapshots = player_snapshots(tmp_path)
    batch = parse_player_values(*snapshots, scope=replace(scope, source_code="GB1"))
    assert not load_batch(pg_engine, batch, snapshots).already_loaded
    assert load_batch(pg_engine, batch, snapshots).already_loaded
    with pg_engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM player_market_values")) == 3
        assert connection.scalar(text("SELECT count(*) FROM source_snapshots")) == 3
        assert connection.scalar(text("SELECT count(*) FROM ingestion_batch_snapshots")) == 3
        assert (
            connection.scalar(
                text("SELECT count(*) FROM player_market_values WHERE club_id IS NOT NULL")
            )
            == 0
        )
