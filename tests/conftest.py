"""Keep unit and app tests independent of developer secrets, files and services."""

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from pl_analytics.config import Settings
from pl_analytics.data.contracts import Scope


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """Create only an isolated test schema; never reset the user's database."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = "analytics_test_" + uuid4().hex
    admin = create_engine(url, poolclass=NullPool)
    engine = create_engine(
        url, poolclass=NullPool, connect_args={"options": f"-csearch_path={schema}"}
    )
    root = Path(__file__).resolve().parents[1]
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        with engine.begin() as connection:
            connection.exec_driver_sql((root / "sql/schema.sql").read_text())
            for _ in range(2):
                connection.exec_driver_sql((root / "sql/migrations/001_ingestion.sql").read_text())
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear application environment variables and avoid the checkout's .env."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def scope() -> Scope:
    """Synthetic EPL window; adapters are also tested with arbitrary competition codes."""
    return Scope(
        "EPL", "Premier League", "England", "E0", "2023/24", date(2023, 7, 1), date(2024, 6, 30)
    )
