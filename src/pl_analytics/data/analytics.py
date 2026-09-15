"""Explicit PostgreSQL analytics installation and invariant checks; no import-time I/O."""

from pathlib import Path

from sqlalchemy.engine import Connection, Engine


def check_analytics(connection: Connection, sql_dir: Path) -> dict[str, int]:
    """Run the SQL invariants and reject invalid marts, including during installation."""
    rows = connection.exec_driver_sql((sql_dir / "checks.sql").read_text(encoding="utf-8"))
    checks = {row.check_name: int(row.violations) for row in rows}
    if not checks:
        raise ValueError("Analytics validation returned no checks")
    failures = {name: count for name, count in checks.items() if count}
    if failures:
        raise ValueError(f"Analytics invariants failed: {failures}")
    return checks


def install_analytics(engine: Engine, sql_dir: Path) -> dict[str, int]:
    """Apply M2 after M1, atomically; invalid existing data leaves M1 unchanged.

    SQL files are repository resources supplied explicitly by the caller. Existing
    canonical rows and raw files are never rewritten. Repeated execution is safe.
    """
    with engine.begin() as connection:
        for relative in ("migrations/002_analytics.sql", "analytics.sql"):
            connection.exec_driver_sql((sql_dir / relative).read_text(encoding="utf-8"))
        return check_analytics(connection, sql_dir)
