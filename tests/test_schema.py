"""Guard the PostgreSQL bootstrap's table dependency order without a server."""

import re
from pathlib import Path


def test_schema_references_existing_tables() -> None:
    schema = (Path(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text()
    created: set[str] = set()
    for statement in schema.split(";"):
        table = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", statement)
        if table:
            references = set(re.findall(r"REFERENCES (\w+)", statement))
            assert references <= created, (
                f"{table[1]} references missing tables: {references - created}"
            )
            created.add(table[1])
    assert "player_season_features" in created
