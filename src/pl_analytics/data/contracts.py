"""Small, source-independent contracts for validated ingestion batches."""

import csv
import gzip
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TextIO


class DataValidationError(ValueError):
    """A source file cannot be safely mapped into canonical records."""


@dataclass(frozen=True)
class Scope:
    """Explicit competition and inclusive date window; no assumed season calendar."""

    competition_id: str
    competition_name: str
    country: str
    source_code: str
    season: str
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        for value in (self.competition_id, self.source_code):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise DataValidationError("Competition identifiers must be non-empty safe codes")
        if not self.season.strip() or not self.competition_name.strip():
            raise DataValidationError("Competition name and season are required")
        if self.end_date < self.start_date:
            raise DataValidationError("Season end date precedes its start")


@dataclass
class Batch:
    """Validated records and explicit scope, prepared without database access."""

    source: str
    scope: Scope
    tables: dict[str, list[dict[str, Any]]]
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    loader_version: str = "1"

    @property
    def row_counts(self) -> dict[str, int]:
        """Counts describe accepted, deduplicated records, not insert counts."""
        return {name: len(rows) for name, rows in self.tables.items()}


@contextmanager
def open_csv(path: Path) -> Iterator[TextIO]:
    """Read UTF-8 CSV or gzip CSV without changing original bytes."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield stream


def csv_rows(path: Path, required: set[str]) -> Iterator[tuple[int, dict[str, str]]]:
    """Validate headers and row shapes; retain physical line numbers in errors."""
    with open_csv(path) as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = required - set(fields)
        if missing or len(fields) != len(set(fields)):
            raise DataValidationError(f"{path.name}: missing/duplicate columns: {sorted(missing)}")
        for row in reader:
            if all(not value or not str(value).strip() for value in row.values()):
                continue
            if None in row or any(value is None for value in row.values()):
                raise DataValidationError(f"{path.name}, line {reader.line_num}: malformed CSV row")
            yield reader.line_num, {key: value.strip() for key, value in row.items()}


def integer(value: str, field_name: str, *, optional: bool = False) -> int | None:
    """Accept non-negative integer counts, rejecting decimals and missing targets."""
    if not value and optional:
        return None
    if not re.fullmatch(r"\d+", value):
        raise DataValidationError(f"{field_name} must be a non-negative integer")
    return int(value)


def source_id(value: str, entity: str) -> str:
    """Namespaced stable source IDs avoid unsupported cross-source entity matches."""
    if not re.fullmatch(r"\d+", value):
        raise DataValidationError(f"Invalid {entity} identifier")
    return f"tm:{entity}:{int(value)}"


def add_unique(records: dict[Any, dict[str, Any]], key: Any, record: dict[str, Any]) -> None:
    """Collapse identical duplicates but reject contradictory observations."""
    if key in records and records[key] != record:
        raise DataValidationError(f"Conflicting duplicate record: {key}")
    records[key] = record
