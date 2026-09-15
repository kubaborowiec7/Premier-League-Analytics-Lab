"""Keep unit and app tests independent of developer secrets, files and services."""

from datetime import date
from pathlib import Path

import pytest

from pl_analytics.config import Settings
from pl_analytics.data.contracts import Scope


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
