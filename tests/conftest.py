"""Keep unit and app tests independent of developer secrets, files and services."""

from pathlib import Path

import pytest

from pl_analytics.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear application environment variables and avoid the checkout's .env."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.chdir(tmp_path)
