"""Run the landing page with no database, datasets or trained artifacts."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "Home.py"


def test_landing_page_without_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTIVE_COMPETITIONS", "EPL,LALIGA")
    monkeypatch.setattr(
        "sqlalchemy.create_engine", Mock(side_effect=AssertionError("Database I/O"))
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert "artifacts are not built yet" in app.info[0].value
    assert "EPL, LALIGA" in app.caption[0].value


def test_landing_page_handles_invalid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "bad://private-password")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    assert "Configuration is invalid" in app.error[0].value
    assert "private-password" not in app.error[0].value
