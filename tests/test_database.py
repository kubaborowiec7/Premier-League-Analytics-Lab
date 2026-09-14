"""Verify smoke-query behavior without a PostgreSQL server."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from pl_analytics.config import Settings
from pl_analytics.data import database


@pytest.mark.parametrize("failure", [None, "connect", "query", "result"])
def test_smoke_query_releases_resources(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    engine = MagicMock()
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(database, "create_engine", factory)
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.return_value = 0 if failure == "result" else 1
    error = OperationalError(None, None, Exception("unavailable"))
    if failure == "connect":
        engine.connect.side_effect = error
    elif failure == "query":
        connection.execute.side_effect = error
    settings = Settings(database_connect_timeout=2)
    if failure:
        with pytest.raises((OperationalError, RuntimeError)):
            database.check_database_connection(settings)
    else:
        database.check_database_connection(settings)
        assert str(connection.execute.call_args.args[0]) == "SELECT 1"
    assert factory.call_args.args == (settings.database_url,)
    assert factory.call_args.kwargs["connect_args"] == {"connect_timeout": 2}
    engine.dispose.assert_called_once()
    if failure != "connect":
        engine.connect.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_cli_exit_and_redacted_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fail: bool
) -> None:
    error = OperationalError(None, None, Exception("private-password"))
    check = MagicMock(side_effect=error if fail else None)
    monkeypatch.setattr(database, "check_database_connection", check)
    assert database.main() == int(fail)
    assert "private-password" not in caplog.text


def test_cli_handles_invalid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "invalid")
    check = MagicMock()
    monkeypatch.setattr(database, "check_database_connection", check)
    assert database.main() == 1
    check.assert_not_called()
