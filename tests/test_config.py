"""Exercise real environment/dotenv parsing, precedence and safe validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pl_analytics.config import Settings, get_settings


@pytest.mark.parametrize("codes", ["EPL", " epl, laliga, EPL ", "CUSTOM_LEAGUE"])
def test_competitions_from_environment(monkeypatch: pytest.MonkeyPatch, codes: str) -> None:
    monkeypatch.setenv("ACTIVE_COMPETITIONS", codes)
    expected = tuple(dict.fromkeys(code.strip().upper() for code in codes.split(",")))
    assert get_settings().active_competitions == expected


def test_dotenv_and_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text(
        "ACTIVE_COMPETITIONS=epl,laliga\nDATA_DIR=local-data\nARTIFACT_DIR=local-models\n"
        "LOG_LEVEL=debug\nFOOTBALL_DATA_API_TOKEN=test-token\nPOSTGRES_USER=ignored\n",
        encoding="utf-8",
    )
    settings = get_settings()
    assert settings.active_competitions == ("EPL", "LALIGA")
    assert settings.data_dir == Path("local-data")
    assert settings.artifact_dir == Path("local-models")
    assert settings.log_level == "DEBUG"
    assert settings.football_data_api_token.get_secret_value() == "test-token"
    assert not settings.data_dir.exists()
    assert not settings.artifact_dir.exists()
    monkeypatch.setenv("ACTIVE_COMPETITIONS", "BUNDESLIGA")
    assert get_settings().active_competitions == ("BUNDESLIGA",)
    assert Settings(active_competitions=("custom",)).active_competitions == ("CUSTOM",)


def test_connection_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+psycopg://test:password@localhost:6543/test_db"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT", "3")
    settings = get_settings()
    assert settings.database_url == url
    assert settings.database_connect_timeout == 3
    assert "password" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ACTIVE_COMPETITIONS", ""),
        ("ACTIVE_COMPETITIONS", "EPL,,LALIGA"),
        ("DATABASE_URL", "invalid-secret-url"),
        ("DATABASE_URL", "sqlite:///test.db"),
        ("DATABASE_URL", "postgresql+psycopg://localhost"),
        ("DATABASE_CONNECT_TIMEOUT", "0"),
        ("DATABASE_CONNECT_TIMEOUT", "61"),
        ("LOG_LEVEL", "invalid-level"),
    ],
)
def test_invalid_environment(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        get_settings()


def test_credentials_are_not_in_validation_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "bad://test:private-password@host/database")
    with pytest.raises(ValidationError) as error:
        get_settings()
    assert "private-password" not in str(error.value)
