"""Validated configuration; importing this module performs no I/O."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional .env."""

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/pl_analytics",
        repr=False,
    )
    database_connect_timeout: int = Field(default=5, ge=1, le=60)
    football_data_api_token: SecretStr | None = None

    # Competition scope is configuration, not business logic.
    # V1 uses ("EPL",), but the data model must support many competitions.
    active_competitions: Annotated[tuple[str, ...], NoDecode] = ("EPL",)

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("active_competitions", mode="before")
    @classmethod
    def parse_competitions(cls, value: object) -> object:
        """Accept ACTIVE_COMPETITIONS=EPL,LALIGA as well as an iterable."""
        if isinstance(value, str):
            return tuple(value.split(","))
        return value

    @field_validator("active_competitions")
    @classmethod
    def normalize_competitions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize canonical codes, preserving order and rejecting empty scopes."""
        codes = tuple(dict.fromkeys(code.strip().upper() for code in value))
        if not codes or any(not code for code in codes):
            raise ValueError("ACTIVE_COMPETITIONS must contain non-empty competition codes")
        return codes

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Require the supported synchronous PostgreSQL driver without connecting."""
        try:
            url = make_url(value)
        except (ArgumentError, ValueError):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL URL") from None
        if url.drivername != "postgresql+psycopg" or not url.database:
            raise ValueError("DATABASE_URL must use postgresql+psycopg and name a database")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Accept case-insensitive standard logging levels."""
        return value.strip().upper() if isinstance(value, str) else value


def get_settings() -> Settings:
    """Return application settings without creating external connections."""
    return Settings()
