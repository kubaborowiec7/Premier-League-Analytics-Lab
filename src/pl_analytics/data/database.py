"""Explicit, read-only database connectivity check; no connection at import time."""

import logging

from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from pl_analytics.config import Settings, get_settings

logger = logging.getLogger(__name__)


def check_database_connection(settings: Settings | None = None) -> None:
    """Run SELECT 1 and release all resources; raise on connection or query failure."""
    settings = settings if settings is not None else get_settings()
    engine = create_engine(
        settings.database_url,
        connect_args={"connect_timeout": settings.database_connect_timeout},
        poolclass=NullPool,
    )
    try:
        with engine.connect() as connection:
            if connection.execute(text("SELECT 1")).scalar_one() != 1:
                raise RuntimeError("Database smoke query returned an unexpected result")
    finally:
        engine.dispose()


def main() -> int:
    """Return a shell-friendly status without logging credentials or driver errors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = get_settings()
        logger.setLevel(settings.log_level)
        check_database_connection(settings)
    except ValidationError:
        logger.error("Invalid configuration. Check .env and environment variables.")
        return 1
    except (SQLAlchemyError, RuntimeError):
        logger.error("Database check failed. Check DATABASE_URL and PostgreSQL availability.")
        return 1
    logger.info("Database connection successful (SELECT 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
