import subprocess
import sys

from pl_analytics import __version__
from pl_analytics.config import Settings, get_settings


def test_package_version() -> None:
    assert __version__ == "1.0.0"


def test_default_settings_load() -> None:
    settings = get_settings()
    assert "postgresql" in settings.database_url
    assert str(settings.data_dir) == "data"
    assert settings.active_competitions == ("EPL",)


def test_competition_scope_is_configurable() -> None:
    settings = Settings(
        active_competitions="EPL,LALIGA,BUNDESLIGA",
    )
    assert settings.active_competitions == ("EPL", "LALIGA", "BUNDESLIGA")


def test_installed_package_import_is_offline() -> None:
    """Import from outside the checkout with invalid config and networking blocked."""
    code = """
import os
import socket
os.environ['ACTIVE_COMPETITIONS'] = ''
os.environ['DATABASE_URL'] = 'invalid'
def fail(*args, **kwargs):
    raise AssertionError('Import attempted network access')
socket.socket.connect = fail
socket.socket.connect_ex = fail
socket.create_connection = fail
import pl_analytics
import pl_analytics.config
import pl_analytics.data.database
import pl_analytics.data.analytics
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
