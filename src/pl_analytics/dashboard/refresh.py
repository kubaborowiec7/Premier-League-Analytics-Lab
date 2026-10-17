"""Controlled execution of the repository's current-data refresh from Streamlit."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class RefreshAlreadyRunningError(RuntimeError):
    """Raised when another dashboard or scheduled refresh owns the lock."""


@dataclass(frozen=True)
class RefreshResult:
    """Outcome safe to present in the dashboard."""

    succeeded: bool
    duration_seconds: float
    output: str


def _project_root(start: Path | None = None) -> Path:
    """Locate the checkout containing the explicit refresh script."""
    candidate = (start or Path.cwd()).resolve()
    for root in (candidate, *candidate.parents):
        if (root / "scripts/refresh_current_data.py").is_file():
            return root
    raise FileNotFoundError("The current-data refresh script is unavailable in this checkout")


def _acquire_lock(path: Path, *, stale_after_seconds: int = 6 * 60 * 60) -> int:
    """Create an exclusive refresh lock, replacing only a clearly stale lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > stale_after_seconds:
        path.unlink(missing_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RefreshAlreadyRunningError("A data refresh is already running") from None
    os.write(descriptor, f"pid={os.getpid()}\nstarted={time.time()}\n".encode())
    return descriptor


@contextmanager
def refresh_lock(artifact_dir: Path) -> Iterator[None]:
    """Serialize refresh producers across the UI, CLI and task scheduler."""
    lock = artifact_dir / ".current-data-refresh.lock"
    descriptor = _acquire_lock(lock)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _ensure_refresh_available(artifact_dir: Path) -> None:
    """Fail quickly in the UI while the producer remains the authoritative lock owner."""
    lock = artifact_dir / ".current-data-refresh.lock"
    if not lock.exists():
        return
    if time.time() - lock.stat().st_mtime > 6 * 60 * 60:
        lock.unlink(missing_ok=True)
        return
    raise RefreshAlreadyRunningError("A data refresh is already running")


def run_current_refresh(
    artifact_dir: Path,
    *,
    project_root: Path | None = None,
    timeout_seconds: int = 30 * 60,
) -> RefreshResult:
    """Run the existing refresh command once and always release its process lock."""
    root = _project_root(project_root)
    script = root / "scripts/refresh_current_data.py"
    _ensure_refresh_available(artifact_dir)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        if completed.returncode == 3 and "already running" in output.lower():
            raise RefreshAlreadyRunningError("A data refresh is already running")
        return RefreshResult(
            succeeded=completed.returncode == 0,
            duration_seconds=time.monotonic() - started,
            output=output,
        )
    except subprocess.TimeoutExpired as error:
        output = "\n".join(
            part.decode(errors="replace") if isinstance(part, bytes) else part
            for part in (error.stdout, error.stderr)
            if part
        ).strip()
        return RefreshResult(False, time.monotonic() - started, output or "Refresh timed out")
