"""Content-addressed raw files and immutable provenance manifests."""

import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx

from pl_analytics.data.contracts import DataValidationError


def sha256_file(path: Path) -> str:
    """Hash a file without loading the entire dataset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """A source file's immutable acquisition record, separate from parsed data."""

    path: Path
    source_name: str
    source_url: str
    dataset_version: str
    license_or_terms_note: str
    retrieved_at: str
    sha256: str
    size_bytes: int

    @property
    def snapshot_id(self) -> str:
        """Stable identity matches the database's source/checksum uniqueness."""
        return str(uuid5(NAMESPACE_URL, f"{self.source_name}:{self.sha256}"))

    def verify(self) -> None:
        """Fail closed when an archived raw file has been modified."""
        if self.path.stat().st_size != self.size_bytes or sha256_file(self.path) != self.sha256:
            raise DataValidationError(f"Raw snapshot checksum mismatch: {self.path}")

    @classmethod
    def read(cls, manifest: Path) -> "Snapshot":
        """Load and verify a portable sidecar stored alongside the raw payload."""
        data = json.loads(manifest.read_text(encoding="utf-8"))
        name = data.pop("file_name")
        if Path(name).name != name or name not in {"payload.csv", "payload.csv.gz"}:
            raise DataValidationError("Invalid snapshot payload filename")
        if manifest.parent.name != data["sha256"]:
            raise DataValidationError("Snapshot manifest does not match its pinned directory hash")
        snapshot = cls(path=manifest.parent / name, **data)
        snapshot.verify()
        return snapshot


class SnapshotStore:
    """Archive byte-identical sources under raw/<source>/<sha256>/; never overwrite."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def import_file(
        self,
        path: Path,
        *,
        source_name: str,
        source_url: str,
        dataset_version: str,
        license_note: str,
        expected_sha256: str | None = None,
        retrieved_at: str | None = None,
    ) -> Snapshot:
        """Archive a provided CSV; repeated identical acquisitions reuse its first metadata."""
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", source_name) or source_name in {".", ".."}:
            raise DataValidationError("Invalid source name")
        if not all(value.strip() for value in (source_url, dataset_version, license_note)):
            raise DataValidationError("Source URL, dataset version and terms note are required")
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise DataValidationError("Downloaded/provided file does not match the pinned SHA-256")
        folder = self.raw_dir / source_name / digest
        manifest = folder / "metadata.json"
        if manifest.exists():
            existing = Snapshot.read(manifest)
            if existing.source_name != source_name or existing.sha256 != digest:
                raise DataValidationError("Snapshot manifest identity mismatch")
            return existing
        folder.mkdir(parents=True, exist_ok=True)
        payload = folder / ("payload.csv.gz" if path.suffix == ".gz" else "payload.csv")
        try:
            with payload.open("xb") as target, path.open("rb") as source:
                shutil.copyfileobj(source, target)
        except FileExistsError:
            if sha256_file(payload) != digest:
                raise DataValidationError("Existing raw payload has been modified") from None
        snapshot = Snapshot(
            payload,
            source_name,
            source_url,
            dataset_version,
            license_note,
            retrieved_at or datetime.now(UTC).isoformat(),
            digest,
            path.stat().st_size,
        )
        snapshot.verify()
        metadata = asdict(snapshot)
        metadata.pop("path")
        metadata["file_name"] = payload.name
        try:
            with manifest.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
        except FileExistsError:
            return Snapshot.read(manifest)
        return snapshot

    def download(
        self,
        url: str,
        *,
        source_name: str,
        dataset_version: str,
        license_note: str,
        expected_sha256: str | None = None,
        client: httpx.Client | None = None,
        max_bytes: int = 128 * 1024 * 1024,
    ) -> Snapshot:
        """Stream a public dataset with finite timeouts, a size cap and three transient retries."""
        if client is None:
            with httpx.Client(timeout=60, follow_redirects=True) as owned:
                return self.download(
                    url,
                    source_name=source_name,
                    dataset_version=dataset_version,
                    license_note=license_note,
                    expected_sha256=expected_sha256,
                    client=owned,
                    max_bytes=max_bytes,
                )
        incoming = self.raw_dir / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        suffix = ".csv.gz" if url.split("?")[0].endswith(".gz") else ".csv"
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(dir=incoming, suffix=suffix, delete=False) as temp:
                temporary = Path(temp.name)
            try:
                with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                    response.raise_for_status()
                    size = 0
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise DataValidationError(
                                    "Dataset exceeds the configured size limit"
                                )
                            stream.write(chunk)
                if size == 0:
                    raise DataValidationError("Downloaded dataset is empty")
                return self.import_file(
                    temporary,
                    source_name=source_name,
                    source_url=str(response.url),
                    dataset_version=dataset_version,
                    license_note=license_note,
                    expected_sha256=expected_sha256,
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                transient = isinstance(error, httpx.TransportError) or (
                    error.response.status_code == 429 or error.response.status_code >= 500
                )
                if not transient or attempt == 2:
                    raise
                time.sleep(2**attempt)
            finally:
                temporary.unlink(missing_ok=True)
        raise RuntimeError("Download attempts exhausted")
