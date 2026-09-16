# Deployment and operations

The app serves prepared historical research artifacts. Training and acquisition
are explicit offline jobs. Run from the repository root with Python 3.12:

```bash
python -m pip install -c requirements/constraints.txt -e ".[dev]"
python -m pip check
python -m compileall src
ruff check .
pytest --cov=pl_analytics --cov-report=term-missing --cov-report=xml:artifacts/coverage.xml
streamlit run app/Home.py --server.address=127.0.0.1
```

Direct runtime dependency versions are pinned to the tested environment in
`requirements/constraints.txt`; transitive dependencies and the Python base image
remain resolver/tag-managed. This is not a fully hermetic lock. Rebuild model bundles
and rerun their frozen evaluation when changing numerical package versions.

## Container

```bash
docker compose -f compose.app.yml config --quiet
docker compose -f compose.app.yml up --build -d
```

Open http://localhost:8501. Stop with `docker compose -f compose.app.yml down`.
The separate PostgreSQL Compose file is for offline ingestion and SQL work;
the dashboard does not depend on it. `APP_PORT` selects another host port.

The image runs as UID 10001. Compose drops capabilities, prevents privilege
escalation, uses a read-only root filesystem and mounts only processed data and
artifacts read-only. `/tmp` is an ephemeral 128MB tmpfs. The build context excludes
raw data, `.env`, Git history, caches and notebooks using an allowlist. Empty
artifact directories are a supported startup state. The health endpoint verifies
the server, not artifact quality; UI tests verify data-dependent behavior.

Build M4–M7 artifacts and the dashboard context on the trusted analysis machine,
then supply `data/processed/m4` and the relevant `artifacts/m5`, `m6`, `m7` and
`dashboard` trees. Never mount raw archives or credentials into the serving app.
Model cards are copied into the image. Preserve source licenses/attribution and
do not redistribute bulk source data as part of a release image.

For remote hosting, use TLS and an authenticated reverse proxy or platform access
control. Keep CORS and XSRF protection enabled. The default Compose port is bound
to loopback; it is not a public deployment. No public URL is claimed by this repo.

## Refresh and rollback

Retain immutable raw archives and source manifests. A mutable source URL whose
bytes no longer match its hash must fail; create a new versioned experiment rather
than altering an evaluated holdout. Stage and verify a complete artifact directory
before swapping it into service. Do not rewrite individual active model bundles.
Rebuild dashboard metadata after a legitimate model rebuild, verify its date and
checksums, then restart the app. Keep the previous code/image and complete artifact
set for rollback. Checksums detect inconsistency, not an untrusted publisher.

CI produces coverage XML and builds/smoke-tests a container without data. A separate
job reproduces pinned data, PostgreSQL integration and all model experiments.
Local Docker validation is limited to Compose parsing while the daemon is absent;
container build/start results must be checked in GitHub Actions.
