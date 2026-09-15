# M1: ingestion and provenance

M1 imports results and published player/value files. It does not build features,
train models, scrape Transfermarkt/FBref, or implement the optional football-data.org client.
Tracking issue: https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/issues/1.

## Reproduce the pinned sample

From the repository root, after installing `.[dev]`:

```bash
python scripts/verify_ingestion.py
```

This validates the versioned checksums in `data/manifests/m1_sources.json`. Existing
archives are read offline; missing files are downloaded from the documented public
CSV endpoints. Raw files stay in ignored `data/raw/`. The JSON report goes to ignored
`artifacts/m1-verification.json`. The expected sample is:

| Source selection | Accepted records |
|---|---:|
| Football-Data.co.uk E0 / 2324 | 380 matches, 20 source club identities |
| Published GB1 valuations dated 2023-07-01 through 2024-06-30 | 2,093 valuations |
| Profiles referenced by these valuations | 948 players, 37 reported club identities |

The 37 reported clubs are **not** a claim that 37 clubs played in that EPL season:
upstream competition membership may reflect snapshot-era information. This selection
is a source-reported cohort, not a verified historical squad universe.

Start PostgreSQL and run the complete load/replay check:

```bash
docker compose up -d --wait db
python scripts/verify_ingestion.py --load --initialize-schema
```

`--initialize-schema` explicitly applies `sql/schema.sql` and the additive
`sql/migrations/001_ingestion.sql`. They may be rerun. Existing M0 volumes need this
upgrade because Docker's initialization scripts only execute on a new volume.
The command verifies record counts, provenance links and repeat-load behavior.
No schema or database is dropped by this command.

GitHub Actions also runs this check against PostgreSQL 16 and uploads the JSON report.
If upstream changes bytes at a published URL, the pinned check intentionally fails;
do not update the expected hash without reviewing the new snapshot.

## Individual commands

`python -m pl_analytics.data.cli` is always available after editable installation.
Reinstall the project to register the new `pl-ingest` console alias.

Download and validate one results season (one line works in PowerShell and Bash):

```bash
pl-ingest matches --competition EPL --competition-name "Premier League" --country England --source-code E0 --season 2023/24 --start-date 2023-07-01 --end-date 2024-06-30 --source-season 2324 --timezone Europe/London --sha256 b2e057b0ed959f198b0f63d2391c01239f3608e6de5db68edab3f88e04d07ff3
```

Add `--load` to commit the validated records to `DATABASE_URL`. Without it, no
database connection is made. Alternatively, replace `--source-season ...` with
`--manifest <path-to-metadata.json>` to replay a locally archived file offline.

Import a directory containing the published `players.csv[.gz]`,
`player_valuations.csv[.gz]`, and `clubs.csv[.gz]`:

```bash
pl-ingest player-values --competition EPL --competition-name "Premier League" --country England --source-code GB1 --season 2023/24 --start-date 2023-07-01 --end-date 2024-06-30 --snapshot-dir data/raw/provided-snapshot --dataset-version published-2026-07-06 --source-url https://github.com/dcaribou/transfermarkt-datasets
```

For public acquisition, replace `--snapshot-dir ...` with `--download-published`.
Optional `--checksums <json>` accepts a mapping of the three exact filenames to SHA-256
values. All three must be present when pinning is requested. For an offline replay,
use `--manifests <players-metadata.json> <valuations-metadata.json> <clubs-metadata.json>`
instead of an acquisition option. `--dataset-version` is required for acquisition.

Competition IDs must be enabled in `ACTIVE_COMPETITIONS`. The CLI requires source
code, display name, country, season label, and inclusive date boundaries explicitly.
There is no EPL-only branch or assumed July-to-June season in the reusable adapters.

## Storage, validation and failure behavior

Raw storage layout:

```text
data/raw/<source>/<sha256>/payload.csv[.gz]
data/raw/<source>/<sha256>/metadata.json
```

Each manifest records source URL, retrieval timestamp, dataset version, terms note,
byte length and SHA-256. Repeated identical acquisitions reuse the first provenance
record. New bytes create a different directory. Input bytes are never cleaned or
rewritten; parsed canonical records exist in memory and PostgreSQL. A changed raw
file fails verification. Downloads have a 128 MiB cap, finite timeouts, and at most
three attempts for transport failures, HTTP 429 or HTTP 5xx.

Validation rejects malformed CSV, missing required columns, contradictory duplicates,
wrong divisions, out-of-season results, invalid goals/statistics, mismatched outcomes,
non-finite/negative valuations and broken selected player/club references. Identical
duplicates collapse. Valuations outside the requested source competition/date scope
are counted as skipped; this is filtering, not silent correction of selected rows.

A load commits dimensions, observations, source snapshots and batch lineage in one
PostgreSQL transaction. A deterministic batch identity includes scope, source hashes,
normalized records and loader version. Replaying it returns `already_loaded=true`.
New batches insert compatible records and reject conflicting existing values, rolling
back the entire transaction. Automatic corrections to previously accepted records
are deliberately deferred to an explicit reconciliation policy.

After an interrupted download, retry the command. Temporary download files are removed
on ordinary failure. An incomplete or damaged archived payload is rejected instead of
silently overwritten; inspect and quarantine that specific archive before reacquiring.

## Identity and time limitations

- Football-Data club IDs hash source names plus country. Match IDs include source,
  competition, season, home and away; match date is not identity, so postponements
  cannot silently create a second match. Replays/cup formats with multiple matches
  for the same ordered pair require a source match ID adapter in a later change.
- Transfermarkt IDs use `tm:player:<id>` / `tm:club:<id>`. No fuzzy name merge is made
  between sources. Cross-source club identity reconciliation is still required before
  joining squad values to Football-Data matches.
- Football-Data dates use day/month/year. Kickoff times are interpreted in the explicit
  source timezone, then converted to UTC. A missing time is stored at local midnight
  with `match_time_known=false`; it must not be used to order same-day features reliably.
- Valuation date is preserved. `club_id` remains NULL because the published club field
  can use a current-club fallback. `source_reported_club_id` preserves that source field;
  `competition_context=source_reported_unverified` labels the competition assignment.
  Neither is suitable as a historical feature without independent date-aligned evidence.
- Player position and nationality describe the profile snapshot. Latest/highest market
  values, contracts and current-club profile fields are excluded from the canonical player
  projection. Later modelling must separately establish feature availability at valuation time.
- `source_snapshots.coverage_start/end` remain NULL unless authoritative full-file coverage
  is known. The explicit **selection window** is stored on each ingestion batch, so a
  filtered subset is never misrepresented as the entire raw file's temporal coverage.

## Verification

```bash
python -m compileall src
ruff check .
pytest
```

The default suite uses synthetic files and mocked HTTP. Three PostgreSQL tests skip
unless `TEST_DATABASE_URL` is set. To run them explicitly:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/pl_analytics"
pytest tests/test_postgres_ingestion.py
```

These integration tests create uniquely named temporary schemas, verify migrations,
idempotence, lineage and rollback, then drop only those schemas. Use a development
database account with schema creation privileges. CI runs the tests against its own
disposable PostgreSQL service, followed by the pinned public-data load and replay.
