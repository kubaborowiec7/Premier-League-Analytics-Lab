# Architecture

## High-level flow

```text
External Sources
      |
      v
Raw immutable files / API snapshots
      |
      v
Ingestion + validation
      |
      v
PostgreSQL canonical tables
      |
      +------------------+
      |                  |
      v                  v
SQL feature marts     Python feature pipelines
      |                  |
      +--------+---------+
               |
               v
        Model training
               |
      +--------+---------+
      |                  |
      v                  v
 Model artifacts     Evaluation reports
      |                  |
      +--------+---------+
               |
               v
        Streamlit app
```

## Layers

### `src/pl_analytics/data`
Responsibilities:
- download/load raw sources,
- schema validation,
- canonical ID mapping,
- database I/O,
- provenance.

### `src/pl_analytics/features`
Responsibilities:
- player per-90 metrics,
- rolling team metrics,
- Elo,
- age features,
- positional peer groups,
- historical squad value summaries.

### `src/pl_analytics/statistics`
Responsibilities:
- bootstrap,
- shrinkage,
- PCA,
- clustering helpers,
- Poisson/Dixon-Coles.

### `src/pl_analytics/models`
Responsibilities:
- market-value models,
- match classifiers,
- probability calibration,
- evaluation utilities,
- artifact persistence.

### `src/pl_analytics/visualization`
Responsibilities:
- Plotly/Matplotlib reusable charts,
- calibration chart,
- radar/profile view,
- residual chart,
- probability matrix.

### `app`
Presentation layer only.
Avoid implementing feature engineering/model logic in Streamlit pages.

---

## Artifact conventions

Suggested local folders:

```text
artifacts/
├── models/
├── preprocessors/
├── metrics/
└── metadata/
```

The folder may be Git-ignored for binary objects.
Keep lightweight JSON model metadata/model cards under `reports/model_cards/`.

---

## Configuration

Configuration must come from environment variables.

Examples:
- `DATABASE_URL`
- `FOOTBALL_DATA_API_TOKEN`
- `DATA_DIR`
- `ARTIFACT_DIR`
- `LOG_LEVEL`

No credentials in source control.

M0 implements `Settings` / `get_settings()` in `src/pl_analytics/config.py`.
Settings load lazily from explicit constructor arguments, environment variables,
the working directory's optional `.env`, and defaults, in that order. Relative
data/artifact paths resolve against the working directory; no folders are created.
`ACTIVE_COMPETITIONS` accepts comma-separated canonical codes. Its explicit parsing
uses Pydantic Settings' `NoDecode` support, requiring version 2.7 or later.

`src/pl_analytics/data/database.py` provides the explicit `pl-db-smoke` command:
a bounded connection attempt, `SELECT 1`, and disposal of the engine. Importing
modules and starting the Streamlit landing page require no database connection.
Connection errors are reported without logging raw driver messages or credentials.
Docker Compose initializes the existing schema in dependency order on an empty volume;
the M1 and M2 migrations extend it, followed by the M2 analytical views.

M1 now adds an explicit additive ingestion migration and these data modules:
`snapshots.py` (immutable acquisition/checksums), `contracts.py` (batch/scope validation),
`football_data.py` and `transfermarkt.py` (source adapters), `repository.py` (atomic
PostgreSQL persistence and lineage), and `cli.py` (explicit orchestration).
See [INGESTION.md](INGESTION.md) for refresh, replay and failure behavior. Analytical
SQL is implemented in M2. No ingestion or model training is performed by app startup.

M2's `data/analytics.py` explicitly installs repository SQL and validates invariants
inside a transaction. `sql/analytics.sql` defines ordinary team-match and player-season
views, including a separate pre-match view with a conservative timestamp cutoff.
`scripts/verify_analytics.py` reports coverage without downloading sources. Retrospective
views are not model-ready historical feature tables. See [SQL_ANALYTICS.md](SQL_ANALYTICS.md).

M3's `data/quality.py` reuses the M1 adapters to analyse checksum-pinned local archives
without a database or network. `scripts/run_quality.py` writes aggregate diagnostics;
`notebooks/01_data_quality.ipynb` uses the same functions for exploratory figures.
`scripts/execute_quality_notebook.py` validates a fresh-kernel run, with outputs kept
under ignored artifacts and uploaded in CI. No canonical/raw data or model features
are changed. See [DATA_QUALITY.md](DATA_QUALITY.md) for the leakage audit and findings.

M4 adds `data/appearances.py` for published source-ID joins, `features/players.py` for
strict-cutoff aggregation and fitted peer normalization, and `statistics/players.py`
for bootstrap intervals, PCA and scoped similarity. `scripts/build_player_analytics.py`
creates version-pinned descriptive artifacts and a real profile example. These outputs
preserve competition/season/reference date and snapshot-position context; they do not
populate PostgreSQL or serve as historically verified predictive features. See
[PLAYER_ANALYTICS.md](PLAYER_ANALYTICS.md).

---

## Reproducibility

M5 uses `data/value_history.py` to read stage-limited, pinned published histories and
`features/value.py` to construct strictly prior observation features. It is an
artifact-based source-ID path, independent of unverified SQL player membership.
`models/value.py` contains fitted benchmarks, SHAP and interval/evaluation utilities;
`models/value_experiment.py` separates selection from hash-checked frozen final
evaluation. `scripts/train_value_models.py` is the explicit CLI and
`scripts/plot_value_report.py` renders existing outputs. See [VALUE_MODELS.md](VALUE_MODELS.md).
No changes to the canonical database or app-startup behavior are required.

Every model run should eventually record:
- run ID,
- timestamp,
- Git commit SHA,
- data snapshot/version,
- feature set version,
- hyperparameters,
- split dates,
- metrics,
- artifact path.

The first implementation can store this in PostgreSQL + JSON.
MLflow is a stretch goal, not a dependency of the MVP.

## Competition-agnostic design

Premier League is the **initial configured scope**, not a hard-coded domain constraint.

V1 configuration:

```env
ACTIVE_COMPETITIONS=EPL
```

Future configuration may become:

```env
ACTIVE_COMPETITIONS=EPL,LALIGA,BUNDESLIGA,SERIE_A,LIGUE_1
```

Rules:

- no `if league == "Premier League"` logic in feature/model modules;
- every match belongs to a `competition_id` and `season`;
- clubs and players are canonical entities independent of competition;
- player appearances identify the club and match, allowing a player to move between leagues;
- player features are keyed by player + competition + season + reference date where appropriate;
- comparison groups explicitly declare whether they are:
  - competition-specific,
  - multi-competition,
  - globally normalized;
- source-specific league codes are translated through adapters into canonical competition codes;
- UI competition filters are populated from available data, not a hard-coded list.

### Scouting architecture

The scouting engine must accept a candidate universe rather than assume the Premier League:

```text
reference player
      +
candidate competitions
      +
position / age / minutes filters
      ↓
competition-aware normalization
      ↓
feature alignment
      ↓
similarity model
      ↓
ranked candidates
```

For V1:

```text
candidate competitions = [EPL]
```

For a future version:

```text
candidate competitions = [EPL, LALIGA, BUNDESLIGA, SERIE_A, LIGUE_1]
```

Cross-league comparisons must account for league context before claiming that raw
per-90 numbers are directly comparable. Candidate approaches include league-season
standardization, competition-strength coefficients, hierarchical models, or learned
league adjustments.
