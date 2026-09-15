# Milestones

Work sequentially. Each milestone should have a GitHub Issue or Milestone and a clean commit history.

## M0 — Repository foundation

Implementation status: implemented. Package configuration, the optional `pl-db-smoke`
command, PostgreSQL Compose setup, offline tests, CI and the landing page are in place.
See README for installation and verification commands.

Local verification (2026-09-14, Python 3.12.14): editable installation succeeded;
`python -m compileall src`, `ruff check .`, `pytest` (28 tests, including Streamlit
startup), `python -m pip check`, and Compose configuration validation passed.
Live PostgreSQL initialization/connectivity remains unverified locally because the
Docker daemon was unavailable. Offline database tests cover query success/failure
and connection cleanup; they do not replace a live PostgreSQL integration check.

Deliver:
- installable Python package,
- configuration,
- PostgreSQL Docker Compose,
- CI,
- lint/test setup,
- Streamlit landing page,
- smoke tests.

Definition of done:
- `ruff check .` passes,
- `pytest` passes without external data,
- app opens gracefully,
- README commands are correct.

Suggested commit:
`chore: initialize analytics project foundation`

---

## M1 — Data ingestion and provenance

Implementation status: complete. Ingestion code, source pins, offline tests and
documentation are implemented. PostgreSQL 16 integration tests and the pinned public
sample load/replay passed in GitHub Actions on commit `2a77694`:
https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/actions/runs/34944298108.
The sample includes 380 matches, 948 player profiles and 2,093 source-reported EPL
valuations. Batch replay and provenance were verified against a real PostgreSQL service.
Local Docker remains unavailable; local raw-file validation is reproducible offline.
The optional API client is deferred. Tracking issue: #1; implementation PR: #2.
M2 builds on these canonical records without acquiring another dataset.

Deliver:
- competition-parameterized Football-Data.co.uk downloader (EPL enabled in V1),
- Transfermarkt dataset import adapter,
- optional football-data.org client,
- source metadata/provenance table,
- raw checksum handling,
- schema validation,
- idempotent loading.

Definition of done:
- one EPL season imports end-to-end,
- one player/value snapshot imports,
- rerunning does not duplicate canonical records,
- source metadata is queryable.

Suggested commits:
- `data: add football match ingestion pipeline`
- `data: add player valuation snapshot loader`

---

## M2 — PostgreSQL analytics layer

Implementation status: complete. Tracking issue: #3; implementation PR: #4.
Local compileall, Ruff and Compose validation passed; pytest passed 62 offline tests
and skipped 13 opt-in PostgreSQL tests. All 13 PostgreSQL 16 tests, M1 replay and
repeated M2 installation/coverage checks passed in GitHub Actions on commit `877e711`:
https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/actions/runs/34946864403.
The pinned sample produces 760 team-match rows, 948 valuation-only player-season rows
and 2,093 source-separated valuations. Local Docker is still unavailable.
The ordinary team-match/player-season marts, conservative prior-five-game histories,
canonical checks/indexes and executable SQL examples are described in
[SQL_ANALYTICS.md](SQL_ANALYTICS.md). M3 has not started.

Deliver:
- finalized canonical schema,
- indexes,
- SQL views/marts,
- example advanced SQL queries.

Must demonstrate:
- CTE,
- window functions,
- `LAG`,
- rolling features,
- conditional aggregation.

Definition of done:
- SQL creates player-season and team-match analytical marts,
- SQL tests/checks validate uniqueness and basic invariants.

Suggested commit:
`feat: build postgres analytics marts`

---

## M3 — EDA and data quality

Deliver:
- reproducible EDA notebook,
- missingness analysis,
- target distributions,
- temporal coverage,
- player-value skew analysis,
- leakage audit,
- data quality report.

Definition of done:
- notebook runs top-to-bottom,
- findings summarized in docs,
- data issues become tracked GitHub Issues.

Suggested commit:
`docs: add exploratory data quality analysis`

---

## M4 — Advanced player statistics

Deliver:
- per-90 features,
- peer groups,
- percentiles/z-scores,
- minutes reliability/shrinkage,
- bootstrap intervals,
- PCA,
- player similarity with an explicit candidate competition universe,
- optional clustering.

Definition of done:
- unit tests for core transformations,
- example player profile,
- methodology documented.

Suggested commit:
`feat: add advanced player analytics`

---

## M5 — Market value model

Deliver:
- baseline models,
- OLS/Ridge/Lasso benchmark,
- XGBoost model,
- chronological evaluation,
- residual/segment analysis,
- SHAP,
- prediction uncertainty,
- undervaluation ranking,
- model card.

Definition of done:
- newest period remains untouched until final evaluation,
- model beats simple baseline on chosen primary metric,
- limitations are explicit.

Suggested commit:
`model: train player market value models`

---

## M6 — Match statistical models

Deliver:
- Elo,
- independent Poisson,
- Dixon-Coles,
- time decay option,
- scoreline probabilities,
- rolling-origin backtest,
- tests.

Definition of done:
- pre-match values are leakage-safe,
- score probability matrix sums appropriately,
- model comparison report exists.

Suggested commit:
`model: add elo poisson and dixon coles models`

---

## M7 — Match ML model

Deliver:
- leakage-safe rolling features,
- multinomial logistic regression,
- XGBoost classifier,
- probability calibration,
- log loss/Brier/RPS,
- time-based backtesting,
- model card.

Definition of done:
- calibration is measured,
- final test is chronological,
- benchmark comparison includes statistical models.

Suggested commit:
`model: add calibrated match outcome predictor`

---

## M8 — Streamlit application

Competition selector must be data-driven; V1 exposes EPL only.

Pages:
- Overview,
- Player Explorer,
- Scouting Finder,
- Market Value,
- Match Predictor,
- Model Lab.

Definition of done:
- no training at request time,
- model/data missing states are handled,
- charts explain uncertainty,
- mobile-ish layout remains usable.

Suggested commit:
`feat: build football analytics dashboard`

---

## M9 — Quality and deployment

Deliver:
- expanded tests,
- coverage report,
- Docker/app deployment configuration,
- cached data access,
- error handling,
- security review,
- documentation polish.

Suggested commit:
`chore: prepare analytics app for deployment`

---

## M10 — Portfolio release

Deliver:
- final README with screenshots,
- architecture diagram,
- model results table,
- limitations,
- source attribution,
- demo link,
- release tag `v1.0.0`,
- CV/LinkedIn project description.

Suggested commit:
`docs: prepare v1 portfolio release`
