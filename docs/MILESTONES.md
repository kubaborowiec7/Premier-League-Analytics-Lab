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
[SQL_ANALYTICS.md](SQL_ANALYTICS.md). M3 audits these data limitations.

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

Implementation status: complete. Tracking issue: #5.
Local verification: compileall and Ruff passed; 72 tests passed and 13 PostgreSQL
tests skipped without a server. The notebook executed from a fresh kernel on both
synthetic fixtures and the pinned 380-match / 2,093-valuation sample. CI repeats
the offline notebook execution after validating M1 ingestion and M2 SQL.
The pinned-sample notebook, offline report generator, synthetic tests and leakage
audit are described in [DATA_QUALITY.md](DATA_QUALITY.md). Data blockers are tracked
in #6 (appearances/identity), #7 (historical membership) and #8 (history/holdouts).
M4 now adds a separate pinned appearance-data path; the M3 findings concern M1 data.

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

Implementation status: complete. Tracking issue: #10.
Local compileall and Ruff passed; 88 tests passed and 13 PostgreSQL tests skipped
without a server. The pinned appearance pipeline produced the expected counts and
real player profile. CI repeats this build after the M1–M3 checks.
The published appearance adapter and player analytics produce 570 player-season rows
from 11,384 appearances, with a real striker profile, bootstrap intervals and PCA.
See [PLAYER_ANALYTICS.md](PLAYER_ANALYTICS.md) for methodology and reproduction.
Snapshot roles remain historically unverified, and source IDs remain separate from
Football-Data IDs. M5 uses a separate strictly prior participant cohort.

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

Implementation status: implemented and evaluated. Tracking issue: #12.
The experiment manifest and model-selection decision were committed before final
test evaluation. Ridge EUR was selected using 552 validation observations after
training on 5,998 rows; 649 separate rows calibrate intervals. On 1,301 final 2025/26
observations it achieved EUR 3.612m MAE versus EUR 17.390m median and EUR 3.687m
persistence. The small advantage over persistence is uncertain, and EUR 11.193m RMSE
reveals serious low-minute extrapolation failures. The model remains a research
benchmark. See [VALUE_MODELS.md](VALUE_MODELS.md) and the
[model card](../reports/model_cards/player_value.md) for full metrics and limitations.
Offline tests cover chronology, future-data invariance, historical joins, fitted
transforms, interval calibration, SHAP additivity and frozen-stage guards. CI repeats
selection, final evaluation and report plotting from pinned public archives.
Local verification: compileall and Ruff passed; 106 tests passed and 13 opt-in
PostgreSQL tests skipped without a server. Selection, final evaluation and diagnostic
plot rendering passed against pinned local archives. Low-exposure successor work is
tracked in #13. M6 and M7 are implemented below.

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

Implementation status: implemented and evaluated; tracking issue #15. Fixed monthly
origins compare seven models on 760 development matches and 380 final 2025/26 matches.
Time-weighted Dixon–Coles was selected before final evaluation: log loss 1.0263 versus
base rate 1.0844. Its difference from Poisson is very small. Score matrices normalize
to one; tests cover temporal isolation, known metrics and numerical likelihood gradients.
Local compileall/Ruff passed; 125 tests passed, 13 PostgreSQL tests skipped without a
server. Both pinned stages and plot generation passed. CI repeats all stages.
See [MATCH_MODELS.md](MATCH_MODELS.md) and the
[model card](../reports/model_cards/match_statistics.md).
Full CI, including real PostgreSQL and M6 reproduction, passed on `610a415`:
https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/actions/runs/35034971888.

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

Implementation status: implemented and evaluated; tracking issue #18. Monthly past-only
features train logistic/XGBoost candidates on 1,520 matches, select on 380, calibrate
on another 380 and evaluate 40 new 2026/27 matches. Calibrated logistic achieved
log loss 1.0491, compared with base rate 1.1323 and Dixon–Coles 1.0115 on the same
fixtures. The limited pilot does not establish robust superiority or calibration.
See [MATCH_ML.md](MATCH_ML.md) and its [model card](../reports/model_cards/match_ml.md).
Tests cover shifted features, fitted-transform isolation and frozen stage boundaries;
CI reproduces both stages and the aligned statistical comparison.
Local verification: compileall and Ruff passed; 130 tests passed, with 13 PostgreSQL
tests skipped without a local server. Pinned selection, final evaluation and plotting
passed. The current-season archive is retained locally because its provider URL is mutable.

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

Implementation status: six pages implemented; tracking issue #20. Cached artifacts,
data-driven competition filters, historical player/scouting views, dated valuation
intervals, frozen match inference and model cards are integrated. See
[DASHBOARD.md](DASHBOARD.md). Local compileall and Ruff pass; 139 tests pass
and 13 PostgreSQL tests skip without a server. All six populated pages were exercised
with Streamlit AppTest; 20 prepared ML fixtures reproduce frozen probabilities
to numerical precision.

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

Implementation status: deployment configuration and security review implemented;
tracking issue #22. Direct runtime constraints, a non-root Docker image, read-only
Compose service, content-aware caching and artifact-context checks are in place.
CI reports coverage with an 80% minimum and builds/smoke-tests the container.
See [DEPLOYMENT.md](DEPLOYMENT.md) and [SECURITY_REVIEW.md](SECURITY_REVIEW.md).
Local verification: 144 tests passed, 13 PostgreSQL tests skipped; package statement
coverage is 87.05%. Compileall, Ruff, pip check and app Compose validation pass.
Local Docker daemon is unavailable. Full hosted CI, including PostgreSQL, pinned
experiments and the non-root container smoke test, passed on `4e21499`:
https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/actions/runs/35071950804.

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

Implementation status: portfolio documentation, three real screenshots, architecture
diagram, results table, source attribution and CV/LinkedIn draft are prepared; issue #24.
The user selected local demo only; no public website is published. Package version is
1.0.0. Release tagging follows successful CI for the release commit.
Local compileall/Ruff/pip check and editable installation pass; 144 tests pass and
13 PostgreSQL tests skip locally. See [RELEASE.md](RELEASE.md).

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
