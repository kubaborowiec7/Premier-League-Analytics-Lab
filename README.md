# Premier League Analytics Lab

A portfolio-grade football analytics project combining:

1. **Player Performance Analytics**
2. **Player Market Value Modelling**
3. **Match Outcome & Score Prediction**
4. **Advanced Statistical Football Lab**
5. **Interactive Streamlit Dashboard**

The project is designed to demonstrate production-minded **Python, SQL, statistics, machine learning, data engineering, explainability, testing, Git/GitHub and deployment** skills.

> Primary scope: Premier League and historical Big-5 / open football data where required for model training.

---

## Business questions

### Player analytics
- Which players outperform peers in the same position?
- Which metrics are stable enough to use for scouting?
- Which players have statistically similar profiles?
- How uncertain is a player's observed performance when minutes are limited?

### Market value
- What should a player be worth based on age, position, playing time and performance?
- Which players appear **undervalued** or **overvalued** relative to the model?
- Which features drive the valuation?
- How wide is the prediction uncertainty?

### Match prediction
- What are the probabilities of home win / draw / away win?
- What scoreline is most likely?
- How does a Poisson/Dixon-Coles model compare with ML?
- Are predicted probabilities well calibrated?
- How does model performance change over time?

---

## Competition scope

**V1 = Premier League only.**

However, Premier League is a configured scope rather than a hard-coded assumption:

```env
ACTIVE_COMPETITIONS=EPL
```

The canonical database, ingestion interfaces, feature engineering and scouting engine
are designed to support additional competitions later.

Future target:

```text
Reference player: Premier League
Candidate universe:
✓ Premier League
✓ La Liga
✓ Bundesliga
✓ Serie A
✓ Ligue 1
```

Cross-league scouting will use competition-aware normalization instead of comparing
raw per-90 numbers blindly.


## Recommended stack

- Python 3.12+
- PostgreSQL 16
- SQLAlchemy + psycopg
- pandas / NumPy / SciPy
- statsmodels
- scikit-learn
- XGBoost
- SHAP
- Plotly + Matplotlib
- Streamlit
- pytest
- Ruff
- GitHub Actions
- Docker Compose

---

## Data strategy

The project separates data sources into three levels:

### Core reproducible sources

**Football-Data.co.uk**
- historical and current match results,
- match statistics,
- optional bookmaker odds used only as a prediction benchmark,
- long history useful for time-aware match models.

**Transfermarkt public dataset snapshots**
- player profiles,
- appearances,
- historical player market values,
- transfers and clubs.
- Use a published dataset snapshot rather than scraping Transfermarkt directly.

**football-data.org**
- fixtures, schedules and league tables for refreshable current-season metadata.

### Advanced event-data source

**StatsBomb Open Data**
- event-level data for selected competitions/seasons,
- suitable for demonstrating event-data engineering, spatial analysis and football analytics methodology.

### Optional/manual source

**FBref export**
- only as a user-provided/manual snapshot where its terms allow.
- Do not build an aggressive automated scraper as a core dependency.

See `docs/DATA_SOURCES.md`.

---

## Repository structure

```text
premier-league-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   └── Home.py
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   ├── DATA_SOURCES.md
│   ├── GITHUB_WORKFLOW.md
│   ├── MILESTONES.md
│   ├── PROJECT_SPEC.md
│   └── STATISTICS.md
├── notebooks/
│   └── README.md
├── reports/
│   ├── figures/
│   └── model_cards/
├── scripts/
│   ├── init_repo.ps1
│   └── init_repo.sh
├── sql/
│   └── schema.sql
├── src/
│   └── pl_analytics/
│       ├── config.py
│       ├── data/
│       │   └── database.py
│       ├── features/
│       ├── models/
│       ├── statistics/
│       └── visualization/
├── tests/
│   └── test_smoke.py
├── .env.example
├── .gitignore
├── AGENTS.md
├── CODEX_START_PROMPT.md
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Modelling principles

1. **No random train/test split for temporal prediction problems.**
2. Features for a match dated `t` may only use information available before `t`.
3. Player market-value models use valuation dates and time-aware splits.
4. Baselines must be implemented before complex ML.
5. Report uncertainty, not only point predictions.
6. Evaluate probability calibration for classification models.
7. Never choose a model solely because it has the highest in-sample R².
8. Keep notebooks exploratory; reusable logic belongs in `src/`.
9. Every model must have a model card.
10. Every important dataset must have provenance documented.

---

## Core metrics

### Market value regression
- MAE
- RMSE
- RMSLE / MAE on log-value
- R²
- Median absolute error
- error by age / position / value band
- temporal out-of-sample performance

### Match probabilities
- Multiclass log loss
- Brier score
- Ranked Probability Score
- calibration curves / ECE
- accuracy as a secondary metric

### Score models
- goal MAE
- Poisson deviance
- log likelihood
- scoreline calibration

---

## Target dashboard

### Player Explorer
- position-adjusted percentiles,
- radar/profile chart,
- per-90 and possession-adjusted statistics,
- confidence/reliability indicator,
- similar players,
- estimated vs observed market value,
- SHAP explanation.

### Scouting
Filters such as:
- position,
- age,
- minutes,
- estimated value,
- actual value,
- undervaluation,
- performance percentile,
- similarity to selected player.

### Match Predictor
- home/draw/away probabilities,
- expected goals,
- most likely scorelines,
- Elo strength,
- model comparison,
- calibrated probabilities.

### Model Lab
- backtest results,
- residual analysis,
- calibration,
- feature importance,
- model cards.

---

## First run

Use Python 3.12+ and run these commands from the repository root. PostgreSQL is
optional for the landing page and tests; no datasets or API tokens are required.

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
python -m pip install -e ".[dev]"
python -m compileall src
ruff check .
pytest
streamlit run app/Home.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
Copy-Item .env.example .env
python -m pip install -e ".[dev]"
python -m compileall src
ruff check .
pytest
streamlit run app/Home.py
```

If PowerShell blocks script activation, use `./.venv/Scripts/python.exe -m pip`,
`./.venv/Scripts/python.exe -m pytest`, and the other executables in `.venv/Scripts/`
directly. Copy `.env.example` only on first setup; preserve an existing `.env`.

### Configuration

`get_settings()` reads environment variables and an optional `.env` in the current
working directory. Explicit `Settings(...)` arguments override environment variables,
which override `.env`, which overrides defaults. Relative paths are relative to the
working directory. Loading settings does not create directories or connections.

| Variable | Default / purpose |
|---|---|
| `ACTIVE_COMPETITIONS` | `EPL`; comma-separated canonical codes, trimmed, uppercased and deduplicated |
| `DATABASE_URL` | Local PostgreSQL URL from `.env.example`; requires `postgresql+psycopg` |
| `DATABASE_CONNECT_TIMEOUT` | `5` seconds; integer from 1 to 60 |
| `FOOTBALL_DATA_API_TOKEN` | Optional; unused in M0 |
| `DATA_DIR` | `data`; immutable inputs belong under `raw/` |
| `ARTIFACT_DIR` | `artifacts` |
| `LOG_LEVEL` | `INFO`; DEBUG, INFO, WARNING, ERROR or CRITICAL |

An empty competition scope is rejected. Codes are configurable; M0 does not validate
them against a dataset or offer competition filters before data is available.

### Optional PostgreSQL smoke check

Start Docker Desktop/the Docker daemon first, then run:

```bash
docker compose --env-file .env.example config --quiet
docker compose up -d --wait db
pl-db-smoke
```

`python -m pl_analytics.data.database` is equivalent to `pl-db-smoke`. It runs
`SELECT 1`, closes the connection, and returns exit code 0 on success or 1 on failure.
Neither package import nor app startup attempts a database connection.

Compose uses PostgreSQL 16, a named persistent volume and a read-only schema mount.
It binds the host port to `127.0.0.1`. The example `postgres` credentials are for local
development only. Set `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` and
`POSTGRES_PORT` in `.env` and keep `DATABASE_URL` in sync (URL-encode credentials
containing reserved characters).

The schema runs only when PostgreSQL initializes an empty volume. Editing credentials
or the schema does not migrate an existing database. `docker compose down` stops the
service while retaining its data. A connection smoke check verifies connectivity,
not schema contents or readiness of analytics artifacts.

CI installs the package on Python 3.12, validates Compose, compiles `src`, and runs
Ruff and the offline test suite, including Streamlit's app test runner. It also runs
PostgreSQL 16 integration tests and the pinned M1 public-data ingestion/replay check.
M2 SQL installation, repeatability and analytical coverage are also checked in CI.

## M1 data ingestion

```bash
python scripts/verify_ingestion.py
```

This validates the pinned 2023/24 sample: 380 matches and 2,093 source-reported EPL
valuations for 948 players. Files are archived with SHA-256 and provenance under
ignored `data/raw/`. Repeat runs reuse the archived files. To also load PostgreSQL:

```bash
docker compose up -d --wait db
python scripts/verify_ingestion.py --load --initialize-schema
```

The explicit schema option also upgrades an existing M0 database. The check loads
and replays both batches, verifies counts and confirms provenance links. See
[the ingestion guide](docs/INGESTION.md) for individual CLI commands, local published
snapshot imports, validation behavior and known identity/time limitations.

Historical competition membership in the valuation source is not fully verified;
reported club context is preserved separately and is not a valid historical feature.
Raw datasets, local reports and credentials are not committed.

## M2 SQL analytics

After loading M1, install and check the analytical views:

```bash
python scripts/verify_analytics.py --install --manifest data/manifests/m1_sources.json
```

The views expose observed team-match results, conservative pre-match histories,
player-season performance and valuation summaries. The current sample has no player
appearances: performance remains missing for its 948 valuation-only players.
Season summaries are retrospective, not valid inputs for earlier predictions.
See [SQL analytics](docs/SQL_ANALYTICS.md) for table grains, temporal cutoffs, upgrades,
advanced SQL examples and verification. No new data is downloaded by this command.

---

## M3 exploratory data quality

With the pinned M1 archives available:

```bash
python scripts/run_quality.py
python scripts/execute_quality_notebook.py
```

Both commands run offline without PostgreSQL. The clean-kernel notebook produces
missingness/coverage diagnostics, match and valuation distributions, and a leakage
audit. [Measured findings and tracked data issues](docs/DATA_QUALITY.md) explain why
the current sample is development data and is not yet sufficient for player models.
Reports, executed notebooks and figures stay under ignored `artifacts/`.

## M4 player statistics

```bash
python scripts/build_player_analytics.py --download
```

This explicitly acquires pinned published appearance data and builds per-90 rates,
position peer percentiles/z-scores, exposure shrinkage, bootstrap intervals, PCA and
player similarity. Repeat without `--download` for offline execution. The real example
and chart are saved under `artifacts/m4/`; scoped analytical tables go to
`data/processed/m4/`. [Methodology and example](docs/PLAYER_ANALYTICS.md) explain the
snapshot-position and limited-metric caveats. The initial sample covers 570 players
and 11,384 appearances. These artifacts are descriptive; no predictor is trained.

## M5 market value models

```bash
python scripts/train_value_models.py --stage select --download
python scripts/train_value_models.py --stage final
python scripts/plot_value_report.py
```

Selection fits median/persistence, OLS, Ridge, Lasso and XGBoost benchmarks on strictly
prior observations. Final evaluation uses frozen models on 2025/26. The selected
Ridge achieved EUR 3.61m MAE versus EUR 3.69m persistence and EUR 17.39m median;
its high EUR 11.19m RMSE exposes serious low-exposure extrapolation failures.
These are research artifacts, not a validated transfer-pricing tool. Predictions,
uncertainty, SHAP, historical review rankings and diagnostics are under `artifacts/m5/`.
See [reproduction and methodology](docs/VALUE_MODELS.md) and the
[model card](reports/model_cards/player_value.md). Reproduction requires a fresh
`--output` directory after selection has been frozen. CI repeats both stages.

## M6 statistical match forecasts

```bash
python scripts/train_match_models.py --stage select --download
python scripts/train_match_models.py --stage final --download
python scripts/plot_match_report.py
```

Seven fixed base-rate/Elo/Poisson/Dixon–Coles benchmarks use monthly rolling origins.
The selected time-weighted Dixon–Coles policy achieved 1.0263 log loss on 380 final
2025/26 matches versus 1.0844 for base rates. Score matrices, calibration diagnostics
and fitted models remain under `artifacts/m6/`. See [methodology](docs/MATCH_MODELS.md)
and the [model card](reports/model_cards/match_statistics.md) for constraints, uncertainty
and the tiny difference from Poisson. These are historical forecasts; no live fixture
service or match UI is added in M6. Use a fresh `--output` directory for reproduction.

## M7 calibrated match ML

```bash
python scripts/train_match_ml.py --stage select --download
python scripts/train_match_ml.py --stage final --download
python scripts/plot_match_ml.py
```

Past-only Elo/form features feed logistic and XGBoost classifiers, with a separate
temperature-calibration season. A new 40-match 2026/27 snapshot is the final pilot.
Selected calibrated logistic log loss is 1.0491 versus base rate 1.1323 and Dixon–Coles
1.0115 on identical fixtures. ML does not outperform the statistical benchmark here;
the small sample limits conclusions. See [methodology](docs/MATCH_ML.md) and the
[model card](reports/model_cards/match_ml.md). Preserve the pinned current-season
archive: its public download URL changes as new results arrive.

## GitHub philosophy

The GitHub history is part of the portfolio.

Do not build the entire project and upload it in one commit. Use small, meaningful commits and GitHub Issues/Milestones. Each milestone must update:
- code,
- tests,
- relevant documentation,
- README progress,
- model/data cards where applicable.

See `docs/GITHUB_WORKFLOW.md`.

---

## Status

| Milestone | Status |
|---|---|
| M0 — Repository foundation | Implemented |
| M1 — Data ingestion | Implemented; PostgreSQL load/replay verified in CI |
| M2 — PostgreSQL analytics | Implemented; SQL tests and sample coverage verified in CI |
| M3 — EDA and data quality | Implemented; offline tests and clean-kernel notebook verified |
| M4 — Advanced player statistics | Implemented; tests and pinned player profile verified |
| M5 — Market value models | Implemented; frozen chronological evaluation and limitations documented |
| M6 — Statistical match models | Implemented; rolling evaluation and score distributions verified |
| M7 — Calibrated match ML | Implemented; separate calibration and 40-match final pilot evaluated |
| M8–M10 | Not started |

The landing page explains the project. Ingestion is available through explicit CLI
commands. M4/M5 analytical and model artifacts are built explicitly; dashboard
integration is planned for M8.

The exact first Codex instruction is in `CODEX_START_PROMPT.md`.
