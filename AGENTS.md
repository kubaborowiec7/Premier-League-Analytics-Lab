# AGENTS.md — Codex operating rules

You are working on a portfolio-grade football analytics repository.

## 1. Read before editing

Before making changes, read:
- `README.md`
- `docs/PROJECT_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/MILESTONES.md`
- `docs/STATISTICS.md`
- `docs/DATA_SOURCES.md`

Work on **one milestone at a time** unless the user explicitly asks otherwise.

## 2. Main objective

Build a reproducible analytics platform that combines:
- player performance analytics,
- player market-value estimation,
- match outcome/score prediction,
- advanced statistical analysis,
- an interactive Streamlit app.

The project must show strong engineering and statistical practice, not only attractive charts.

## 2A. Competition-agnostic requirement

V1 uses only Premier League data, but **never hard-code Premier League as the only
supported competition**.

Required:
- competition scope comes from settings/data;
- every relevant analytical row keeps `competition_id` and `season`;
- loaders accept competition/source-code parameters;
- source-specific league codes are mapped to canonical codes;
- UI filters are populated from available competitions;
- scouting similarity accepts a candidate competition universe;
- feature normalization records its competition/season/position context.

Forbidden:
- logic such as `if competition == "Premier League"` in reusable analytics modules;
- fixed arrays of Premier League clubs in model code;
- feature tables that lose `competition_id`;
- copying a whole pipeline to add another league.

V1 tests may use EPL fixtures/data, but the interfaces must remain generic.

## 3. Data rules

- Never commit secrets.
- Never commit large raw datasets to Git unless explicitly appropriate and licensed.
- Store local datasets under `data/raw/`; this folder is ignored by Git.
- Keep small schemas, samples or test fixtures under version control.
- Every source must be documented in `docs/DATA_SOURCES.md`.
- Record retrieval date, source URL, licence/terms notes and dataset version where possible.
- Do not create an automated scraper for Transfermarkt or FBref unless the user explicitly approves it after checking terms.
- Prefer published datasets, official/open APIs and downloadable CSVs.
- Preserve raw data unchanged; transformations go to `interim` and `processed`.

## 4. Data leakage rules

These are non-negotiable.

For any observation dated `t`:
- no feature may use information created after `t`;
- rolling features must be shifted before aggregation;
- season-end statistics cannot be used to predict earlier matches;
- market values after the target valuation date cannot enter features;
- preprocessing must be fitted on training data only.

Add automated leakage tests when practical.

## 5. Statistical rules

Always implement a simple baseline before a complex model.

### Player value
At minimum compare:
- naive/median baseline,
- linear regression,
- Ridge/Lasso or Elastic Net,
- tree boosting model.

Use `log1p(market_value_eur)` as a candidate target and compare against direct-value modelling.

### Match prediction
At minimum compare:
- league/base-rate baseline,
- Elo,
- independent Poisson,
- Dixon-Coles,
- multinomial logistic regression,
- gradient boosting.

Use chronological / rolling-origin evaluation.

### Uncertainty
Where feasible report:
- bootstrap confidence intervals,
- prediction intervals,
- calibration of probabilities,
- sample/minutes reliability.

## 6. Evaluation rules

Do not optimize to accuracy alone.

Regression:
- MAE,
- RMSE,
- R²,
- median AE,
- segment errors,
- time-split error.

Classification/probabilities:
- log loss,
- Brier score,
- Ranked Probability Score,
- calibration plot,
- accuracy secondarily.

Document why the final model was selected.

## 7. Code quality

- Python 3.12+
- type hints on public functions,
- docstrings for non-trivial modules/functions,
- functions should be small and testable,
- avoid duplicated transformations,
- no hard-coded local absolute paths,
- configuration through environment variables / config objects,
- use `pathlib`,
- raise meaningful errors,
- use logging instead of scattered prints in production modules.

Run before considering a task complete:

```bash
ruff check .
pytest
```

If type checking has been configured for the touched module, run it too.

## 8. SQL quality

Use PostgreSQL-compatible SQL.

Demonstrate:
- CTEs,
- window functions,
- `LAG`/`LEAD`,
- rolling calculations,
- joins,
- conditional aggregation,
- indexing decisions.

SQL transformations must be readable and commented when logic is non-obvious.

## 9. Notebook rules

Notebooks are for:
- EDA,
- statistical exploration,
- explanatory visualizations.

Production transformations/models must live in `src/pl_analytics/`.

A notebook should run top-to-bottom from a clean kernel.

## 10. Documentation rules

Every meaningful change should update the relevant docs.

When a model is finalized, create/update a model card containing:
- purpose,
- data,
- target,
- features,
- split strategy,
- metrics,
- limitations,
- leakage controls,
- interpretation,
- version/date.

When a data pipeline is added, update:
- data source,
- schema/data dictionary,
- refresh instructions.

## 11. Git rules

Use small coherent commits. Suggested prefixes:

- `chore:`
- `docs:`
- `data:`
- `feat:`
- `model:`
- `test:`
- `fix:`
- `refactor:`

Do not rewrite Git history unless asked.
Do not push forcefully.
Do not commit `.env`, API keys or downloaded bulk datasets.

At the end of a milestone:
1. run tests/lint,
2. update docs,
3. summarize files changed,
4. summarize commands run and results,
5. recommend the exact commit message,
6. stop and wait for the user before starting the next milestone unless explicitly told to continue.

## 12. UX rules

The Streamlit application must:
- load without training models at startup,
- show graceful errors when data/model artifacts are missing,
- explain metrics in plain language,
- distinguish observed values from predictions,
- surface uncertainty,
- avoid implying predictions are certainties.

## 13. Definition of done

A task is not done merely because code runs once.

It is done when:
- code is implemented,
- tests exist where appropriate,
- lint passes,
- docs are current,
- results are reproducible,
- assumptions and limitations are explicit.
