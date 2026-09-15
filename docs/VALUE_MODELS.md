# M5 market value experiment

M5 estimates the next published EUR valuation of a recent competition participant.
The target is the provider's editorial estimate, not an observed transfer fee or
intrinsic player value. Tracking issue: #12. See the final
[model card](../reports/model_cards/player_value.md) for measured results.

## Reproduction

From the repository root after installing `.[dev]`:

```bash
python scripts/train_value_models.py --stage select --download
python scripts/train_value_models.py --stage final
python scripts/plot_value_report.py
```

The first command acquires only pinned published archives if absent. Omit `--download`
for fully offline runs. No scraper, database or notebook is needed. Both commands
default to `artifacts/m5/`. Use `--output artifacts/m5-reproduction` for a fresh
reproduction; selection refuses to overwrite an existing frozen record. This is not
permission to tune against the now-observed final test: methodological changes need
a new future holdout. Never load joblib bundles from untrusted sources.

`data/manifests/m5_experiment.json` fixes source hashes, competition mapping, dates,
cohort rules, primary EUR MAE, seed and interval settings. It was committed before
model selection (a profile-hash transcription error was corrected before loading).
Selection returns no test outcomes, fits all preprocessing on training alone, and
saves all 12 fitted candidates. Final evaluation checks configuration and bundle
hashes, performs no fitting or reselection, and evaluates the frozen candidates.
`reports/model_cards/value_selection.json` records the original selection before
the final test was opened. Small floating-point differences across libraries/platforms
are possible; the original runtime is recorded alongside that selection.

## Temporal design and cohort

| Window | Dates, inclusive | Purpose |
|---|---|---|
| History | From 2018-07-01 | Prior competition appearances |
| Training | 2019-07-01–2024-06-30 | Fit models, imputation and scaling |
| Validation | 2024-07-01–2024-12-31 | Choose minimum EUR MAE |
| Calibration | 2025-01-01–2025-06-30 | Residual interval quantile only |
| Final test | 2025-07-01–2026-06-30 | Frozen final evaluation |

A target requires a competition appearance strictly before its date and within
90 days. Performance aggregates use only appearances in that configured competition
in the preceding 365 days. Same-day appearances and valuations are excluded. Earlier
valuations may come from any competition. Earlier observed test valuations can inform
later test predictions: this is sequential updating, not a forecast made once in July.

Current club, snapshot position, current market value and source-reported valuation
league are excluded. This avoids the known current-club fallback in the published
valuation source. Participation does not establish membership on the valuation date:
recent transfers can remain in this cohort. `competition_id`, `season` and this cohort
label accompany analytical rows. The adapter maps source competition code to the
canonical code through the manifest. Source IDs stay separate from Football-Data IDs.

Only date of birth and a display name come from player profiles. Dates of observations
are available, but original publication timestamps and revision vintages are not;
historical as-published availability cannot be proven from a present-day archive.

## Models and uncertainty

The fixed candidate set is median, previous-value persistence, OLS, Ridge (alpha 10),
Lasso (alpha 10000 EUR / .001 log units), and XGBoost depths 3 and 5. All five learned
specifications run with EUR and log1p EUR targets. XGBoost uses 250 trees, learning
rate .05, minimum child weight 10, absolute-error objective, full row/column sampling
and one CPU thread. Learned candidates use training median imputation and standard
scaling; no player ID or competition/season identity enters the estimator. Predictions
are clipped nonnegative. Log predictions are clipped to [-50, 30] before expm1 for
numerical safety; no retransformation bias correction is fitted.

The eight features are age, previous EUR value, days since that valuation, days since
last appearance, trailing appearance count, minutes, goals/90 and assists/90. Missing
exposure/counts produce missing rates; absence of earlier valuation stays missing
until training-fitted imputation. No historically unverified positional normalization
or invented cross-league strength adjustment is applied.

Intervals use absolute calibration residuals divided by max(previous value, EUR 1m),
with the training median replacing missing prior value. The finite-sample order
statistic at ceil((n+1)*.9) gives a nominal 90% interval, clipped at zero. Repeated
players, time drift and unequal valuations violate exchangeability, so nominal
coverage is not a guarantee; final empirical coverage and widths are reported.

SHAP uses native XGBoost TreeSHAP contributions and independent-feature linear SHAP
relative to the transformed training mean. Local contributions sum to the raw fitted
output, in EUR or log1p EUR units before clipping/inversion. These are model
explanations, not causal effects; correlated performance features complicate attribution.
The first 200 test rows in chronological/player order form a declared explanatory
sample, not a representative full-test importance estimate.

## Outputs and interpretation

All bulky outputs remain ignored under the selected artifact directory:

- `selection.json`, `models.joblib`: frozen development decision and fitted transforms.
- `evaluation.json`: all-candidate test metrics, age/value/season/new-player segment
  errors, residual summaries, empirical coverage and SHAP summary.
- `test_features.parquet`: auditable prior dates and features.
- `predictions.parquet`: dated observations, predictions, intervals and residuals.
- `value_ranking.parquet`: latest test observation per player/competition, sorted by
  `(prediction - observed value) / interval half-width`. Rows have different dates;
  this is a historical review list, not a current simultaneous scouting shortlist.
- `shap_<model>.parquet`: local feature contributions for the selected model and the
  best validation XGBoost benchmark.
- `model_diagnostics.png`: static comparison, full residual scatter and SHAP summary.

Residual = observed minus predicted. A positive `model_undervaluation_eur` means the
model predicts above the published value; `below_model_interval` requires the observed
value below the lower bound. These flags do not demonstrate a profitable transaction.
Review rows retain trailing exposure and missing-prior-value flags. `low_exposure`
means fewer than 90 known minutes; it is a post-evaluation warning, not a fitted
threshold, row exclusion or alteration to the frozen predictions. The final report
also includes a paired player-cluster bootstrap of MAE differences as a
post-evaluation diagnostic, without affecting model selection.
Segment value-band thresholds use training target terciles only. Position segments
are deferred because historical roles are not verified. No models train during app
startup; dashboard integration belongs to M8.
