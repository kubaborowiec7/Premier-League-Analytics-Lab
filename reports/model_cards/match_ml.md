# Calibrated match classifier — M7 v1

Date: 2026-09-16. Status: **small-sample research pilot**. Selected classifier:
multinomial logistic regression C=.1, with temperature **1.257243**. Configuration
was predeclared in `77f250e`; selection, calibration and code were frozen in `d6fe10f`
before opening the 2026/27 final outcomes. Tracking issue #18.

See [methodology](../../docs/MATCH_ML.md), [frozen selection](match_ml_selection.json)
and [final aggregate metrics](match_ml_evaluation.json).

## Data, features and split

Purpose: estimate H/D/A probabilities conditional on a known fixture and results
available before its forecast month. Data comes from pinned published Football-Data
CSV archives; no scraped sources, odds, shots or player values enter the model.
Canonical competition and season are retained, and IDs/names are never predictors.

Eleven features contain pre-origin Elo difference and each team's last-five-match
mean points/goals for/goals against, sample count and days since last observed result.
States update only between months. All current-month fixtures share the pre-month
information cutoff, including the M6 comparison models. Result dates have calendar-day
granularity; same-day or later results cannot enter features. Unseen teams retain
missing averages and zero exposure. Exact formulas and fixed settings are in the
methodology; tests mutate future outcomes to verify invariance.

1,520 training matches span 2020/21–2023/24 after a historical warm-up. The 380 matches
in 2024/25 select the raw classifier; a separate 380 in 2025/26 fit only temperature.
Imputation, indicators and scaling fit on training only. Classifier parameters never
refit during calibration or test. The final sample is **40 matches**, 2026-08-21 through
2026-09-14, across two monthly origins. M6's previously examined 2025/26 period is
calibration data here, not called untouched test data.

## Model selection and calibration

The fixed candidate grid contains logistic C=.1/1 and XGBoost depth 2/3. Validation
log losses are respectively .995869, .996437, .998459 and 1.007120. The differences
between the first three are small. The selected classifier's calibration log loss
improves from 1.029827 to 1.025391 after fitting T in [.5,5]. This transformation
reduces confidence while preserving predicted-class order. It was not chosen using
final results. Runtime versions and model/checksum metadata are saved in the selection.

## Aligned final comparison

All rows below use the **same 40 final fixtures**. These are not M6's earlier-season
metrics. Lower log loss/Brier/RPS are better; accuracy is secondary.

| Model | Log loss | Brier | RPS | Accuracy |
|---|---:|---:|---:|---:|
| Base rate | 1.132330 | .688931 | .225363 | 32.5% |
| Elo K20 | 1.077007 | .648430 | .205036 | 40.0% |
| Elo K40 | 1.070423 | .644383 | .203198 | 40.0% |
| Poisson flat | 1.034960 | .618495 | .191428 | 47.5% |
| Poisson half-life 365 | 1.023321 | .610828 | .188434 | 45.0% |
| Dixon–Coles flat | 1.025102 | .612978 | .190495 | 47.5% |
| Dixon–Coles half-life 365 | 1.011529 | .604263 | .187266 | 45.0% |
| Logistic C=.1 raw | 1.059624 | .641145 | .202764 | 45.0% |
| **Logistic C=.1 calibrated — selected ML** | **1.049079** | **.633589** | **.200692** | **45.0%** |
| Logistic C=1 | 1.059632 | .641755 | .202687 | 45.0% |
| XGBoost depth 2 | 1.075883 | .646537 | .203914 | 40.0% |
| XGBoost depth 3 | 1.076261 | .648491 | .204716 | 40.0% |

Calibration improves this pilot's ML log loss by .01055, without changing accuracy.
The selected classifier beats base rates but **does not beat the statistical goal
models**. XGBoost's complexity does not improve the measured result. M6's selected
Dixon–Coles policy remains the stronger measured benchmark; M7 does not replace it
on the strength of model complexity. Forty matches and two origins cannot establish
a stable ranking or statistically reliable calibration across probability ranges.

## Reliability, uncertainty and limitations

Class-wise fixed-bin calibration counts are saved for every candidate. The figure
uses marker sizes to expose small bins, including single observations. Probability
forecasts express outcome uncertainty, but do not include parameter uncertainty or
guaranteed calibration. Confidence intervals over only two monthly clusters would
be unreliable; no significance or profitability claim is made.

The model omits injuries, lineups, shots and tactical information; coefficients were
last fitted through June 2024 and may age. Monthly features are stale within a month;
days since an observed result is not actual rest. Neutral/new-team handling, source
revisions and unavailable historical fixture announcements limit realism. The raw
current-season URL changes with new matches; reproducing this exact experiment may
require its archived local bytes after the provider changes the file.

## Outputs and validation

`artifacts/m7/` contains the frozen classifier bundle, selection/evaluation reports,
auditable test features, all candidate probabilities and statistical score grids.
ML output is H/D/A only; no score matrix is inferred from a three-class classifier.
The chart is `ml_diagnostics.png`. Dashboard inference must load these artifacts
without fitting and show the evaluation period and limited sample size.

Tests cover future/current-month invariance, unknown-team missingness, training-only
preprocessing, temperature normalization, calibration improvement on controlled data,
equal benchmark fixture coverage and unchanged bundles across final evaluation.
Both real pinned stages and plot generation passed locally; CI reproduces them.
Full local verification: compileall and Ruff passed; 130 tests passed, 13 opt-in
PostgreSQL tests skipped without a server.
