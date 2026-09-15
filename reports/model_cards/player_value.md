# Player value model — M5 v1

Date: 2026-09-15. Status: evaluated research benchmark; unsuitable for unattended
transfer/scouting decisions. Selected model: **Ridge, direct EUR target, alpha 10**.
The split was predeclared in `38a9dad`; model selection and implementation were frozen
in `d8c12c8` before reading final outcomes. Subsequent changes add reporting, bootstrap
diagnostics and exposure flags without refitting, reselection or deleting test rows.

## Purpose, data and target

Estimate the next published editorial market value for players with a strictly prior
appearance in the configured competition within 90 days. V1 maps `GB1` to `EPL`.
This is sequential valuation updating, not a fee forecast or an independent assessment
of intrinsic player ability. Recent participants can include transferred players.

The four published archives and their exact hashes are in
[the experiment manifest](../../data/manifests/m5_experiment.json). Original URLs,
retrieval times, runtime versions, model checksum and validation metrics are recorded
in [the frozen selection record](value_selection.json). No website scraping is used.

Training contains 5,998 dated observations (2019-07-01–2024-06-30); validation 552
(2024-07-01–2024-12-31); calibration 649 (2025-01-01–2025-06-30). Final test contains
1,301 observations for 502 players in the reserved 2025/26 window, with actual target
dates 2025-08-11–2026-06-03. Training appearances begin 2018-07-01. No random split.

## Features and leakage controls

Eight allowlisted features: age, previous EUR valuation, days since previous valuation,
days since last appearance, previous 365-day appearances and minutes, goals/90 and
assists/90. The previous value is usually the dominant signal; high R² is not evidence
that performance alone explains price. Player identity and names never enter fitting.
Missing inputs use training medians; scaling is also fitted on training only.

All appearances and lagged valuations must precede the target date, excluding same-day
events. Earlier observed test valuations can inform later targets without refitting.
Current club, current value, source valuation league and snapshot position are excluded.
Date of birth is treated as a fixed fact. Present-day archives do not expose original
publication or revision timestamps, so strict historical as-published availability is
unverified. Valuation-date club membership and historical positions remain unverified.

## Selection and measured results

The minimum validation EUR MAE selects Ridge (2.696m), ahead of persistence (2.896m)
and median (16.500m). The 336 EUR validation advantage over OLS is practically tiny.
The fixed grid contains median/persistence and five learned specifications each using
direct EUR and log1p targets. Details and all hyperparameters are in
[the methodology](../../docs/VALUE_MODELS.md).

All figures below use final-test observed EUR; amounts are **million EUR**.

| Frozen candidate | MAE | RMSE | Median AE | R² |
|---|---:|---:|---:|---:|
| Median | 17.390 | 26.614 | 10.000 | -0.316 |
| Last-value persistence | 3.687 | 5.452 | 3.000 | 0.945 |
| OLS EUR | 3.609 | 11.114 | 2.090 | 0.770 |
| **Ridge EUR — selected** | **3.612** | **11.193** | **2.103** | **0.767** |
| Lasso EUR | 3.610 | 10.849 | 2.110 | 0.781 |
| XGBoost depth 3 EUR | 3.540 | 6.262 | 2.077 | 0.927 |
| XGBoost depth 5 EUR | 3.438 | 6.116 | 1.957 | 0.930 |
| OLS log1p | 8,241.640 | 296,275.422 | 5.716 | -163,120,453.526 |
| Ridge log1p | 8,241.450 | 296,275.419 | 5.741 | -163,120,450.300 |
| Lasso log1p | 8,241.555 | 296,275.421 | 5.700 | -163,120,452.147 |
| XGBoost depth 3 log1p | 3.494 | 5.898 | 2.123 | 0.935 |
| XGBoost depth 5 log1p | 3.288 | 5.479 | 1.943 | 0.944 |

Ridge beats the median baseline by 79.23% MAE and persistence by 2.03%. Its much
worse RMSE than persistence exposes unacceptable tail behavior. XGBoost depth 5 log1p
performs better on this final test but is not substituted after seeing test results.
The log-linear models are severe failures: the unchanged linear EUR predictors and
exponential retransformation extrapolate explosively. Their results are retained.

The paired player-cluster bootstrap (2,000 draws, seed 42), added as a post-evaluation
diagnostic, estimates Ridge-minus-persistence MAE difference at **-0.075m EUR**, with
95% interval **[-0.452m, +0.638m]**. This does not establish reliable superiority over
persistence. Against median the interval is [-15.653m, -12.032m]. These intervals
preserve within-player dependence, but not shared date shocks or future drift.

## Uncertainty, residuals and segments

The nominal 90% calibrated intervals achieve **92.39%** empirical final-test coverage;
mean total width is **26.16m EUR**, often too wide for a useful scouting decision.
Calibration residuals are scaled by prior value with a EUR 1m floor; temporal data
are not exchangeable, so coverage is descriptive rather than guaranteed. There are
78 observations above the interval and 21 below it; 1,202 lie within it.

Residual means observed minus predicted. Mean residual is +0.377m EUR; the 5th, 50th
and 95th percentiles are -5.765m, +0.066m and +10.235m EUR.

| Selected Ridge segment | n | MAE, million EUR |
|---|---:|---:|
| Under 23 | 296 | 5.790 |
| Age 23–29 | 826 | 3.266 |
| Age 30+ | 179 | 1.609 |
| Low training-tercile value band | 221 | 1.572 |
| Middle band | 446 | 3.038 |
| High band | 634 | 4.727 |
| Player observed in training | 801 | 3.114 |
| Player new since training | 500 | 4.411 |

Position segments are unavailable because historic roles are unverified. There are
73 test observations with fewer than 90 known trailing minutes and seven without a
prior value. The largest error is Rio Ngumoha on 2025-08-26: the source records one
goal in one minute of prior competition exposure, yielding 90 goals/90. Ridge predicts
370.61m EUR against a 10m EUR observation. This is an extrapolation/reliability failure,
not evidence of a bargain. The record remains in all reported metrics and rankings.
Exposure and missing-lag flags were added for review after evaluation, without changing
predictions. Performance-rate shrinkage or a fallback needs a separately evaluated
successor with a new untouched holdout; this test cannot be reused for selection.
The follow-up is tracked in issue #13.

## Explanations and review ranking

For the first 200 chronological test rows, Ridge's mean absolute SHAP contribution
is largest for previous value (14.68m EUR), followed by goals/90 (2.51m), minutes
(2.03m), appearance count (1.16m) and age (0.89m). Linear contributions use the training
mean background with independent-feature semantics. Native XGBoost TreeSHAP is also
saved for the best validation tree model, in log1p target units. Additivity applies
before inversion/clipping, and neither explanation is causal.

The historical review ranking covers 502 players' latest available test observations,
retaining competition, season, date, observed and predicted values, intervals, exposure
flags and model-relative gaps. Different dates and possible transfers mean it is not
a current squad list. Low-exposure extrapolation can lead the ranking; inspect flags
and intervals before interpretation. Labels describe differences from the model only.

## Reproduction and release limitations

See [VALUE_MODELS.md](../../docs/VALUE_MODELS.md) for commands, output schemas and
selection/final guards. The complete small aggregate report is
[value_evaluation.json](value_evaluation.json). Raw archives, player-level outputs and
joblib models are excluded from Git; CI uploads artifacts. No model trains on app
startup, and M5 does not add the Market Value dashboard page (M8).

M5 meets the research milestone's median-baseline MAE comparison, but no production
readiness or dependable improvement over persistence is claimed. Membership, source
vintage, limited football metrics, unshrunk low-minute rates, missing lag history,
repeated-player dependence and model drift constrain interpretation.
