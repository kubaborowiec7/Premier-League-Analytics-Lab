# M7 calibrated match classification

Tracking issue #18. Predeclared design: `data/manifests/m7_experiment.json`, first
committed in `77f250e`. No shot, odds, player-value or future-season features are used.
Historical shots remain blocked by issue #16. M7 adds multinomial logistic regression
and XGBoost to the M6 statistical comparison without changing M6's models.

## Reproduction and source scope

```bash
python scripts/train_match_ml.py --stage select --download
python scripts/train_match_ml.py --stage final --download
python scripts/plot_match_ml.py
```

Use the same `--output` directory for all stages; a new directory is required for
reproduction after selection has been frozen. Offline runs omit `--download` and use
the existing immutable archives. Final checks both code/configuration and fitted-bundle
hashes before loading data. Never load untrusted joblib files. No training runs at
app startup. New model choices need a new future holdout.

The Football-Data 2026/27 snapshot adds **40 completed matches**, dated 2026-08-21
through 2026-09-14, with a fixed test cutoff 2026-09-15. These outcomes were not opened
for M7 development. The 2025/26 scores already evaluated in M6 are explicitly reused
as calibration data, not misrepresented as an untouched M7 test.

The provider's current-season URL changes as matches are added. The checksum pin
fails closed on changes; it does not silently train on a newer version. Preserve
the local content-addressed archive to reproduce this exact experiment. A clean
download can fail after the provider updates that URL; refreshing a manifest creates
a new experiment and must not overwrite old raw data or selection records.

## Temporal contract

| Window | Dates | Use |
|---|---|---|
| History | From 2019/20 | Warm-up for form and Elo |
| Training | 2020-08-01–2024-06-30 | Fit transforms and classifiers |
| Validation | 2024-07-01–2025-06-30 | Minimum raw H/D/A log loss selects classifier |
| Calibration | 2025-07-01–2026-06-30 | Fit only the selected model's temperature |
| Final | 2026-07-01–2026-09-15 | Frozen evaluation on 40 newly held-out matches |

Features and the M6 benchmarks share the same monthly information horizon. Before
each month, use only the preceding 1,095 days of results. Current-month outcomes are
unavailable to every fixture forecast that month. A month's completed results become
available at the following origin. Elo/form states update; classifier coefficients,
preprocessing and temperature do not refit during validation, calibration or test.

Feature generation needs at least 100 previous matches to permit the first training
season after its warm-up; M6 statistical fitting retains its 500-match requirement.
The additional feature warm-up parameter was fixed before classifier selection.
Local-calendar day handling and retrospective fixture-schedule caveats follow M6.

## Features, candidates and calibration

The eleven numeric features are pre-origin home-minus-away Elo, and each team's last
five observed matches' mean points, goals for, goals against, sample count and days
since its latest observed result at origin. The last quantity is not actual rest
time within the forecast month. Missing/unseen-team history produces missing averages
and zero sample count; no retrospective promotion labels are invented.

Elo uses K=20, home advantage 60 and .75 season retention. Only numeric football
features enter classifiers; competition/season/team/match IDs remain audit metadata.
Each fitted pipeline uses training median imputation, missingness indicators and
standard scaling. Multinomial logistic candidates use C=.1 and C=1, L-BFGS, and at
most 2,000 iterations. XGBoost depths 2 and 3 use 200 trees, learning rate .03,
minimum child weight 10, L2 regularization 1, full row/column sampling, histogram
training and a fixed seed, with one CPU thread. There is no random split or early
stopping on test data.

After choosing the classifier on validation, temperature scaling fits one T in
[.5,5] by minimizing calibration-period log loss:

`p_calibrated = softmax(log(max(p_raw, 1e-15)) / T)`

Temperature changes confidence but preserves predicted-class order. The calibration
period cannot select another classifier. Both raw and calibrated final results are
reported even when calibration makes final performance worse. It does not provide
guaranteed probability accuracy under drift.

## Evaluation and artifacts

The selected calibrated classifier is compared with the other fixed ML candidates
and all seven M6 statistical policies on **identical final match IDs and dates**.
Report natural-log loss, unscaled multiclass Brier, H/D/A-ordered RPS and accuracy,
plus fixed class-wise calibration bins with counts. Scores from different seasons
must not be directly substituted for this aligned comparison. Only two final monthly
origins and 40 matches are available: this is a small pilot, not a stable ranking of
model families. No significance claim is supported by this sample alone.

`artifacts/m7/` holds the frozen `selection.json` and `classifiers.joblib`, final
`evaluation.json`, `test_features.parquet`, combined `predictions.parquet` and a
calibration/comparison figure. Statistical benchmark subdirectory retains fitted
origin models, convergence metadata and indexed score matrices. ML models predict
H/D/A only; no score distribution or expected goals are fabricated from classifiers.

Model versions, measured results and limitations belong in the
[model card](../reports/model_cards/match_ml.md). Bulk/local artifacts remain ignored.
