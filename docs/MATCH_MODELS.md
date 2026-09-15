# M6 statistical match models

Tracking issue: #15. The predeclared manifest is `data/manifests/m6_experiment.json`
(commit `25b9057`). Sources are seven immutable Football-Data CSVs, 2019/20–2025/26,
mapped from source code to canonical competition through configuration. Original
bytes and retrieval/terms metadata remain under ignored `data/raw/`. The existing
2023/24 archive is reused. No odds, shots or player valuation information enters M6.

## Reproduction

```bash
python scripts/train_match_models.py --stage select --download
python scripts/train_match_models.py --stage final --download
python scripts/plot_match_report.py
```

Omit `--download` to work offline. A fresh reproduction uses a new `--output` directory
for all commands. Selection refuses to overwrite an existing decision. Final evaluation
checks configuration and modelling-source checksums; it cannot silently run a changed
policy. The selection does not open the final-season archive. Future changes require
a new untouched evaluation period, not reuse of 2025/26 for tuning.

## Forecast contract

Each nonempty calendar month is a rolling origin. All models fit once to the preceding
1,095 days, requiring at least 500 matches, and forecast all that month's fixtures
without consuming their outcomes. Historical local match dates become UTC-labelled
calendar days, avoiding uncertain intra-day result availability. Training end day is
strictly before origin, which is at or before fixture day. The forecasting function
does not access fixture scores; observed results join only after probabilities exist.
Actual fixture dates are retrospective: this evaluation assumes the fixture is known,
and does not reconstruct historical scheduling/postponement announcements.

Validation covers 2023/24 and 2024/25 (760 matches, 20 nonempty monthly origins).
The final 2025/26 period is reserved until selection is frozen. It evaluates a fixed
rolling policy, not a single season-ahead model: results from earlier test months
can legitimately enter subsequent training windows. All candidates share identical
origins and histories. Archived 2019/20 results fall outside the active validation
windows but remain available for future historical experiments.

Forecasts retain `competition_id`, `season`, canonical match/club IDs, origin, training
end day and unseen-team flags. Training rejects mixed competitions. Unseen clubs
receive neutral goal-model effects or Elo 1500; this is a fallback, not a measured
promotion adjustment. Source club names determine stable canonical IDs, so renamed
clubs would need an explicit alias migration. No cross-source club matching is inferred.

## Seven fixed benchmarks

- Base rate: training H/D/A counts with one pseudocount per class.
- Elo K=20 and K=40: initial 1500, home advantage 60, logistic expected-score updates,
  and 75% retention of deviation from 1500 at season boundaries. Same-day games update
  in a batch. H/D/A forecasts use a Davidson draw extension, with draw parameter
  `nu=2*d/(1-d)` from smoothed training draw frequency; for strength ratio `q`, weights
  are `sqrt(q), nu, 1/sqrt(q)`. This draw extension differs from the binary expectation
  used in Elo updates. All ratings are frozen within each forecast month.
- Independent Poisson and Dixon–Coles, each with flat weights or exponential decay
  with half-life 365 days. These estimate home advantage plus centered team attack
  and defensive-weakness effects by joint penalized likelihood.

For the goal models:

```text
log(lambda_home) = intercept + home_advantage + attack_home + defense_away
log(lambda_away) = intercept + attack_away + defense_home
weight(age_days) ∝ exp(-log(2) * age_days / half_life)
```

Attack and defense effects each sum to zero. The objective is weighted mean negative
log likelihood plus `0.5 * 0.01 * sum(raw_team_effects²)`. Intercept and home advantage
are unpenalized. L-BFGS-B uses an analytic gradient checked numerically in tests;
failed convergence raises an error. Raw team effects/intercept are bounded [-2,2],
home advantage [-1,1], and rate log values clipped to [-6,log(8)]. These are explicit
regularization/stability choices, not an unconstrained replication of the paper.

The Dixon–Coles multiplier is `1-lambda*mu*rho` for 0–0, `1+lambda*rho` for 0–1,
`1+mu*rho` for 1–0, `1-rho` for 1–1, and one elsewhere. This adjusts low-score dependence
while preserving Poisson marginals. Rho is restricted to [-0.12,0.015]; combined with
the rate cap this ensures positive correction for every possible fixture. The
restriction particularly limits positive rho; fits reaching a boundary are reported.
Independent Poisson fixes rho at zero. Method reference:
[Dixon and Coles (1997)](https://doi.org/10.1111/1467-9876.00065).

## Probability and evaluation conventions

Score matrices use home goals as rows and away goals as columns. The default grid is
0–30 for both teams. The omitted probability mass is recorded before normalization;
grids losing more than 1e-6 are rejected. H/D/A is lower triangle / diagonal / upper
triangle. Negative or nonfinite probabilities are rejected; no arbitrary repair is
applied to invalid rho. Output expected goals are Poisson rate parameters, not xG
event-data measurements. They omit parameter-estimation uncertainty.

Selection minimizes pooled validation natural-log loss; ties are not tuned against
the final season. Additional metrics are unscaled multiclass Brier (range 0–2),
RPS (mean squared error of the first two cumulative H/D/A probabilities), and accuracy.
Goal models additionally report joint observed-score negative log likelihood and
mean absolute goal error averaged over home/away. Base rate and Elo do not invent
score distributions. Fixed-width, class-wise reliability bins show empirical
calibration; no probability recalibration is fitted in M6.

The final selected-policy minus base-rate log-loss difference receives a paired
bootstrap interval over complete forecast months (2,000 draws, seed 42). With only
one final season and few monthly clusters it is descriptive; inter-month/team
dependence and drift are not fully captured. A lower score does not establish
profitability or certainty about an individual match.

## Data quality and artifacts

The 2021/22 archive reports Newcastle–West Ham on 2021-08-15 with eight away shots
but nine on target. M6 explicitly requests result-only parsing, excluding **all** shot
fields, while preserving date/team/score/result validation. The default M1 importer
still rejects inconsistent shots. No raw value is corrected and no match is dropped.
M7 must not silently trust these shot statistics.

`artifacts/m6/selection.json` freezes the choice/configuration/code hashes. Each stage
directory contains `predictions.parquet`, `fits.json`, `latest_origin_models.joblib`,
compressed `score_matrices.npz` and matching row-order `score_matrix_index.parquet`.
These binaries are trusted local outputs, never imported from untrusted sources.
`evaluation.json` contains final metrics, seasonal results and calibration summaries.
The latest bundle reflects the last evaluated origin, not a model refreshed to today.
No data is loaded into PostgreSQL and no training occurs during app startup.
