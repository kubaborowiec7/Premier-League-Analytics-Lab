# Statistical and modelling methodology

This project should be statistically stronger than a typical "train XGBoost and show accuracy" portfolio project.

M4 implements the player-statistics subset below. Exact percentile tie handling,
population z-scores, exposure shrinkage, match-resampled intervals and PCA/similarity
contracts are documented in [PLAYER_ANALYTICS.md](PLAYER_ANALYTICS.md), with a measured
example and source limitations. Snapshot position context is descriptive, not a
verified historical role. Modelling sections remain specifications for later milestones.

## 1. Player metrics

### Per-90 normalization

For count statistic `x`:

`x_per90 = 90 * x / minutes`

Use a minimum-minutes threshold for rankings.

### Position-adjusted percentiles

Compare players against an appropriate peer group:
- GK,
- CB,
- FB/WB,
- DM/CM,
- AM/W,
- ST.

Do not compare raw attacking output for centre-backs against forwards.

### Standard scores

For metric `x` within a peer group:

`z = (x - mean) / sd`

Use robust variants when distributions are strongly skewed:
- median,
- MAD,
- winsorization where justified.

### Small-sample reliability / shrinkage

Per-90 rates from 150 minutes should not be treated like rates from 2,500 minutes.

Candidate approach:
- empirical-Bayes style shrinkage toward the positional mean,
- or weighted shrinkage based on minutes/exposure.

Document the formula and sensitivity.

### Bootstrap confidence intervals

For selected player metrics:
- bootstrap match/appearance observations,
- estimate 95% intervals,
- show uncertainty visually.

---

## 2. Player similarity

Pipeline:
1. choose role-specific features,
2. transform skewed metrics if needed,
3. standardize,
4. optionally reduce dimension with PCA,
5. calculate cosine similarity or k-nearest neighbours.

Validate whether nearest neighbours make football sense.

Do not use market value as an input to the similarity model if the goal is football-style similarity.

---

## 3. PCA and clustering

PCA goals:
- understand correlated dimensions,
- visualize player profiles,
- reduce noise before clustering.

Clustering candidates:
- K-means,
- Gaussian mixture,
- hierarchical clustering.

Choose number of clusters using:
- silhouette score,
- stability,
- interpretability.

Clusters should be described as **archetypes**, not as objective player quality tiers.

---

## 4. Market-value modelling

### Target

Primary candidate:

`y = log1p(market_value_eur)`

Reason:
market values are strongly right-skewed and multiplicative errors are often more meaningful.

Always back-transform predictions carefully and report error in EUR too.

### Baselines

1. global median,
2. median by position + age band,
3. linear regression.

### Statistical models

- OLS with diagnostics,
- Ridge,
- Lasso / Elastic Net,
- robust regression as an optional extension.

Diagnostics:
- residual plots,
- heteroskedasticity,
- influential points,
- multicollinearity/VIF for explanatory models.

### ML models

- Random Forest as a benchmark,
- XGBoost as primary boosted-tree candidate.

### Split design

Prefer chronological split:
- train on older valuation dates,
- validate on later dates,
- final test on the newest held-out period.

If multiple rows belong to the same player, make sure the design does not create unrealistic leakage.

### Evaluation

- MAE in EUR,
- RMSE,
- R²,
- MAE in log space,
- median AE,
- MAPE only with care,
- errors by age,
- position,
- value decile,
- league/club strength where applicable.

### Undervaluation score

Raw residual:

`residual_eur = actual_value - predicted_value`

For "model-undervalued":
`predicted_value - actual_value > 0`

Also calculate:
`relative_gap = (predicted - actual) / max(actual, floor)`

A better ranking should incorporate uncertainty:

`undervaluation_score = (predicted - actual) / prediction_uncertainty`

This prevents ranking noisy estimates too highly.

### Explainability

Use:
- global feature importance,
- permutation importance,
- SHAP.

SHAP must be presented as **model explanation**, not causal effect.

---

## 5. Elo rating

A basic implementation:

`E_home = 1 / (1 + 10^(-(R_home + H - R_away)/400))`

Update after match:

`R_new = R_old + K * (S - E)`

Tune:
- `K`,
- home advantage `H`,
- season-regression factor.

Use only pre-match Elo as a predictive feature.

---

## 6. Poisson goals model

For goals `X`:

`P(X=k) = exp(-lambda) * lambda^k / k!`

Model home and away scoring intensities using:
- attack strength,
- defence strength,
- home advantage.

Then construct a score probability matrix.

Derive:
- home-win probability,
- draw probability,
- away-win probability.

---

## 7. Dixon-Coles

Implement Dixon-Coles as an improved football-specific baseline.

Benefits:
- adjusts dependence for low-scoring outcomes,
- supports time weighting,
- widely interpretable.

Estimate:
- attack parameters,
- defence parameters,
- home advantage,
- low-score correlation parameter,
- optional time-decay parameter.

Compare against independent Poisson on out-of-sample log likelihood / scoring metrics.

---

## 8. Match ML model

Candidate target:
- H / D / A multiclass.

Candidate features:
- pre-match Elo difference,
- rolling points,
- rolling goal difference,
- rolling shots,
- rolling shots on target,
- rolling xG/xGA only when safely available,
- days rest,
- promoted indicator,
- home advantage,
- team-value summaries aligned to date,
- opponent-adjusted strength.

Every rolling feature must be shifted so the current match is excluded.

Models:
- multinomial logistic regression,
- XGBoost classifier.

---

## 9. Temporal validation

Use rolling-origin evaluation, e.g.:

- Train seasons 1–3 → validate season 4
- Train seasons 1–4 → validate season 5
- ...
- newest season/period → untouched final test

Do not randomly shuffle football matches for the primary evaluation.

---

## 10. Probability calibration

Assess:
- reliability diagram,
- Brier score,
- log loss,
- Expected Calibration Error,
- class-wise calibration.

If necessary:
- isotonic calibration,
- sigmoid/Platt calibration.

Calibration itself must be fitted without leaking test data.

---

## 11. Ranked Probability Score

For ordered outcomes such as home/draw/away, add Ranked Probability Score as a football-relevant probability metric.

Keep the implementation tested against known examples.

---

## 12. Statistical significance and uncertainty

Avoid simplistic "p < 0.05 therefore important" reporting.

Where comparisons matter:
- report effect size,
- confidence interval,
- bootstrap uncertainty,
- multiple-testing caveats if many hypotheses are tested.

---

## 13. Optional advanced extensions

High-value portfolio extensions:
- generalized additive model for nonlinear age curves,
- Bayesian hierarchical team-strength model,
- conformal prediction for market value,
- survival analysis for time-to-transfer,
- change-point analysis for form,
- event-level xT-style spatial value model on open event data.


## 14. Cross-league comparability

V1 uses Premier League players only, but feature engineering must retain enough
context to support future cross-league scouting.

Never assume that the same raw per-90 value has identical meaning in every league.

Minimum future comparison design:
- keep `competition_id` and `season` on analytical player rows;
- normalize metrics within competition-season-position peer groups;
- store both raw values and normalized values;
- expose the normalization context in model metadata.

Candidate future methods:

### League-season positional z-scores

`z = (x - mean_{competition,season,position}) / sd_{competition,season,position}`

This gives a player's standing relative to peers in the same environment.

### Percentile normalization

Useful for interpretable scouting profiles across leagues, while retaining the
original raw metric for context.

### Competition-strength adjustment

A later version may estimate league-strength multipliers from:
- UEFA/inter-league results where meaningful,
- player transfers followed by performance changes,
- hierarchical models with competition effects.

### Hierarchical modelling

For sufficiently rich data, model:

`player performance ~ player effect + competition effect + season effect + age + role`

This is preferred to manually assigning arbitrary league-strength constants.

V1 must not invent league-strength adjustments before multi-league validation data exists.

## Implemented M5 experiment

[VALUE_MODELS.md](VALUE_MODELS.md) fixes the train/validation/calibration/test design,
direct-EUR and log1p candidate comparison, native TreeSHAP and linear SHAP semantics,
and scaled residual intervals. The selected model is frozen before final testing.
Paired MAE comparisons additionally resample complete player histories (2,000 draws,
seed 42), preserving within-player dependence but not common valuation-date shocks.
This diagnostic was added after the first final evaluation; it does not change model
selection or predictions. Model-relative review lists retain exposure and missing-lag
flags. Low exposure is flagged below 90 minutes; no test rows are removed or metrics
improved by applying this post-evaluation diagnostic. See the
[model card](../reports/model_cards/player_value.md) for measured failures and uncertainty.
