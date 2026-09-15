# Statistical match forecasts — M6 v1

Date: 2026-09-16. Research benchmark; no betting profitability claim. Selected policy:
**Dixon–Coles with 365-day half-life**, fixed before final evaluation in `ab4f963`.
The design was predeclared in `25b9057`; tracking issue #15. See
[methodology](../../docs/MATCH_MODELS.md), [selection record](match_selection.json)
and [final aggregate report](match_evaluation.json).

## Purpose and data

Estimate conditional H/D/A and exact-score probabilities for configured competition
fixtures, using only strictly prior full-time results. V1 maps Football-Data `E0` to
`EPL`; every prediction preserves competition, season, origin and training cutoff.
Seven published season CSVs are checksum-pinned. Raw data and terms/retrieval metadata
remain local. Source: [Football-Data historical CSVs](https://www.football-data.co.uk/englandm.php).
The statistical framework follows [Dixon and Coles](https://doi.org/10.1111/1467-9876.00065),
with the explicit bounded/penalized implementation documented in the methodology.

No odds, shots, xG, player values or future season summaries are features. Shot fields
are excluded because a development-season row has more shots on target than total
shots. The default ingestion validator remains strict. Historical names define club
IDs; renamed clubs require alias maintenance. Unknown clubs receive neutral effects.

## Fit, split and selection

Each month fits all candidates on the previous 1,095 calendar days. The minimum is
500 matches; all models use the same pre-month history. Outcomes join forecasts only
after prediction. Elo replays days in batches and applies season regression; forecast
ratings remain frozen during a month. Poisson/Dixon–Coles jointly estimate centered
attack/defensive-weakness effects, intercept and home advantage, optionally with decay.
Unknown kickoff times cause no intra-day leakage because calendar days are atomic.

The candidate set is smoothed base rates, Elo K=20/40, and Poisson/Dixon–Coles each
with flat/365-day decay weights. Fixed settings include home Elo advantage 60,
season retention .75, goal penalty .01, maximum rate 8, rho [-.12,.015] and a 0–30
score grid. Rho limits ensure all possible low-score correction factors are positive.
They constrain the model, particularly positive dependence parameters; this is not
an unconstrained reproduction of the original paper.

Selection minimizes validation log loss over 760 matches and 20 monthly origins in
2023/24–2024/25. The selected score was 0.959667; Poisson with the same decay scored
0.959839, a very small difference. No additional settings were tuned on final data.
The newest season was not opened by the M6 pipeline until selection was committed.
M5 previously used appearances from this period for a separate player task; thus
project-wide exposure to related data existed, although M6 match outcomes were not
used for selection. This is not a prospectively preregistered external trial.

## Final results: 2025/26

380 matches across ten forecast months. Earlier completed test months may enter later
training windows under the frozen policy. This is monthly updating, not a season-ahead
forecast. Lower probability/score errors are better; accuracy is secondary.

| Candidate | Log loss | Brier | RPS | Accuracy | Score NLL | Goal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Base rate | 1.084438 | .656310 | .227935 | 42.63% | — | — |
| Elo K20 | 1.036118 | .621225 | .210053 | 49.47% | — | — |
| Elo K40 | 1.036839 | .623140 | .210514 | 48.68% | — | — |
| Poisson flat | 1.027979 | .617654 | .208965 | 46.84% | 2.890234 | .897752 |
| Poisson half-life 365 | 1.027401 | .616763 | .208498 | 46.32% | 2.889916 | .895761 |
| Dixon–Coles flat | 1.027730 | .617478 | .208941 | 46.84% | 2.890557 | .897665 |
| **Dixon–Coles half-life 365** | **1.026291** | **.616227** | **.208428** | **46.32%** | **2.888891** | **.895712** |

The selected model reduces base-rate log loss by 5.36%. Paired resampling of complete
forecast months (2,000 draws) gives selected-minus-baseline difference -0.05815 with
95% interval [-0.08801,-0.02540]. Only ten monthly clusters are available, and common
team/time dependence persists; this is descriptive uncertainty, not a guarantee.
No convincing claim of superiority over the very similar Poisson model is made.

All 40 final goal-model fits converged; none reached a rho boundary. In development,
14 of 40 Dixon–Coles fits reached a rho bound, emphasizing sensitivity to constraints.
Maximum final omitted grid probability was below 5e-16; saved grids normalize to one.
Outcome uncertainty is expressed by distributions, not confidence in a point score.
Rates are expected goal counts, not measured event-data xG.

## Calibration and limitations

Full class-wise fixed-bin reliability counts are saved in `evaluation.json`; the plot
shows bins containing at least five games. Empty/sparse bins are retained in the report.
No calibrator was fitted in M6; calibration improvements belong to M7. Parameter
uncertainty is not propagated into score distributions.

The method ignores lineups, injuries, shots and current tactical changes. Monthly
updates are stale within a month. A neutral unseen-club fallback is weak for promoted
teams; fixed season regression is not a measured promotion model. Archived results
can be retrospectively revised, and original publication timestamps are unavailable.
Evaluation assumes actual fixture dates/participants were known at forecast time;
postponement history is not reconstructed. A single test season limits generalization.

## Artifacts and verification

`predictions.parquet` contains dated outcome probabilities, goal-model rates, Elo
ratings, training cutoffs and observed scores. `score_matrices.npz` and its index hold
the full score distributions; `fits.json` records convergence. The latest saved model
bundle predates the last evaluated month, not today's date. No app startup trains models.

Local `compileall` and Ruff passed; 125 tests passed, 13 opt-in PostgreSQL tests skipped
without a local server. Tests cover numerical gradients, probability conservation,
score orientation, finite tails, known metrics, same-day Elo batching, future-result
invariance, scope rejection and frozen-stage guards. Both pinned stages and plotting
ran successfully; CI repeats the full pipeline and real PostgreSQL checks.
