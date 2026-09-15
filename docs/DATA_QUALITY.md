# M3 — Exploratory data-quality report

Measured on 2026-09-15 using the immutable M1 manifest
`data/manifests/m1_sources.json`. This is a **development sample**, already explored;
it cannot subsequently be described as an untouched final test set. No model was
trained and no data was imputed, winsorized or removed as an outlier.

## Reproduce

From the repository root, using the installed project and dev dependencies:

```bash
# Explicit acquisition only if the pinned archives are not already available:
python scripts/verify_ingestion.py
# Both following commands are offline and require no PostgreSQL:
python scripts/run_quality.py
python scripts/execute_quality_notebook.py
```

The report is `artifacts/m3-quality.json`; the executed notebook is
`artifacts/m3-executed.ipynb`; three figures are under `artifacts/m3/`. These outputs
are ignored by Git and uploaded by CI. The committed notebook has cleared outputs
and executes top-to-bottom from a new kernel. You can also open
`notebooks/01_data_quality.ipynb` in Jupyter. The reusable analysis is in
`src/pl_analytics/data/quality.py`.

`--manifest` on the report script selects another M1-compatible pinned manifest.
For the notebook, set `PL_ANALYTICS_QUALITY_MANIFEST`; it defaults to the M1 manifest.
Both respect configured `DATA_DIR`, `ARTIFACT_DIR` and `ACTIVE_COMPETITIONS`.
Missing or changed archives fail explicitly; M3 never silently downloads replacements.
Report contents are deterministic for the same records/provenance and dependency
versions; every source hash, acquisition timestamp, URL and terms note is recorded.

## Coverage and missingness

Scope: canonical `EPL`, season `2023/24`, requested 2023-07-01 through 2024-06-30.
Diagnostics describe the canonical **post-ingestion** subset, not every raw row.
M1 validation rejects malformed inputs and filters records outside the requested scope.
Zero missingness here does not establish completeness of the underlying sources.

| Measure | Observed result | Interpretation |
|---|---|---|
| Finished matches | 380, 20 clubs | Match sample covers the selected season |
| Match dates (UTC) | 2023-08-11 to 2024-05-19 | July/June have zero matches within the requested window |
| Player profiles | 948 | Snapshot attributes, not proven historical profiles |
| Valuations | 2,093, for 948 player/provider pairs | Repeated observations are correlated |
| Valuation dates | 2023-07-04 to 2024-06-28 | Irregular refresh cadence |
| Canonical duplicate keys | 0 in matches, players and valuations | Not evidence against duplicate real-world identities across sources |
| Missing dates / out-of-window dates | 0 / 0 | Within this validated selected subset |
| Unknown match kickoffs | 0 | M2 still tests conservative behavior for other samples |
| Missing match/profile fields | 0 in the selected canonical columns | Upstream excluded fields are not audited here |
| Missing canonical valuation club | 2,093 / 2,093 (100%) | Intentionally unknown historical membership |
| Unverified valuation competition context | 2,093 / 2,093 (100%) | Cannot call these verified historical EPL player observations |
| Source-reported valuation clubs | 37 | Separate identity namespace; not a join to the 20 match clubs |
| Valuations without selected player profile | 0 | Profile linkage succeeds; historical feature availability remains unproven |
| Appearance records loaded by M1 | 0 | Player performance/minutes are unavailable, not zero |

Valuation observations cluster in October (309), December (780), March (199) and
May (745): approximately 97.1% of records fall in those four months. January,
February and April have no observations. This does not establish missing files:
valuation publications are not a daily measurement process. Do not manufacture
independent daily observations by forward-filling editorial estimates.

## Observed match targets

| Outcome | Matches | Share |
|---|---:|---:|
| Home win | 175 | 46.05% |
| Draw | 82 | 21.58% |
| Away win | 123 | 32.37% |

Mean home goals: **1.800**; mean away goals: **1.479**. Maximum observed scores for
one side: six home goals and eight away goals. All 380 matches have eligible outcomes.
These are descriptive full-sample summaries, not fitted baseline probabilities or
out-of-sample metrics. Future baselines must estimate frequencies only from training data.

## Market-value target skew

Values are editorial estimates, not transfer fees. No causal claims follow from
their distribution. The cohort is source-reported and historically unverified.

| Statistic | All 2,093 valuation records | Latest per player/provider (948) |
|---|---:|---:|
| Median | €9.00m | €5.00m |
| Mean | €16.006m | €13.428m |
| 95th percentile | €60.00m | €50.00m |
| Minimum / maximum | €10,000 / €180m | €10,000 / €180m |
| Sample skew in EUR | 2.445 | 3.007 |
| Sample skew after `log1p(EUR)` | −0.538 | −0.407 |

There are no zero values. Players have one to five observations (median two, mean
2.208). Keeping each player's latest record changes both observation weighting and
the dates represented; it is a sensitivity comparison, not an estimate of a causal
sampling effect. Provider identities remain separate if more sources are introduced.

The raw target is right-skewed. `log1p` reduces that skew in this sample but leaves
negative skew; this is not evidence of normality and does not select the model target.
M5 must compare direct-EUR and transformed targets using chronological validation,
then evaluate in EUR and inspect segment errors. No distribution was trimmed to
make a transformation look better.

## Leakage audit

| Risk | Current control/evidence | Remaining work |
|---|---|---|
| Current result used as a feature | M2 separates outcome and pre-match views; PostgreSQL tests alter current/future results without changing earlier features | Maintain explicit model feature projections |
| Same-day / unknown kickoff ordering | M2 uses a strict UTC cutoff with a conservative 24–48h embargo and tests timezone independence | Revisit policy for new sources without weakening temporal tests |
| Season totals / latest values used in earlier predictions | Notebook labels these retrospective and trains nothing | Build as-of player features before M5 |
| Current club / highest value leaked from snapshot | M1 excludes these fields from canonical player profiles; M3 confirms forbidden profile columns absent | Snapshot position/nationality still need historical availability review |
| Historical league membership inferred from current club | M1 preserves source-reported context separately and keeps historical club NULL | Resolve dated membership; issue #7 |
| Final snapshot corrections known too early | Hashes establish exactly which file was used | Publication-time availability is not reconstructable; disclose limitation in model cards |
| Repeated valuations randomly split | M3 reports one to five records per player/provider; no splits or models are fitted | Specify known-player vs unseen-player task; chronological evaluation and train-only preprocessing |
| EDA exposes the final evaluation target | Current sample explicitly designated development data | Reserve a separate untouched period before model selection; issue #8 |
| Missing appearances silently become zero performance | M2 retains NULL metrics; M3 reports absence of the dataset | Acquire/validate appearance histories; issue #6 |

Passing structural checks is not a declaration of model readiness. No market-value
feature table or fitted preprocessing pipeline exists yet, so no audit can certify
their future point-in-time correctness.

## Tracked actions

- [#6 — Appearance histories and identity mapping](https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/issues/6): prerequisite for data-backed M4 player statistics.
- [#7 — Historical valuation membership](https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/issues/7): prerequisite for reliable historical cohort/club features in M5.
- [#8 — Additional history and untouched holdouts](https://github.com/kubaborowiec7/Premier-League-Analytics-Lab/issues/8): expand seasons through M1's generic interfaces before M5/M6 model selection.

Additional seasons are planned as data preparation before modelling; M3 does not
download them or select a final holdout after inspecting its targets.

## Verification

`python -m compileall src` and `ruff check .` passed. Local `pytest` passed 72 tests;
13 PostgreSQL integration tests skip without `TEST_DATABASE_URL`. M3 tests cover
immutable/checksummed archives, source/scope validation, missing outcomes, empty
samples, date coverage, repeated valuations and undefined skew. The notebook executed
from a fresh kernel on synthetic fixtures and on the pinned public sample. CI repeats
the report and notebook run and uploads their outputs alongside M1/M2 verification.
