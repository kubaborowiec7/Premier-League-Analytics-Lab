# Session player import

Player Explorer, Scouting Finder and Market Value accept session-only player inputs.
The feature is intended for placing an external observation into an existing league
comparison. It does not append to raw/processed data, retrain a model or treat the
input as verified source data. Closing the browser session clears the records.

## Performance profiles

Select a season first, then open **Import a player for league comparison**. Manual
entry supports totals or per-90 count statistics. Percentages use a 0–100 scale.
The CSV template accepts 1–20 rows and requires:

- `player_name`, `position_group`, `minutes`;
- optional `competition_id`, `season`, `club_name`, `appearances`;
- one or more metric columns from the advanced registry;
- either `metric` or `metric_per90` for a count, never both;
- optional `metric_coverage` from 0–1; supplied values otherwise mean 100% coverage.

Competition and season must equal the selected cohort. Position aliases are normalized
to `GK`, `DEF`, `MID` and `FWD`. Unknown columns, non-finite/negative values, percentages
above 100, ambiguous total/per-90 inputs and duplicate identities fail with a readable
message. Missing metrics stay missing.

Imported players receive ranks against the native competition/season/position reference
players. They are excluded from the reference distribution, so importing a player does
not change league players' ranks. The imported record can be the primary radar player,
an overlay, a scatterplot point or the Scouting Finder reference profile.

## Market-value scenarios

Market Value accepts the eight allowlisted M5 features: age, previous editorial value,
days since that value, days since the last appearance, prior-365-day appearances and
minutes, goals/90 and assists/90. The observed editorial value is required for the
model-gap comparison. CSV currency inputs use EUR; the manual form uses million EUR.

Inference loads the trusted local M5 bundle only after submission and verifies it against
the frozen selection checksum. The displayed estimate and nominal interval use the same
Ridge model and calibrated residual rule as the historical evaluation. This is a
hypothetical application of a frozen historical benchmark. It is not a live valuation,
transfer-fee estimate or evidence that a player is under/overvalued.

Uploaded CSV content is parsed as data only. Uploaded joblib/pickle models are never
accepted because those formats can execute code.
