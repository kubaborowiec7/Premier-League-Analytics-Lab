# M4 — Advanced player statistics

M4 builds descriptive player features from published appearance histories. It does
not train a valuation/match predictor. The initial example is Premier League 2023/24;
reusable functions retain competition/season scope and accept other competitions.

## Reproduction and source boundaries

```bash
# First acquisition only; published CSV mirror, never scraping:
python scripts/build_player_analytics.py --download
# Repeat offline from the same immutable snapshots:
python scripts/build_player_analytics.py
```

Use `--manifest` for an alternative scope and source-code/season mapping. Settings
control active competitions and data/artifact directories. The manifest
`data/manifests/m4_sources.json` pins games, appearances and players by SHA-256.
Missing archives fail unless acquisition is explicitly enabled; changed files fail
checksum verification. Acquisition records retain URL, timestamp, version and terms.
The games/appearances files were acquired 2026-09-15; the player profile pin is the
same as M1. File hashes, rather than an assumed synchronized publication label,
identify the exact combination used.

Games and appearances join by source game ID, players by source player ID, and
appearance clubs must be participants in the game. Dates and competition codes must
agree. Conflicting duplicates and missing profiles/appearance coverage fail explicitly.
Identical duplicate observations collapse. Current-club columns are ignored.
The adapter checks every selected game has appearances; it does not claim every
player event or statistic was independently verified against an official match feed.

Source identities remain `tm:match:*`, `tm:player:*`, `tm:club:*`. This artifact path
does not add another copy of matches to PostgreSQL or equate them to Football-Data
IDs. Cross-source reconciliation and historical valuation membership remain separate
tasks. M2's database marts are not populated by this command. M3's report remains an
accurate description of the earlier M1 sample, which did not load appearances.

Outputs (ignored by Git; uploaded in CI):

- `data/processed/m4/appearances.parquet`: scoped match/player observations.
- `data/processed/m4/player_features.parquet`: one player/competition/season row at the reference date.
- `data/processed/m4/player_pca.parquet`: striker projection with scope and position context.
- `artifacts/m4/profile.json`: example, peer parameters, bootstrap intervals, neighbors and PCA metadata.
- `artifacts/m4/player_profile.png`: observed profile and uncertainty plot.

## Exposure and missingness

The aggregation includes only matches strictly before `reference_date` (UTC). Current
and later matches are excluded before summation. Incomplete minutes/count totals stay
NULL; they are not silently treated as zero. A zero exposure has undefined per-90
statistics. Duplicate appearance keys are rejected to prevent double-counting.
Transfers can change the recorded club while retaining one player-season row.

For count `x`, `x_per90 = 90 × sum(x) / sum(minutes)`. M4 supplies goals, assists,
yellow cards and red cards. These four counts are an intentionally limited description;
they cannot measure passing, defending, movement or overall player quality.

## Peer groups and normalization

The published profile's sub-position maps explicitly to GK, CB, FB/WB, DM/CM, AM/W,
or ST; unmapped values become UNKNOWN and are not ranked. **These are snapshot
positions, not verified historical positions.** Every row retains
`position_context=snapshot_unverified`. Consequently, the current artifacts are
retrospective and must not enter historical prediction as if their role labels had
been available on the reference date. A future as-of feature pipeline needs dated roles.

`fit_peers(training)` stores a copy of the supplied reference cohort. Transforming
candidates does not refit it. Peers share competition, season, position and position
context. Defaults: at least 450 minutes per reference player and at least five complete
peers per metric. Reference dates later than the transformed row are rejected.
The current demo fits the full descriptive season; it is not a train/test evaluation.

Percentile = `100 × (number below + 0.5 × number tied) / peer count`. An all-equal
group has percentile 50. Z-score uses the peer population standard deviation (`ddof=0`);
constant groups have undefined z-scores. Players below the minutes threshold retain
raw values and shrinkage estimates but are not percentile-ranked. A high card percentile
means more cards per 90, not better performance. No overall quality score is computed.

The minimum minutes and group sizes are transparent heuristics, not optimized claims.
Callers can change them and should assess sensitivity before deploying rankings.

## Reliability and shrinkage

The unsmoothed per-90 rate is the simple baseline. Reliability weight is
`w = minutes / (minutes + 900)`. The stabilized rate is
`w × player_rate + (1 − w) × peer_rate`, where the peer rate is exposure-weighted:
`90 × sum(peer counts) / sum(peer minutes)`.

The 900-minute prior is a configurable heuristic, not an estimated empirical-Bayes
hyperparameter. Weight measures exposure under this rule, not a probability that the
player is good or a validated confidence score. Missing exposure/counts or insufficient
peer support produce missing outputs. The raw and stabilized rates are both retained.

## Bootstrap intervals

For one player/competition/season, resample complete positive-minute appearance rows
with replacement and recompute the ratio of total count to total minutes. This differs
from averaging per-match per-90 rates, which overweights brief appearances.
Defaults: 1,000 repetitions, seed 42, percentile interval at 2.5% and 97.5%.
Draws are chunked to bound memory. Missing counts/exposure invalidate the interval;
fewer than two positive-minute appearances give no interval. A positive count recorded
with zero minutes also gives no interval rather than silently discarding that count.

Intervals describe empirical match variation under an exchangeability assumption.
They do not account for opponent strength, serial dependence, injuries, selection bias
or uncertainty in the source data, and are not guaranteed future prediction intervals.
All-zero observations yield a degenerate empirical interval; this does not prove the
true event probability is zero. A later study can use block/bootstrap or count models.

## PCA and similarity

PCA explicitly fits a scaler and projection on supplied reference rows within one
position/context. Only the declared performance z-score columns are accepted;
market value cannot enter. Missing or constant inputs require an explicit decision,
not automatic imputation. The demo removes constant columns and uses complete eligible
strikers, then saves two components, loadings, scaler values and explained variance.
The projection checks that its reference date is not later than projected observations.
PCA sign has no intrinsic interpretation; it is dimensional reduction, not a quality rank.

Similarity uses Euclidean distance in the declared standardized performance space,
not the truncated PCA projection. The caller must supply a candidate competition
universe. Candidates must match season, reference date, position and position context,
have enough minutes and complete features; the reference player is excluded across
all competitions. Lower distance means closer values on these metrics. Normalization
within each competition does not establish equal league strength or playing style.
Cross-league use needs separate validation. Optional clustering is deferred.

## Measured example

Pinned coverage: **380 games, 11,384 appearances, 570 players**. Of these, 404 meet
the 450-minute threshold; the complete striker PCA cohort contains 47 players.
The script selects the eligible striker with most observed goals, breaking ties by
minutes then ID. For this sample that is **Erling Haaland**, `tm:player:418560`.

| Measure | Result |
|---|---:|
| Recorded appearances / minutes | 31 / 2,558 |
| Recorded goals / assists | 27 / 4 |
| Goals per 90 | 0.950 |
| Assists per 90 | 0.141 |
| Goals percentile among eligible strikers | 96.81 |
| Exposure weight | 0.740 |
| Stabilized goals per 90 | 0.815 |
| Goals per 90 bootstrap interval | 0.599–1.343 |
| Assists per 90 bootstrap interval | 0.035–0.282 |

The two PCA components explain approximately 28.73% and 28.25% of standardized
variance (56.98% combined). The limited metrics produce neighbors Alexander Isak,
Callum Wilson, Chris Wood, Jean-Philippe Mateta and Richarlison. This is a count-statistic
comparison; it does not establish equivalence of movement, role or overall ability.
Assists and minutes reflect this source's definitions and are not claimed as official
league totals. Different metric sets or reliability thresholds can change the neighbors.

## Verification

Offline tests cover ratios/exposure, missing and zero minutes, duplicate/invalid counts,
strict time cutoffs, frozen references, peer ties/constant groups, small cohorts,
candidate universes, PCA feature restrictions and deterministic bootstrap intervals.
Source fixtures validate date, player and club integrity using an arbitrary competition.
CI also reproduces the complete pinned-source example and checks expected row counts.
Local verification passed compileall, Ruff and 88 tests; 13 PostgreSQL tests skip
without a server. The descriptive PCA has a model card at
`reports/model_cards/player_pca.md`.
