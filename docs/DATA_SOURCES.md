# Data sources

The core project should remain reproducible and should not depend on fragile web scraping.

M1 implementation and refresh commands: [INGESTION.md](INGESTION.md).
M4 extends the same published CSV mirror with `games.csv.gz` and `appearances.csv.gz`.
Hashes, scope and expected counts are pinned in `data/manifests/m4_sources.json`;
acquisition uses the existing immutable snapshot store. It joins appearances to game
participants through source IDs and does not infer cross-source club aliases. Profile
sub-positions remain explicitly historically unverified. Acquisition, output schemas
and reproduction are described in [PLAYER_ANALYTICS.md](PLAYER_ANALYTICS.md).
The pinned sample manifest is `data/manifests/m1_sources.json`; acquisition timestamps
and source URLs are retained in raw sidecars and PostgreSQL provenance records.

## 1. Football-Data.co.uk — match-level core source

Website:
https://www.football-data.co.uk/

Use for:
- historical match results,
- full-time / half-time scores,
- common match statistics,
- historical odds as an optional external benchmark.

Advantages:
- long history,
- simple downloadable CSV files,
- ideal for time-series/backtesting work.

Implementation:
- create a versioned downloader with explicit season/league URLs,
- persist source URL and retrieval timestamp,
- cache raw files unchanged.

Do not treat bookmaker odds as model features in the primary football model.
They can be used later as a benchmark for predictive information/calibration.

---

M6 extends the existing Football-Data path to seven pinned `E0` seasons from 2019/20
through 2025/26. Hashes and season coverage are in `data/manifests/m6_experiment.json`;
2019/20 permits the COVID-delayed July 2020 finish. Competition/country/timezone are
configured, not inferred in model code. The 2021/22 Newcastle–West Ham row (15 August
2021) contains eight away shots but nine on target. M6 excludes all shot fields through
explicit result-only parsing and continues to validate teams, dates and scores.
The default ingestion path remains strict and raw archives are unchanged. See
[MATCH_MODELS.md](MATCH_MODELS.md) for acquisition and refresh commands.

M7 adds a pinned partial 2026/27 Football-Data `E0` archive: 40 matches dated
2026-08-21–2026-09-14, test cutoff 2026-09-15. The exact hash is recorded in
`data/manifests/m7_experiment.json`. Only row count/date coverage was inspected before
selection; outcomes were reserved for final evaluation. The active-season URL is
mutable, so a fresh download may fail its pin after new results arrive. Preserve the
immutable local archive; any refresh creates a new manifest/experiment. No raw dataset
is committed. [MATCH_ML.md](MATCH_ML.md) explains the chronological windows and refresh.

## 2. Transfermarkt dataset snapshot — valuation core source

Recommended published dataset:
https://www.kaggle.com/datasets/davidcariboo/player-scores

Associated project:
https://github.com/dcaribou/transfermarkt-datasets

Use for:
- players,
- clubs,
- games,
- appearances,
- transfers,
- historical player valuations.

Important status:
The upstream dataset metadata currently notes that automated updates are paused and that published data remains valid up to the last successful 2026 snapshots.

Therefore:
- pin the dataset version used,
- document its coverage end date,
- use it primarily for historical supervised learning,
- do not pretend it contains live 2026/27 valuation data.

Do not make Transfermarkt scraping a core project dependency.

M1 uses the upstream project's published CSV mirror, documented in its README:
`https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/` with `players.csv.gz`,
`player_valuations.csv.gz` and `clubs.csv.gz`. The chosen publication is labelled
`published-2026-07-06`, with exact file hashes pinned locally and in CI. The upstream
repository declares CC0-1.0; attribution and terms notes are preserved, and no raw
third-party datasets are committed to this repository.

The upstream `player_valuations` transformation resolves clubs through transfers when
possible but falls back to current player-club information. Its competition column is
therefore not sufficient proof of historical league membership. M1 labels this context
as unverified and leaves canonical historical `club_id` unset. See the
[source transformation](https://github.com/dcaribou/transfermarkt-datasets/blob/master/dbt/models/curated/player_valuations.sql).

---

## 3. football-data.org — lightweight current metadata API

Website:
https://www.football-data.org/

Use for:
- competition metadata,
- schedules,
- fixtures,
- standings,
- optional current-season refresh.

Store the API token in `.env`:
`FOOTBALL_DATA_API_TOKEN`.

The free tier can be sufficient for lightweight project refreshes but is rate limited.
Implement retry/backoff and caching.

---

## 4. StatsBomb Open Data — event-data lab

Repository:
https://github.com/statsbomb/open-data
(or its current Hudl-hosted equivalent)

Use for:
- event-level football analytics,
- event coordinates,
- lineups,
- selected 360 data,
- advanced methodology demonstrations.

Important:
Open data covers selected competitions/seasons; do not assume it includes the complete current Premier League season.

If publishing analysis, follow the source attribution requirements.

---

## 5. FBref — optional/manual enrichment

Website:
https://fbref.com/

Use only if:
- the data is manually exported/provided,
- or usage is clearly compliant with current terms.

Do not make an automated FBref scraper necessary to reproduce the main project.

Possible enrichment:
- player standard stats,
- shooting,
- passing,
- possession,
- defensive actions,
- goalkeeping.

The ingestion layer should accept an exported CSV and map it into the canonical schema.

---

## Source priority

For the first implementation:

1. Football-Data.co.uk → match model
2. published Transfermarkt dataset snapshot → player/value model
3. football-data.org → current fixture metadata
4. StatsBomb Open Data → advanced event-data extension
5. FBref manual export → optional enrichment

---

## Provenance record

M5 reuses the published `games`, `appearances`, `players` and `player_valuations`
archives pinned in `data/manifests/m5_experiment.json`; it adds no scraper or source.
Game/appearance facts from 2018-07-01 provide strictly prior competition participation;
valuation history supplies lagged values and dated targets through 2026-06-30.
Selection exposes targets only through 2025-06-30, with 2025/26 reserved for final
evaluation. Current-club and valuation-league snapshot fields are ignored. The cohort
is recent prior participants, not confirmed valuation-date league membership.
Original publication times/revision vintages are unavailable. Local snapshot metadata
retains retrieval dates, published URLs and existing licence/terms notes; the selected
run records source hashes and URLs. See [VALUE_MODELS.md](VALUE_MODELS.md) for refresh.

For every raw dataset store a metadata record with:

- `source_name`
- `source_url`
- `retrieved_at`
- `dataset_version`
- `coverage_start`
- `coverage_end`
- `license_or_terms_note`
- `sha256`
- `local_path`

Never silently replace a raw source file.


## Competition adapters

Source-specific competition identifiers must never leak into modelling logic.

Example concept:

```text
canonical: EPL
  football-data.co.uk: E0
  football-data.org: PL
  other_dataset: source-specific id
```

Maintain this translation in ingestion/configuration or the
`source_competition_mappings` table.

V1 only activates `EPL`, but adapters should be written so another competition can be
enabled by supplying the correct source mapping rather than copying the entire loader.

## Advanced player statistics: FPL Core published snapshots

The extension uses [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights)
CSV exports at commit `d2b7c3a7f9bc7ba4781efef1fda2408b12d4d5cb`, pinned with 89 file
hashes in `data/manifests/advanced_players.json`. Coverage: 2024/25, 2025/26 and
40 matches of 2026/27. The publisher permits reuse with attribution requested;
upstream-provider rights are not independently granted here. No underlying sports
website is scraped. Snapshots retain retrieval metadata and remain outside Git.
See [ADVANCED_PLAYERS.md](ADVANCED_PLAYERS.md) for metric definitions and gaps.
