# Data sources

The core project should remain reproducible and should not depend on fragile web scraping.

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
