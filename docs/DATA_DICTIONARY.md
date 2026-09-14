# Data dictionary

This file starts as a canonical target schema. Update it whenever ingestion changes.

## competitions

| Column | Type | Meaning |
|---|---|---|
| competition_id | text | canonical competition id |
| name | text | competition name |
| country | text | country |
| source | text | source system |

## clubs

| Column | Type | Meaning |
|---|---|---|
| club_id | text | canonical club id |
| name | text | club name |
| country | text | country |

## players

| Column | Type | Meaning |
|---|---|---|
| player_id | text | canonical player id |
| name | text | display name |
| date_of_birth | date | birth date |
| position | text | normalized position |
| nationality | text | nationality |

## matches

| Column | Type | Meaning |
|---|---|---|
| match_id | text | canonical match id |
| competition_id | text | competition |
| season | text | season label |
| match_date | timestamptz | scheduled/played datetime |
| home_club_id | text | home club |
| away_club_id | text | away club |
| home_goals | int | full-time home goals |
| away_goals | int | full-time away goals |
| status | text | match status |

## player_appearances

| Column | Type | Meaning |
|---|---|---|
| match_id | text | match |
| player_id | text | player |
| club_id | text | club |
| minutes | int | minutes played |
| goals | int | goals |
| assists | int | assists |
| yellow_cards | int | yellow cards |
| red_cards | int | red cards |

## player_market_values

| Column | Type | Meaning |
|---|---|---|
| player_id | text | player |
| valuation_date | date | date of valuation |
| market_value_eur | numeric | editorial market value |
| club_id | text | club at/near valuation date |
| source | text | source dataset |

## source_snapshots

| Column | Type | Meaning |
|---|---|---|
| snapshot_id | uuid | snapshot identifier |
| source_name | text | source |
| source_url | text | retrieval URL |
| retrieved_at | timestamptz | retrieval time |
| dataset_version | text | version |
| coverage_start | date | coverage start |
| coverage_end | date | coverage end |
| sha256 | text | file checksum |
| local_path | text | raw local file |

## model_runs

| Column | Type | Meaning |
|---|---|---|
| run_id | uuid | model run |
| model_name | text | model |
| task | text | task name |
| git_sha | text | code version |
| train_start | date | train window start |
| train_end | date | train window end |
| test_start | date | test window start |
| test_end | date | test window end |
| params | jsonb | parameters |
| metrics | jsonb | metrics |
| created_at | timestamptz | creation time |


## player_season_features

Analytical/model-ready table. Exact metric columns will evolve.

| Column | Type | Meaning |
|---|---|---|
| player_id | text | canonical player |
| competition_id | text | canonical competition |
| season | text | season |
| reference_date | date | latest data included in the row |
| position_group | text | normalized scouting peer group |
| club_id | text | club for the observation |
| minutes | numeric | exposure |
| feature_payload | jsonb / columns | raw and derived football metrics |
| normalization_context | text | e.g. competition-season-position |
| created_at | timestamptz | materialization time |

Primary modelling grain should make `competition_id` explicit even while V1 only contains EPL.
