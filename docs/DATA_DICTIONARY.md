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
| match_time_known | boolean | true only when the source supplied kickoff time; false means local-midnight placeholder |
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
| source_reported_club_id | text | source club field, retained separately from verified historical membership |
| competition_id | text | canonical code mapped from source-reported competition |
| season | text | explicit ingestion season label; new M1 rows always retain scope |
| competition_context | text | `source_reported_unverified`; must not be used as historical league evidence |
| source | text | source dataset |

M1 leaves valuation `club_id` NULL. Scope fields are nullable in SQL only to accommodate
pre-existing M0 records without inventing missing historical context.

## source_snapshots

| Column | Type | Meaning |
|---|---|---|
| snapshot_id | uuid | snapshot identifier |
| source_name | text | source |
| source_url | text | retrieval URL |
| retrieved_at | timestamptz | retrieval time |
| dataset_version | text | version |
| license_or_terms_note | text | source attribution and licence/terms notes |
| coverage_start | date | coverage start |
| coverage_end | date | coverage end |
| sha256 | text | file checksum |
| local_path | text | raw local file |

The source/checksum pair is unique. Full-file coverage stays NULL when not established;
filtered selection windows belong to ingestion batches instead.

## ingestion_batches and ingestion_batch_snapshots

| Column (ingestion_batches) | Type | Meaning |
|---|---|---|
| batch_id | uuid | deterministic identity of scope, loader version, source bytes and accepted records |
| source | text | adapter source |
| competition_id / season | text | canonical scope |
| coverage_start / coverage_end | date | inclusive requested selection window, not guaranteed fixture coverage |
| loader_version | text | adapter contract version |
| row_counts | jsonb | accepted deduplicated rows per canonical table |
| skipped_rows | integer | records filtered out by source competition/date scope |
| warnings | jsonb | contextual limitations such as unverified historical membership |
| created_at | timestamptz | successful ingestion transaction time |

`ingestion_batch_snapshots` has a composite primary key `(batch_id, snapshot_id)` and
foreign keys to batches and raw source snapshots. It links multi-file imports to all
their inputs. Reruns reuse a batch; conflicting canonical observations roll back.

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
