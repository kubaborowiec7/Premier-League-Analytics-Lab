# PostgreSQL analytics (M2)

M2 adds five ordinary views over the canonical tables. They update when canonical
data changes; there is no materialized-view refresh or model training. No additional
seasons or external datasets are downloaded by M2.

## Install and validate

Use the configured `DATABASE_URL` and run from the repository root after M1:

```bash
python scripts/verify_analytics.py --install --manifest data/manifests/m1_sources.json
python scripts/verify_analytics.py --manifest data/manifests/m1_sources.json
```

The first command applies `sql/migrations/002_analytics.sql` and `sql/analytics.sql`
in one transaction and runs `sql/checks.sql`. Repeating installation is supported.
Invalid existing rows abort installation without rewriting canonical data. Investigate
the offending records and their source before attempting the migration again.
The optional manifest asserts the pinned M1 sample's coverage: 760 team-match rows,
948 player-season rows and 2,093 valuations. Omit it for an empty or different database;
structural checks permit empty tables and the JSON report records player coverage.

New Compose volumes apply the bootstrap, M1 migration, M2 constraints and views in
that order. Existing volumes require the explicit command above. Do not delete the
volume to upgrade. A database created using only `sql/schema.sql` also needs
`sql/migrations/001_ingestion.sql` before M2. M1's verification script with
`--load --initialize-schema` supplies that prerequisite.

Reports are written to ignored `artifacts/m2-verification.json` by default. Nothing
connects to PostgreSQL during package import or Streamlit startup.

## Contracts

| View | Unique grain | Meaning |
|---|---|---|
| `mart_team_match` | match, club | Two observed sides of each finished match; goals, shots, home/away and points |
| `mart_team_prematch` | match, club | Finished/scheduled fixtures with prior five eligible games, no current result columns |
| `mart_player_season_performance` | player, competition, season | Full-season appearance totals and completeness counts |
| `mart_player_season_value` | player, competition, season, source | First/latest valuation dates, latest/min/max values, counts and context labels |
| `mart_player_season` | player, competition, season | Coverage overview joining performance and valuation populations |

All views retain `competition_id` and `season`. Cancelled and unrecognized statuses
are not treated as finished results. Stored `finished` rows require both scores;
canonical match scope, clubs and nonnegative recorded statistics are validated.
Scheduled fixtures may have NULL scores. Statistics that are missing stay NULL.

Player performance aggregates only recorded appearances in finished matches. A
missing minutes/goals/assists value makes that season's corresponding total NULL;
completeness counts remain available. Transfers do not create duplicate player-season
rows. The M1 sample has no appearances, so it produces valuation-only player-season
rows with `has_performance=false` and NULL performance, not invented zero totals.

Valuations from different sources remain separate. `latest_value_eur` is the last
dated observation, not the season maximum. `has_unverified_context` and `context_labels`
preserve source limitations; absence of that particular label is not proof of verified
membership. Reported clubs are not joined to Football-Data clubs. See
[INGESTION.md](INGESTION.md) for source identity and historical membership caveats.

## Temporal boundaries

`mart_team_match` and all player-season views are **retrospective**. Season-end
totals/latest values must not be used for predictions earlier in that season.
They do not populate the future model-ready `player_season_features` table.

`mart_team_prematch` has an explicit `history_cutoff`: the start of the target's UTC
date minus one day. Only results with a strictly earlier stored timestamp qualify.
This imposes a conservative 24–48 hour embargo because unknown kickoffs may be
represented by local midnight, which can fall on the previous UTC date. Same-day
and preceding-day results cannot enter history. The cutoff is independent of the
database session timezone. It intentionally omits some otherwise usable recent games.

History resets each competition and season. Eligible games are ordered by timestamp
and match ID and limited to the latest five; ID breaks ties only among already eligible
results. `history_matches` reports 0–5. Cold starts have zero history and NULL means.
`points_last_5` and goal fields are **means per eligible game**, not sums. No imputation
is fitted in SQL. The cutoff policy needs review before extending to sources with
different timing semantics. Delayed result publication/corrections cannot be reconstructed
from a final historical CSV: this is event-time protection, not proof of original
publication-time availability. Models still need chronological evaluation.

## Advanced queries and indexes

`sql/examples.sql` contains three independent SQLAlchemy `text()` statements accepting
bound `competition_id` and `season` parameters. They demonstrate CTEs, conditional
aggregation, ranking, LAG/LEAD, a cumulative window, and explicit rolling-feature
selection. The league table uses observed points and does not apply disciplinary
deductions. The timeline's LEAD and current-result cumulative total are descriptive.

M2's home/away history indexes start with competition, season and club, followed by
date/match ID, with a `finished` partial predicate. They support per-club lookups;
the planner may still choose scans on small datasets or through view expansion.
The valuation index matches competition/season/player/source/date grouping and order.
Existing primary keys support appearance-to-match joins and enforce canonical grains.
Do not claim a speedup without `EXPLAIN (ANALYZE, BUFFERS)` on representative data.
Ordinary lateral histories trade simplicity for repeated lookups; benchmark before
large multi-season use and consider persisted as-of features in a later milestone.

## Tests

```bash
python -m compileall src
ruff check .
pytest
# Set TEST_DATABASE_URL to a disposable PostgreSQL 16 database, then:
pytest tests/test_postgres_ingestion.py tests/test_postgres_analytics.py
```

PostgreSQL tests create/drop only their own randomly named schemas. With no test URL
they skip, so ordinary pytest requires no external service or data. Integration tests
execute the actual SQL, check empty/repeated installation, invalid legacy rollback,
five-game windows, current/future outcome independence, timestamp ties, unknown
kickoffs, timezone independence, competition/season separation, transfers, incomplete
statistics, source-separated valuations and the example queries. SQL invariants cover
view uniqueness, two sides per result, balanced goals/points, cutoff bounds and
appearance club participation. CI also installs/replays M2 on the pinned M1 sample.
