# FPL Decision Lab

The FPL Decision Lab is a read-only planning workspace built during the current-data
refresh. It contains player rankings for the next 1, 3 and 5 gameweeks, a five-week
fixture ticker, transfer comparisons, a captaincy shortlist and an optional view of a
public FPL squad.

## Rebuild

```bash
python scripts/refresh_current_data.py
```

The refresh pins one FPL-Core-Insights commit, archives every fixture snapshot and
writes `data/processed/fpl/players.parquet`, `fixtures.parquet` and `metadata.json`.
Raw snapshots remain immutable. GitHub Actions uploads these files with the other
portable current-data artifacts.

To show a personal squad, add only its public numeric ID to local configuration:

```dotenv
FPL_ENTRY_ID=123456
```

The next refresh reads the official public entry and picks endpoints and creates the
ignored local `squad.parquet`. It never requests FPL credentials. The metadata retains
the entry ID and team name, but no manager name.

## Projection baseline

For each fixture, the transparent baseline blends:

- season FPL points per 90, shrunk toward the position median according to minutes;
- expected goals and assists per 90;
- clean-sheet probability and attacking difficulty derived from pre-refresh club Elo;
- official FPL defensive-contribution thresholds where source coverage exists;
- expected minutes and the source availability flag.

The 80% display range scales historical gameweek volatility by projected minutes.
It is an indicative uncertainty range, not a calibrated prediction interval. Blank and
double gameweeks are represented by the number of fixture rows in each horizon.

## Limitations

This is a simple reproducible baseline. It does not model predicted line-ups, bookmaker
odds, transfer price changes, chip strategy or late injury news. Expected minutes are a
season average adjusted by availability, so rotations can dominate the forecast. The
tool supports decisions; it does not claim that the highest-ranked player will score the
most points. A historical rolling-origin FPL backtest and an optimization solver remain
future work.
