# FPL Decision Lab

The FPL Decision Lab is a planning workspace built during the current-data
refresh. It contains player rankings for the next 1, 3 and 5 gameweeks, a five-week
fixture ticker, transfer comparisons, a captaincy shortlist and an optional view of a
public FPL squad.

M15 adds an interactive expected-points pitch, a mixed-integer squad/line-up/captain
optimizer, transfer-hit comparison, transparent chip signals and a rolling historical
backtest.

## Rebuild

```bash
python scripts/refresh_current_data.py
```

The refresh pins one FPL-Core-Insights commit, archives every fixture snapshot and
writes `data/processed/fpl/players.parquet`, `fixtures.parquet` and `metadata.json`.
It also writes `backtest.parquet` and `backtest.json`.
Raw snapshots remain immutable. GitHub Actions uploads these files with the other
portable current-data artifacts.

To show a personal squad, add only its public numeric ID to local configuration:

```dotenv
FPL_ENTRY_ID=123456
```

The next refresh reads the official public entry and picks endpoints and creates the
ignored local `squad.parquet`. It never requests FPL credentials. The metadata retains
the entry ID and team name, but no manager name.

For an immediate session-only lookup, open **My team**, enter the same public ID and
select **Load team**. This explicit action calls the two official public endpoints and
keeps the response only in Streamlit session memory. It does not modify `.env`, raw
snapshots or processed artifacts.

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

## Optimization

The optimizer maximizes expected starter points, the extra captain copy and a small
bench value. It enforces a 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD), at most three
players per club, a user-specific budget and a legal XI (1 GK, 3–5 DEF, 2–5 MID,
1–3 FWD). Transfer runs retain at least `15 - max_transfers` current players and deduct
four projected points per transfer above the selected free-transfer allowance.

Current public endpoints do not expose each manager's historic purchase/selling price,
so the optimization uses current prices and may overstate the available budget. For each
of the next five gameweeks, the chip optimizer calculates the extra Triple Captain copy,
bench points and the gap to an unrestricted Free Hit squad, then assigns the three chips
to distinct gameweeks for the largest combined increment. Threshold labels still say
Hold when that best increment is weak. This is a five-week plan rather than a full-season
chip schedule, and it never executes an FPL action.

## Backtest

For each target gameweek, `rolling_backtest` uses only the latest cumulative snapshot
strictly before that gameweek. Exposure-weighted points per game are shrunk toward a
position median and compared with the following gameweek's observed points. The report
shows MAE, RMSE, the prior position-median baseline and mean gameweek Spearman rank
correlation. The current pilot covers GW2–GW5 (2,583 player-gameweek rows): model MAE
1.19 versus 1.49 for the baseline, RMSE 2.38 and mean Spearman 0.63. Four early-season
targets are too few to establish stable calibration or expected future performance.

## Limitations

This is a simple reproducible baseline. It does not model predicted line-ups, bookmaker
odds, transfer price changes, season-wide chip scheduling or late injury news. Expected minutes are a
season average adjusted by availability, so rotations can dominate the forecast. The
tool supports decisions; it does not claim that the highest-ranked player will score the
most points. The optimizer and early backtest do not remove those forecast limitations.
