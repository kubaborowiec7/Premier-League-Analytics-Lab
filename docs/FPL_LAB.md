# FPL Decision Lab

The FPL Decision Lab is a planning workspace built during the current-data
refresh. It contains player rankings for the next 1, 3 and 5 gameweeks, a five-week
fixture ticker, transfer comparisons, a captaincy shortlist and an optional view of a
public FPL squad.

M15 adds an interactive expected-points pitch, a mixed-integer squad/line-up/captain
optimizer, transfer-hit comparison, transparent chip signals and a rolling historical
backtest.

M16 replaces the season-average minutes baseline with a recency-weighted availability
model and expands expected points to the main position-specific FPL scoring events.

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

### Expected minutes

For every player, the latest six completed fixtures receive exponentially declining
weights (`0.65 ** games_ago`). Position priors with the weight of one match stabilize
small samples. The model separately estimates:

- probability of starting;
- probability of any appearance;
- probability of reaching 60 minutes;
- minutes conditional on starting or entering as a substitute.

Official availability probability or status caps all four outputs. A recent run of
starts can therefore outweigh early-season absences, while one isolated start cannot
produce certain 90 minutes. The UI exposes the three probabilities and a session-only
xMins scenario control. It does not ingest press conferences or inferred team news.

### Expected points

For each fixture, the transparent event model combines:

- expected appearance points from `P(appearance) + P(60+)`;
- expected goals and assists per 90, shrunk toward active positional peers;
- clean-sheet probability and attacking difficulty derived from pre-refresh club Elo;
- goalkeeper saves and penalty saves;
- official defensive-contribution thresholds for outfield players;
- historical bonus rate, cards, own goals and missed penalties;
- expected deductions for every two goals conceded by goalkeepers and defenders.

The Player projection expander shows the one-gameweek contribution of each component.
This is an interpretable expectation model rather than a joint match simulation: goal,
assist and bonus events are not yet simulated together and Elo clean-sheet probabilities
remain intentionally simple.

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

For each target gameweek, `rolling_backtest` uses only cumulative snapshots strictly
before that gameweek. It reconstructs prior fixture minutes, applies the same recency
minute model and scales a shrunk prior points-per-90 rate. Players with no senior minutes
before the forecast origin are excluded using only prior information; this prevents
hundreds of predictable reserve zeroes from flattering aggregate error.

The current pilot covers GW2–GW5 (1,467 eligible player-gameweek rows): model MAE 1.83
versus 2.08 for the active-position baseline, RMSE 2.75, mean Spearman 0.53 and xMins
MAE 20.4 minutes. Mean predicted points are 2.24 versus 2.40 observed; 28.0% of eligible
rows score zero. Four early-season targets are too few to establish stable calibration
or expected future performance.

## Limitations

This remains a reproducible planning model rather than a commercial projection feed. It
does not model predicted line-ups, bookmaker odds, press conferences, transfer price
changes, season-wide chip scheduling or late injury news. It has no explicit dependency
between competing players for one starting place. Goalkeeper save and bonus expectations
use shrunk historical rates; the scoreline is not jointly simulated. The tool supports
decisions and does not claim that the highest-ranked player will score the most points.
