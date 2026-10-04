# FPL Decision Lab

The FPL Decision Lab is a planning workspace built during the current-data refresh.
One planning horizon (next gameweek, 3 or 5 gameweeks) in the page header drives the
tabs:

- **This gameweek** – captain picks with indicative ranges, the top picks per position,
  an availability watch and, for a loaded team, the projected XI and best single transfer.
- **Players** – a filterable projection table; selecting a row shows the next-gameweek
  points breakdown and a session-only minutes scenario.
- **Fixtures** – a five-gameweek ticker with Overall, Attack and Defence difficulty.
- **My team & transfers** – the recommended XI on a pitch, a transfer planner with a
  head-to-head comparison, the squad optimizer and a five-gameweek chip plan.

- **Mini-league** – compare the loaded squad with rivals in one of its private classic
  leagues: differentials, players most rivals own, and rivals' captains.

The rolling backtest lives in **Model Lab → FPL expected points**.

### Price watch

"This gameweek" lists the five most-bought and five most-sold players by net transfers
in the current gameweek (`features.fpl.price_watch`). It is a watch list, not a model of
the game's undisclosed price thresholds. `price_watch_backtest` checks it on the previous
season's snapshots: over 36 gameweeks of 2025/26, 56% of the ten most-bought players had
risen by the next gameweek (2% of all players did) and 52% of the ten most-sold had
fallen (5% of all players). The result is stored in `metadata.json`.

### Transfer plan

`plan_transfers` proposes transfers week by week over the five prepared gameweeks. Each
week it takes the best single swap while that adds at least one point to the best XI over
the remaining weeks (plus four once free transfers run out, if hits are allowed). A swap
that would not help that week's XI is deferred, which saves the free transfer; one free
transfer is added per week, up to five. It is a greedy heuristic, not an optimal plan: it
never combines moves, and it uses current prices rather than selling prices.

### Mini-league

Pressing **Load** reads one page of the chosen league's table and up to 20 rival squads
from the official public endpoints (`fetch_league_standings`, `fetch_league_picks`),
either the top of the league or the teams nearest the user's rank. The data stays in
Streamlit session memory. Manager names are discarded on read; only team names are shown.
`league_ownership` gives each player's share of rival squads owning, starting and
captaining them. Only private classic leagues from the team's entry are listed.

## Rebuild

```bash
python scripts/refresh_current_data.py
```

The refresh pins one FPL-Core-Insights commit, archives every fixture snapshot, fits the
live match models and then writes `data/processed/fpl/players.parquet`,
`fixtures.parquet`, `standings.parquet`, `results.parquet`, `backtest.parquet`,
`backtest.json` and `metadata.json`. Raw snapshots remain immutable. GitHub Actions
uploads these files with the other portable current-data artifacts.

To show a personal squad, add only its public numeric ID to local configuration:

```dotenv
FPL_ENTRY_ID=123456
```

The next refresh reads the official public entry and picks endpoints and creates the
ignored local `squad.parquet`, which the page loads automatically. It never requests FPL
credentials. The metadata retains the entry ID and team name, but no manager name.

For a session-only lookup, use **Load my team / Change team** in the page header. This
explicit action calls the two official public endpoints and keeps the response only in
Streamlit session memory. It does not modify `.env`, raw snapshots or processed artifacts.

## Projection model (`recency-events-goalmodel-v3`)

### Fixture context

Each scheduled match gets a home and an away goal expectation from the live
`dixon_coles_half365` goal model (`data.fpl.model_goal_rates`; clubs are matched by
normalised name). A fixture the model cannot score falls back to
`features.fpl.season_goal_rates`: this season's xG-based attack and defence ratios,
shrunk toward the league average with the weight of six matches. `rate_source` records
which was used. From the two rates each club row gets:

- `clean_sheet_probability = exp(-expected_goals_against)`;
- `attack_multiplier`, the fixture's expected goals relative to that club's average over
  its listed schedule, because player per-90 rates already contain the club's typical
  attack;
- `difficulty` (expected goals against minus for), `attack_difficulty` and
  `defence_difficulty`, each banded 1 (easiest) to 5 (hardest).

The source's club Elo column is empty, so the earlier Elo-based context reduced to home
versus away only; it has been removed.

### Expected minutes

For every player, the latest six completed fixtures receive exponentially declining
weights (`0.65 ** games_ago`). A position prior with the weight of a quarter of a match
stabilizes players with no history. The model separately estimates the probability of
starting, of any appearance and of reaching 60 minutes, and minutes conditional on
starting or coming on. Official availability probability or status caps all four outputs.
It does not ingest press conferences or inferred team news.

### Expected points

For each fixture, the transparent event model combines:

- expected appearance points from `P(appearance) + P(60+)`;
- expected goals and assists per 90, scaled by the fixture's attack multiplier;
- clean-sheet points from the fixture's clean-sheet probability;
- goalkeeper saves and penalty saves;
- official defensive-contribution thresholds for outfield players;
- historical bonus rate, cards, own goals and missed penalties;
- deductions for every two goals conceded, from the fixture's expected goals against.

Every per-90 rate is a minutes-weighted blend of three sources: this season, the
player's previous season (`prior_season_rates`, half weight, capped at 1,800 minutes) and
the position's minutes-weighted mean (450 minutes; 900 for rare events). The anchor is a
mean rather than a median because a median sits near zero for skewed events and biased
low-minute players downwards.

This is an interpretable expectation model rather than a joint match simulation: goal,
assist and bonus events are not simulated together.

The 80% display range scales gameweek volatility by projected minutes. A player's own
volatility is pooled with the position's typical spread (weight of ten gameweeks) so one
early haul does not inflate it. It is an indicative range, not a calibrated prediction
interval. Blank and double gameweeks are represented by the number of fixture rows.

## Transfers and optimization

`replacement_options` ranks same-position replacements affordable from the sale price
plus the bank, with at least 30 expected minutes. For a loaded squad it also excludes
current players, applies the three-per-club limit and ranks by `lineup_gain`: the change
in the best legal XI (captain counted twice), so upgrading a player who stays on the
bench is worth nothing. `best_single_transfers` returns the best swap per squad player.

The optimizer maximizes expected starter points, the extra captain copy and a small
bench value. It enforces a 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD), at most three
players per club, a user-specific budget and a legal XI (1 GK, 3–5 DEF, 2–5 MID,
1–3 FWD). Transfer runs retain at least `15 - max_transfers` current players and deduct
four projected points per transfer above the selected free-transfer allowance.

Current public endpoints do not expose each manager's historic purchase/selling price,
so transfers use current prices and may overstate the available budget. For each of the
next five gameweeks, the chip optimizer calculates the extra Triple Captain copy, bench
points and the gap to an unrestricted Free Hit squad, then assigns the three chips to
distinct gameweeks for the largest combined increment. Threshold labels still say Hold
when that best increment is weak. It never executes an FPL action.

## Backtest

`rolling_backtest` scores the production event model. For each target gameweek it
rebuilds the player snapshot, appearance history and season-xG fixture strengths from
strictly earlier gameweeks, runs `project_players` and compares with the points scored.
Players with no minutes before the forecast origin are excluded so predictable reserve
zeroes do not flatter the error. Previous-season priors are fixed before the season and
therefore leak-free. The live goal model is not used here because it is fitted on all
results to date.

Three references are stored beside the model: the position median, a points-per-90 rate
model (the previous backtest's subject) and the provider's own `ep_next` published before
the target gameweek.

Pilot on GW2–GW5 of 2026/27 (1,467 player-gameweeks, refreshed 2026-10-04):

| Model | MAE | Mean Spearman |
| --- | --- | --- |
| Event model | 1.85 | 0.52 |
| Points-per-90 rate model | 1.85 | 0.52 |
| Provider `ep_next` | 1.97 | 0.45 |
| Position median | 2.08 | – |

Event model RMSE is 2.73, mean projection 2.31 against 2.40 scored, and expected-minutes
MAE 17.6. Before the v3 changes the same event model scored RMSE 2.82, Spearman 0.50,
mean projection 1.95 and minutes MAE 21.6; its MAE was lower (1.80) only because it
under-projected a right-skewed target, and its top-decile picks were projected at 3.3
while scoring 4.2 (now 4.2 against 4.5).

The backtest also scores the anytime-scorer probability used by the Betting Lab against
whether each player scored; see [BETTING_LAB.md](BETTING_LAB.md).

The minutes prior weight was chosen on these same four gameweeks, so the figures are
optimistic for that choice. Four early-season targets are too few to establish stable
calibration; on one-gameweek accuracy the event model is level with the simple rate
model, and its advantages (fixture context, components, multi-week horizons) are not
measured by this backtest.

## Limitations

This remains a reproducible planning model rather than a commercial projection feed. It
does not model predicted line-ups, bookmaker odds, press conferences, transfer price
changes, season-wide chip scheduling or late injury news. Availability flags apply
unchanged to every gameweek in the horizon. It has no explicit dependency between
competing players for one starting place. Previous-season rates follow the player, not
the club, so a transfer or a changed role is only corrected as new minutes accumulate.
The tool supports decisions and does not claim that the highest-ranked player will score
the most points.
