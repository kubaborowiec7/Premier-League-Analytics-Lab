# Dashboard

M8 and later extensions provide nine pages: Overview, Player Explorer,
Scouting Finder, Teams, League Table, FPL Decision Lab, Market Value, Match Predictor
and Model Lab. Competition options are populated
from available artifacts within `ACTIVE_COMPETITIONS`. Player and value pages
have independent season filters; they do not imply matching historical coverage.

After reproducing M4–M7 using their documented commands, run:

```bash
python scripts/build_dashboard_data.py
streamlit run app/Home.py
```

The explicit builder reads pinned local match archives and frozen M7 models. It
creates `artifacts/dashboard/catalog.json`, `clubs.parquet`, `team_states.parquet`
and, when M4 appearances are available, `player_intervals.parquet`. It does not
retrain models. `--download` explicitly permits missing pinned raw downloads.

`team_states` retains club, competition, season, forecast origin, Elo and the
prior-only feature summaries used by M7. Each club has one state at the latest
evaluated monthly origin. The catalog records that date and SHA-256 model hashes.
Player intervals retain player, competition, season and metric. They resample
recorded appearances; they are not prospective prediction intervals.

Teams uses the advanced FPL Core profiles only. It defaults to 2026/27 when that
season exists and offers every available advanced season. The squad table can show
all supported metrics or one metric family, with Total/Per-90 values and native
two-direction column sorting. League ranks use competition/season/broad-position
peers; Team ranks use the currently filtered squad. The source player snapshot assigns
one club per season, so transfers are not split into separate club totals.

The app caches tables/reports by path and content hash, and trusted model
bundles by checksum (verified on every load request). It never fits, downloads or connects
to PostgreSQL at startup or during prediction. Models must come from this project's
trusted local build: pickle/joblib is executable and must never be user-uploaded.
Missing modules remain usable as explanatory empty states; malformed artifacts
produce sanitized rebuild messages without connection strings or private content.

Player Explorer, Scouting Finder and Market Value may accept CSV/manual observations
into Streamlit session memory. Performance records are placed in the selected peer
cohort without changing its native reference distribution. Market-value scenarios load
the checksum-verified frozen M5 bundle only after explicit submission. Neither path
writes artifacts, trains models or accepts executable model uploads. See
[PLAYER_IMPORT.md](PLAYER_IMPORT.md).

Player statistics are limited to recorded goals, assists and cards; positions
remain snapshot-based. Similarity holds season, date and position context fixed,
with an explicit candidate competition universe and selectable metrics; constant
peer features are unavailable for similarity. Valuations show observed editorial
values separately from model estimates and nominal 90% intervals. Fragile rows
can be revealed explicitly; all rows remain in Model Lab evaluation metrics.

When `live_catalog.json` exists, Match Predictor uses the newest validated operational
snapshot and displays both its UTC origin and last included result. Without it, the
page falls back to the evaluated frozen origin. The selector uses short names and a
plain-language description for every model. Only goal models return score matrices; H/D/A classifiers do
not fabricate scorelines. The 0–6 display states its omitted probability mass.
The ML pilot contains only 40 final fixtures. Model Lab exposes full model cards,
including poor results and limitations, with comparisons confined to each experiment.

Run from the repository root. Layout uses responsive Streamlit columns and wide
tables with horizontal scrolling. Tests cover navigation with absent data, synthetic
competition filters, player/scouting interaction, corrupt files and probability mass.

## Advanced player comparison

Run `python scripts/build_advanced_players.py --download` to add 2024/25, 2025/26
and partial 2026/27 profiles. Player Explorer, Scouting Finder and Teams then expose
role-specific radar overlays (up to five players), cross-season comparisons,
selectable scatterplot axes and dark Total/Per-90 performance panels. The 2023/24
view remains the original limited M4 profile. See [ADVANCED_PLAYERS.md](ADVANCED_PLAYERS.md).

## Current refresh

Run `python scripts/refresh_current_data.py` for an immediate update. The supplied
GitHub Actions workflow and Windows Task Scheduler installer automate this outside
the web process. Streamlit never fits models or downloads datasets on a button click.
The interactive data requests are all explicit buttons in FPL Decision Lab: the
public-team and mini-league lookups, the news update from the official player feed and
the headline search for one player. They remain in session memory and send no
credentials. The only file
the app writes is the user's own bet log, `data/user/bets.csv` (see
[BETTING_LAB.md](BETTING_LAB.md)). See
[CURRENT_REFRESH.md](CURRENT_REFRESH.md).
FPL Decision Lab uses separately prepared current-season projections. One planning
horizon in its header drives four tabs: This gameweek, Players, Fixtures and My team &
transfers. My team shows the recommended legal XI on a pitch, a transfer planner and the
constrained transfer/chip optimizer, which runs only when requested. The rolling
backtest is the fourth Model Lab experiment. See [FPL_LAB.md](FPL_LAB.md) for the
projection formula and limits.

## Matchday Hub and visual system

The default page is the Matchday Hub. It shows the next gameweek's fixtures with
home/draw/away probabilities from the live goal model (Dixon–Coles, recent weighting),
a match preview with a 0–5 scoreline grid, the table with xGD and last-five form, the
latest results with xG, next-gameweek FPL projections and an availability watch.
It reads `data/processed/fpl/{standings,fixtures,results,players}.parquet` and the
live match snapshot; `results.parquet` is written by `refresh_current_data.py`.
FPL club names are matched to model club names by `data.fpl.normalise_club`.

Navigation is grouped (Matchday, Clubs & players, Fantasy, Models & data). The former
Overview counts live on the Data Coverage page. All pages share one dark theme:
`.streamlit/config.toml` sets Streamlit colours and `pl_analytics.dashboard.theme`
holds the palette, the default Plotly template and small HTML components (club chips,
form pills, probability bars, stat cards). Clubs are shown as colour-coded
three-letter chips; no crests or league marks are used.

Every page follows the same layout: a row of stat cards for context, one compact filter
row, then tabs or cards for the content.

- **League Table** – summary cards, the full table with xG columns and form, a "Season
  outlook" tab and a "Points vs expected" tab with diverging bars. The outlook comes from
  `statistics.season.simulate_season`: 10,000 seasons of the remaining fixtures drawn from
  the live goal model's score matrices, written to `season_simulation.parquet` by the
  refresh. It shows projected points with a 90% range, title, top-four and relegation
  probabilities and a finishing-position heatmap. Goal rates are held fixed for the whole
  season, so the probabilities are overconfident, and the simulator has no backtest yet.
- **Match Centre** – one upcoming fixture from the next five gameweeks: the live model's
  forecast, each club's position, record, xG per match, season outlook and last five
  results, expected goals by match, and each side's likeliest scorers and availability
  flags. A fixture the model cannot price still shows everything but the forecast.
- **Match Predictor** – the forecast follows the club and model selectors (inference only,
  no button): outcome tiles, probability bar, derived markets and the 0–6 score grid.
  `dashboard.components` holds these pieces.
- **Teams** – filters, squad summary cards, then "Squad statistics" and "Compare players"
  (radar and scatter side by side).
- **Player Explorer / Scouting Finder** – filters and a player header, then Profile, Peers
  and Season statistics tabs; Scouting Finder adds Similar players first. The page opens on
  the role's most-used player, and the peer minutes threshold starts at half the most-played
  total until a season reaches 900 minutes.
- **Market Value** – "Player review" (values and an interval bar) and "All reviews".
- **Model Lab** – headline cards and a bar chart of the experiment's primary metric above
  the full metric table; model cards open on demand.
- **Data Coverage** – historical module counts and the freshness of current data.
- **Betting Lab** – fair odds from the live goal model, a price checker for typed-in
  bookmaker odds and stake-tax/staking calculators, under a standing note that bookmakers
  have been more accurate than the model. The benchmark is the fifth Model Lab experiment.
  See [BETTING_LAB.md](BETTING_LAB.md).
