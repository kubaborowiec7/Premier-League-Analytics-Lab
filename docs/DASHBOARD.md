# Dashboard

M8 and later extensions provide eight read-only pages: Overview, Player Explorer,
Scouting Finder, Teams, FPL Decision Lab, Market Value, Match Predictor and Model Lab. Competition options are populated
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
the web process. Streamlit remains read-only and never downloads or fits on a button
click. See [CURRENT_REFRESH.md](CURRENT_REFRESH.md).
FPL Decision Lab uses separately prepared current-season projections. Its five tabs
cover player picks, fixtures, transfers, captaincy and an optional public squad. See
[FPL_LAB.md](FPL_LAB.md) for the projection formula and limits.
