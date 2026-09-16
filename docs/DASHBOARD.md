# Dashboard

M8 provides six read-only pages: Overview, Player Explorer, Scouting Finder,
Market Value, Match Predictor and Model Lab. Competition options are populated
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

The app caches tables/reports by path and content hash, and trusted model
bundles by checksum and modification time. It never fits, downloads or connects
to PostgreSQL at startup or during prediction. Models must come from this project's
trusted local build: pickle/joblib is executable and must never be user-uploaded.
Missing modules remain usable as explanatory empty states; malformed artifacts
produce sanitized rebuild messages without connection strings or private content.

Player statistics are limited to recorded goals, assists and cards; positions
remain snapshot-based. Similarity holds season, date and position context fixed,
with an explicit candidate competition universe. Valuations show observed editorial
values separately from model estimates and nominal 90% intervals. Fragile rows
can be revealed explicitly; all rows remain in Model Lab evaluation metrics.

Match predictions are hypothetical fixtures at the displayed frozen origin,
not live forecasts. Only goal models return score matrices; H/D/A classifiers do
not fabricate scorelines. The 0–6 display states its omitted probability mass.
The ML pilot contains only 40 final fixtures. Model Lab exposes full model cards,
including poor results and limitations, with comparisons confined to each experiment.

Run from the repository root. Layout uses responsive Streamlit columns and wide
tables with horizontal scrolling. Tests cover navigation with absent data, synthetic
competition filters, player/scouting interaction, corrupt files and probability mass.
