# Current data refresh

Current match results and advanced player profiles are refreshed outside Streamlit:

```bash
python scripts/refresh_current_data.py
```

The command resolves the latest published FPL Core commit, archives every downloaded
CSV by SHA-256, downloads the mutable current-season Football-Data CSV into the same
immutable store, and validates both sources. It includes only completed fixtures and
appearances. Scheduled gameweeks published in advance do not enter player totals.

The command then publishes two independent outputs:

- current multi-season player profiles and a metric-coverage report;
- a versioned match-inference snapshot fitted using results strictly before the UTC
  origin, plus the frozen M7 classifier and past-only team features.

The dashboard switches to a new match snapshot only after its bundles, club names and
team states have been written. `live_catalog.json` is the final pointer and contains
the origin, last result date, source hash and bundle hashes. The evaluated M6/M7
reports remain frozen; an operational refit is not presented as a new evaluation.

## Automatic execution

GitHub Actions runs `.github/workflows/refresh-current-data.yml` daily at 05:15 UTC
and supports a manual run. It uploads the prepared bundle for 14 days. Scheduled
workflows run only after the workflow is present on the repository's default branch;
an uploaded Actions artifact does not automatically modify a local checkout.

For the local dashboard on Windows, register the supplied Task Scheduler job once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_refresh_task.ps1
```

The default is 07:15 local time, daily, with `StartWhenAvailable` for a missed run.
The computer and network must be available. Override the time with `-DailyTime HH:mm`.
The task uses the repository `.venv` and working directory. Re-running the installer
updates the same task.

There is deliberately no dashboard refresh button. A web request never downloads
data, writes raw archives or fits models, which keeps the UI responsive and prevents
concurrent visitors from creating inconsistent artifacts. To force an immediate
local refresh, run the Python command above and reload the page.

## Provenance and limitations

- Raw payloads are never overwritten and remain ignored by Git.
- Historical pinned seasons retain their expected-row checks.
- The frozen M7 current-season URL is mutable. A clean reproduction may use newer
  bytes only when the rows at its frozen cutoff still match both the expected count
  and a canonical result-fact hash; a locally archived original snapshot remains
  preferred.
- Each current player refresh resolves all files to one Git commit. Season-level card
  totals use the latest published `gw` row for each player.
- Progressive passes, progressive carries and forward-pass completion remain absent;
  the refresh does not infer them from unrelated fields.
- Source publication can lag the final whistle. “Results through” in Match Predictor
  and `coverage.json` are the authoritative freshness indicators.
- A failed validation leaves the previous dashboard pointer and prepared player file
  in place. The command exits non-zero so Task Scheduler or Actions records failure.

