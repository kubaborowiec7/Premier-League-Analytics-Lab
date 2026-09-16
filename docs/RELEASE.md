# v1.0.0 release preparation

Release scope: M0–M10 portfolio implementation, with local demo only as requested.
No public hosting or social publication is included. The milestone pull requests
remain stacked for review; a release tag does not imply that they were merged into main.

## Included

- Validated settings, optional PostgreSQL connection check and Python packaging.
- Provenance-aware ingestion, PostgreSQL analytics, EDA and leakage audits.
- Player profiles, similarity, shrinkage and bootstrap uncertainty.
- Chronological value and match experiments with frozen selection, calibration,
  benchmarks and model cards.
- Six-page Streamlit UI, cached artifacts, graceful missing states and frozen inference.
- Non-root deployment configuration, coverage gate and security review.
- Portfolio README, actual screenshots and CV/LinkedIn draft.

## Verification and limitations

Local compileall, Ruff, pip check and Compose configuration validation pass.
144 tests pass; 13 opt-in PostgreSQL tests skip without a local server; statement
coverage is 87.05%. GitHub Actions supplies real PostgreSQL and container execution.
M9 full CI passed on `4e21499` (run 35071950804), including the container and
PostgreSQL jobs. Release tagging waits for successful CI for the release commit.

The current-season CSV URL is mutable; preserve the exact pinned raw archive for
reproduction. Editorial market values, incomplete historical membership/roles and
missing publication vintages limit causal or real-time interpretations. The small ML
holdout and unstable low-minute valuation estimates are research limitations, not
production-ready claims. See the three model cards and open data-quality issues.

## Screenshot reproduction

Prepare the artifacts, start the local app, then:

```bash
python -m pip install -e ".[screenshots]"
python -m playwright install chromium
python scripts/capture_dashboard.py
```

On Windows with Edge installed, use `python scripts/capture_dashboard.py --channel msedge`
instead of downloading Chromium. The script opens a fresh headless browser, captures
three real populated pages and closes it. The player screenshot intentionally selects
the documented 2023/24 example, Erling Haaland; this display-only choice is not a
competition assumption in reusable analytics. Run from the repository root.
