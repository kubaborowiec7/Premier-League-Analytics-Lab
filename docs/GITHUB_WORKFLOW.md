# GitHub workflow

The repository history is part of the portfolio.

## Branching

For a solo project use lightweight branches:

- `main` — stable
- `data/<topic>`
- `feat/<topic>`
- `model/<topic>`
- `docs/<topic>`
- `fix/<topic>`

Examples:
- `data/football-results-ingestion`
- `feat/player-similarity`
- `model/dixon-coles`
- `model/player-valuation`

## Commit style

Use Conventional-Commit-like prefixes:

```text
chore: initialize project tooling
data: ingest premier league match results
feat: calculate rolling team features
model: add dixon coles backtest
test: add leakage checks
docs: document player valuation methodology
fix: exclude current match from rolling window
```

Avoid:
- `update`
- `final`
- `stuff`
- one giant commit containing the whole project.

## GitHub Issues

Create one Issue for each milestone and smaller Issues for meaningful subproblems.

Suggested labels:
- `data`
- `sql`
- `statistics`
- `model`
- `dashboard`
- `documentation`
- `bug`
- `enhancement`
- `data-quality`

## Pull requests

Even when working solo, PRs can document important changes.

PR template:
- What changed?
- Why?
- Data impact?
- Leakage risk?
- Tests run?
- Screenshots/results?
- Docs updated?
- Known limitations?

## Release checkpoints

Recommended:
- `v0.1.0` — ingestion works
- `v0.2.0` — player analytics
- `v0.3.0` — market value model
- `v0.4.0` — match models
- `v0.5.0` — dashboard
- `v1.0.0` — portfolio release

## README progress section

Keep a simple milestone table:

| Milestone | Status |
|---|---|
| M0 Foundation | ✅ |
| M1 Data | 🚧 |
| M2 SQL | ⬜ |

Do not claim a milestone complete until tests/docs satisfy its definition of done.

## GitHub Actions

CI should run on:
- push,
- pull request.

Minimum:
- install,
- Ruff,
- pytest.

Later:
- coverage threshold,
- type checking,
- SQL validation,
- Docker build.

## Repository presentation

Before v1.0:
- add 3–5 high-quality screenshots,
- include an architecture diagram,
- add a clear 30-second project summary,
- show model metrics with chronological test dates,
- clearly state data source limitations,
- include setup instructions that work from a clean clone.
