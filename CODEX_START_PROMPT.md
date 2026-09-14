# First prompt to paste into Codex

You are starting a new portfolio-grade project called **Premier League Analytics Lab**.

First, read all existing repository documentation, especially:

- `AGENTS.md`
- `README.md`
- `docs/PROJECT_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_SOURCES.md`
- `docs/STATISTICS.md`
- `docs/MILESTONES.md`
- `docs/GITHUB_WORKFLOW.md`
- `sql/schema.sql`
- `pyproject.toml`

## Goal

Build an analytics platform combining:

1. advanced player performance analytics,
2. market-value prediction and undervalued/overvalued player detection,
3. Premier League match outcome and score prediction,
4. advanced statistical models such as Elo, Poisson and Dixon-Coles,
5. an interactive Streamlit dashboard.

The project must demonstrate strong Python, SQL, statistics, ML, testing, Git/GitHub and data-engineering practice.

## Important constraints

- Do not scrape Transfermarkt or FBref as part of the core pipeline.
- Do not hard-code Premier League into reusable code. V1 sets `ACTIVE_COMPETITIONS=EPL`, but interfaces, schema, ingestion and scouting must support future additional competitions.
- Use reproducible public datasets/APIs described in `docs/DATA_SOURCES.md`.
- Avoid data leakage.
- Use chronological splits for time-dependent models.
- Keep raw data immutable.
- Keep notebooks exploratory; reusable code belongs under `src/pl_analytics/`.
- Never commit secrets or large raw datasets.
- Follow `AGENTS.md`.

## Your first task: implement M0 only

Implement **M0 — Repository Foundation** from `docs/MILESTONES.md`.

Specifically:

1. Inspect the starter structure and identify anything missing for a clean Python package.
2. Make the package importable.
3. Implement settings/configuration loading from environment variables, including a generic `ACTIVE_COMPETITIONS` setting whose V1 value is `EPL`.
4. Ensure PostgreSQL Docker Compose configuration is valid.
5. Add a database connection smoke utility without requiring it during import.
6. Add/adjust tests so `pytest` works without external data.
7. Add Ruff configuration if needed.
8. Add a minimal Streamlit landing page that explains the project and safely reports that data/model artifacts are not built yet.
9. Add GitHub Actions CI that installs the project and runs Ruff + pytest.
10. Update README/docs only where implementation details changed.

## Verification

Run:

```bash
python -m compileall src
ruff check .
pytest
```

Do not proceed to M1.

At the end, return:
- a concise summary of changes,
- files created/modified,
- verification commands and results,
- any unresolved issue,
- the recommended Git commit message.

Recommended commit message:

`chore: initialize analytics project foundation`
