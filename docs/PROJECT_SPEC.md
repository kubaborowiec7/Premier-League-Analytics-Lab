# Project specification

## Working title

**Premier League Analytics Lab: Player Valuation, Scouting & Match Prediction**

## Portfolio objective

Create a single coherent project that demonstrates the full analytics lifecycle.

**V1 analysis scope is Premier League only, but the system must be competition-agnostic. No analytical pipeline may depend on Premier League being the only possible competition.**

The full lifecycle:

**data acquisition → storage → SQL transformation → statistical analysis → ML → explainability → evaluation → dashboard → documentation → deployment**

The project should be understandable to:
- a data analyst,
- BI/data team,
- data scientist,
- technical recruiter.

## Product modules

### A. Player Performance Explorer

Purpose: compare players fairly within role/position.

Core outputs:
- per-90 metrics,
- position-adjusted percentiles,
- age and minutes context,
- reliability indicator,
- radar/profile visualization,
- league/position ranking.

Advanced outputs:
- possession-adjusted defensive metrics where source data permits,
- shrinkage for low-minute samples,
- principal components,
- player archetype clusters,
- nearest-neighbour similarity.

### B. Market Value Engine

Target:
- `market_value_eur` at a clearly defined valuation date.

Questions:
- how much of market value can measurable performance explain?
- which players have the largest positive/negative valuation residual?
- how much does age drive value?
- where does the model become unreliable?

Outputs:
- predicted market value,
- prediction interval,
- observed market value,
- absolute and relative residual,
- SHAP drivers,
- model diagnostics.

Important:
Market value is an **editorial estimate**, not the same as an actual transfer fee. State this throughout the project.

### C. Scouting Finder

Example filters:
- age <= 23,
- minimum minutes,
- position,
- performance percentile >= X,
- estimated value <= Y,
- undervaluation score >= Z.

Similarity:
- select a player,
- standardize role-specific features,
- rank nearest neighbours by cosine distance / kNN.

### D. Match Prediction Engine

Targets:
- `P(Home Win)`,
- `P(Draw)`,
- `P(Away Win)`,
- expected home goals,
- expected away goals,
- scoreline probability matrix.

Candidate features:
- pre-match Elo,
- rolling goals/xG where legally/reproducibly available,
- shots / shots on target,
- rolling points,
- home advantage,
- rest days,
- rolling attack/defence strength,
- team squad value summaries where historically aligned,
- promoted-team indicator,
- season phase.

### E. Statistical Model Lab

Compare statistical and ML approaches:
- Elo,
- Poisson,
- Dixon-Coles,
- multinomial logistic regression,
- gradient boosting.

Focus on:
- out-of-sample performance,
- calibration,
- temporal stability,
- uncertainty.

---

## MVP

The MVP is complete when the repository can:

1. ingest at least one historical match dataset and one player/value dataset;
2. load normalized data to PostgreSQL;
3. build leakage-safe rolling match features;
4. compute Elo ratings;
5. train a Poisson/Dixon-Coles baseline;
6. train one ML match model;
7. build player season features;
8. train baseline + boosted market-value models;
9. calculate player valuation residuals;
10. expose results in a Streamlit app;
11. reproduce tests and CI from a clean checkout.

---

## Stretch goals

- Bayesian hierarchical model for player performance / team strength,
- conformal prediction intervals,
- survival/time-to-transfer analysis,
- injury-adjusted availability,
- network analysis of passing/event data,
- event-level expected-threat style analysis using open event data,
- scheduled refresh workflow,
- dbt transformations,
- MLflow experiment tracking,
- public API layer via FastAPI.

---

## Non-goals for the initial version

- real-time betting recommendations,
- live in-play predictions,
- scraping websites in ways that violate terms,
- building a full React frontend before the analytics pipeline is stable,
- claiming causal effects from observational correlations.


## Future multi-league scouting

V1 candidate universe:
- Premier League only.

Architecture requirement from day one:
- allow additional competitions without rewriting core analytics logic.

Future use case:
- choose a Premier League reference player;
- search similar players across selected leagues;
- normalize features within appropriate competition/season/position contexts;
- optionally adjust for league strength;
- return comparable candidates with similarity, age, playing time and valuation context.

The future expansion should be mostly:
1. add a source/competition mapping,
2. ingest the competition,
3. run the same canonical transformations,
4. enable the competition in configuration/UI.

It should **not** require duplicating the Premier League pipeline.
