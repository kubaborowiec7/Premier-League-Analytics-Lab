# Descriptive player PCA — M4

Version/date: M4 initial implementation, 2026-09-15.

Purpose: summarize variation in four recorded count rates among eligible strikers.
This is unsupervised dimensional reduction, with no prediction target or quality label.

Data: pinned published games, appearances and profiles in
`data/manifests/m4_sources.json`; EPL 2023/24, strictly before 2024-07-01.
The example uses 47 complete striker records with at least 450 minutes.
Snapshot positions are historically unverified; the artifacts are retrospective.

Features: competition/season/position/context z-scores of goals, assists, yellow cards
and red cards per 90. No market-value input. Constant or missing features require an
explicit decision. The example retains all four features.

Fit: StandardScaler followed by full-SVD PCA with two components. Preprocessing is
fit on the explicitly provided cohort; transform does not refit it and rejects dates
earlier than the fitted reference date. The whole descriptive cohort is used; there
is no train/test split and no out-of-sample predictive claim.

Metrics: explained variance ratios approximately 0.2873 and 0.2825, or 56.98% combined.
These measure retained sample variance, not accuracy or evidence of player quality.
Loadings, scaler parameters and feature order are saved in `artifacts/m4/profile.json`.

Interpretation/limitations: signs are arbitrary; components are mathematical directions.
Rare cards can materially influence components. Goals/assists/cards omit most football
behavior, and position labels may reflect later roles. No causal, scouting-equivalence,
league-strength or future-performance interpretation is validated. Similarity uses the
full declared standardized feature space rather than these two components.

Validation: synthetic tests cover feature allowlists, minimum/complete cohorts, frozen
transforms and temporal rejection. CI reproduces the pinned real-data profile.
See `docs/PLAYER_ANALYTICS.md` for source boundaries, exposure rules and uncertainty.
