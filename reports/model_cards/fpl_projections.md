# FPL projection model card

**Version:** `recency-events-elo-v2`  
**Evaluation date:** 2026-09-30

## Purpose

Rank selectable players and support transparent Fantasy Premier League line-up,
captaincy, transfer and chip scenarios over the next one, three and five gameweeks.
Outputs are decision support, not guaranteed points or automated account actions.

## Data and scope

- Competition: configured `EPL`; interfaces preserve competition and season context.
- Season: partial 2026/27 through GW5 for the current evaluation.
- Inputs: commit-pinned FPL Core player totals, match appearances, teams and fixtures.
- Availability: published FPL status and chance-of-playing fields where populated.

## Targets and features

The minutes layer predicts expected minutes plus `P(start)`, `P(appearance)` and
`P(60+)`. It uses six prior fixtures, exponential decay 0.65, position priors and
minutes conditional on starting or entering as a substitute.

The points layer adds expectations for appearance points, goals, assists, clean sheets,
saves, penalty saves, defensive contributions, bonus, cards, own goals, missed penalties
and goals conceded. Fixture attack and clean-sheet context use pre-refresh club Elo.

## Evaluation

GW2–GW5 are evaluated chronologically. Every target uses only earlier cumulative
snapshots. Players without senior minutes before the origin are excluded before the
target is observed.

| Metric | Result |
|---|---:|
| Eligible player-gameweeks | 1,467 |
| xPts MAE | 1.83 |
| Position baseline MAE | 2.08 |
| xPts RMSE | 2.75 |
| Mean weekly Spearman | 0.53 |
| xMins MAE | 20.4 minutes |
| Mean predicted / actual points | 2.24 / 2.40 |
| Zero-point rate | 28.0% |

Four early-season gameweeks do not establish stable future performance or calibration.

## Leakage controls

- Appearance weights use completed fixtures at or before the forecast origin.
- Cumulative minutes and starts are differenced only through the prior snapshot.
- Eligibility is based on prior minutes, never target participation.
- Future fixtures provide schedule context but no future outcomes.

## Interpretation

xMins is an average over playing-time scenarios, not a claim that a player will be
substituted at that exact minute. xPts is the mean of scoring-event expectations. The
component table shows why the total changes. The 80% display range is indicative and
is not a calibrated prediction interval.

## Limitations

- No press-conference, predicted-line-up or licensed injury-news feed.
- No constraint linking teammates competing for the same starting role.
- Elo fixture probabilities are simpler than bookmaker or event-data models.
- Goal, assist, clean-sheet and bonus events are not jointly simulated.
- Save and bonus rates are unstable after few matches.
- Current-price squad optimization does not know manager selling prices.
- Backtest covers only four early gameweeks and one competition-season.
