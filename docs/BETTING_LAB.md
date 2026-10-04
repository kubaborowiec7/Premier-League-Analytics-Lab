# Betting Lab

The Betting Lab prices fixtures with the live goal model and helps check a bookmaker's
price. It does not download odds and does not recommend bets. Its standing warning comes
from the benchmark below: the bookmakers have been more accurate than the model, so a
positive expected value on this page is a disagreement with the price, not an edge.

## Page

- **Header** – how many next-gameweek fixtures are priced, the model-versus-bookmaker log
  loss gap, and the stake tax (default 12%, editable).
- **Fair odds** – every fixture's fair odds for home, draw, away, over/under 2.5 and both
  teams to score; then, for one fixture, match result, double chance, total goals
  (1.5/2.5/3.5), both teams to score, Asian handicap half-lines and the likeliest scores.
- **Price checker** – type one bookmaker price to see the odds after tax, the break-even
  probability and the expected value against the model; or type the three match-result
  prices to see the bookmaker's margin and margin-free probabilities beside the model's.
- **Player markets** – each player's probability of scoring and of assisting in the next
  fixture, with fair odds and a price check for a typed-in price.
- **Calculators** – a stake-tax calculator and a fractional-Kelly staking calculator with
  a cap on the share of bankroll. The staking calculator takes the user's own probability.

- **Bet log** – record a bet (event, market, selection, odds, stake), then set its result
  and, optionally, the closing odds. Shows profit, return on stakes, win rate and
  closing-line value, with a cumulative profit chart.

The bet log is the one file the dashboard writes: `data/user/bets.csv`, ignored by Git.
It holds only what the user typed, is validated before every save and is replaced
atomically (`data.bet_log`). A win returns `stake × (1 − tax) × odds`, a void bet returns
the stake, and closing-line value is `odds taken / closing odds − 1`, both before tax.

Fixtures come from `data/processed/fpl/fixtures.parquet`; probabilities come from the
`dixon_coles_half365` model in the live snapshot, the same forecasts shown on the
Matchday Hub. Only markets that settle as win or lose are priced, so every probability
converts directly to fair odds (handicap and totals use half-goal lines).

The fixture markets also include clean sheets for each side, read from the same score
matrix.

## Player markets

`features.fpl.scorer_markets` treats a player's goals and assists in a fixture as Poisson
with the projected per-90 rate (see [FPL_LAB.md](FPL_LAB.md)) scaled by the fixture's
attack multiplier. Minutes are a three-way mixture — starts, comes on for about 20
minutes, or is not used — because the chance of scoring is not linear in minutes.

The FPL rolling backtest scores the scorer probability on gameweeks it had not seen.
Pilot on GW2–GW5 of 2026/27 (1,467 player-gameweeks): mean probability 7.3% against 6.2%
who scored; Brier 0.0524 against 0.0563 for each position's own scoring rate in the same
sample. Players given 20–30% scored 21% of the time, and those given 30–45% scored 33%.
The model therefore runs slightly high. There are no bookmaker scorer prices in the
data, so this market has not been compared with bookmakers, and assists have no backtest.
Predicted line-ups, penalty-taker changes and late injuries are not modelled.

## Stake tax

`tax` is a stake tax: the bookmaker deducts that share of the stake before applying the
odds, so a win returns `stake × (1 − tax) × odds`. Polish licensed bookmakers apply 12%.
Effective odds are `odds × (1 − tax)`, break-even probability is `1 / effective odds`,
and expected value per unit handed over is `p × effective odds − 1`. With a typical 5%
margin and the 12% tax, an average bettor gets back about 84% of stakes.

## Model versus bookmakers

```bash
python scripts/build_market_benchmark.py
```

The script reads the rolling-origin predictions already written by the M6 and M7
pipelines and the odds columns of the same archived Football-Data.co.uk files
(`data.market_odds`), and writes `artifacts/market/evaluation.json`. Nothing is fitted
or downloaded. The frozen results adapter still ignores odds; no frozen model uses them
as a feature. Model Lab shows the result as **Match models vs bookmakers**.

Odds become probabilities by removing the margin proportionally. Models and the market
average are scored on the matches that have average closing odds; Pinnacle is absent
from the 2026/27 file and is scored on the matches it priced.

Result on 1,180 matches, 2023/24–2026/27 (built 2026-10-04):

| Source | Log loss |
| --- | --- |
| Pinnacle, closing (970 matches) | 0.945 |
| Bookmaker average, closing | 0.963 |
| Bookmaker average, opening | 0.968 |
| Dixon–Coles, 365-day half-life | 0.983 |
| Base rate | 1.076 |

The average closing margin on the match-result market is 4.7%. Backing every selection
the model priced at least 5% above the average closing odds would have returned −6.9% of
stakes (95% interval −17.3% to +4.4%) and −18.0% after a 12% stake tax, over 1,048 bets.
At the best available closing price the same rule returned −3.5% untaxed.

### Blending the model with the odds

`statistics.betting.market_blend` tests whether the model adds information to the odds.
It pools the two with a log-linear weight (`model^w × market^(1 − w)`), fits `w` on the
model-selection seasons (2023/24–2024/25, 760 matches) and scores the blend on the later
ones (420 matches). Odds enter only this secondary combination; the match model itself
still uses no odds, as [DATA_SOURCES.md](DATA_SOURCES.md) requires.

Result: the fitted weight on the model is **0** for both opening and closing average
odds, so the blend is the odds. On the test matches the odds alone score 1.018 (opening)
and 1.015 (closing) against 1.025 for the model. On this evidence the model carries no
information that the odds do not already contain, and no blended probability is offered
in the price checker. A model that used inputs the market prices slowly (line-ups,
injuries, xG-based strength) would have to be a new experiment with a fresh holdout.

## Limitations

- 2023/24 and 2024/25 were the model-selection seasons, so the model is slightly
  flattered there; 2025/26 and the 40 matches of 2026/27 are final-test periods.
- The odds are from international bookmakers. Polish bookmakers' prices are not in this
  data and are typically lower because of the stake tax.
- The simulation uses closing prices, flat stakes and the match-result market only.
- Model prices ignore line-ups, injuries and news; closing odds do not.
- No odds are stored for upcoming fixtures; closing odds in the bet log are typed by hand.
- This is a research and learning tool for adults. Betting involves a real risk of
  losing money.
