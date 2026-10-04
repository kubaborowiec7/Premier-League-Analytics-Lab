"""Odds arithmetic, markets derived from a score matrix and a model-versus-market benchmark.

Decimal odds throughout. ``tax`` is a stake tax: the bookmaker deducts that share of the
stake before applying the odds, so a win returns ``stake * (1 - tax) * odds``. Poland's
licensed bookmakers apply 12%. Nothing here recommends a bet; it prices and compares.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pl_analytics.statistics.matches import probability_metrics

OUTCOME_COLUMNS = ("p_home", "p_draw", "p_away")
SIDES = ("home", "draw", "away")


def _check(probability: float, odds: float, tax: float) -> None:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must lie in [0, 1]")
    if not np.isfinite(odds) or odds <= 1:
        raise ValueError("Decimal odds must be above 1")
    if not 0 <= tax < 1:
        raise ValueError("Stake tax must lie in [0, 1)")


def fair_odds(probability: float) -> float:
    """Decimal odds with no margin; an impossible event has no price."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must lie in [0, 1]")
    return float("inf") if probability == 0 else 1 / probability


def effective_odds(odds: float, tax: float = 0.0) -> float:
    """Return per unit actually handed over, after the stake tax."""
    _check(0.5, odds, tax)
    return odds * (1 - tax)


def break_even_probability(odds: float, tax: float = 0.0) -> float:
    """Win probability at which the bet has zero expected value."""
    return min(1.0, 1 / effective_odds(odds, tax))


def expected_value(probability: float, odds: float, tax: float = 0.0) -> float:
    """Expected profit per unit handed over; -0.05 means losing 5% of stakes on average."""
    _check(probability, odds, tax)
    return probability * effective_odds(odds, tax) - 1


def kelly_fraction(probability: float, odds: float, tax: float = 0.0) -> float:
    """Full-Kelly share of bankroll; zero whenever the bet has no positive expected value."""
    _check(probability, odds, tax)
    net = effective_odds(odds, tax) - 1
    if net <= 0:
        return 0.0
    return float(np.clip((probability * (net + 1) - 1) / net, 0.0, 1.0))


def implied_probabilities(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise each row of odds to probabilities and return the bookmaker margin."""
    prices = np.asarray(odds, dtype=float)
    if prices.ndim == 1:
        prices = prices[None, :]
    if not np.isfinite(prices).all() or (prices <= 1).any():
        raise ValueError("Decimal odds must be finite and above 1")
    raw = 1 / prices
    total = raw.sum(axis=1, keepdims=True)
    return raw / total, total[:, 0] - 1


def markets_from_matrix(matrix: np.ndarray, *, scores: int = 8) -> pd.DataFrame:
    """Price common markets from a score matrix (rows home goals, columns away goals).

    Only markets that settle as win or lose are included, so every probability converts
    directly to fair odds: handicap and totals lines are half-goal lines.
    """
    grid = np.asarray(matrix, dtype=float)
    if grid.ndim != 2 or not np.isclose(grid.sum(), 1, atol=1e-6) or (grid < 0).any():
        raise ValueError("A normalised score matrix is required")
    home, away = np.indices(grid.shape)
    margin, total = home - away, home + away
    win, draw, lose = (
        float(grid[margin > 0].sum()),
        float(grid[margin == 0].sum()),
        float(grid[margin < 0].sum()),
    )
    rows = [
        ("Match result", "Home", win),
        ("Match result", "Draw", draw),
        ("Match result", "Away", lose),
        ("Double chance", "Home or draw", win + draw),
        ("Double chance", "Home or away", win + lose),
        ("Double chance", "Draw or away", draw + lose),
    ]
    for line in (1.5, 2.5, 3.5):
        over = float(grid[total > line].sum())
        rows += [("Total goals", f"Over {line}", over), ("Total goals", f"Under {line}", 1 - over)]
    both = float(grid[(home > 0) & (away > 0)].sum())
    rows += [("Both teams to score", "Yes", both), ("Both teams to score", "No", 1 - both)]
    home_clean, away_clean = float(grid[away == 0].sum()), float(grid[home == 0].sum())
    rows += [
        ("Clean sheet", "Home keeps one", home_clean),
        ("Clean sheet", "Home concedes", 1 - home_clean),
        ("Clean sheet", "Away keeps one", away_clean),
        ("Clean sheet", "Away concedes", 1 - away_clean),
    ]
    for line in (-1.5, -0.5, 0.5, 1.5):
        covers = float(grid[margin + line > 0].sum())
        rows += [
            ("Asian handicap", f"Home {line:+.1f}", covers),
            ("Asian handicap", f"Away {-line:+.1f}", 1 - covers),
        ]
    flat = np.argsort(grid, axis=None)[::-1][:scores]
    for index in flat:
        h, a = np.unravel_index(int(index), grid.shape)
        rows.append(("Correct score", f"{h}–{a}", float(grid[h, a])))
    result = pd.DataFrame(rows, columns=["market", "selection", "probability"])
    result["probability"] = result.probability.clip(0, 1)
    result["fair_odds"] = [fair_odds(value) for value in result.probability]
    return result


def settle_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Add each bet's return, profit and closing-line value.

    A win returns ``stake × (1 − tax) × odds``; a void bet returns the stake; an open bet
    has no profit yet. Closing-line value compares the price taken with the closing price,
    both before tax: positive means the bet was placed at better odds than the close.
    """
    result = bets.copy()
    won, void, lost = result.result.eq("won"), result.result.eq("void"), result.result.eq("lost")
    returned = pd.Series(np.nan, index=result.index, dtype=float)
    returned[won] = result.stake[won] * (1 - result.tax[won]) * result.odds[won]
    returned[void] = result.stake[void]
    returned[lost] = 0.0
    result["returned"] = returned
    result["profit"] = returned - result.stake
    result["closing_line_value"] = result.odds / result.closing_odds - 1
    return result


def bet_summary(bets: pd.DataFrame) -> dict:
    """Totals over settled bets; rates are missing rather than zero when undefined."""
    settled = settle_bets(bets)
    done = settled.loc[settled.result.isin(["won", "lost", "void"])]
    decided = done.loc[done.result.isin(["won", "lost"])]
    staked = float(done.stake.sum())
    priced = settled.closing_line_value.dropna()
    return {
        "bets": int(len(settled)),
        "open": int(settled.result.eq("open").sum()),
        "settled": int(len(done)),
        "staked": staked,
        "profit": float(done.profit.sum()),
        "return_on_stakes": float(done.profit.sum() / staked) if staked else None,
        "win_rate": float(decided.result.eq("won").mean()) if len(decided) else None,
        "closing_line_value": float(priced.mean()) if len(priced) else None,
        "beat_the_close": float(priced.gt(0).mean()) if len(priced) else None,
        "closing_prices": int(len(priced)),
    }


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return np.where(
        frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2)
    )


def value_simulation(
    frame: pd.DataFrame,
    *,
    book: str,
    threshold: float,
    tax: float,
    seed: int = 42,
    resamples: int = 2000,
) -> dict:
    """Flat one-unit bets wherever the model's expected value at ``book`` beats ``threshold``.

    The expected value used to pick bets ignores the tax (the model's view of the price);
    the profit applies it. The interval resamples whole matches. This is a historical
    what-if on closing prices, not a forecast of future returns.
    """
    probabilities = frame[list(OUTCOME_COLUMNS)].to_numpy(float)
    prices = frame[[f"{book}_{side}" for side in SIDES]].to_numpy(float)
    valid = np.isfinite(prices).all(axis=1)
    probabilities, prices = probabilities[valid], prices[valid]
    actual = _labels(frame.loc[valid])
    picked = probabilities * prices - 1 > threshold
    won = np.eye(3, dtype=bool)[actual]
    profit = np.where(picked, np.where(won, prices * (1 - tax) - 1, -1.0), 0.0)
    bets_per_match, profit_per_match = picked.sum(axis=1), profit.sum(axis=1)
    bets = int(bets_per_match.sum())
    result = {
        "book": book,
        "threshold": threshold,
        "tax": tax,
        "matches": int(valid.sum()),
        "bets": bets,
        "roi": float(profit_per_match.sum() / bets) if bets else None,
        "roi_lower_95": None,
        "roi_upper_95": None,
    }
    if bets:
        draws = np.random.default_rng(seed).integers(0, len(actual), (resamples, len(actual)))
        staked = bets_per_match[draws].sum(axis=1)
        returns = profit_per_match[draws].sum(axis=1)[staked > 0] / staked[staked > 0]
        result["roi_lower_95"], result["roi_upper_95"] = (
            float(value) for value in np.quantile(returns, [0.025, 0.975])
        )
    return result


def blend_probabilities(model: np.ndarray, market: np.ndarray, weight: float) -> np.ndarray:
    """Log-linear pool: ``model^weight × market^(1 − weight)``, renormalised per row.

    A weight of 0 returns the market's probabilities and 1 the model's.
    """
    if not 0 <= weight <= 1:
        raise ValueError("Blend weight must lie in [0, 1]")
    first, second = np.atleast_2d(model).astype(float), np.atleast_2d(market).astype(float)
    if first.shape != second.shape or (first <= 0).any() or (second <= 0).any():
        raise ValueError("Blending needs matching, strictly positive probabilities")
    pooled = np.exp(weight * np.log(first) + (1 - weight) * np.log(second))
    return pooled / pooled.sum(axis=1, keepdims=True)


def fit_blend_weight(model: np.ndarray, market: np.ndarray, actual: np.ndarray) -> float:
    """Weight on the model that minimises log loss on the supplied (training) matches."""
    rows = np.arange(len(actual))

    def loss(weight: float) -> float:
        return float(-np.log(blend_probabilities(model, market, weight)[rows, actual]).mean())

    fitted = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    # The optimiser never lands exactly on a bound; prefer the bound when it is as good.
    return float(min((0.0, 1.0, float(fitted.x)), key=loss))


def market_blend(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    train_seasons: tuple[str, ...],
    model: str = "dixon_coles_half365",
    books: tuple[str, ...] = ("avg_open", "avg_close"),
    seed: int = 42,
    resamples: int = 2000,
) -> dict:
    """Does the model add information to the odds? Fit a blend weight, then test it.

    The weight is fitted on ``train_seasons`` and scored on every other season, so the
    test never informs it. The interval for blend-minus-market log loss resamples test
    matches. Odds enter only this secondary combination, never the match model itself.
    """
    rows = predictions.loc[predictions.model.eq(model)].merge(
        odds.drop(columns=["competition_id", "season"], errors="ignore"),
        on="match_id",
        validate="one_to_one",
    )
    result = {}
    for book in books:
        columns = [f"{book}_{side}" for side in SIDES]
        if not set(columns) <= set(rows):
            continue
        priced = rows.loc[rows[columns].notna().all(axis=1)]
        train = priced.loc[priced.season.isin(train_seasons)]
        test = priced.loc[~priced.season.isin(train_seasons)]
        if train.empty or test.empty:
            continue

        def parts(frame: pd.DataFrame, columns: list[str] = columns) -> tuple:
            return (
                frame[list(OUTCOME_COLUMNS)].to_numpy(float),
                implied_probabilities(frame[columns].to_numpy(float))[0],
                _labels(frame),
            )

        weight = fit_blend_weight(*parts(train))
        own, market, actual = parts(test)
        index = np.arange(len(actual))
        blended = blend_probabilities(own, market, weight)
        gap = -np.log(blended[index, actual]) + np.log(market[index, actual])
        gap[np.abs(gap) < 1e-12] = 0.0  # a zero weight reproduces the market exactly
        draws = np.random.default_rng(seed).integers(0, len(gap), (resamples, len(gap)))
        lower, upper = np.quantile(gap[draws].mean(axis=1), [0.025, 0.975])
        result[book] = {
            "weight": weight,
            "train_matches": int(len(train)),
            "test_matches": int(len(test)),
            "model_log_loss": float(-np.log(own[index, actual]).mean()),
            "market_log_loss": float(-np.log(market[index, actual]).mean()),
            "blend_log_loss": float(-np.log(blended[index, actual]).mean()),
            "blend_minus_market": float(gap.mean()),
            "blend_minus_market_lower_95": float(lower),
            "blend_minus_market_upper_95": float(upper),
        }
    return {"model": model, "train_seasons": list(train_seasons), "books": result}


def market_benchmark(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    books: tuple[str, ...] = ("avg_open", "avg_close", "pinnacle_close"),
    value_model: str = "dixon_coles_half365",
    value_books: tuple[str, ...] = ("avg_close", "max_close"),
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.10),
    taxes: tuple[float, ...] = (0.0, 0.12),
) -> dict:
    """Score every model and bookmaker source on the same matches, season by season.

    ``predictions`` holds one row per match and model with outcome probabilities and the
    final score; ``odds`` one row per match. Only matches priced by ``avg_close`` are
    used, so models and the market average are compared like for like. A source that
    covers fewer matches (Pinnacle is absent from some files) reports its own ``n``.
    """
    required = {"match_id", "model", "season", "home_goals", "away_goals", *OUTCOME_COLUMNS}
    if not required <= set(predictions) or "match_id" not in odds:
        raise ValueError("Benchmark inputs are incomplete")
    if predictions.duplicated(["match_id", "model"]).any() or odds.match_id.duplicated().any():
        raise ValueError("Benchmark inputs contain duplicate matches")
    joined = predictions.merge(
        odds.drop(columns=["competition_id", "season"], errors="ignore"),
        on="match_id",
        validate="many_to_one",
    )
    reference = [f"avg_close_{side}" for side in SIDES]
    joined = joined.loc[joined[reference].notna().all(axis=1)]
    if joined.empty:
        raise ValueError("No predicted match has closing odds")
    matches = joined.drop_duplicates("match_id")

    def score(frame: pd.DataFrame, columns: list[str], *, from_odds: bool) -> dict | None:
        values = frame[columns].to_numpy(float)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            return None
        probabilities = implied_probabilities(values[valid])[0] if from_odds else values[valid]
        return probability_metrics(_labels(frame.loc[valid]), probabilities)

    def table(frame_models: pd.DataFrame, frame_matches: pd.DataFrame) -> dict:
        scores = {
            str(name): score(group, list(OUTCOME_COLUMNS), from_odds=False)
            for name, group in frame_models.groupby("model", sort=True)
        }
        for book in books:
            columns = [f"{book}_{side}" for side in SIDES]
            if set(columns) <= set(frame_matches):
                scores[f"market_{book}"] = score(frame_matches, columns, from_odds=True)
        return {name: value for name, value in scores.items() if value is not None}

    margins = {}
    for book in (*books, "max_close"):
        columns = [f"{book}_{side}" for side in SIDES]
        values = matches[columns].to_numpy(float) if set(columns) <= set(matches) else None
        if values is not None and np.isfinite(values).all(axis=1).any():
            margins[book] = float(
                implied_probabilities(values[np.isfinite(values).all(axis=1)])[1].mean()
            )
    selected = joined.loc[joined.model.eq(value_model)]
    simulations = [
        value_simulation(selected, book=book, threshold=threshold, tax=tax)
        for book in value_books
        for threshold in thresholds
        for tax in taxes
        if not selected.empty and f"{book}_home" in selected
    ]
    return {
        "metrics": table(joined, matches),
        "seasonal": {
            str(season): table(group, matches.loc[matches.season.eq(season)])
            for season, group in joined.groupby("season", sort=True)
        },
        "margins": margins,
        "value_model": value_model,
        "value_simulation": simulations,
        "matches": int(len(matches)),
        "seasons": sorted(str(value) for value in matches.season.unique()),
    }
