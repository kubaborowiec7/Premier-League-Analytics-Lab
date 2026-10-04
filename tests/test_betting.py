"""Odds arithmetic, derived markets, archived odds parsing and the market benchmark."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pl_analytics.data.bet_log import BET_COLUMNS, add_bet, bet_log_path, read_bets, write_bets
from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import parse_results
from pl_analytics.data.market_odds import parse_odds
from pl_analytics.data.snapshots import SnapshotStore
from pl_analytics.statistics.betting import (
    bet_summary,
    blend_probabilities,
    break_even_probability,
    effective_odds,
    expected_value,
    fair_odds,
    fit_blend_weight,
    implied_probabilities,
    kelly_fraction,
    market_benchmark,
    market_blend,
    markets_from_matrix,
    settle_bets,
    value_simulation,
)
from pl_analytics.statistics.matches import outcome_probabilities, score_matrix


def test_stake_tax_lowers_the_price_and_raises_the_break_even():
    assert fair_odds(0.25) == 4 and fair_odds(0) == float("inf")
    assert effective_odds(2.0, 0.12) == pytest.approx(1.76)
    assert break_even_probability(2.0) == pytest.approx(0.5)
    assert break_even_probability(2.0, 0.12) == pytest.approx(1 / 1.76)
    assert expected_value(0.5, 2.0) == pytest.approx(0)
    assert expected_value(0.5, 2.0, 0.12) == pytest.approx(-0.12)
    # A fair coin at 2.20 is worth a bet untaxed, but not once 12% of the stake is taken.
    assert kelly_fraction(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    assert kelly_fraction(0.5, 2.2, 0.12) == 0
    assert kelly_fraction(0.5, 1.1, 0.12) == 0  # taxed odds below 1 can never pay
    for bad in ((1.2, 2.0, 0.0), (0.5, 1.0, 0.0), (0.5, 2.0, 1.0)):
        with pytest.raises(ValueError):
            expected_value(*bad)


def test_implied_probabilities_remove_the_margin():
    probabilities, margin = implied_probabilities(np.array([[1.9, 3.8, 3.8], [2.0, 4.0, 4.0]]))
    assert probabilities.sum(axis=1) == pytest.approx([1, 1])
    assert margin[1] == pytest.approx(0) and margin[0] > 0
    assert probabilities[1].tolist() == pytest.approx([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        implied_probabilities(np.array([1.0, 3.0, 3.0]))


def test_markets_are_consistent_with_the_score_matrix():
    matrix, _ = score_matrix(1.7, 1.1, -0.05)
    markets = markets_from_matrix(matrix).set_index(["market", "selection"]).probability
    home, draw, away = outcome_probabilities(matrix)
    assert markets["Match result", "Home"] == pytest.approx(home)
    assert markets["Double chance", "Home or draw"] == pytest.approx(home + draw)
    assert markets["Asian handicap", "Home -0.5"] == pytest.approx(home)
    assert markets["Asian handicap", "Away -0.5"] == pytest.approx(away)
    assert markets["Asian handicap", "Home +0.5"] == pytest.approx(home + draw)
    for market in ("Total goals", "Both teams to score"):
        assert markets[market].sum() == pytest.approx(3 if market == "Total goals" else 1)
    assert markets["Total goals", "Over 1.5"] > markets["Total goals", "Over 2.5"]
    assert markets["Clean sheet", "Home keeps one"] == pytest.approx(matrix[:, 0].sum())
    assert markets["Clean sheet", "Away keeps one"] == pytest.approx(matrix[0, :].sum())
    assert markets["Clean sheet"].sum() == pytest.approx(2)
    scores = markets["Correct score"]
    assert scores.is_monotonic_decreasing and scores.iloc[0] == pytest.approx(matrix.max())
    with pytest.raises(ValueError):
        markets_from_matrix(matrix * 2)


def test_archived_odds_share_match_identities_with_the_results_adapter(tmp_path):
    payload = tmp_path / "payload.csv"
    payload.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA,"
        "AvgC>2.5,AvgC<2.5\n"
        "T0,01/09/2026,15:00,Alpha,Beta,2,1,H,2.10,3.40,3.60,2.00,3.50,3.80,1.90,1.95\n"
        "T0,02/09/2026,15:00,Beta,Gamma,0,0,D,2.50,3.20,2.90,0,3.20,2.95,,\n",
        encoding="utf-8",
    )
    snapshot = SnapshotStore(tmp_path / "raw").import_file(
        payload,
        source_name="football-data.co.uk",
        source_url="https://example.invalid/T0.csv",
        dataset_version="T0-2627",
        license_note="test",
    )
    scope = Scope(
        "TEST", "Test League", "Testland", "T0", "2026/27", date(2026, 7, 1), date(2027, 6, 30)
    )
    odds = parse_odds(snapshot, scope).set_index("match_id")
    results = parse_results(snapshot, scope, timezone="Europe/London").tables["matches"]
    assert set(odds.index) == {row["match_id"] for row in results}
    first = odds.loc[results[0]["match_id"]]
    assert first.avg_close_home == 2.0 and first.avg_open_draw == 3.4
    assert first.avg_close_over25 == 1.9 and pd.isna(first.pinnacle_close_home)
    # A price of 0 is a source error and an empty cell is missing: neither becomes odds.
    second = odds.loc[results[1]["match_id"]]
    assert pd.isna(second.avg_close_home) and pd.isna(second.avg_close_over25)


def test_bet_log_round_trips_and_settles_with_tax(tmp_path):
    path = bet_log_path(tmp_path)
    assert read_bets(path).empty and list(read_bets(path).columns) == list(BET_COLUMNS)
    add_bet(path, event="A v B", market="Match result", selection="Home", odds=2.0,
            stake=100.0, tax=0.12, model_probability=0.55)
    bets = add_bet(path, event="C v D", market="Total goals", selection="Over 2.5",
                   odds=1.9, stake=50.0, tax=0.0)
    assert len(bets) == 2 and bets.bet_id.is_unique and bets.result.eq("open").all()
    assert bet_summary(bets)["return_on_stakes"] is None
    bets.loc[0, ["result", "closing_odds"]] = ["won", 1.8]
    bets.loc[1, "result"] = "lost"
    saved = write_bets(path, bets)
    pd.testing.assert_frame_equal(read_bets(path), saved)
    settled = settle_bets(saved)
    # 100 staked, 12 taxed, 88 × 2.0 returned.
    assert settled.profit.tolist() == pytest.approx([76.0, -50.0])
    assert settled.closing_line_value.iloc[0] == pytest.approx(2.0 / 1.8 - 1)
    summary = bet_summary(saved)
    assert summary["staked"] == 150 and summary["profit"] == pytest.approx(26.0)
    assert summary["win_rate"] == 0.5 and summary["closing_prices"] == 1
    assert summary["beat_the_close"] == 1.0
    void = saved.assign(result=["void", "open"])
    assert settle_bets(void).profit.iloc[0] == 0 and pd.isna(settle_bets(void).profit.iloc[1])
    # An invalid edit is rejected and leaves the saved file untouched.
    for column, value in (("odds", 1.0), ("stake", 0.0), ("result", "maybe"), ("tax", 1.0)):
        broken = saved.copy()
        broken.loc[0, column] = value
        with pytest.raises(ValueError):
            write_bets(path, broken)
    pd.testing.assert_frame_equal(read_bets(path), saved)
    assert not path.with_suffix(".csv.tmp").exists()


def _benchmark_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    truth = rng.dirichlet([4, 2.5, 3], size=300)
    outcome = np.array([rng.choice(3, p=row) for row in truth])
    goals = {0: (1, 0), 1: (1, 1), 2: (0, 1)}
    base = pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(300)],
            "season": np.where(np.arange(300) < 150, "2024/25", "2025/26"),
            "home_goals": [goals[value][0] for value in outcome],
            "away_goals": [goals[value][1] for value in outcome],
        }
    )
    sharp = base.assign(model="dixon_coles_half365")
    sharp[["p_home", "p_draw", "p_away"]] = truth
    naive = base.assign(model="base_rate", p_home=0.45, p_draw=0.25, p_away=0.30)
    odds = pd.DataFrame({"match_id": base.match_id})
    # The market knows the truth less precisely and adds a 5% margin.
    noisy = 0.5 * truth + 0.5 * np.array([0.45, 0.25, 0.30])
    for book, margin in (("avg_close", 1.05), ("max_close", 1.0), ("avg_open", 1.06)):
        for column, side in enumerate(("home", "draw", "away")):
            odds[f"{book}_{side}"] = 1 / (noisy[:, column] * margin)
    odds.loc[odds.index[:20], "avg_close_home"] = np.nan
    return pd.concat([sharp, naive], ignore_index=True), odds


def test_market_benchmark_scores_models_and_books_on_the_same_matches():
    predictions, odds = _benchmark_inputs()
    report = market_benchmark(predictions, odds)
    metrics = report["metrics"]
    # Matches without the reference closing price are dropped for every source.
    assert report["matches"] == 280 and metrics["base_rate"]["n"] == 280
    assert metrics["market_avg_close"]["n"] == 280 and "market_pinnacle_close" not in metrics
    assert (
        metrics["dixon_coles_half365"]["log_loss"]
        < metrics["market_avg_close"]["log_loss"]
        < metrics["base_rate"]["log_loss"]
    )
    assert set(report["seasonal"]) == {"2024/25", "2025/26"}
    assert report["margins"]["avg_close"] == pytest.approx(0.05)
    assert report["margins"]["max_close"] == pytest.approx(0, abs=1e-9)
    rows = {(row["book"], row["threshold"], row["tax"]): row for row in report["value_simulation"]}
    untaxed, taxed = rows["max_close", 0.05, 0.0], rows["max_close", 0.05, 0.12]
    assert untaxed["bets"] == taxed["bets"] > 0 and taxed["roi"] < untaxed["roi"]
    assert untaxed["roi_lower_95"] <= untaxed["roi"] <= untaxed["roi_upper_95"]
    with pytest.raises(ValueError):
        market_benchmark(pd.concat([predictions, predictions]), odds)


def test_blend_weight_follows_whichever_source_is_informative():
    rng = np.random.default_rng(11)
    truth = rng.dirichlet([4, 2.5, 3], size=1500)
    actual = np.array([rng.choice(3, p=row) for row in truth])
    flat = np.full_like(truth, 1 / 3)
    assert blend_probabilities(truth, flat, 0.0) == pytest.approx(flat)
    assert blend_probabilities(truth, flat, 1.0) == pytest.approx(truth)
    assert blend_probabilities(truth, flat, 0.4).sum(axis=1) == pytest.approx(np.ones(1500))
    # An uninformative model earns almost no weight; against flat odds a good one earns
    # almost all. (Not exactly 0 or 1: a flat partner tempers the other, which sampling
    # noise can slightly favour.)
    assert fit_blend_weight(flat, truth, actual) < 0.1
    assert fit_blend_weight(truth, flat, actual) > 0.9
    # A model identical to the market cannot improve on it at any weight.
    assert blend_probabilities(truth, truth, 0.37) == pytest.approx(truth)
    with pytest.raises(ValueError):
        blend_probabilities(truth, flat, 1.5)

    predictions, odds = _benchmark_inputs()
    report = market_blend(predictions, odds, train_seasons=("2024/25",))
    close = report["books"]["avg_close"]
    # Here the model is the truth and the market a noisy copy, so the model gets the weight.
    assert close["weight"] > 0.5 and close["blend_log_loss"] < close["market_log_loss"]
    assert close["train_matches"] + close["test_matches"] == 280
    assert close["blend_minus_market_lower_95"] <= close["blend_minus_market"] < 0
    assert market_blend(predictions, odds, train_seasons=("1999/00",))["books"] == {}


def test_value_simulation_settles_bets_with_the_stake_tax():
    frame = pd.DataFrame(
        {
            "p_home": [0.6, 0.6],
            "p_draw": [0.2, 0.2],
            "p_away": [0.2, 0.2],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "book_home": [2.0, 2.0],
            "book_draw": [3.0, 3.0],
            "book_away": [3.0, 3.0],
        }
    )
    plain = value_simulation(frame, book="book", threshold=0.1, tax=0.0, resamples=200)
    taxed = value_simulation(frame, book="book", threshold=0.1, tax=0.12, resamples=200)
    # Only the home price (0.6 × 2.0 − 1 = +20%) clears the threshold: one win, one loss.
    assert plain["bets"] == 2 and plain["roi"] == pytest.approx(0.0)
    assert taxed["roi"] == pytest.approx((2.0 * 0.88 - 1 - 1) / 2)
    empty = value_simulation(frame, book="book", threshold=0.9, tax=0.0)
    assert empty["bets"] == 0 and empty["roi"] is None
