"""Betting Lab: model prices, a price checker and calculators.

Everything is computed from the prepared match model; odds are typed in by the user.
The page prices and compares. It does not recommend bets: the model-versus-market
benchmark shown here is the reason its "value" figures are disagreements, not edges.
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard import theme
from pl_analytics.dashboard.data import ArtifactError, report, table
from pl_analytics.dashboard.hub import fixture_predictions, next_games
from pl_analytics.data.bet_log import (
    RESULTS,
    TEXT_COLUMNS,
    add_bet,
    bet_log_path,
    read_bets,
    write_bets,
)
from pl_analytics.features.fpl import scorer_markets
from pl_analytics.statistics.betting import (
    bet_summary,
    break_even_probability,
    effective_odds,
    expected_value,
    implied_probabilities,
    kelly_fraction,
    markets_from_matrix,
    settle_bets,
)

DEFAULT_TAX_PERCENT = 12.0
SOURCE_LABELS = {
    "market_avg_close": "Bookmaker average, closing",
    "market_avg_open": "Bookmaker average, opening",
    "market_pinnacle_close": "Pinnacle, closing",
    "dixon_coles_half365": "Dixon–Coles — recent (used here)",
    "dixon_coles_flat": "Dixon–Coles",
    "poisson_half365": "Poisson — recent",
    "poisson_flat": "Poisson",
    "elo_k20": "Elo — steady",
    "elo_k40": "Elo — responsive",
    "base_rate": "League baseline",
}


def _odds(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.2f}" if value < 100 else f"{value:.0f}"


def _benchmark(settings: Settings, competition: str) -> dict:
    result = report(settings.artifact_dir / "market/evaluation.json")
    if result.get("config", {}).get("competition_id") != competition:
        return {}
    return result


def _simulation(benchmark: dict, *, book: str, threshold: float, tax: float) -> dict | None:
    return next(
        (
            row
            for row in benchmark.get("value_simulation", [])
            if row["book"] == book
            and abs(row["threshold"] - threshold) < 1e-9
            and abs(row["tax"] - tax) < 1e-9
            and row.get("roi") is not None
        ),
        None,
    )


def _honesty(benchmark: dict) -> None:
    """The standing warning: what the benchmark says about trusting the model's prices."""
    if not benchmark:
        st.warning(
            "No model-versus-market benchmark has been built, so this page cannot show how "
            "these prices compare with bookmakers. Run `python scripts/build_market_benchmark.py`. "
            "Treat every figure as a model opinion, not an edge."
        )
        return
    metrics = benchmark["metrics"]
    model, market = metrics[benchmark["value_model"]], metrics["market_avg_close"]
    text = (
        f"**The bookmakers were more accurate than this model.** Over {benchmark['matches']:,} "
        f"matches the model's log loss was {model['log_loss']:.3f} against "
        f"{market['log_loss']:.3f} for average closing odds (lower is better)."
    )
    plain = _simulation(benchmark, book="avg_close", threshold=0.05, tax=0.0)
    taxed = _simulation(benchmark, book="avg_close", threshold=0.05, tax=0.12)
    if plain and taxed:
        text += (
            f" Backing every selection the model rated 5% or more above the price would have "
            f"returned {plain['roi']:+.1%} of stakes, and {taxed['roi']:+.1%} after a 12% stake "
            f"tax ({plain['bets']:,} bets)."
        )
    blends = (benchmark.get("blend") or {}).get("books") or {}
    weights = [row["weight"] for row in blends.values()]
    if weights and max(weights) == 0:
        text += (
            " Blending the model with the odds did not improve on the odds alone: the best "
            "weight on the model was zero."
        )
    text += (
        " A positive figure below means the model disagrees with the price, not that the bet "
        "is good."
    )
    st.warning(text)


def _header(games: pd.DataFrame, predictions: dict, benchmark: dict) -> float:
    first, second, third = st.columns([2, 2, 1.4], vertical_alignment="center")
    gameweek = int(games.gameweek.iloc[0])
    first.markdown(
        theme.stat_card(
            f"Gameweek {gameweek}",
            f"{len(predictions)} of {len(games)} fixtures priced",
            "Prices from the live Dixon–Coles goal model",
        ),
        unsafe_allow_html=True,
    )
    if benchmark:
        metrics = benchmark["metrics"]
        gap = (
            metrics[benchmark["value_model"]]["log_loss"]
            - metrics["market_avg_close"]["log_loss"]
        )
        second.markdown(
            theme.stat_card(
                "Model vs bookmakers",
                f"{gap:+.3f} log loss",
                f"Worse than closing odds over {benchmark['matches']:,} matches"
                if gap > 0
                else f"Against closing odds over {benchmark['matches']:,} matches",
            ),
            unsafe_allow_html=True,
        )
    else:
        second.markdown(
            theme.stat_card("Model vs bookmakers", "Not built", "Run the market benchmark"),
            unsafe_allow_html=True,
        )
    tax = third.number_input(
        "Stake tax (%)",
        min_value=0.0,
        max_value=50.0,
        value=DEFAULT_TAX_PERCENT,
        step=1.0,
        key="bet-tax",
        help="Polish licensed bookmakers deduct 12% of every stake before applying the odds. "
        "Set 0 for a bookmaker that covers the tax or does not charge it.",
    )
    return float(tax) / 100


def _label(game: object) -> str:
    return f"{game.club_name} v {game.opponent_name} · {game.kickoff_uk:%a %d %b %H:%M}"


def _pick_fixture(games: pd.DataFrame, predictions: dict, key: str) -> object:
    scored = [game for game in games.itertuples(index=False) if game.match_id in predictions]
    labels = {game.match_id: _label(game) for game in scored}
    match_id = st.selectbox("Fixture", list(labels), format_func=labels.get, key=key)
    return next(game for game in scored if game.match_id == match_id)


def _overview(games: pd.DataFrame, predictions: dict) -> str:
    head = (
        "<tr><th>KO</th><th>FIXTURE</th><th>HOME</th><th>DRAW</th><th>AWAY</th>"
        "<th>OVER 2.5</th><th>UNDER 2.5</th><th>BOTH SCORE</th></tr>"
    )
    rows = []
    for game in games.itertuples(index=False):
        prediction = predictions.get(game.match_id)
        fixture = (
            f'<td style="white-space:nowrap;color:{theme.TEXT}">'
            f"{theme.club_chip(game.club_short_name)} v {theme.club_chip(game.opponent_short_name)}"
            f"</td>"
        )
        kickoff = f'<td style="white-space:nowrap">{game.kickoff_uk:%a %H:%M}</td>'
        if prediction is None or prediction["matrix"] is None:
            rows.append(f'<tr>{kickoff}{fixture}<td colspan="6">No model price</td></tr>')
            continue
        markets = markets_from_matrix(prediction["matrix"]).set_index(["market", "selection"])
        cells = "".join(
            f"<td>{_odds(markets.loc[key, 'fair_odds'])}</td>"
            for key in (
                ("Match result", "Home"),
                ("Match result", "Draw"),
                ("Match result", "Away"),
                ("Total goals", "Over 2.5"),
                ("Total goals", "Under 2.5"),
                ("Both teams to score", "Yes"),
            )
        )
        rows.append(f"<tr>{kickoff}{fixture}{cells}</tr>")
    return (
        f'<div style="overflow-x:auto"><table class="pl-table" style="font-size:14px">{head}'
        f'{"".join(rows)}</table></div>'
        '<div class="pl-muted" style="margin-top:10px">Fair odds: the price at which the model '
        "expects neither side to profit, before any bookmaker margin or tax. Times are UK.</div>"
    )


def _market_card(title: str, frame: pd.DataFrame) -> str:
    rows = "".join(
        f'<tr><td style="color:{theme.TEXT}">{escape(str(row.selection))}</td>'
        f"<td>{row.probability:.1%}</td>"
        f'<td style="color:{theme.TEXT};font-weight:700">{_odds(row.fair_odds)}</td></tr>'
        for row in frame.itertuples(index=False)
    )
    head = "<tr><th>SELECTION</th><th>MODEL</th><th>FAIR ODDS</th></tr>"
    return theme.card(title, f'<table class="pl-table" style="font-size:14px">{head}{rows}</table>')


def _fair_odds(games: pd.DataFrame, predictions: dict) -> None:
    gameweek = int(games.gameweek.iloc[0])
    st.markdown(
        theme.card(f"Gameweek {gameweek} fair odds", _overview(games, predictions)),
        unsafe_allow_html=True,
    )
    game = _pick_fixture(games, predictions, "bet-fair-fixture")
    markets = markets_from_matrix(predictions[game.match_id]["matrix"])
    rates = predictions[game.match_id].get("rates")
    if rates is not None:
        st.caption(
            f"Model goal expectation: {game.club_name} {float(rates[0]):.2f} – "
            f"{float(rates[1]):.2f} {game.opponent_name}. Home is the first-named club."
        )
    names = list(dict.fromkeys(markets.market))
    for index in range(0, len(names), 3):
        for column, name in zip(st.columns(3), names[index : index + 3], strict=False):
            column.markdown(
                _market_card(name, markets.loc[markets.market.eq(name)]), unsafe_allow_html=True
            )


def _verdict(value: float) -> tuple[str, str]:
    if value > 0.02:
        return "Model rates this above the price", "good"
    if value < -0.02:
        return "Model rates this below the price", "bad"
    return "Model and price roughly agree", "warning"


def _price_checker(games: pd.DataFrame, predictions: dict, tax: float) -> None:
    game = _pick_fixture(games, predictions, "bet-check-fixture")
    markets = markets_from_matrix(predictions[game.match_id]["matrix"])
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown('<div class="pl-card-title">Check one price</div>', unsafe_allow_html=True)
        first, second = st.columns(2)
        market = first.selectbox("Market", list(dict.fromkeys(markets.market)), key="bet-market")
        options = markets.loc[markets.market.eq(market)]
        selection = second.selectbox("Selection", options.selection.tolist(), key="bet-selection")
        row = options.loc[options.selection.eq(selection)].iloc[0]
        price = st.number_input(
            "Bookmaker odds (decimal)",
            min_value=1.01,
            max_value=1000.0,
            value=float(np.clip(round(row.fair_odds, 2), 1.01, 1000.0)),
            step=0.01,
            key=f"bet-price-{game.match_id}-{market}-{selection}",
            help="Starts at the model's fair odds. Type the price your bookmaker shows.",
        )
        value = expected_value(float(row.probability), price, tax)
        verdict, tone = _verdict(value)
        facts = (
            ("Model probability", f"{row.probability:.1%}"),
            ("Model fair odds", _odds(row.fair_odds)),
            ("Odds after stake tax", f"{effective_odds(price, tax):.2f}"),
            ("Break-even probability", f"{break_even_probability(price, tax):.1%}"),
            ("Expected value per 100 staked", f"{100 * value:+.1f}"),
        )
        st.markdown(
            "".join(
                f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
                f"<b>{text}</b></div>"
                for label, text in facts
            )
            + f'<div style="margin-top:12px">{theme.badge(verdict, tone)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Expected value uses the model's probability. Because bookmakers have been more "
            "accurate than the model, a positive figure is a disagreement to investigate."
        )
    with right, st.container(border=True):
        st.markdown(
            '<div class="pl-card-title">Check the whole match-result market</div>',
            unsafe_allow_html=True,
        )
        result = markets.loc[markets.market.eq("Match result")].set_index("selection")
        columns = st.columns(3)
        prices = [
            column.number_input(
                label,
                min_value=1.01,
                max_value=1000.0,
                value=float(np.clip(round(result.loc[label, "fair_odds"], 2), 1.01, 1000.0)),
                step=0.01,
                key=f"bet-1x2-{game.match_id}-{label}",
            )
            for column, label in zip(columns, ("Home", "Draw", "Away"), strict=True)
        ]
        implied, margin = implied_probabilities(np.array(prices))
        head = (
            "<tr><th>SELECTION</th><th>BOOKMAKER</th><th>MODEL</th><th>DIFFERENCE</th>"
            "<th>EV / 100</th></tr>"
        )
        rows = []
        for label, price, bookmaker in zip(
            ("Home", "Draw", "Away"), prices, implied[0], strict=True
        ):
            model = float(result.loc[label, "probability"])
            gap = model - float(bookmaker)
            colour = (
                theme.ACCENT_TEXT
                if gap > 0.02
                else theme.BAD_TEXT
                if gap < -0.02
                else theme.TEXT_SOFT
            )
            rows.append(
                f'<tr><td style="color:{theme.TEXT}">{label}</td><td>{bookmaker:.1%}</td>'
                f'<td>{model:.1%}</td><td style="color:{colour}">{gap:+.1%}</td>'
                f"<td>{100 * expected_value(model, price, tax):+.1f}</td></tr>"
            )
        st.markdown(
            f'<table class="pl-table" style="font-size:14px;margin-top:8px">{head}'
            f'{"".join(rows)}</table>'
            '<div class="pl-list-row" style="margin-top:8px"><span class="grow pl-muted">'
            f"Bookmaker margin</span><b>{margin[0]:.1%}</b></div>"
            '<div class="pl-list-row"><span class="grow pl-muted">Expected return on stakes, '
            f"margin and tax combined</span><b>{(1 - tax) / (1 + margin[0]):.1%}</b></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Bookmaker: the three prices with the margin removed proportionally. The last line "
            "is what an average bettor gets back per 100 staked on this market."
        )


def _calculators(tax: float) -> None:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown('<div class="pl-card-title">Stake tax calculator</div>', unsafe_allow_html=True)
        first, second = st.columns(2)
        stake = first.number_input(
            "Stake", min_value=0.0, value=100.0, step=10.0, key="bet-tax-stake"
        )
        price = second.number_input(
            "Odds (decimal)", min_value=1.01, value=2.00, step=0.01, key="bet-tax-odds"
        )
        taxed_stake = stake * (1 - tax)
        payout = taxed_stake * price
        facts = (
            (f"Tax taken from the stake ({tax:.0%})", f"{stake - taxed_stake:,.2f}"),
            ("Stake actually placed", f"{taxed_stake:,.2f}"),
            ("Payout if it wins", f"{payout:,.2f}"),
            ("Profit if it wins", f"{payout - stake:+,.2f}"),
            ("Odds you really get", f"{effective_odds(price, tax):.2f}"),
            ("Win probability needed to break even", f"{break_even_probability(price, tax):.1%}"),
        )
        st.markdown(
            "".join(
                f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
                f"<b>{text}</b></div>"
                for label, text in facts
            ),
            unsafe_allow_html=True,
        )
        st.caption("Uses the stake tax set at the top of the page. Amounts are in your currency.")
    with right, st.container(border=True):
        st.markdown('<div class="pl-card-title">Staking calculator</div>', unsafe_allow_html=True)
        first, second = st.columns(2)
        probability = (
            first.number_input(
                "Your win probability (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                key="bet-kelly-probability",
            )
            / 100
        )
        price = second.number_input(
            "Odds (decimal)", min_value=1.01, value=2.20, step=0.01, key="bet-kelly-odds"
        )
        third, fourth = st.columns(2)
        bankroll = third.number_input(
            "Bankroll", min_value=0.0, value=1000.0, step=100.0, key="bet-kelly-bankroll"
        )
        cap = (
            fourth.number_input(
                "Maximum stake (% of bankroll)",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5,
                key="bet-kelly-cap",
            )
            / 100
        )
        share = st.segmented_control(
            "Kelly share",
            [0.25, 0.5, 1.0],
            default=0.25,
            format_func=lambda value: {0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[value],
            key="bet-kelly-share",
            help="Full Kelly assumes your probability is exactly right. Fractions reduce the "
            "damage when it is not.",
        )
        value = expected_value(probability, price, tax)
        full = kelly_fraction(probability, price, tax)
        fraction = min(full * float(share or 0.25), cap)
        if value <= 0:
            st.markdown(
                theme.badge("No stake: the expected value is not positive", "bad"),
                unsafe_allow_html=True,
            )
        facts = (
            ("Expected value per 100 staked", f"{100 * value:+.1f}"),
            ("Full Kelly share of bankroll", f"{full:.1%}"),
            ("Share after fraction and cap", f"{fraction:.1%}"),
            ("Stake", f"{bankroll * fraction:,.2f}"),
        )
        st.markdown(
            "".join(
                f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
                f"<b>{text}</b></div>"
                for label, text in facts
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "The stake is only as good as the probability you enter. Using this page's model "
            "probability here is not advised: the benchmark shows it loses to bookmakers."
        )


def _record_bet(path: Path, games: pd.DataFrame, predictions: dict, tax: float) -> None:
    with st.container(border=True):
        st.markdown('<div class="pl-card-title">Record a bet</div>', unsafe_allow_html=True)
        scored = [game for game in games.itertuples(index=False) if game.match_id in predictions]
        labels = {game.match_id: _label(game) for game in scored} | {"other": "Another event"}
        first, second, third = st.columns([3, 2, 2])
        match_id = first.selectbox(
            "Event", list(labels), format_func=labels.get, key="bet-log-event"
        )
        probability = None
        if match_id == "other":
            event = first.text_input("Event name", key="bet-log-event-name").strip()
            market = second.text_input("Market", key="bet-log-market-text").strip()
            selection = third.text_input("Selection", key="bet-log-selection-text").strip()
        else:
            markets = markets_from_matrix(predictions[match_id]["matrix"])
            event = labels[match_id]
            market = second.selectbox(
                "Market", list(dict.fromkeys(markets.market)), key="bet-log-market"
            )
            options = markets.loc[markets.market.eq(market)]
            selection = third.selectbox(
                "Selection", options.selection.tolist(), key="bet-log-selection"
            )
            probability = float(options.loc[options.selection.eq(selection), "probability"].iloc[0])
        fourth, fifth, sixth = st.columns([2, 2, 2], vertical_alignment="bottom")
        price = fourth.number_input(
            "Odds taken (decimal)", min_value=1.01, value=2.00, step=0.01, key="bet-log-odds"
        )
        stake = fifth.number_input(
            "Stake", min_value=0.0, value=10.0, step=1.0, key="bet-log-stake"
        )
        if sixth.button("Add to log", type="primary", width="stretch"):
            if not event or not market or not selection or stake <= 0:
                st.error("Enter the event, market, selection and a positive stake.")
            else:
                try:
                    add_bet(
                        path,
                        event=event,
                        market=market,
                        selection=selection,
                        odds=float(price),
                        stake=float(stake),
                        tax=tax,
                        model_probability=probability,
                    )
                except OSError:
                    st.error("This server cannot save files, so the bet was not recorded.")
                else:
                    st.rerun()
        st.caption(
            f"Saved with the {tax:.0%} stake tax set at the top of the page. The model's "
            "probability is stored with the bet when the event is a priced fixture."
        )


def _backup(path: Path, bets: pd.DataFrame) -> None:
    """Download and restore: a hosted app can lose its files when it restarts."""
    with st.expander("Back up or restore the log"):
        st.caption(
            "When the app runs on a hosting service its files can be wiped by a restart or "
            "update. Download a copy after recording bets, and restore it here if the log "
            "comes back empty."
        )
        left, right = st.columns(2)
        left.download_button(
            "Download bet log",
            bets.to_csv(index=False),
            "bets.csv",
            "text/csv",
            disabled=bets.empty,
            key="bet-log-download",
        )
        uploaded = right.file_uploader(
            "Restore from a downloaded log", type="csv", key="bet-log-upload"
        )
        if uploaded is not None and right.button(
            "Replace the log with this file", key="bet-log-restore"
        ):
            try:
                write_bets(path, pd.read_csv(uploaded, dtype=dict.fromkeys(TEXT_COLUMNS, str)))
            except (ValueError, OSError) as error:
                st.error(f"Not restored: {error}")
            else:
                st.rerun()


def _bet_log(settings: Settings, games: pd.DataFrame, predictions: dict, tax: float) -> None:
    path = bet_log_path(settings.data_dir)
    try:
        bets = read_bets(path)
    except (ValueError, OSError) as error:
        st.error(f"The bet log could not be read and was left untouched: {error}")
        return
    _record_bet(path, games, predictions, tax)
    _backup(path, bets)
    if bets.empty:
        st.info(
            "No bets recorded yet. Bets you add are saved to a file where this app runs and "
            "nowhere else."
        )
        return
    summary = bet_summary(bets)

    def share(value: float | None, signed: bool = True) -> str:
        return "—" if value is None else f"{value:+.1%}" if signed else f"{value:.0%}"

    cards = (
        ("Bets", str(summary["bets"]), f"{summary['open']} open · {summary['settled']} settled"),
        ("Profit", f"{summary['profit']:+,.2f}", f"on {summary['staked']:,.2f} staked"),
        ("Return on stakes", share(summary["return_on_stakes"]), "Settled bets, after tax"),
        ("Win rate", share(summary["win_rate"], signed=False), "Won out of won and lost"),
        (
            "Closing-line value",
            share(summary["closing_line_value"]),
            f"{share(summary['beat_the_close'], signed=False)} beat the close · "
            f"{summary['closing_prices']} with a closing price",
        ),
    )
    theme.stat_row(cards)
    settled = settle_bets(bets)
    done = settled.loc[settled.profit.notna()].sort_values("placed_at")
    if len(done) > 1:
        figure = go.Figure(
            go.Scatter(
                x=list(range(1, len(done) + 1)),
                y=done.profit.cumsum(),
                mode="lines+markers",
                line={"color": theme.ACCENT},
                hovertemplate="After bet %{x}: %{y:+.2f}<extra></extra>",
            )
        )
        figure.add_hline(y=0, line_color=theme.MUTED, line_width=1)
        figure.update_layout(
            height=260,
            margin={"l": 10, "r": 10, "t": 10, "b": 40},
            xaxis={"title": "Settled bets, oldest first", "dtick": max(1, len(done) // 10)},
            yaxis_title="Cumulative profit",
        )
        st.plotly_chart(
            figure, width="stretch", key="bet-log-profit", config={"displayModeBar": False}
        )
    edited = st.data_editor(
        settled.assign(delete=False)[
            [
                "bet_id",
                "placed_at",
                "event",
                "market",
                "selection",
                "odds",
                "stake",
                "tax",
                "model_probability",
                "closing_odds",
                "result",
                "profit",
                "delete",
            ]
        ],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="bet-log-editor",
        disabled=["placed_at", "model_probability", "profit"],
        column_config={
            "bet_id": None,
            "placed_at": st.column_config.TextColumn("Placed (UTC)"),
            "event": "Event",
            "market": "Market",
            "selection": "Selection",
            "odds": st.column_config.NumberColumn("Odds", min_value=1.01, format="%.2f"),
            "stake": st.column_config.NumberColumn("Stake", min_value=0.01, format="%.2f"),
            "tax": st.column_config.NumberColumn(
                "Tax", min_value=0.0, max_value=0.99, format="percent"
            ),
            "model_probability": st.column_config.NumberColumn("Model", format="percent"),
            "closing_odds": st.column_config.NumberColumn(
                "Closing odds",
                min_value=1.01,
                format="%.2f",
                help="The last price before kick-off, if you noted it.",
            ),
            "result": st.column_config.SelectboxColumn("Result", options=list(RESULTS)),
            "profit": st.column_config.NumberColumn("Profit", format="%+.2f"),
            "delete": st.column_config.CheckboxColumn("Delete"),
        },
    )
    if st.button("Save changes", key="bet-log-save"):
        try:
            write_bets(path, edited.loc[~edited["delete"]].drop(columns=["profit", "delete"]))
        except (ValueError, OSError) as error:
            st.error(f"Not saved: {error}")
        else:
            st.rerun()
    st.caption(
        "Set each bet's result, add the closing odds if you noted them, tick Delete to remove "
        "a row, then save. Closing-line value is the price you took divided by the closing "
        "price, minus one: beating the close consistently is the usual sign of real skill, "
        f"more reliable than short-run profit. The file is {path.as_posix()}."
    )


PLAYER_MARKETS = {"Anytime scorer": "scorer_probability", "Anytime assist": "assist_probability"}


def _player_markets(settings: Settings, competition: str, games: pd.DataFrame, tax: float) -> None:
    root = settings.data_dir / "processed/fpl"
    players = table(root / "players.parquet", ("competition_id", "player_name", "club_id"))
    players = players.loc[players.competition_id.eq(competition)]
    fixtures = table(root / "fixtures.parquet")
    if players.empty or "projected_xg_per90" not in players:
        st.info("Player projections are not built yet. Run the current-data refresh.")
        return
    markets = scorer_markets(players, fixtures, int(games.gameweek.iloc[0]))
    markets = markets.loc[markets.appearance_probability.gt(0)].copy()
    if markets.empty:
        st.info("No player is expected to appear in the next gameweek.")
        return
    labels = {"all": "All fixtures", **{game.match_id: _label(game) for game in games.itertuples()}}
    first, second, third = st.columns([3, 2, 2])
    match_id = first.selectbox(
        "Fixture", list(labels), format_func=labels.get, key="bet-player-fixture"
    )
    positions = sorted(markets.position_group.dropna().unique())
    chosen = second.pills(
        "Position", positions, selection_mode="multi", default=positions, key="bet-player-pos"
    )
    search = third.text_input("Search", placeholder="Surname", key="bet-player-search").strip()
    view = markets.loc[markets.position_group.isin(chosen or positions)]
    if match_id != "all":
        view = view.loc[view.match_id.eq(match_id)]
    if search:
        view = view.loc[view.player_name.str.contains(search, case=False, regex=False, na=False)]
    if view.empty:
        st.info("No players match these filters.")
        return
    view = view.sort_values("scorer_probability", ascending=False).reset_index(drop=True)
    scores, assists = view.scorer_probability, view.assist_probability
    st.dataframe(
        pd.DataFrame(
            {
                "Player": view.player_name,
                "Club": view.club_short_name,
                "Pos": view.position_group,
                "Opponent": view.opponent_short_name + " (" + view.venue + ")",
                "Start": view.start_probability,
                "xMins": view.expected_minutes,
                "Scores": view.scorer_probability,
                "Scorer fair odds": 1 / scores.where(scores.gt(0)),
                "Assists": assists,
                "Assist fair odds": 1 / assists.where(assists.gt(0)),
            }
        ),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Start": st.column_config.NumberColumn(format="percent"),
            "xMins": st.column_config.NumberColumn(format="%.0f"),
            "Scores": st.column_config.ProgressColumn(
                "Scores",
                format="percent",
                min_value=0.0,
                max_value=float(markets.scorer_probability.max()),
                help="Model probability of at least one goal in this fixture.",
            ),
            "Scorer fair odds": st.column_config.NumberColumn(format="%.2f"),
            "Assists": st.column_config.NumberColumn(
                format="percent", help="Model probability of at least one assist."
            ),
            "Assist fair odds": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    with st.container(border=True):
        st.markdown('<div class="pl-card-title">Check a player price</div>', unsafe_allow_html=True)
        first, second, third = st.columns([3, 2, 2])
        names = {
            index: f"{row.player_name} · {row.club_short_name} v {row.opponent_short_name}"
            for index, row in view.iterrows()
        }
        index = first.selectbox(
            "Player", list(names), format_func=names.get, key="bet-player-choice"
        )
        market = second.selectbox("Market", list(PLAYER_MARKETS), key="bet-player-market")
        row = view.loc[index]
        probability = float(row[PLAYER_MARKETS[market]])
        price = third.number_input(
            "Bookmaker odds (decimal)",
            min_value=1.01,
            max_value=1000.0,
            value=float(np.clip(round(1 / max(probability, 1e-3), 2), 1.01, 1000.0)),
            step=0.05,
            key=f"bet-player-price-{row.player_id}-{market}",
        )
        value = expected_value(probability, price, tax)
        verdict, tone = _verdict(value)
        facts = (
            ("Model probability", f"{probability:.1%}"),
            ("Model fair odds", _odds(1 / probability if probability > 0 else float("inf"))),
            ("Start probability · expected minutes",
             f"{row.start_probability:.0%} · {row.expected_minutes:.0f}"),
            ("Break-even probability after tax", f"{break_even_probability(price, tax):.1%}"),
            ("Expected value per 100 staked", f"{100 * value:+.1f}"),
        )
        st.markdown(
            "".join(
                f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
                f"<b>{text}</b></div>"
                for label, text in facts
            )
            + f'<div style="margin-top:12px">{theme.badge(verdict, tone)}</div>',
            unsafe_allow_html=True,
        )
    scorer = report(root / "backtest.json").get("scorer")
    note = (
        "Probabilities come from each player's expected goals and assists per 90, the chances "
        "of starting or coming on, and the fixture's expected goals. They ignore predicted "
        "line-ups, penalties being reassigned and late injuries."
    )
    if scorer:
        note += (
            f" Backtest over {scorer['observations']:,} player-gameweeks: the model gave an "
            f"average {scorer['mean_probability']:.1%} chance of scoring and "
            f"{scorer['scoring_rate']:.1%} of players scored (Brier {scorer['brier']:.4f} "
            f"against {scorer['position_rate_brier']:.4f} for each position's own scoring "
            "rate). There are no bookmaker scorer prices in the data, so this market has not "
            "been compared with bookmakers."
        )
    st.caption(note)


def render_market_benchmark(settings: Settings, competition: str) -> None:
    """Model Lab view: every match model against bookmaker odds on the same matches."""
    benchmark = _benchmark(settings, competition)
    if not benchmark:
        st.info("Analytics artifacts are not built yet for this view.")
        st.caption("Run `python scripts/build_market_benchmark.py` after the match pipelines.")
        return
    metrics = pd.DataFrame(benchmark["metrics"]).T
    model, market = benchmark["value_model"], "market_avg_close"
    st.write(
        f"Each model's forecasts and the bookmakers' odds are scored on the same "
        f"{benchmark['matches']:,} matches ({benchmark['seasons'][0]}–{benchmark['seasons'][-1]}). "
        "Odds are converted to probabilities by removing the margin proportionally."
    )
    cards = [
        ("Model log loss", f"{metrics.loc[model, 'log_loss']:.3f}", SOURCE_LABELS[model]),
        ("Bookmaker log loss", f"{metrics.loc[market, 'log_loss']:.3f}", "Average closing odds"),
        (
            "Average bookmaker margin",
            f"{benchmark['margins'].get('avg_close', float('nan')):.1%}",
            "Match-result market, closing",
        ),
    ]
    simulated = _simulation(benchmark, book="avg_close", threshold=0.05, tax=0.12)
    if simulated:
        cards.append(
            (
                "Backing the model's value picks",
                f"{simulated['roi']:+.1%}",
                f"Return on {simulated['bets']:,} bets after 12% tax",
            )
        )
    theme.stat_row(cards)
    ordered = metrics.log_loss.sort_values()
    with st.container(border=True):
        st.markdown('<div class="pl-card-title">Log loss by source</div>', unsafe_allow_html=True)
        figure = go.Figure(
            go.Bar(
                x=ordered.to_numpy(),
                y=[SOURCE_LABELS.get(name, name) for name in ordered.index],
                orientation="h",
                marker_color=[
                    theme.WARNING
                    if str(name).startswith("market_")
                    else theme.ACCENT
                    if name == model
                    else theme.DRAW
                    for name in ordered.index
                ],
                text=[f"{value:.3f}" for value in ordered],
                textposition="outside",
                cliponaxis=False,
            )
        )
        figure.update_layout(
            height=34 * len(ordered) + 90,
            margin={"l": 10, "r": 60, "t": 10, "b": 40},
            xaxis={"title": "Log loss (lower is better)", "range": [0.9, ordered.max() * 1.02]},
            yaxis={"autorange": "reversed", "ticksuffix": "  "},
        )
        st.plotly_chart(
            figure, width="stretch", key="market-benchmark", config={"displayModeBar": False}
        )
        st.caption(
            "Amber: bookmaker odds. Green: the model used across the app. Grey: other models. "
            "Pinnacle is scored on the matches it priced, so its bar is not strictly comparable."
        )
    seasonal = pd.DataFrame(
        {
            season: {
                SOURCE_LABELS.get(name, name): values["log_loss"]
                for name, values in scores.items()
                if name in (model, market, "market_pinnacle_close")
            }
            for season, scores in benchmark["seasonal"].items()
        }
    ).T
    periods = benchmark.get("periods", {})
    seasonal.insert(0, "Period", [periods.get(season, "") for season in seasonal.index])
    st.markdown('<div class="pl-eyebrow">Log loss by season</div>', unsafe_allow_html=True)
    st.dataframe(seasonal, width="stretch")
    blend = (benchmark.get("blend") or {}).get("books") or {}
    if blend:
        st.markdown(
            '<div class="pl-eyebrow" style="margin-top:8px">Does the model add anything to '
            "the odds?</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Odds blended with": {
                            "avg_open": "Bookmaker average, opening",
                            "avg_close": "Bookmaker average, closing",
                        }.get(book, book),
                        "Best weight on the model": row["weight"],
                        "Fitted on": row["train_matches"],
                        "Tested on": row["test_matches"],
                        "Model": row["model_log_loss"],
                        "Odds alone": row["market_log_loss"],
                        "Blend": row["blend_log_loss"],
                        "Blend minus odds": row["blend_minus_market"],
                    }
                    for book, row in blend.items()
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Best weight on the model": st.column_config.NumberColumn(format="percent"),
                **{
                    name: st.column_config.NumberColumn(format="%.4f")
                    for name in ("Model", "Odds alone", "Blend", "Blend minus odds")
                },
            },
        )
        zero = all(row["weight"] == 0 for row in blend.values())
        st.caption(
            "The blend multiplies the model's and the bookmakers' probabilities with a weight "
            f"fitted on {', '.join(benchmark['blend']['train_seasons'])} and tested on the later "
            "seasons; the last four columns are log loss on the test matches. "
            + (
                "The best weight on the model was zero: on these matches it carried no "
                "information the odds did not already contain, so the blend is the odds."
                if zero
                else "A negative last column means the blend beat the odds alone."
            )
        )
    simulations = pd.DataFrame(benchmark["value_simulation"])
    if not simulations.empty:
        st.markdown(
            '<div class="pl-eyebrow" style="margin-top:8px">What if you had backed the '
            "model's value picks at closing odds?</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            simulations.assign(
                book=simulations.book.map(
                    {"avg_close": "Average price", "max_close": "Best available price"}
                )
            ).rename(
                columns={
                    "book": "Price taken",
                    "threshold": "Minimum model edge",
                    "tax": "Stake tax",
                    "bets": "Bets",
                    "roi": "Return on stakes",
                    "roi_lower_95": "Low (95%)",
                    "roi_upper_95": "High (95%)",
                }
            )[
                [
                    "Price taken",
                    "Minimum model edge",
                    "Stake tax",
                    "Bets",
                    "Return on stakes",
                    "Low (95%)",
                    "High (95%)",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="percent")
                for name in (
                    "Minimum model edge",
                    "Stake tax",
                    "Return on stakes",
                    "Low (95%)",
                    "High (95%)",
                )
            },
        )
    st.caption(
        "Flat one-unit bets on every selection the model priced above the bookmaker by at "
        "least the stated edge. Intervals resample whole matches. 2023/24 and 2024/25 were the "
        "model-selection seasons, so the model is slightly flattered there. International "
        "bookmakers' prices are used; Polish bookmakers' odds are not in this data."
    )
    st.warning(
        "This is a historical comparison, not a prediction of future returns. Betting involves "
        "a real risk of losing money."
    )


def render_betting(settings: Settings, competition: str) -> None:
    """Render model prices and calculators from prepared artifacts only."""
    fixtures = table(
        settings.data_dir / "processed/fpl/fixtures.parquet",
        ("match_id", "gameweek", "kickoff_time", "club_name", "opponent_name", "venue"),
    )
    games = next_games(fixtures)
    if games.empty:
        raise ArtifactError(
            "Betting Lab artifacts are not built yet. Run the current-data refresh."
        )
    predictions, _ = fixture_predictions(settings, competition, games)
    predictions = {key: value for key, value in predictions.items() if value["matrix"] is not None}
    if not predictions:
        raise ArtifactError(
            "No fixture could be priced: the live match models are missing or do not cover "
            "these clubs. Run the current-data refresh."
        )
    benchmark = _benchmark(settings, competition)
    tax = _header(games, predictions, benchmark)
    _honesty(benchmark)
    fair, checker, player, calculators, log = st.tabs(
        ["Fair odds", "Price checker", "Player markets", "Calculators", "Bet log"]
    )
    with log:
        _bet_log(settings, games, predictions, tax)
    with fair:
        _fair_odds(games, predictions)
    with checker:
        _price_checker(games, predictions, tax)
    with player:
        _player_markets(settings, competition, games, tax)
    with calculators:
        _calculators(tax)
    st.caption(
        "For adults only. Betting involves a real risk of losing money; nothing here is advice "
        "or a recommendation to bet. No bookmaker odds are downloaded: you type them in."
    )
    st.page_link(
        "pages/5_Model_Lab.py",
        label="See the full model-versus-bookmaker benchmark in Model Lab",
        icon=":material/query_stats:",
    )
