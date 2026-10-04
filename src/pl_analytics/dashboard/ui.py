"""Read-only Streamlit views of offline analytics and frozen model artifacts."""

import logging
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from pl_analytics.config import get_settings
from pl_analytics.dashboard import components, theme
from pl_analytics.dashboard.data import (
    ArtifactError,
    available_competitions,
    catalog,
    model_clubs,
    prepared_models,
    report,
    scenario,
    table,
)
from pl_analytics.dashboard.formatting import top_percent_label
from pl_analytics.statistics.players import similar_players

PAGES = {
    "Matchday Hub": "Home.py",
    "League Table": "pages/3_League_Table.py",
    "Match Centre": "pages/8_Match_Centre.py",
    "Match Predictor": "pages/4_Match_Predictor.py",
    "Teams": "pages/3_Teams.py",
    "Player Explorer": "pages/1_Player_Explorer.py",
    "Scouting Finder": "pages/2_Scouting_Finder.py",
    "FPL Decision Lab": "pages/4_FPL_Decision_Lab.py",
    "Betting Lab": "pages/7_Betting_Lab.py",
    "Model Lab": "pages/5_Model_Lab.py",
    "Market Value": "pages/3_Market_Value.py",
    "Data Coverage": "pages/6_Data_Coverage.py",
}

SECTIONS = {
    "Matchday": ["Matchday Hub", "Match Centre", "League Table", "Match Predictor"],
    "Clubs & players": ["Teams", "Player Explorer", "Scouting Finder"],
    "Fantasy": ["FPL Decision Lab"],
    "Betting": ["Betting Lab"],
    "Models & data": ["Model Lab", "Market Value", "Data Coverage"],
}

MODEL_PRESENTATION = {
    "base_rate": (
        "League baseline",
        "Historical home/draw/away frequencies. It is the simplest reference model.",
    ),
    "elo_k20": (
        "Elo — steady",
        "Updates club strength after each result and reacts gradually to new matches.",
    ),
    "elo_k40": (
        "Elo — responsive",
        "Uses the same rating system with larger updates, so recent results matter more quickly.",
    ),
    "poisson_flat": (
        "Poisson",
        "Estimates home and away goal rates from attack, defence and home advantage.",
    ),
    "poisson_half365": (
        "Poisson — recent",
        "Poisson goal model with match weights halving every 365 days.",
    ),
    "dixon_coles_flat": (
        "Dixon–Coles",
        "Poisson model with a dependence correction for low scorelines such as 0–0 and 1–1.",
    ),
    "dixon_coles_half365": (
        "Dixon–Coles — recent",
        "Low-score correction plus greater weight for recent matches; the preferred score model.",
    ),
    "calibrated_ml": (
        "Logistic ML",
        "Calibrated outcome model using Elo and each club's five-match form. "
        "It does not predict scores.",
    ),
}


# name -> (label, display multiplier, number format)
PRIMARY_METRICS = {
    "log_loss": ("Log loss", 1.0, ",.3f"),
    "mae_eur": ("Mean absolute error, € million", 1e-6, ",.2f"),
    "mae": ("Mean absolute error", 1.0, ",.3f"),
}


def _season(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(frame.season.unique(), reverse=True)
    requested = st.query_params.get("season")
    index = seasons.index(requested) if requested in seasons else 0
    selected = st.selectbox("Season", seasons, index=index)
    return frame.loc[frame.season.eq(selected)]


def _player(frame: pd.DataFrame) -> pd.Series:
    names = frame.set_index("player_id").player_name.to_dict()
    selected = st.selectbox(
        "Player", sorted(names, key=lambda item: names[item]), format_func=lambda item: names[item]
    )
    return frame.loc[frame.player_id.eq(selected)].iloc[0]


def _date(value: object, *, with_time: bool = False) -> str:
    """Readable UTC dates instead of raw ISO timestamps."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(stamp):
        return "—"
    return stamp.strftime("%d %b %Y, %H:%M UTC" if with_time else "%d %b %Y")


def _missing() -> None:
    st.info("Analytics artifacts are not built yet for this view.")
    st.caption("Prepare the offline artifacts using the commands in docs/DASHBOARD.md.")


def _overview(tables: dict, settings) -> None:
    if all(frame.empty for frame in tables.values()):
        _missing()
        return
    st.markdown(
        '<div class="pl-eyebrow" style="margin-bottom:10px">Historical modules · selected '
        "competition</div>",
        unsafe_allow_html=True,
    )
    labels = {
        "players": "Player profiles",
        "values": "Valuation reviews",
        "matches": "Evaluated matches",
    }
    for column, (name, frame), identity in zip(
        st.columns(3), tables.items(), ("player_id", "player_id", "match_id"), strict=True
    ):
        seasons = sorted(frame.season.dropna().astype(str).unique())
        span = f"{seasons[0]} – {seasons[-1]}" if len(seasons) > 1 else "".join(seasons)
        column.markdown(
            theme.stat_card(
                labels[name],
                f"{frame[identity].nunique():,}",
                f"{len(seasons)} season{'s' if len(seasons) != 1 else ''}: {span}"
                if seasons
                else "Not built yet",
            ),
            unsafe_allow_html=True,
        )
    fpl = report(settings.data_dir / "processed/fpl/metadata.json")
    live = report(settings.artifact_dir / "dashboard/live_catalog.json")
    current = []
    if fpl:
        current.append(
            (
                "FPL and league data",
                f"Gameweek {fpl.get('current_gameweek', '—')}",
                f"Season {fpl.get('season', '—')} · {fpl.get('players', '—')} players · "
                f"source {str(fpl.get('source_commit', ''))[:8]}",
            )
        )
    if live:
        current.append(
            (
                "Live match models",
                _date(live.get("last_result_date")),
                f"Last result used · {live.get('training_matches', '—')} matches · "
                f"refreshed {_date(live.get('refreshed_at'), with_time=True)}",
            )
        )
    if current:
        st.markdown(
            '<div class="pl-eyebrow" style="margin:18px 0 10px">Current data · refreshed by '
            "<code>refresh_current_data.py</code></div>",
            unsafe_allow_html=True,
        )
        theme.stat_row(current)
    st.write("")
    st.info(
        "These modules cover different historical periods. Dates shown in each view matter; "
        "a past player profile is not a current scouting assessment."
    )
    st.caption(
        "Observed statistics describe recorded matches. Predictions are uncertain estimates."
    )


def _players(frame: pd.DataFrame, settings) -> None:
    all_seasons = frame
    frame = _season(frame)
    if "source" in frame and frame.source.eq("FPL-Core-Insights").all():
        from pl_analytics.dashboard.advanced_players import render_advanced
        from pl_analytics.dashboard.player_import import (
            performance_importer,
            performance_session_rows,
        )

        imported = performance_importer(
            frame,
            competition_id=str(frame.competition_id.iloc[0]),
            season=str(frame.season.iloc[0]),
            prefix="explorer-import",
        )
        current = pd.concat([frame, imported], ignore_index=True)
        advanced = all_seasons.loc[all_seasons.source.eq("FPL-Core-Insights")]
        advanced = pd.concat([advanced, performance_session_rows()], ignore_index=True)
        render_advanced(current, advanced)
        return
    first, second = st.columns(2)
    position = first.selectbox("Position", ["All", *sorted(frame.position_group.unique())])
    minimum = second.slider(
        "Minimum minutes", 0, max(1, int(frame.minutes.max())), min(450, int(frame.minutes.max()))
    )
    frame = frame.loc[frame.minutes.ge(minimum)]
    if position != "All":
        frame = frame.loc[frame.position_group.eq(position)]
    if frame.empty:
        st.info("No players match these filters.")
        return
    player = _player(frame)
    st.markdown(
        f'<div class="pl-card-title" style="margin-top:6px">{escape(str(player.player_name))}'
        f'</div><div class="pl-muted" style="margin:-8px 0 12px">'
        f"{escape(str(player.position_group))} · {escape(str(player.season))} · as of "
        f"{_date(player.reference_date)}</div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "Positions come from a snapshot and are not historically verified. "
        "These metrics cover goals, assists and cards, not every aspect of performance."
    )
    for column, label, value in zip(
        st.columns(4),
        ("Minutes", "Appearances", "Goals / 90", "Assists / 90"),
        (player.minutes, player.appearances, player.goals_per90, player.assists_per90),
        strict=True,
    ):
        column.metric(label, f"{value:,.2f}" if "/" in label else f"{value:,.0f}")
    st.progress(float(player.reliability_weight), text="Minutes-based reliability weight")
    st.caption(
        "More exposure reduces shrinkage toward the peer average. This is not a quality score."
    )
    values = pd.DataFrame(
        [
            {"Metric": metric.title(), "Estimate": label, "Per 90": player[f"{metric}_{suffix}"]}
            for metric in ("goals", "assists")
            for label, suffix in (("Observed", "per90"), ("Shrunk toward peers", "shrunk_per90"))
        ]
    )
    st.plotly_chart(
        px.bar(
            values,
            x="Metric",
            y="Per 90",
            color="Estimate",
            barmode="group",
            color_discrete_sequence=[theme.ACCENT, theme.MUTED],
        ),
        width="stretch",
    )
    if player.ranking_eligible:
        peers = pd.DataFrame(
            {
                "Metric": ["Goals", "Assists"],
                "Percentile": [player.goals_percentile, player.assists_percentile],
                "Peer rank": [
                    top_percent_label(player.goals_percentile),
                    top_percent_label(player.assists_percentile),
                ],
            }
        )
        st.plotly_chart(
            px.bar(
                peers,
                x="Metric",
                y="Percentile",
                text="Peer rank",
                range_y=[0, 100],
                labels={"Percentile": "Peer standing"},
            ),
            width="stretch",
        )
        st.caption("Top-% ranks compare the recorded competition, season and position peer group.")
    intervals = table(settings.artifact_dir / "dashboard/player_intervals.parquet")
    if not intervals.empty:
        selected = intervals.loc[
            intervals.player_id.eq(player.player_id)
            & intervals.competition_id.eq(player.competition_id)
            & intervals.season.eq(player.season)
        ]
        st.subheader("Bootstrap uncertainty")
        st.dataframe(
            selected.drop(columns=["player_id", "competition_id", "season"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Resampled appearance intervals describe sampling variation, not future guarantees."
        )


def _scouting(frame: pd.DataFrame, all_players: pd.DataFrame) -> None:
    frame = _season(frame)
    if "source" in frame and frame.source.eq("FPL-Core-Insights").all():
        from pl_analytics.dashboard.advanced_players import render_advanced
        from pl_analytics.dashboard.player_import import (
            performance_importer,
            performance_session_rows,
        )

        imported = performance_importer(
            frame,
            competition_id=str(frame.competition_id.iloc[0]),
            season=str(frame.season.iloc[0]),
            prefix="scouting-import",
        )
        current = pd.concat([frame, imported], ignore_index=True)
        advanced = all_players.loc[all_players.source.eq("FPL-Core-Insights")]
        advanced = pd.concat([advanced, performance_session_rows()], ignore_index=True)
        render_advanced(
            current, advanced, scouting=True
        )
        return
    frame = frame.loc[frame.minutes.ge(450) & frame.ranking_eligible]
    metrics = st.multiselect(
        "Similarity metrics",
        ["goals", "assists", "yellow_cards", "red_cards"],
        default=["goals", "assists"],
    )
    if not metrics:
        st.info("Select at least one similarity metric.")
        return
    features = tuple(f"{metric}_zscore" for metric in metrics)
    frame = frame.loc[np.isfinite(frame.loc[:, features].to_numpy(dtype=float)).all(axis=1)]
    if frame.empty:
        st.info(
            "No eligible reference players with 450 minutes and complete selected peer metrics."
        )
        return
    reference = _player(frame)
    universe = st.multiselect(
        "Candidate competitions",
        sorted(all_players.competition_id.unique()),
        default=[reference.competition_id],
    )
    if not universe:
        st.info("Select at least one candidate competition.")
        return
    result = similar_players(
        reference,
        all_players,
        candidate_competitions=tuple(universe),
        features=features,
        limit=10,
    )
    st.caption(
        "Closest statistical profiles, not best players. Same season, reference date and "
        "position context; at least 450 minutes. No cross-league strength adjustment."
    )
    result = result[
        [
            "player_name",
            "competition_id",
            "season",
            "position_group",
            "minutes",
            "goals_per90",
            "assists_per90",
            "distance",
        ]
    ]
    st.dataframe(result, hide_index=True, width="stretch")
    st.download_button(
        "Download comparison", result.to_csv(index=False), "scouting.csv", "text/csv"
    )


def _values(
    frame: pd.DataFrame, settings, competition: str, default_season: str
) -> None:
    from pl_analytics.dashboard.player_import import value_importer

    imported = value_importer(
        settings=settings,
        competition_id=competition,
        default_season=default_season,
        prefix="market-import",
    )
    frame = frame.assign(is_user_import=False)
    frame = pd.concat([frame, imported], ignore_index=True)
    first, second = st.columns([1, 2], vertical_alignment="bottom")
    with first:
        frame = _season(frame)
    include = second.checkbox("Include low-exposure or missing-prior-value observations")
    st.warning(
        "Research benchmark: serious errors occur for low-exposure players. "
        "A model gap is not evidence of a bargain. Editorial market values are not transfer fees."
    )
    if not include:
        excluded = (frame.low_exposure | frame.missing_previous_value) & ~frame.is_user_import
        st.caption(
            f"{int(excluded.sum())} fragile observations hidden; "
            "evaluation metrics remain unchanged."
        )
        frame = frame.loc[~excluded]
    if frame.empty:
        st.info("No valuation observations match these filters.")
        return
    review, everything = st.tabs(["Player review", "All reviews"])
    with review:
        player = _player(frame)
        st.markdown(
            f'<div class="pl-card-title" style="margin-top:6px">{escape(str(player.player_name))}'
            f'</div><div class="pl-muted" style="margin:-8px 0 12px">Valuation date '
            f"{_date(player.valuation_date)} · {player.minutes_365:,.0f} minutes in the prior "
            "year</div>",
            unsafe_allow_html=True,
        )
        for column, label, value in zip(
            st.columns(3),
            ("Observed editorial value", "Model estimate", "Model minus observed"),
            (player.market_value_eur, player.predicted_value_eur, player.model_undervaluation_eur),
            strict=True,
        ):
            column.metric(label, f"€{value / 1e6:,.2f}m")
        scale = 1.05 * max(float(player.upper_eur), float(player.market_value_eur), 1.0)
        observed = 100 * min(max(float(player.market_value_eur) / scale, 0.0), 1.0)
        st.markdown(
            '<div class="pl-card" style="margin-top:14px">'
            f"Nominal 90% prediction interval: <b>€{player.lower_eur / 1e6:,.2f}m – "
            f"€{player.upper_eur / 1e6:,.2f}m</b>. Coverage is not guaranteed for each player."
            '<div style="position:relative;margin:16px 0 6px">'
            + theme.range_bar(
                float(player.lower_eur),
                float(player.predicted_value_eur),
                float(player.upper_eur),
                scale,
            )
            + f'<span title="Observed value" style="position:absolute;top:-5px;'
            f"left:calc({observed:.1f}% - 5px);width:10px;height:16px;border-radius:3px;"
            f'background:{theme.WARNING}"></span></div>'
            '<div class="pl-muted">Band: model interval · white marker: model estimate · '
            f'<b style="color:{theme.WARNING}">■</b> observed editorial value. Scale runs from '
            f"€0 to €{scale / 1e6:,.1f}m.</div></div>",
            unsafe_allow_html=True,
        )
    with everything:
        st.caption(
            "Latest recorded review per player in this period; dates differ. "
            "Not a current shortlist."
        )
        ranking = frame.sort_values("uncertainty_scaled_gap", ascending=False)[
            [
                "player_name",
                "valuation_date",
                "minutes_365",
                "market_value_eur",
                "predicted_value_eur",
                "lower_eur",
                "upper_eur",
                "assessment",
            ]
        ]
        euros = st.column_config.NumberColumn(format="€%.0f")
        st.dataframe(
            ranking.rename(
                columns={
                    "player_name": "Player",
                    "valuation_date": "Valuation date",
                    "minutes_365": "Minutes (prior year)",
                    "market_value_eur": "Observed",
                    "predicted_value_eur": "Model",
                    "lower_eur": "Low (90%)",
                    "upper_eur": "High (90%)",
                    "assessment": "Assessment",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Observed": euros,
                "Model": euros,
                "Low (90%)": euros,
                "High (90%)": euros,
                "Minutes (prior year)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.download_button(
            "Download dated reviews", ranking.to_csv(index=False), "values.csv", "text/csv"
        )


def _match(settings, competition: str) -> None:
    statistics, ml, states, metadata = prepared_models(settings)
    states = states.loc[states.competition_id.eq(competition)]
    if len(states) < 2:
        st.info("At least two prepared clubs are required for this competition.")
        return
    names = model_clubs(settings, metadata)
    names = names.loc[names.competition_id.eq(competition)].set_index("club_id")["name"].to_dict()
    clubs = sorted(states.club_id.unique(), key=lambda club: names.get(club, club))
    operational = metadata.get("mode") == "operational"
    if operational:
        through = _date(metadata["last_result_date"])
        cards = (
            ("Model data", "Live snapshot", f"Results through {through}"),
            ("Refreshed", _date(metadata["refreshed_at"]), "Refit by the current-data refresh"),
            ("Clubs covered", str(len(clubs)), f"{metadata.get('training_matches', '—')} matches"),
        )
    else:
        cards = (
            ("Model data", "Frozen evaluation", f"As of {_date(metadata['origin'])}"),
            ("Clubs covered", str(len(clubs)), "At the evaluated origin"),
        )
    theme.stat_row(cards)
    if not operational:
        st.info(
            f"Frozen features and models as of {_date(metadata['origin'])}. "
            "This is a hypothetical fixture at that origin, not a live forecast."
        )
    first, second, third = st.columns(3)
    home = first.selectbox("Home club", clubs, format_func=lambda club: names.get(club, club))
    away = second.selectbox(
        "Away club",
        [club for club in clubs if club != home],
        format_func=lambda club: names.get(club, club),
    )
    models = [*statistics["models"], "calibrated_ml"]
    model = third.selectbox(
        "Model",
        models,
        index=models.index("dixon_coles_half365") if "dixon_coles_half365" in models else 0,
        format_func=lambda name: MODEL_PRESENTATION.get(name, (name, ""))[0],
    )
    st.caption(MODEL_PRESENTATION.get(model, (model, "No description available."))[1])
    # Inference from prepared bundles is instant, so the forecast follows the selectors.
    result = scenario(home, away, competition, model, statistics, ml, states)
    home_name, away_name = str(names.get(home, home)), str(names.get(away, away))
    probabilities = [float(value) for value in result["probabilities"]]
    matrix = result["matrix"]
    with st.container(border=True):
        st.markdown(
            f'<div class="pl-card-title">{escape(home_name)} v {escape(away_name)}</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([2, 3], gap="medium")
        with left:
            st.markdown(
                components.outcome_cards(*probabilities)
                + '<div style="margin-top:12px">'
                + theme.probability_bar(*probabilities)
                + "</div>"
                + (
                    '<div style="margin-top:10px">'
                    + components.score_facts(matrix, result["rates"])
                    + "</div>"
                    if matrix is not None
                    else ""
                ),
                unsafe_allow_html=True,
            )
        with right:
            if matrix is None:
                st.info("This model estimates outcomes only; it does not produce scorelines.")
            else:
                st.plotly_chart(
                    components.score_heatmap(matrix, home_name, away_name, goals=7),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.caption(
                    f"The 0–6 grid holds {matrix[:7, :7].sum():.2%} of the probability. "
                    "Expected goals are model goal counts, not event-based xG."
                )
    st.caption(
        "Probabilities express uncertainty; the most likely result can still fail. "
        "The calibrated ML pilot has only 40 held-out matches and trails Dixon–Coles "
        "on log loss. Consult Model Lab before interpreting estimates."
    )


def _lab(settings, competition: str) -> None:
    experiment = st.selectbox(
        "Experiment",
        ["m5", "m6", "m7", "fpl", "market"],
        format_func=lambda item: {
            "m5": "Player value",
            "m6": "Statistical matches",
            "m7": "Calibrated match ML",
            "fpl": "FPL expected points",
            "market": "Match models vs bookmakers",
        }[item],
    )
    if experiment == "market":
        from pl_analytics.dashboard.betting import render_market_benchmark

        render_market_benchmark(settings, competition)
        return
    if experiment == "fpl":
        from pl_analytics.dashboard.fpl import render_backtest

        render_backtest(settings)
        return
    result = report(settings.artifact_dir / experiment / "evaluation.json")
    if not result:
        _missing()
        return
    selection = report(settings.artifact_dir / experiment / "selection.json")
    scope = result.get("config", selection.get("config", {})).get("competition_id")
    if scope is not None and scope != competition:
        st.info("This experiment does not cover the selected competition.")
        return
    metrics = pd.DataFrame(result["metrics"]).T
    chosen = str(result["chosen"])
    primary = next(
        (name for name in PRIMARY_METRICS if name in metrics),
        next((name for name in metrics.columns if name != "n"), metrics.columns[0]),
    )
    primary_label, multiplier, number = PRIMARY_METRICS.get(primary, (primary, 1.0, ",.3f"))

    def display(name: object) -> str:
        return MODEL_PRESENTATION.get(str(name), (str(name).replace("_", " "), ""))[0]

    cards = [("Frozen selected model", display(chosen), "Chosen before the final test")]
    if chosen in metrics.index and pd.notna(metrics.loc[chosen, primary]):
        cards.append(
            (
                primary_label,
                f"{metrics.loc[chosen, primary] * multiplier:{number}}",
                "Selected model, final test · lower is better",
            )
        )
    sample = result.get("test_matches") or metrics.get("n", pd.Series(dtype=float)).max()
    if pd.notna(sample):
        cards.append(("Test sample", f"{int(sample):,}", "Observations in the final evaluation"))
    cards.append(("Candidates compared", str(len(metrics)), "Including simple baselines"))
    theme.stat_row(cards)

    ordered = (metrics[primary].dropna() * multiplier).sort_values()
    with st.container(border=True):
        st.markdown(
            f'<div class="pl-card-title">{primary_label} by model</div>',
            unsafe_allow_html=True,
        )
        # A failed candidate can be thousands of times worse; a log axis keeps the rest legible.
        wide = len(ordered) > 1 and ordered.min() > 0 and ordered.max() / ordered.min() > 20
        figure = go.Figure(
            go.Bar(
                x=ordered.to_numpy(),
                y=[display(name) for name in ordered.index],
                orientation="h",
                marker_color=[
                    theme.ACCENT if name == chosen else theme.DRAW for name in ordered.index
                ],
                text=[f"{value:{number}}" for value in ordered],
                textposition="outside",
                cliponaxis=False,
            )
        )
        figure.update_layout(
            height=max(220, 34 * len(ordered) + 70),
            margin={"l": 10, "r": 70, "t": 10, "b": 40},
            xaxis={
                "title": primary_label + (" (log scale)" if wide else ""),
                "type": "log" if wide else "linear",
            },
            yaxis={"autorange": "reversed", "ticksuffix": "  "},
        )
        st.plotly_chart(
            figure, width="stretch", key=f"lab-{experiment}", config={"displayModeBar": False}
        )
        st.caption("The selected model is highlighted. Shorter bars are better.")
    st.dataframe(metrics.rename(index=display), width="stretch")
    st.caption(
        "Lower MAE/RMSE, log loss, Brier and RPS are better. "
        "Log loss penalizes confident mistakes; Brier measures probability error; "
        "RPS accounts for the ordering of home/draw/away. Accuracy is secondary."
    )
    st.warning(
        "Results belong to different time windows: compare models within an experiment. "
        "Market-value extrapolation is fragile; the ML match test has only 40 fixtures."
    )
    card = {"m5": "player_value", "m6": "match_statistics", "m7": "match_ml"}[experiment]
    path = Path("reports/model_cards") / f"{card}.md"
    if path.exists():
        with st.expander("Methodology, dates and limitations"):
            st.markdown(path.read_text(encoding="utf-8"))


def render(page: str = "Matchday Hub") -> None:
    """Render a standalone page safely; all expensive preparation is an explicit CLI task."""
    st.set_page_config(page_title=f"{page} · Analytics Lab", page_icon="⚽", layout="wide")
    theme.apply()
    try:
        settings = get_settings()
    except ValidationError:
        st.error(
            "Configuration is invalid. Check .env and environment variables, then restart the app."
        )
        return
    with st.sidebar:
        st.markdown(
            '<div class="pl-eyebrow" style="margin-bottom:2px">Analytics Lab</div>'
            '<div class="pl-muted">Premier League · performance, probability, FPL</div>',
            unsafe_allow_html=True,
        )
    tables, errors = catalog(settings)
    for error in errors:
        st.warning(error)
    competitions = available_competitions(tables)
    if len(competitions) > 1:
        competition = st.sidebar.selectbox("Competition", competitions)
    else:
        competition = competitions[0] if competitions else None
    all_players = tables["players"]
    tables = {
        name: frame.loc[frame.competition_id.eq(competition)] for name, frame in tables.items()
    }
    if page != "Matchday Hub":
        theme.page_header(page)
    try:
        if page == "Matchday Hub":
            from pl_analytics.dashboard.hub import render_hub

            render_hub(settings, competition or settings.active_competitions[0])
        elif page == "Data Coverage":
            _overview(tables, settings)
        elif competition is None:
            _missing()
        elif page == "League Table":
            from pl_analytics.dashboard.standings import render_standings

            render_standings(settings, competition)
        elif page in ("Player Explorer", "Scouting Finder", "Teams", "Market Value"):
            frame = tables["values" if page == "Market Value" else "players"]
            if frame.empty:
                _missing()
            elif page == "Player Explorer":
                _players(frame, settings)
            elif page == "Scouting Finder":
                _scouting(frame, all_players)
            elif page == "Teams":
                from pl_analytics.dashboard.teams import render_teams

                advanced = frame.loc[
                    frame.get("source", pd.Series(index=frame.index, dtype=object)).eq(
                        "FPL-Core-Insights"
                    )
                ]
                render_teams(advanced)
            else:
                competition_players = all_players.loc[
                    all_players.competition_id.eq(competition)
                ]
                seasons = sorted(competition_players.season.dropna().astype(str).unique())
                default_season = seasons[-1] if seasons else str(frame.season.max())
                _values(frame, settings, competition, default_season)
        elif page == "Match Predictor":
            _match(settings, competition)
        elif page == "FPL Decision Lab":
            from pl_analytics.dashboard.fpl import render_fpl

            render_fpl(settings, competition)
        elif page == "Match Centre":
            from pl_analytics.dashboard.match_centre import render_match_centre

            render_match_centre(settings, competition)
        elif page == "Betting Lab":
            from pl_analytics.dashboard.betting import render_betting

            render_betting(settings, competition)
        elif page == "Model Lab":
            _lab(settings, competition)
    except ArtifactError as error:
        st.info(str(error))
    except (OSError, KeyError, ValueError, TypeError, StopIteration) as error:
        logging.warning("Dashboard view %s failed: %s", page, type(error).__name__)
        st.error(
            "Prepared artifacts are incompatible with this view. "
            "Rebuild them using the documented commands."
        )
