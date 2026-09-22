"""Read-only Streamlit views of offline analytics and frozen model artifacts."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from pl_analytics.config import get_settings
from pl_analytics.dashboard.data import (
    ArtifactError,
    available_competitions,
    catalog,
    prepared_models,
    report,
    scenario,
    table,
)
from pl_analytics.statistics.players import similar_players

PAGES = {
    "Overview": "Home.py",
    "Player Explorer": "pages/1_Player_Explorer.py",
    "Scouting Finder": "pages/2_Scouting_Finder.py",
    "Market Value": "pages/3_Market_Value.py",
    "Match Predictor": "pages/4_Match_Predictor.py",
    "Model Lab": "pages/5_Model_Lab.py",
}


def _season(frame: pd.DataFrame) -> pd.DataFrame:
    selected = st.selectbox("Season", sorted(frame.season.unique(), reverse=True))
    return frame.loc[frame.season.eq(selected)]


def _player(frame: pd.DataFrame) -> pd.Series:
    names = frame.set_index("player_id").player_name.to_dict()
    selected = st.selectbox(
        "Player", sorted(names, key=lambda item: names[item]), format_func=lambda item: names[item]
    )
    return frame.loc[frame.player_id.eq(selected)].iloc[0]


def _missing() -> None:
    st.info("Analytics artifacts are not built yet for this view.")
    st.caption("Prepare the offline artifacts using the commands in docs/DASHBOARD.md.")


def _overview(tables: dict) -> None:
    st.subheader("Football, measured.")
    st.write(
        "Explore player performance, compare editorial valuations and estimate match outcomes."
    )
    if all(frame.empty for frame in tables.values()):
        _missing()
        return
    for column, (name, frame), identity in zip(
        st.columns(3), tables.items(), ("player_id", "player_id", "match_id"), strict=True
    ):
        column.metric(
            {
                "players": "Player profiles",
                "values": "Valuation reviews",
                "matches": "Evaluated matches",
            }[name],
            frame[identity].nunique(),
        )
        column.caption("Seasons: " + ", ".join(sorted(frame.season.unique())))
    st.info(
        "These modules cover different historical periods. Dates shown in each view matter; "
        "a past player profile is not a current scouting assessment."
    )
    st.write(
        "Use the sidebar to explore profiles, scouting, valuations, match predictions and models."
    )
    st.caption(
        "Observed statistics describe recorded matches. Predictions are uncertain estimates."
    )


def _players(frame: pd.DataFrame, settings) -> None:
    all_seasons = frame
    frame = _season(frame)
    if "source" in frame and frame.source.eq("FPL-Core-Insights").all():
        from pl_analytics.dashboard.advanced_players import render_advanced

        render_advanced(frame, all_seasons.loc[all_seasons.source.eq("FPL-Core-Insights")])
        return
    position = st.selectbox("Position", ["All", *sorted(frame.position_group.unique())])
    minimum = st.slider(
        "Minimum minutes", 0, max(1, int(frame.minutes.max())), min(450, int(frame.minutes.max()))
    )
    frame = frame.loc[frame.minutes.ge(minimum)]
    if position != "All":
        frame = frame.loc[frame.position_group.eq(position)]
    if frame.empty:
        st.info("No players match these filters.")
        return
    player = _player(frame)
    st.subheader(player.player_name)
    st.caption(f"{player.position_group} · {player.season} · as of {player.reference_date}")
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
        px.bar(values, x="Metric", y="Per 90", color="Estimate", barmode="group"), width="stretch"
    )
    if player.ranking_eligible:
        peers = pd.DataFrame(
            {
                "Metric": ["Goals", "Assists"],
                "Percentile": [player.goals_percentile, player.assists_percentile],
            }
        )
        st.plotly_chart(
            px.bar(peers, x="Metric", y="Percentile", range_y=[0, 100]), width="stretch"
        )
        st.caption("Percentiles compare the recorded competition, season and position peer group.")
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

        render_advanced(
            frame, all_players.loc[all_players.source.eq("FPL-Core-Insights")], scouting=True
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


def _values(frame: pd.DataFrame) -> None:
    frame = _season(frame)
    st.warning(
        "Research benchmark: serious errors occur for low-exposure players. "
        "A model gap is not evidence of a bargain. Editorial market values are not transfer fees."
    )
    if not st.checkbox("Include low-exposure or missing-prior-value observations"):
        excluded = frame.low_exposure | frame.missing_previous_value
        st.caption(
            f"{int(excluded.sum())} fragile observations hidden; "
            "evaluation metrics remain unchanged."
        )
        frame = frame.loc[~excluded]
    if frame.empty:
        st.info("No valuation observations match these filters.")
        return
    player = _player(frame)
    st.subheader(player.player_name)
    st.caption(
        f"Valuation date: {player.valuation_date} · prior-year minutes: {player.minutes_365:,.0f}"
    )
    for column, label, value in zip(
        st.columns(3),
        ("Observed editorial value", "Model estimate", "Model minus observed"),
        (player.market_value_eur, player.predicted_value_eur, player.model_undervaluation_eur),
        strict=True,
    ):
        column.metric(label, f"€{value / 1e6:,.2f}m")
    st.write(
        f"Nominal 90% prediction interval: **€{player.lower_eur / 1e6:,.2f}m – "
        f"€{player.upper_eur / 1e6:,.2f}m**. Coverage is not guaranteed for each player."
    )
    st.caption(
        "Latest recorded review per player in this period; dates differ. Not a current shortlist."
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
    st.dataframe(ranking, hide_index=True, width="stretch")
    st.download_button(
        "Download dated reviews", ranking.to_csv(index=False), "values.csv", "text/csv"
    )


def _match(settings, competition: str) -> None:
    statistics, ml, states, metadata = prepared_models(settings)
    states = states.loc[states.competition_id.eq(competition)]
    if len(states) < 2:
        st.info("At least two prepared clubs are required for this competition.")
        return
    names = table(settings.artifact_dir / "dashboard/clubs.parquet")
    names = names.loc[names.competition_id.eq(competition)].set_index("club_id")["name"].to_dict()
    clubs = sorted(states.club_id.unique(), key=lambda club: names.get(club, club))
    st.info(
        f"Frozen features and models as of {metadata['origin']}. "
        "This is a hypothetical fixture at that origin, not a live forecast."
    )
    home = st.selectbox("Home club", clubs, format_func=lambda club: names.get(club, club))
    away = st.selectbox(
        "Away club",
        [club for club in clubs if club != home],
        format_func=lambda club: names.get(club, club),
    )
    models = [*statistics["models"], "calibrated_ml"]
    model = st.selectbox(
        "Model",
        models,
        index=models.index("dixon_coles_half365") if "dixon_coles_half365" in models else 0,
    )
    if st.button("Predict match", type="primary"):
        result = scenario(home, away, competition, model, statistics, ml, states)
        for column, label, probability in zip(
            st.columns(3), ("Home win", "Draw", "Away win"), result["probabilities"], strict=True
        ):
            column.metric(label, f"{probability:.1%}")
        st.caption("Probabilities express uncertainty; the most likely result can still fail.")
        if result["matrix"] is not None:
            matrix = result["matrix"]
            st.write(
                f"Expected goal counts: {result['rates'][0]:.2f} home, "
                f"{result['rates'][1]:.2f} away (not event-based xG)."
            )
            st.plotly_chart(
                px.imshow(
                    matrix[:7, :7],
                    origin="lower",
                    text_auto=".1%",
                    labels={"x": "Away goals", "y": "Home goals", "color": "Probability"},
                ),
                width="stretch",
            )
            st.caption(
                f"Displayed 0–6 grid contains {matrix[:7, :7].sum():.2%} of probability; "
                f"the complete grid sums to {matrix.sum():.6f}."
            )
        else:
            st.info("This model estimates outcomes only; it does not produce scorelines.")
    st.caption(
        "The calibrated ML pilot has only 40 held-out matches and trails Dixon–Coles "
        "on log loss. Consult Model Lab before interpreting estimates."
    )


def _lab(settings, competition: str) -> None:
    experiment = st.selectbox(
        "Experiment",
        ["m5", "m6", "m7"],
        format_func=lambda item: {
            "m5": "Player value",
            "m6": "Statistical matches",
            "m7": "Calibrated match ML",
        }[item],
    )
    result = report(settings.artifact_dir / experiment / "evaluation.json")
    if not result:
        _missing()
        return
    selection = report(settings.artifact_dir / experiment / "selection.json")
    scope = result.get("config", selection.get("config", {})).get("competition_id")
    if scope is not None and scope != competition:
        st.info("This experiment does not cover the selected competition.")
        return
    st.write("Frozen selected model: **" + result["chosen"] + "**")
    st.dataframe(pd.DataFrame(result["metrics"]).T, width="stretch")
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
        with st.expander("Methodology, dates and limitations", expanded=True):
            st.markdown(path.read_text(encoding="utf-8"))


def render(page: str = "Overview") -> None:
    """Render a standalone page safely; all expensive preparation is an explicit CLI task."""
    st.set_page_config(page_title=f"{page} · Analytics Lab", page_icon="⚽", layout="wide")
    st.markdown(
        "<style>.block-container{max-width:1440px;padding-top:4rem}"
        "[data-testid=stMetric]{background:white;padding:1rem;border:1px solid #dce3eb;"
        "border-radius:12px}[data-testid=stMetricLabel] p{white-space:normal}"
        "[data-testid=stMetricValue]{font-size:1.6rem}"
        "[data-testid=stMetricValue] div{white-space:normal;overflow-wrap:anywhere}</style>",
        unsafe_allow_html=True,
    )
    try:
        settings = get_settings()
    except ValidationError:
        st.error(
            "Configuration is invalid. Check .env and environment variables, then restart the app."
        )
        return
    st.caption("Configured competition scope: " + ", ".join(settings.active_competitions))
    with st.sidebar:
        st.title("⚽ Analytics Lab")
        st.caption("Performance · Value · Probability")
    tables, errors = catalog(settings)
    for error in errors:
        st.warning(error)
    competitions = available_competitions(tables)
    competition = st.sidebar.selectbox("Competition", competitions) if competitions else None
    all_players = tables["players"]
    tables = {
        name: frame.loc[frame.competition_id.eq(competition)] for name, frame in tables.items()
    }
    st.title(page)
    try:
        if page == "Overview":
            _overview(tables)
        elif competition is None:
            _missing()
        elif page in ("Player Explorer", "Scouting Finder", "Market Value"):
            frame = tables["values" if page == "Market Value" else "players"]
            if frame.empty:
                _missing()
            elif page == "Player Explorer":
                _players(frame, settings)
            elif page == "Scouting Finder":
                _scouting(frame, all_players)
            else:
                _values(frame)
        elif page == "Match Predictor":
            _match(settings, competition)
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
