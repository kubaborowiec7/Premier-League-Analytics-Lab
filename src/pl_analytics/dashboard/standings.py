"""Observed league table with xG-derived expected performance."""

import plotly.express as px
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import ArtifactError, table


def render_standings(settings: Settings, competition: str) -> None:
    """Render current standings and explain xG-derived expected points."""
    standings = table(
        settings.data_dir / "processed/fpl/standings.parquet",
        (
            "competition_id",
            "season",
            "position",
            "club_name",
            "played",
            "points",
            "expected_goals",
            "expected_goals_against",
            "expected_points",
        ),
    )
    standings = standings.loc[standings.competition_id.eq(competition)].copy()
    if standings.empty:
        raise ArtifactError(
            "League-table artifacts are not built yet. Run the current-data refresh."
        )
    season = standings.season.iloc[0]
    complete_xg = int(standings.xg_matches.sum() / 2)
    matches = int(standings.played.sum() / 2)
    st.caption(f"Season {season} · {matches} completed matches · xG available for {complete_xg}")

    view = standings[
        [
            "position",
            "club_name",
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "points",
            "expected_goals",
            "expected_goals_against",
            "expected_goal_difference",
            "expected_points",
            "expected_points_difference",
        ]
    ].rename(
        columns={
            "position": "Pos",
            "club_name": "Team",
            "played": "P",
            "wins": "W",
            "draws": "D",
            "losses": "L",
            "goals_for": "GF",
            "goals_against": "GA",
            "goal_difference": "GD",
            "points": "Pts",
            "expected_goals": "xG",
            "expected_goals_against": "xGA",
            "expected_goal_difference": "xGD",
            "expected_points": "xPts",
            "expected_points_difference": "Pts − xPts",
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=760,
        column_config={
            name: st.column_config.NumberColumn(format="%.2f")
            for name in ("xG", "xGA", "xGD", "xPts", "Pts − xPts")
        },
    )
    st.caption(
        "xPts uses independent Poisson score probabilities from each completed match's xG: "
        "3 × P(win) + P(draw). Pts − xPts above zero means the team has collected more points "
        "than its xG scorelines implied. It is descriptive and does not prove luck or quality."
    )

    chart = standings.sort_values("expected_points_difference")
    figure = px.bar(
        chart,
        x="expected_points_difference",
        y="club_name",
        orientation="h",
        color="expected_points_difference",
        color_continuous_scale=["#2563eb", "#e2e8f0", "#ef4444"],
        labels={"expected_points_difference": "Actual points minus xPts", "club_name": "Team"},
    )
    figure.update_layout(height=620, coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch")
