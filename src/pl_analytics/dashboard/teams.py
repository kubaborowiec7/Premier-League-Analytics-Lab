"""Squad tables and player comparisons for prepared advanced profiles."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.dashboard.advanced_players import COLORS, radar_chart
from pl_analytics.dashboard.formatting import top_percent_label
from pl_analytics.features.advanced_players import (
    METRICS,
    PROFILES,
    cohort_percentiles,
    peer_percentiles,
)


@st.cache_data(show_spinner=False)
def _league_ranked(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    return peer_percentiles(frame, min_minutes=minimum)


def _metric_value(row: pd.Series, name: str, per90: bool) -> float:
    metric = METRICS[name]
    return row[f"{name}_per90"] if per90 and metric.kind == "count" else row[name]


def _scatter(
    peers: pd.DataFrame,
    selected: pd.DataFrame,
    x: str,
    y: str,
    *,
    context: str,
) -> go.Figure:
    eligible = peers.loc[
        peers[f"{x}_coverage"].ge(0.9)
        & peers[f"{y}_coverage"].ge(0.9)
        & peers[f"{x}_per90"].notna()
        & peers[f"{y}_per90"].notna()
    ]
    figure = go.Figure(
        go.Scatter(
            x=eligible[f"{x}_per90"],
            y=eligible[f"{y}_per90"],
            text=eligible.player_name,
            mode="markers",
            name=f"{context} peers",
            marker={"color": "#64748b", "size": 8, "opacity": 0.65},
            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )
    for color, (_, player) in zip(COLORS, selected.iterrows(), strict=False):
        if pd.isna(player[f"{x}_per90"]) or pd.isna(player[f"{y}_per90"]):
            continue
        figure.add_trace(
            go.Scatter(
                x=[player[f"{x}_per90"]],
                y=[player[f"{y}_per90"]],
                mode="markers",
                name=player.player_name,
                marker={"color": color, "size": 15, "line": {"color": "white", "width": 1}},
            )
        )
    suffix_x = " / 90" if METRICS[x].kind == "count" else ""
    suffix_y = " / 90" if METRICS[y].kind == "count" else ""
    figure.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=METRICS[x].label + suffix_x,
        yaxis_title=METRICS[y].label + suffix_y,
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def render_teams(frame: pd.DataFrame) -> None:
    """Render one season squad with sortable metrics and explicit comparison cohorts."""
    required = {"club_id", "club_name", "club_short_name"}
    if not required <= set(frame) or frame.club_id.isna().all():
        st.info("Team identities are not prepared yet. Rebuild the advanced player profiles.")
        return

    seasons = sorted(frame.season.dropna().unique(), reverse=True)
    default_season = "2026/27" if "2026/27" in seasons else seasons[0]
    season = st.selectbox("Season", seasons, index=seasons.index(default_season))
    season_frame = frame.loc[frame.season.eq(season)].copy()
    clubs = (
        season_frame[["club_id", "club_name"]]
        .drop_duplicates()
        .sort_values("club_name")
        .set_index("club_id")
        .club_name.to_dict()
    )
    club_id = st.selectbox(
        "Team", list(clubs), format_func=lambda value: clubs[value], key="teams_club"
    )
    squad = season_frame.loc[season_frame.club_id.eq(club_id)].copy()
    st.caption(
        "Team membership comes from the season player snapshot. A transferred player's season "
        "statistics are assigned to that snapshot team because the source does not split totals."
    )

    left, middle, right = st.columns(3)
    position = left.selectbox(
        "Position", ["All", *sorted(squad.position_group.dropna().unique())]
    )
    maximum = max(1, int(squad.minutes.max()))
    minimum = int(
        middle.number_input(
            "Minimum minutes", min_value=0, max_value=maximum, value=min(90, maximum), step=30
        )
    )
    context = right.radio("Comparison cohort", ["League", "Team"], horizontal=True)
    value_mode = st.radio("Statistics", ["Per 90", "Total"], horizontal=True)
    per90 = value_mode == "Per 90"

    filtered = squad.loc[squad.minutes.ge(minimum)].copy()
    if position != "All":
        filtered = filtered.loc[filtered.position_group.eq(position)]
    if filtered.empty:
        st.info("No players match the selected team, position and minutes filters.")
        return

    groups = list(dict.fromkeys(metric.group for metric in METRICS.values()))
    metric_group = st.selectbox("Metric group", ["All statistics", *groups])
    metrics = [
        name
        for name, metric in METRICS.items()
        if (metric_group == "All statistics" or metric.group == metric_group)
        and filtered[name].notna().any()
    ]
    if not metrics:
        st.info("The source has no recorded values for this metric group and selection.")
        return

    ranked_league = _league_ranked(season_frame, minimum)
    league_squad = ranked_league.loc[ranked_league.club_id.eq(club_id)].copy()
    if position != "All":
        league_squad = league_squad.loc[league_squad.position_group.eq(position)]
    league_squad = league_squad.loc[league_squad.minutes.ge(minimum)]
    ranked_team = cohort_percentiles(filtered, metrics=tuple(metrics), min_minutes=minimum)
    ranked = (league_squad if context == "League" else ranked_team).copy()

    ranking_role = position if position != "All" else "DEF"
    ranking_defaults = [name for name in PROFILES.get(ranking_role, ()) if name in metrics]
    rank_default = ranking_defaults[0] if ranking_defaults else metrics[0]
    rank_metric = st.selectbox(
        "Rank displayed in the table",
        metrics,
        index=metrics.index(rank_default),
        format_func=lambda name: METRICS[name].label,
    )
    percentile_column = (
        f"{rank_metric}_percentile"
        if context == "League"
        else f"{rank_metric}_cohort_percentile"
    )
    peer_column = (
        f"{rank_metric}_peers" if context == "League" else f"{rank_metric}_cohort_peers"
    )
    table_rows = []
    for _, player in ranked.iterrows():
        row = {
            "Player": player.player_name,
            "Position": player.position_group,
            "Minutes": player.minutes,
            "Appearances": player.appearances,
            f"{context} rank · {METRICS[rank_metric].label}": top_percent_label(
                player[percentile_column]
            ),
            "Eligible peers": player[peer_column],
        }
        for name in metrics:
            label = METRICS[name].label
            if per90 and METRICS[name].kind == "count":
                label += " / 90"
            row[label] = _metric_value(player, name, per90)
        table_rows.append(row)
    st.subheader(f"{clubs[club_id]} squad statistics")
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", height=520)
    st.caption(
        "Click any column header to sort ascending or descending. League Top-% ranks compare "
        "the same competition, season and broad position. Team ranks compare the filtered "
        "squad shown above; unavailable values remain blank."
    )

    choices = ranked.set_index("player_id").player_name.to_dict()
    default_role = (
        "DEF" if "DEF" in ranked.position_group.unique() else ranked.position_group.mode()[0]
    )
    defaults = list(
        ranked.loc[ranked.position_group.eq(default_role)]
        .sort_values("minutes", ascending=False)
        .player_id.astype(str)
        .head(min(2, len(ranked)))
    )
    selected_ids = st.multiselect(
        "Compare players (up to 5)",
        sorted(choices, key=lambda item: choices[item]),
        default=defaults,
        max_selections=5,
        format_func=lambda item: choices[item],
    )
    if not selected_ids:
        st.info("Select at least one player to show comparison charts.")
        return
    selected = ranked.set_index("player_id", drop=False).loc[selected_ids].copy()

    radar_rows = selected.copy()
    if context == "Team":
        for name in metrics:
            radar_rows[f"{name}_percentile"] = radar_rows[f"{name}_cohort_percentile"]
    available = [name for name in metrics if radar_rows[f"{name}_percentile"].notna().all()]
    primary_role = selected.position_group.iloc[0]
    profile_defaults = [name for name in PROFILES.get(primary_role, ()) if name in available]
    radar_metrics = st.multiselect(
        "Radar metrics",
        available,
        default=(profile_defaults or available)[:8],
        max_selections=12,
        format_func=lambda name: METRICS[name].label,
    )
    if len(radar_metrics) >= 3:
        st.plotly_chart(radar_chart(radar_rows, radar_metrics), width="stretch", theme=None)
    else:
        st.info("Choose at least three metrics with eligible percentile comparisons.")

    st.subheader("Peer scatterplot")
    axes = [name for name in metrics if filtered[f"{name}_per90"].notna().any()]
    if len(axes) >= 2:
        axis_defaults = [name for name in PROFILES.get(primary_role, ()) if name in axes]
        x_col, y_col = st.columns(2)
        x_default = axis_defaults[0] if axis_defaults else axes[0]
        y_default = axis_defaults[1] if len(axis_defaults) > 1 else axes[1]
        x = x_col.selectbox(
            "X metric",
            axes,
            index=axes.index(x_default),
            format_func=lambda name: METRICS[name].label,
        )
        y = y_col.selectbox(
            "Y metric",
            axes,
            index=axes.index(y_default),
            format_func=lambda name: METRICS[name].label,
        )
        if context == "Team":
            peers = filtered
        else:
            peers = season_frame.loc[
                season_frame.minutes.ge(minimum)
                & season_frame.position_group.eq(primary_role)
            ]
        st.plotly_chart(
            _scatter(peers, selected, x, y, context=context), width="stretch", theme=None
        )
        st.caption(
            "Grey points show the active cohort. Counts use per 90 minutes on both axes; "
            "percentages keep their original scale."
        )

    open_id = st.selectbox(
        "Player Explorer profile",
        selected_ids,
        format_func=lambda item: choices[item],
    )
    if st.button("Open in Player Explorer", type="primary"):
        st.query_params["season"] = season
        st.query_params["player"] = open_id
        st.switch_page("pages/1_Player_Explorer.py")
