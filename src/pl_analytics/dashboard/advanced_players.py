"""Position-aware advanced player profiles, multi-player radars and peer scatterplots."""

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.features.advanced_players import METRICS, PROFILES, peer_percentiles

COLORS = ("#38bdf8", "#fb7185", "#a3e635", "#c084fc", "#fbbf24")


@st.cache_data(show_spinner=False)
def _ranked(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    return peer_percentiles(frame, min_minutes=minimum)


def radar_chart(players: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Plot comparable percentile axes, leaving absent observations as gaps, never zero."""
    figure = go.Figure()
    labels = [METRICS[name].label for name in metrics]
    for color, (_, player) in zip(COLORS, players.iterrows(), strict=False):
        values = [player[f"{name}_percentile"] for name in metrics]
        complete = all(pd.notna(value) for value in values)
        values = [None if pd.isna(value) else float(value) for value in values]
        figure.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=labels + labels[:1],
                name=f"{player.player_name} · {player.season}",
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 7},
                fill="toself" if complete else "none",
                opacity=0.8,
                connectgaps=False,
                hovertemplate="%{theta}<br>Percentile: %{r:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        template="plotly_dark",
        height=620,
        paper_bgcolor="#171b22",
        font={"color": "#edf2f7", "size": 13},
        polar={"bgcolor": "#202632", "radialaxis": {"range": [0, 100], "dtick": 20}},
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 95, "r": 95, "t": 50, "b": 110},
    )
    return figure


def _performance(player: pd.Series, per90: bool) -> None:
    st.subheader("Season performance")
    st.caption(
        "— means unavailable, not zero. Coverage is the share of player minutes with a value. "
        "A partial total is not a complete season total."
    )
    groups = list(dict.fromkeys(metric.group for metric in METRICS.values()))
    if player.position_group != "GK":
        groups.remove("Goalkeeping")
    for index in range(0, len(groups), 2):
        for column, group in zip(st.columns(2), groups[index : index + 2], strict=False):
            rows = []
            for name, metric in METRICS.items():
                if metric.group != group:
                    continue
                value = (
                    player[f"{name}_per90"] if per90 and metric.kind == "count" else player[name]
                )
                percentile = player[f"{name}_percentile"]
                coverage = player[f"{name}_coverage"]
                display = (
                    "—"
                    if pd.isna(value)
                    else f"{value:,.2f}"
                    if per90 or metric.kind != "count"
                    else f"{value:,.2f}".rstrip("0").rstrip(".")
                )
                rank = "—" if pd.isna(percentile) else f"P{percentile:.0f}"
                label = html.escape(metric.label)
                rows.append(
                    f"<tr><td>{label}</td><td>{display}</td><td>{rank}</td>"
                    f"<td>{coverage:.0%}</td></tr>"
                )
            column.markdown(
                "<div class='performance-card'><h3>" + group + "</h3>"
                "<table><thead><tr><th>Metric</th><th>Value</th><th>Rank</th>"
                "<th>Coverage</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
                unsafe_allow_html=True,
            )


def render_advanced(
    frame: pd.DataFrame, all_seasons: pd.DataFrame, *, scouting: bool = False
) -> None:
    """Render actual prepared observations; role-specific metrics replace goal-only comparisons."""
    st.markdown(
        "<style>.performance-card{background:#1c2027;color:#edf2f7;padding:20px;"
        "border-radius:16px;margin-bottom:16px}.performance-card h3{color:#edf2f7}"
        ".performance-card table{width:100%;font-size:14px}.performance-card td,"
        ".performance-card th{padding:9px 5px;border-bottom:1px solid #343b46;text-align:right}"
        ".performance-card td:first-child,.performance-card th:first-child{text-align:left}"
        "</style>",
        unsafe_allow_html=True,
    )
    roles = sorted(frame.position_group.unique())
    role = st.selectbox("Position group", roles, index=roles.index("DEF") if "DEF" in roles else 0)
    minimum = st.number_input(
        "Peer minimum minutes",
        min_value=1,
        max_value=4000,
        value=450 if frame.minutes.max() >= 450 else 90,
        step=30,
    )
    st.caption(
        "Percentiles use the same competition, season and broad FPL position; at least five "
        "eligible peers and 90% metric coverage. Higher is the chosen direction, "
        "not overall quality. "
        "Fewer than 450 minutes gives an unstable early-season comparison."
    )
    ranked = _ranked(all_seasons, int(minimum))
    current = ranked.loc[
        ranked.season.eq(frame.season.iloc[0])
        & ranked.position_group.eq(role)
        & ranked.competition_id.eq(frame.competition_id.iloc[0])
    ]
    names = current.set_index("player_id").player_name.to_dict()
    selected_id = st.selectbox(
        "Player", sorted(names, key=lambda key: names[key]), format_func=lambda key: names[key]
    )
    primary = current.loc[current.player_id.eq(selected_id)].iloc[0]
    universe = ranked.loc[ranked.position_group.eq(role)].copy()
    universe["selection_id"] = (
        universe.competition_id + "|" + universe.player_id + "|" + universe.season
    )
    primary_key = primary.competition_id + "|" + primary.player_id + "|" + primary.season
    labels = (
        universe.set_index("selection_id")
        .apply(lambda row: f"{row.player_name} · {row.season} · {row.competition_id}", axis=1)
        .to_dict()
    )
    others = st.multiselect(
        "Overlay players (up to 4; other seasons allowed)",
        sorted([key for key in labels if key != primary_key], key=lambda key: labels[key]),
        format_func=lambda key: labels[key],
        max_selections=4,
    )
    selected = universe.set_index("selection_id").loc[[primary_key, *others]].copy().reset_index()
    st.subheader(primary.player_name)
    st.caption(
        f"{primary.competition_id} · {primary.season} · {role} · "
        f"last recorded match: {str(primary.last_match_date)[:10]}"
    )
    a, b, c = st.columns(3)
    a.metric("Minutes", f"{primary.minutes:,.0f}")
    b.metric("Appearances", int(primary.appearances))
    c.metric("Comparison players", len(selected))
    st.info(
        "Published FPL Core match statistics. Roles are broad season snapshots: DEF includes "
        "centre-backs and full-backs. Different-season radars compare relative standing in each "
        "season, not identical opponents or a strength-adjusted performance level."
    )
    if frame.season.iloc[0] == sorted(all_seasons.season.unique())[-1]:
        st.caption("The newest snapshot may be a partial season and has uneven metric coverage.")
    available = [name for name in METRICS if selected[f"{name}_percentile"].notna().all()]
    defaults = [name for name in PROFILES.get(role, PROFILES["MID"]) if name in available]
    metrics = st.multiselect(
        "Radar metrics",
        available,
        default=defaults[:8],
        max_selections=12,
        format_func=lambda name: METRICS[name].label,
    )
    if len(metrics) >= 3:
        st.plotly_chart(radar_chart(selected, metrics), width="stretch", theme=None)
        st.caption(
            "All curves use the same 0–100 axes. Missing or insufficiently covered metrics "
            "are omitted from the selector; their absence is not poor performance."
        )
    else:
        st.info(
            "Choose at least three shared eligible metrics. Lower the minutes threshold or "
            "choose another player/season if the sample is insufficient."
        )
    if metrics:
        comparison = selected[
            [
                "player_name",
                "competition_id",
                "season",
                "minutes",
                *[f"{name}_per90" for name in metrics],
            ]
        ]
        st.dataframe(
            comparison.rename(
                columns={
                    f"{name}_per90": METRICS[name].label
                    + (" / 90" if METRICS[name].kind == "count" else "")
                    for name in metrics
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if scouting and len(metrics) >= 3:
        candidate_competitions = st.multiselect(
            "Candidate competitions",
            sorted(universe.competition_id.unique()),
            default=[primary.competition_id],
        )
        columns = [f"{name}_percentile" for name in metrics]
        candidates = (
            universe.loc[
                universe.competition_id.isin(candidate_competitions)
                & universe.season.eq(primary.season)
                & universe.reference_date.eq(primary.reference_date)
            ]
            .dropna(subset=columns)
            .copy()
        )
        candidates = candidates.loc[candidates.player_id.ne(primary.player_id)]
        candidates["distance"] = np.linalg.norm(
            candidates[columns].to_numpy(float) - primary[columns].to_numpy(float), axis=1
        )
        st.subheader("Closest profiles in the selected competitions and season")
        st.dataframe(
            candidates.sort_values("distance").head(10)[
                ["player_name", "competition_id", "season", "minutes", "distance"]
            ],
            hide_index=True,
        )
    st.subheader("Peer scatterplot")
    axes = [name for name in METRICS if current[f"{name}_per90"].notna().any()]
    if len(axes) >= 2:
        defaults = [name for name in PROFILES.get(role, PROFILES["MID"]) if name in axes]
        left, right = st.columns(2)
        x = left.selectbox(
            "X metric",
            axes,
            index=axes.index(defaults[0]) if defaults else 0,
            format_func=lambda name: METRICS[name].label,
        )
        y = right.selectbox(
            "Y metric",
            axes,
            index=axes.index(defaults[1]) if len(defaults) > 1 else 1,
            format_func=lambda name: METRICS[name].label,
        )
        peers = current.loc[
            current.minutes.ge(minimum)
            & current[f"{x}_coverage"].ge(0.9)
            & current[f"{y}_coverage"].ge(0.9)
        ]
        fig = go.Figure(
            go.Scatter(
                x=peers[f"{x}_per90"],
                y=peers[f"{y}_per90"],
                text=peers.player_name,
                mode="markers",
                name="Eligible peers",
                marker={"color": "#94a3b8", "size": 8},
                hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )
        for color, (_, player) in zip(COLORS, selected.iterrows(), strict=False):
            if min(player[f"{x}_coverage"], player[f"{y}_coverage"]) < 0.9:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[player[f"{x}_per90"]],
                    y=[player[f"{y}_per90"]],
                    mode="markers",
                    name=f"{player.player_name} · {player.season}",
                    marker={"color": color, "size": 15, "line": {"color": "white", "width": 1}},
                )
            )
        fig.update_layout(
            template="plotly_dark",
            height=480,
            xaxis_title=METRICS[x].label + (" / 90" if METRICS[x].kind == "count" else ""),
            yaxis_title=METRICS[y].label + (" / 90" if METRICS[y].kind == "count" else ""),
        )
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption(
            "Grey points: selected season and role. Colored points: comparison selections, "
            "possibly other seasons. Counts are per 90 observed metric minutes."
        )
    per90 = st.radio("Performance values", ["Total", "Per 90"], horizontal=True) == "Per 90"
    _performance(primary, per90)
    with st.expander("Definitions and missing statistics"):
        st.write(
            "Percentage fields labelled match avg are minutes-weighted means of the provider's "
            "rounded match percentages, not season success/attempt ratios. Duels won % and goal "
            "conversion use summed known numerators/denominators. No attempts are fabricated. "
            "Progressive passes/carries, forward-pass accuracy, headed goals, big chances created "
            "and player cards are absent from this adapter. Final-third passes are not progressive "
            "passes. These descriptive features do not enter the frozen prediction models."
        )
