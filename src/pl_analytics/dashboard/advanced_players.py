"""Position-aware advanced player profiles, multi-player radars and peer scatterplots."""

import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.dashboard import theme
from pl_analytics.dashboard.formatting import top_percent_label
from pl_analytics.features.advanced_players import METRICS, PROFILES, peer_percentiles

COLORS = theme.SERIES[:5]


@st.cache_data(show_spinner=False)
def _ranked(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    return peer_percentiles(frame, min_minutes=minimum)


def _chip(short_name: object) -> str:
    """Club chip when the profile carries a club code; older profiles do not."""
    return theme.club_chip(short_name) if isinstance(short_name, str) and short_name else ""


def radar_chart(players: pd.DataFrame, metrics: list[str], *, height: int = 480) -> go.Figure:
    """Plot comparable percentile axes, leaving absent observations as gaps, never zero."""
    figure = go.Figure()
    # Long axis names wrap onto two lines so the radar keeps its size on a phone.
    labels = [
        "<br>".join(textwrap.wrap(METRICS[name].label, 16)) or METRICS[name].label
        for name in metrics
    ]
    for color, (_, player) in zip(COLORS, players.iterrows(), strict=False):
        values = [player[f"{name}_percentile"] for name in metrics]
        top_shares = [None if pd.isna(value) else max(1, 100 - float(value)) for value in values]
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
                customdata=top_shares + top_shares[:1],
                hovertemplate="%{theta}<br>Top %{customdata:.0f}%<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        template=theme.TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": theme.TEXT, "size": 12},
        polar={
            "bgcolor": theme.CARD_RAISED,
            "radialaxis": {"range": [0, 100], "dtick": 25, "tickfont": {"size": 10}},
        },
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 64, "r": 64, "t": 36, "b": 70},
    )
    return figure


def _performance(player: pd.Series, per90: bool) -> None:
    groups = list(dict.fromkeys(metric.group for metric in METRICS.values()))
    if player.position_group != "GK":
        groups.remove("Goalkeeping")
    head = "<tr><th>METRIC</th><th>VALUE</th><th>RANK</th><th>COVERAGE</th></tr>"
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
                rank = top_percent_label(percentile)
                colour = (
                    theme.ACCENT_TEXT
                    if pd.notna(percentile) and percentile >= 75
                    else theme.TEXT_SOFT
                )
                rows.append(
                    f'<tr><td style="color:{theme.TEXT}">{html.escape(metric.label)}</td>'
                    f'<td>{display}</td><td style="color:{colour}">{rank}</td>'
                    f"<td>{coverage:.0%}</td></tr>"
                )
            column.markdown(
                theme.card(
                    group,
                    f'<table class="pl-table" style="font-size:13.5px">{head}{"".join(rows)}'
                    "</table>",
                ),
                unsafe_allow_html=True,
            )


def _similar(universe: pd.DataFrame, primary: pd.Series, metrics: list[str]) -> None:
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
    closest = candidates.sort_values("distance").head(10)
    if closest.empty:
        st.info("No eligible candidates share these metrics in the selected competitions.")
        return
    scale = max(float(closest.distance.max()), 1e-9)
    rows = "".join(
        '<div class="pl-list-row">'
        f"{_chip(getattr(row, 'club_short_name', None))}"
        f'<span class="grow"><b>{html.escape(str(row.player_name))}</b>'
        f'<span class="pl-muted">{html.escape(str(row.competition_id))} · {row.season} · '
        f"{row.minutes:,.0f} minutes</span></span>"
        f'<span style="width:120px"><span style="display:block;height:8px;border-radius:4px;'
        f'background:{theme.GRIDLINE}"><i style="display:block;height:100%;border-radius:4px;'
        f"width:{100 * (1 - row.distance / scale * 0.85):.0f}%;background:{theme.ACCENT}\"></i>"
        f'</span></span><span class="big" style="width:52px;text-align:right">'
        f"{row.distance:.0f}</span></div>"
        for row in closest.itertuples(index=False)
    )
    note = (
        '<div class="pl-muted" style="margin-top:10px">Number: distance across the selected '
        "percentile metrics, smaller is closer; the bar is longer for closer profiles. Closest "
        "statistical profiles, not best players, and no cross-league strength adjustment.</div>"
    )
    st.markdown(
        theme.card("Closest profiles in the selected competitions and season", rows + note),
        unsafe_allow_html=True,
    )


def _peer_scatter(
    current: pd.DataFrame, selected: pd.DataFrame, role: str, minimum: int
) -> None:
    axes = [name for name in METRICS if current[f"{name}_per90"].notna().any()]
    if len(axes) < 2:
        st.info("Fewer than two metrics have recorded values for this season and role.")
        return
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
            marker={"color": theme.MUTED, "size": 8},
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
        template=theme.TEMPLATE,
        height=480,
        xaxis_title=METRICS[x].label + (" / 90" if METRICS[x].kind == "count" else ""),
        yaxis_title=METRICS[y].label + (" / 90" if METRICS[y].kind == "count" else ""),
        legend={"orientation": "h", "y": -0.2},
    )
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption(
        "Grey points: selected season and role. Colored points: comparison selections, "
        "possibly other seasons. Counts are per 90 observed metric minutes."
    )


def render_advanced(
    frame: pd.DataFrame, all_seasons: pd.DataFrame, *, scouting: bool = False
) -> None:
    """Render actual prepared observations; role-specific metrics replace goal-only comparisons."""
    roles = sorted(frame.position_group.unique())
    first, second, third = st.columns([1, 1, 2])
    role = first.selectbox(
        "Position group", roles, index=roles.index("DEF") if "DEF" in roles else 0
    )
    # Early in a season nobody has 450 minutes yet, so start from half the most played.
    most_played = float(pd.to_numeric(frame.minutes, errors="coerce").max())
    default_minimum = 450 if most_played >= 900 else max(30, int(most_played / 2 // 30) * 30)
    minimum = second.number_input(
        "Peer minimum minutes",
        min_value=1,
        max_value=4000,
        value=default_minimum,
        step=30,
        help="Top-% ranks need at least five eligible peers and 90% metric coverage. Fewer "
        "than 450 minutes gives an unstable early-season comparison.",
    )
    ranked = _ranked(all_seasons, int(minimum))
    current = ranked.loc[
        ranked.season.eq(frame.season.iloc[0])
        & ranked.position_group.eq(role)
        & ranked.competition_id.eq(frame.competition_id.iloc[0])
    ]
    names = current.set_index("player_id").player_name.to_dict()
    requested_player = st.query_params.get("player")
    player_options = sorted(names, key=lambda key: names[key])
    # Without a requested player, open on the role's most-used player, who has a full profile.
    busiest = current.loc[pd.to_numeric(current.minutes, errors="coerce").idxmax(), "player_id"]
    player_index = player_options.index(
        requested_player if requested_player in names else busiest
    )
    selected_id = third.selectbox(
        "Player",
        player_options,
        index=player_index,
        format_func=lambda key: names[key],
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

    chip = _chip(primary.get("club_short_name"))
    partial = (
        " · newest snapshot, may be a partial season"
        if frame.season.iloc[0] == sorted(all_seasons.season.unique())[-1]
        else ""
    )
    st.markdown(
        f'<div class="pl-card-title" style="margin-top:8px">{chip} '
        f"{html.escape(str(primary.player_name))}</div>"
        f'<div class="pl-muted" style="margin:-8px 0 12px">'
        f"{html.escape(str(primary.competition_id))} · {primary.season} · {role} · last recorded "
        f"match {str(primary.last_match_date)[:10]}{partial}</div>",
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Minutes", f"{primary.minutes:,.0f}")
    b.metric("Appearances", int(primary.appearances))
    c.metric("Comparison players", len(selected))

    available = [name for name in METRICS if selected[f"{name}_percentile"].notna().all()]
    defaults = [name for name in PROFILES.get(role, PROFILES["MID"]) if name in available]
    metrics = st.multiselect(
        "Radar metrics",
        available,
        default=defaults[:8],
        max_selections=12,
        format_func=lambda name: METRICS[name].label,
    )
    enough = len(metrics) >= 3
    if not enough:
        st.info(
            "Choose at least three shared eligible metrics. Lower the minutes threshold or "
            "choose another player/season if the sample is insufficient."
        )
    tab_names = ["Profile", "Peers", "Season statistics"]
    if scouting:
        tab_names.insert(0, "Similar players")
    tabs = dict(zip(tab_names, st.tabs(tab_names), strict=True))
    if scouting:
        with tabs["Similar players"]:
            if enough:
                _similar(universe, primary, metrics)
            else:
                st.caption("Similarity needs at least three shared metrics.")
    with tabs["Profile"]:
        if enough:
            st.plotly_chart(radar_chart(selected, metrics), width="stretch", theme=None)
            st.caption(
                "All curves use the same 0–100 axes. Missing or insufficiently covered metrics "
                "are omitted from the selector; their absence is not poor performance."
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
                        "player_name": "Player",
                        "competition_id": "Competition",
                        "season": "Season",
                        "minutes": "Minutes",
                        **{
                            f"{name}_per90": METRICS[name].label
                            + (" / 90" if METRICS[name].kind == "count" else "")
                            for name in metrics
                        },
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    with tabs["Peers"]:
        _peer_scatter(current, selected, role, int(minimum))
    with tabs["Season statistics"]:
        per90 = st.radio("Performance values", ["Total", "Per 90"], horizontal=True) == "Per 90"
        st.caption(
            "— means unavailable, not zero. Coverage is the share of player minutes with a "
            "value. A partial total is not a complete season total."
        )
        _performance(primary, per90)
    with st.expander("Definitions, sources and missing statistics"):
        st.write(
            "Published FPL Core match statistics. Roles are broad season snapshots: DEF includes "
            "centre-backs and full-backs. Different-season radars compare relative standing in "
            "each season, not identical opponents or a strength-adjusted performance level. "
            "Top-% ranks use the same competition, season and broad FPL position; higher is the "
            "chosen direction, not overall quality."
        )
        st.write(
            "Percentage fields labelled match avg are minutes-weighted means of the provider's "
            "rounded match percentages, not season success/attempt ratios. Duels won % and goal "
            "conversion use summed known numerators/denominators. No attempts are fabricated. "
            "Progressive passes/carries, forward-pass accuracy, headed goals, big chances created "
            "and player cards are absent from this adapter. Final-third passes are not progressive "
            "passes. These descriptive features do not enter the frozen prediction models."
        )
