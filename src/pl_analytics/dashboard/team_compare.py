"""Club-against-club comparison of recorded match statistics."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.dashboard import theme
from pl_analytics.dashboard.advanced_players import COLORS, radar_chart
from pl_analytics.features.team_stats import GROUPS, SCOPES, STATS, summarize_teams

RADAR_DEFAULTS = (
    "xg",
    "shots",
    "box_touches",
    "possession",
    "pass_accuracy",
    "corners",
    "xg_against",
    "shots_against",
    "tackles_won",
)
LABELS = {name: stat.label for name, stat in STATS.items()}


@st.cache_data(show_spinner=False)
def _summary(rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    return summarize_teams(rows, scope)


def _ordinal(rank: float) -> str:
    number = int(rank)
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return str(number) + {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def _value(value: float, name: str) -> str:
    if pd.isna(value):
        return "—"
    stat = STATS[name]
    if stat.denominator or stat.percentage:
        return f"{value:.1f}%"
    return f"{value:.2f}"


def _table(selected: pd.DataFrame, names: list[str]) -> str:
    """Statistics as rows and clubs as columns, with the best club in each row marked."""
    right = "text-align:right"
    head = "<tr><th>STATISTIC</th>" + "".join(
        f'<th style="{right}">{theme.club_chip(club.club_short_name)}</th>'
        for club in selected.itertuples()
    )
    body, group = [], None
    for name in names:
        stat = STATS[name]
        if stat.group != group:
            group = stat.group
            body.append(
                f'<tr><td colspan="{len(selected) + 1}" style="color:{theme.MUTED};font-size:11px;'
                f'letter-spacing:.06em;font-weight:600;padding-top:14px">{escape(group.upper())}'
                "</td></tr>"
            )
        values = selected[name]
        best = values.min() if not stat.higher_is_better else values.max()
        cells = []
        for value, rank in zip(values, selected[f"{name}_rank"], strict=True):
            lead = len(selected) > 1 and pd.notna(value) and value == best
            style = f"{right};" + (f"color:{theme.ACCENT_TEXT};font-weight:700" if lead else "")
            place = "" if pd.isna(rank) else (
                f' <span style="color:{theme.MUTED};font-size:11px">{_ordinal(rank)}</span>'
            )
            cells.append(f'<td style="{style}">{_value(value, name)}{place}</td>')
        arrow = "" if stat.higher_is_better else (
            f' <span style="color:{theme.MUTED}" title="Lower is better">↓</span>'
        )
        body.append(f"<tr><td>{escape(stat.label)}{arrow}</td>{''.join(cells)}</tr>")
    return (
        '<div style="overflow-x:auto"><table class="pl-table">'
        f"<thead>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _scatter(league: pd.DataFrame, selected: pd.DataFrame, x: str, y: str) -> go.Figure:
    others = league.loc[~league.club_id.isin(selected.club_id)]
    figure = go.Figure(
        go.Scatter(
            x=others[x],
            y=others[y],
            text=others.club_short_name,
            mode="markers+text",
            textposition="top center",
            textfont={"color": theme.MUTED, "size": 10},
            name="Other clubs",
            marker={"color": theme.MUTED, "size": 8, "opacity": 0.65},
            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )
    for color, club in zip(COLORS, selected.itertuples(), strict=False):
        figure.add_trace(
            go.Scatter(
                x=[getattr(club, x)],
                y=[getattr(club, y)],
                text=[club.club_short_name],
                mode="markers+text",
                textposition="top center",
                textfont={"color": theme.TEXT, "size": 11},
                name=club.club_name,
                marker={"color": color, "size": 15, "line": {"color": "white", "width": 1}},
                hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )
    average = {"color": theme.MUTED, "width": 1, "dash": "dot"}
    figure.add_vline(x=league[x].mean(), line=average)
    figure.add_hline(y=league[y].mean(), line=average)
    figure.update_layout(
        template=theme.TEMPLATE,
        height=480,
        xaxis_title=STATS[x].label,
        yaxis_title=STATS[y].label,
        xaxis_autorange="reversed" if not STATS[x].higher_is_better else True,
        yaxis_autorange="reversed" if not STATS[y].higher_is_better else True,
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def _season_rows(rows: pd.DataFrame | None, season: str) -> pd.DataFrame | None:
    if rows is None or rows.empty or not rows.season.eq(season).any():
        st.info(
            "Club match statistics are not prepared for this season. "
            "Run `python scripts/refresh_current_data.py` to build them."
        )
        return None
    return rows.loc[rows.season.eq(season)]


def render_team_table(rows: pd.DataFrame | None, *, season: str, key: str = "team_table") -> None:
    """Every club of one season in a table that sorts by any statistic."""
    season_rows = _season_rows(rows, season)
    if season_rows is None:
        return
    left, middle, right = st.columns([2, 1, 1])
    scope = left.radio("Period", SCOPES, horizontal=True, key=f"{key}_scope")
    group = middle.selectbox("Statistics group", GROUPS, key=f"{key}_group")
    show = right.radio("Show", ["Per match", "League rank"], horizontal=True, key=f"{key}_show")
    league = _summary(season_rows, scope)
    names = [
        name
        for name, stat in STATS.items()
        if stat.group == group and league[name].notna().any()
    ]
    if not names:
        st.info("The source has no recorded values for this group and period.")
        return
    ranks = show == "League rank"
    frame = pd.DataFrame({"Club": league.club_name, "Matches": league.matches.astype(int)})
    config = {"Club": st.column_config.TextColumn(pinned=True)}
    for name in names:
        stat = STATS[name]
        label = stat.label if stat.higher_is_better else f"{stat.label} \u2193"
        frame[label] = league[f"{name}_rank"] if ranks else league[name]
        if ranks:
            config[label] = st.column_config.NumberColumn(format="%d")
        elif stat.denominator or stat.percentage:
            config[label] = st.column_config.NumberColumn(format="%.1f%%")
        else:
            config[label] = st.column_config.NumberColumn(format="%.2f")
    first = STATS[names[0]]
    frame = frame.sort_values(
        frame.columns[2], ascending=ranks or not first.higher_is_better, ignore_index=True
    )
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=38 * (len(frame) + 1),
        column_config=config,
    )
    st.caption(
        "Click any column header to sort ascending or descending. Values are averages per "
        "match (percentages are totals over the period); \u2193 marks statistics where lower "
        "is better, and League rank puts the best club at 1 either way. Recorded match "
        "statistics, not predictions."
    )


def render_team_compare(
    rows: pd.DataFrame | None,
    *,
    season: str,
    club_id: str = "",
    default_clubs: tuple[str, ...] = (),
    key: str = "team_compare",
) -> None:
    """Overlay up to five clubs of one season on a table, a radar and a scatterplot.

    ``default_clubs`` are club short names to preselect instead of ``club_id``.
    """
    season_rows = _season_rows(rows, season)
    if season_rows is None:
        return
    clubs = (
        season_rows[["club_id", "club_name"]]
        .drop_duplicates()
        .sort_values("club_name")
        .set_index("club_id")
        .club_name.to_dict()
    )
    first = club_id if club_id in clubs else next(iter(clubs))
    played = season_rows.groupby("club_id").match_id.nunique()
    rival = next((item for item in clubs if item != first), None)
    shorts = season_rows.drop_duplicates("club_id").set_index("club_short_name").club_id
    preset = [shorts[name] for name in default_clubs if name in shorts.index]
    left, middle, right = st.columns([2, 2, 1])
    selected_ids = left.multiselect(
        "Compare teams (up to 5)",
        list(clubs),
        default=preset or [item for item in (first, rival) if item],
        max_selections=5,
        format_func=lambda item: clubs[item],
        key=f"{key}_{season}_{first}",
    )
    scope = middle.radio("Period", SCOPES, horizontal=True, key=f"{key}_scope")
    group = right.selectbox("Statistics group", ["All", *GROUPS], key=f"{key}_group")
    if not selected_ids:
        st.info("Select at least one team to compare.")
        return

    league = _summary(season_rows, scope)
    selected = league.set_index("club_id", drop=False).loc[selected_ids].reset_index(drop=True)
    names = [
        name
        for name, stat in STATS.items()
        if (group == "All" or stat.group == group) and league[name].notna().any()
    ]
    if not names:
        st.info("The source has no recorded values for this group and period.")
        return
    theme.stat_row(
        [
            (
                club.club_name,
                f"{club.xg:.2f} xG" if pd.notna(club.xg) else "—",
                f"{club.xg_against:.2f} conceded · {int(club.matches)} "
                f"match{'es' if club.matches != 1 else ''}",
            )
            for club in selected.itertuples()
        ]
    )

    table_column, radar_column = st.columns([5, 4], gap="medium")
    with table_column:
        st.markdown(theme.card("Per match", _table(selected, names)), unsafe_allow_html=True)
    with radar_column:
        st.markdown('<div class="pl-card-title">League rank radar</div>', unsafe_allow_html=True)
        defaults = RADAR_DEFAULTS if group == "All" else names
        radar_stats = st.multiselect(
            "Radar statistics",
            names,
            default=[name for name in defaults if name in names][:10],
            max_selections=12,
            format_func=LABELS.get,
            key=f"{key}_radar_{group}",
        )
        if len(radar_stats) >= 3:
            st.plotly_chart(
                radar_chart(
                    selected, radar_stats, axis_labels=LABELS, name_column="club_name"
                ),
                width="stretch",
                theme=None,
            )
            st.caption(
                "The outer edge is the best club in the league for that statistic and the "
                "centre the worst. For statistics where lower is better (conceded, fouls, "
                "cards) the scale is flipped, so further out is always better."
            )
        else:
            st.info("Choose at least three statistics for the radar.")

    st.markdown('<div class="pl-card-title">League scatterplot</div>', unsafe_allow_html=True)
    all_names = [name for name in STATS if league[name].notna().any()]
    x_column, y_column = st.columns(2)
    x_default = "xg" if "xg" in all_names else all_names[0]
    y_default = "xg_against" if "xg_against" in all_names else all_names[-1]
    x = x_column.selectbox(
        "X statistic",
        all_names,
        index=all_names.index(x_default),
        format_func=LABELS.get,
        key=f"{key}_x",
    )
    y = y_column.selectbox(
        "Y statistic",
        all_names,
        index=all_names.index(y_default),
        format_func=LABELS.get,
        key=f"{key}_y",
    )
    st.plotly_chart(_scatter(league, selected, x, y), width="stretch", theme=None)

    fewest = int(played.min())
    sample = (
        f" Clubs have played as few as {fewest} matches this season, so every figure is a "
        "small sample."
        if fewest < 10
        else ""
    )
    period = {
        "Season": "all matches of the season",
        "Last 5 matches": "each club's five most recent matches",
        "Home": "home matches only",
        "Away": "away matches only",
    }[scope]
    st.caption(
        f"Averages per match over {period}; percentages are totals over the period. Ranks "
        "compare all clubs of the season over the same period; ↓ marks statistics where "
        "lower is better and the leader among the selected clubs is highlighted. Dotted "
        "lines on the scatterplot mark the league "
        "average and axes are flipped where lower is better, so the top-right corner is "
        f"always the strongest. These are recorded match statistics, not predictions.{sample}"
    )
