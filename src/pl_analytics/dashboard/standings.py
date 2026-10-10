"""Observed league table with xG-derived expected performance."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard import theme
from pl_analytics.dashboard.data import ArtifactError, table
from pl_analytics.dashboard.team_compare import render_team_compare, render_team_table

STANDING_COLUMNS = (
    "competition_id",
    "season",
    "position",
    "club_name",
    "played",
    "points",
    "expected_goals",
    "expected_goals_against",
    "expected_points",
)


def form_guide(results: pd.DataFrame) -> dict[str, str]:
    """Last five results per club short name, oldest first."""
    form: dict[str, str] = {}
    if results.empty:
        return form
    for match in results.sort_values("kickoff_time").itertuples(index=False):
        home, away = int(match.home_score), int(match.away_score)
        home_letter = "W" if home > away else "D" if home == away else "L"
        away_letter = {"W": "L", "D": "D", "L": "W"}[home_letter]
        form[match.home_short_name] = (form.get(match.home_short_name, "") + home_letter)[-5:]
        form[match.away_short_name] = (form.get(match.away_short_name, "") + away_letter)[-5:]
    return form


WIDE = ' class="pl-wide"'  # table cells hidden on phone-width screens


def _signed(value: float, digits: int = 1, *, wide: bool = False) -> str:
    colour = (
        theme.ACCENT_TEXT if value > 0.05 else theme.BAD_TEXT if value < -0.05 else theme.TEXT_SOFT
    )
    return f'<td{WIDE if wide else ""} style="color:{colour}">{value:+.{digits}f}</td>'


def _summary(standings: pd.DataFrame, matches: int, with_xg: int) -> None:
    leader = standings.sort_values("position").iloc[0]
    over = standings.nlargest(1, "expected_points_difference").iloc[0]
    under = standings.nsmallest(1, "expected_points_difference").iloc[0]
    cards = (
        ("Leaders", str(leader.club_name), f"{int(leader.points)} pts from {int(leader.played)}"),
        (
            "Most points above xPts",
            str(over.club_name),
            f"{over.expected_points_difference:+.1f} against its xG scorelines",
        ),
        (
            "Most points below xPts",
            str(under.club_name),
            f"{under.expected_points_difference:+.1f} against its xG scorelines",
        ),
        ("Matches played", str(matches), f"xG recorded for {with_xg}"),
    )
    theme.stat_row(cards)


def _table(standings: pd.DataFrame, form: dict[str, str]) -> None:
    head = (
        f"<tr><th>#</th><th>CLUB</th><th>P</th><th{WIDE}>W</th><th{WIDE}>D</th><th{WIDE}>L</th>"
        f"<th{WIDE}>GF</th><th{WIDE}>GA</th><th>GD</th><th>PTS</th><th{WIDE}>xG</th>"
        f"<th{WIDE}>xGA</th><th{WIDE}>xGD</th><th>xPTS</th><th>PTS − xPTS</th>"
        f"<th{WIDE}>FORM</th></tr>"
    )
    last = int(standings.position.max())
    rows = []
    for club in standings.sort_values("position").itertuples(index=False):
        position = int(club.position)
        if position <= 4:
            zone = f"background:{theme.ACCENT};color:{theme.ON_ACCENT}"
        elif position > last - 3:
            zone = f"background:{theme.BAD};color:{theme.ON_BAD}"
        else:
            zone = f"color:{theme.TEXT_SOFT}"
        short = getattr(club, "club_short_name", "")
        rows.append(
            "<tr>"
            f'<td><span class="pl-badge" style="{zone};border-radius:6px;padding:2px 7px">'
            f"{position}</span></td>"
            f'<td style="white-space:nowrap;color:{theme.TEXT}">{theme.club_chip(short)} '
            f"{escape(str(club.club_name))}</td>"
            f"<td>{int(club.played)}</td><td{WIDE}>{int(club.wins)}</td>"
            f"<td{WIDE}>{int(club.draws)}</td><td{WIDE}>{int(club.losses)}</td>"
            f"<td{WIDE}>{int(club.goals_for)}</td><td{WIDE}>{int(club.goals_against)}</td>"
            f"<td>{int(club.goal_difference):+d}</td>"
            f'<td style="color:{theme.TEXT};font-weight:700;font-size:14px">{int(club.points)}</td>'
            f"<td{WIDE}>{club.expected_goals:.1f}</td>"
            f"<td{WIDE}>{club.expected_goals_against:.1f}</td>"
            f"{_signed(float(club.expected_goal_difference), wide=True)}"
            f"<td>{club.expected_points:.1f}</td>"
            f"{_signed(float(club.expected_points_difference))}"
            f'<td{WIDE} style="white-space:nowrap">{theme.form_pills(form.get(short, ""))}</td>'
            "</tr>"
        )
    legend = (
        '<div class="pl-muted" style="margin-top:10px">'
        f'<b style="color:{theme.ACCENT}">■</b> Champions League places · '
        f'<b style="color:{theme.BAD}">■</b> Relegation · xPts = 3 × P(win) + P(draw) from each '
        "match's xG, assuming independent Poisson scores."
        '<span class="pl-narrow"> Turn the phone sideways to see every column.</span></div>'
    )
    st.markdown(
        theme.card(
            "Table",
            f'<div style="overflow-x:auto"><table class="pl-table">{head}{"".join(rows)}'
            f"</table></div>{legend}",
        ),
        unsafe_allow_html=True,
    )


def _performance(standings: pd.DataFrame) -> None:
    """Diverging bars: points collected minus the points the xG scorelines implied."""
    ordered = standings.sort_values("expected_points_difference", ascending=False)
    scale = max(float(ordered.expected_points_difference.abs().max()), 1e-9)
    rows = []
    for club in ordered.itertuples(index=False):
        value = float(club.expected_points_difference)
        width = 50 * abs(value) / scale
        colour = theme.ACCENT if value >= 0 else theme.BAD
        offset = 50 if value >= 0 else 50 - width
        rows.append(
            '<div class="pl-list-row">'
            f'<span style="width:150px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f"{theme.club_chip(getattr(club, 'club_short_name', ''))} "
            f"{escape(str(club.club_name))}</span>"
            '<span class="grow"><span style="position:relative;height:10px;border-radius:5px;'
            f'background:{theme.GRIDLINE};display:block">'
            f'<i style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;'
            f'background:{theme.MUTED}"></i>'
            f'<i style="position:absolute;left:{offset:.1f}%;width:{width:.1f}%;top:0;bottom:0;'
            f'border-radius:5px;background:{colour}"></i></span></span>'
            f'<span class="big" style="width:52px;text-align:right">{value:+.1f}</span></div>'
        )
    note = (
        '<div class="pl-muted" style="margin-top:10px">Right of the centre line: more points '
        "than the xG scorelines implied; left: fewer. This is descriptive and does not prove "
        "luck or quality.</div>"
    )
    st.markdown(
        theme.card("Points collected minus expected points", "".join(rows) + note),
        unsafe_allow_html=True,
    )


def _share(value: float) -> str:
    """Probability as text; tiny non-zero chances are not rounded to nothing."""
    if value <= 0:
        return "—"
    if value < 0.005:
        return "&lt;1%"
    if 0.995 < value < 1:
        return "&gt;99%"
    return f"{value:.0%}"


def _bar_cell(value: float, colour: str) -> str:
    return (
        '<td><span style="display:inline-flex;align-items:center;gap:8px;'
        'justify-content:flex-end;width:100%">'
        f'<span style="width:56px;height:8px;border-radius:4px;background:{theme.GRIDLINE};'
        f'display:inline-block"><i style="display:block;height:100%;border-radius:4px;'
        f'width:{100 * min(max(value, 0.0), 1.0):.0f}%;background:{colour}"></i></span>'
        f'<span style="width:38px;text-align:right">{_share(value)}</span></span></td>'
    )


def _outlook(simulation: pd.DataFrame) -> None:
    """Simulated end-of-season outcomes: points range and title, top and relegation odds."""
    if simulation.empty:
        st.info("The season simulation is not built yet. Run the current-data refresh.")
        return
    runs = int(simulation.simulations.iloc[0])
    top, bottom = int(simulation.top_places.iloc[0]), int(simulation.relegation_places.iloc[0])
    ordered = simulation.sort_values("expected_position")
    scale = float(ordered.points_high.max())
    head = (
        f"<tr><th>#</th><th>CLUB</th><th{WIDE}>NOW</th><th>PROJECTED</th><th{WIDE}>RANGE</th>"
        f"<th>TITLE</th><th>TOP {top}</th><th>RELEGATION</th></tr>"
    )
    rows = []
    for place, club in enumerate(ordered.itertuples(index=False), start=1):
        rows.append(
            "<tr>"
            f"<td>{place}</td>"
            f'<td style="white-space:nowrap;color:{theme.TEXT}">'
            f"{theme.club_chip(club.club_short_name)} {escape(str(club.club_name))}</td>"
            f"<td{WIDE}>{club.current_points:.0f}</td>"
            f'<td style="color:{theme.TEXT};font-weight:700;font-size:14px">'
            f"{club.mean_points:.0f}</td>"
            f'<td{WIDE} style="min-width:150px">'
            '<span style="display:flex;align-items:center;gap:8px">'
            f'<span style="flex:1">'
            f"{theme.range_bar(club.points_low, club.mean_points, club.points_high, scale)}"
            f'</span><span style="white-space:nowrap">{club.points_low:.0f}–'
            f"{club.points_high:.0f}</span></span></td>"
            f"{_bar_cell(float(club.title_probability), theme.ACCENT)}"
            f"{_bar_cell(float(club.top_probability), '#7AA7FF')}"
            f"{_bar_cell(float(club.relegation_probability), theme.BAD)}"
            "</tr>"
        )
    note = (
        f'<div class="pl-muted" style="margin-top:10px">{runs:,} simulated seasons of the '
        f"{int(simulation.remaining_matches.iloc[0])} remaining matches, using the live goal "
        "model's expected goals. Clubs are ordered by average finishing position; the range "
        "covers 90% of simulations. Team strength is held fixed all season, so form changes, "
        f"injuries and transfers are not included and the probabilities are too confident. "
        f"Relegation means the bottom {bottom}.</div>"
    )
    st.markdown(
        theme.card(
            "Season outlook",
            f'<div style="overflow-x:auto"><table class="pl-table">{head}{"".join(rows)}'
            f"</table></div>{note}",
        ),
        unsafe_allow_html=True,
    )
    places = [column for column in simulation if column.startswith("position_")]
    figure = go.Figure(
        go.Heatmap(
            z=ordered[places].to_numpy(float),
            x=[column.removeprefix("position_") for column in places],
            y=ordered.club_short_name,
            colorscale=theme.SEQUENTIAL,
            zmin=0,
            zmax=0.5,
            xgap=2,
            ygap=2,
            showscale=False,
            hovertemplate="%{y} finishing %{x}: %{z:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=26 * len(ordered) + 90,
        margin={"l": 10, "r": 10, "t": 10, "b": 40},
        xaxis={"title": "Final position", "showgrid": False, "type": "category"},
        yaxis={"autorange": "reversed", "showgrid": False, "ticksuffix": "  "},
    )
    with st.container(border=True):
        st.markdown(
            '<div class="pl-card-title">Where each club finishes</div>', unsafe_allow_html=True
        )
        st.plotly_chart(
            figure, width="stretch", key="season-positions", config={"displayModeBar": False}
        )
        st.caption(
            "Brighter cells are more likely finishing positions; hover for the exact share. "
            "The colour scale tops out at 50%."
        )


def render_standings(settings: Settings, competition: str) -> None:
    """Render current standings and explain xG-derived expected points."""
    root = settings.data_dir / "processed/fpl"
    standings = table(root / "standings.parquet", STANDING_COLUMNS)
    standings = standings.loc[standings.competition_id.eq(competition)].copy()
    if standings.empty:
        raise ArtifactError(
            "League-table artifacts are not built yet. Run the current-data refresh."
        )
    results = table(root / "results.parquet")
    required = {"kickoff_time", "home_short_name", "away_short_name", "home_score", "away_score"}
    form = form_guide(results) if required <= set(results) else {}
    matches = int(standings.played.sum() / 2)
    with_xg = int(standings.xg_matches.sum() / 2)
    st.markdown(
        f'<div class="pl-eyebrow" style="margin-bottom:10px">Season '
        f"{escape(str(standings.season.iloc[0]))}</div>",
        unsafe_allow_html=True,
    )
    _summary(standings, matches, with_xg)
    simulation = table(root / "season_simulation.parquet")
    if not simulation.empty:
        simulation = simulation.loc[simulation.competition_id.eq(competition)]
    table_tab, outlook_tab, performance_tab, stats_tab, compare_tab = st.tabs(
        ["Table", "Season outlook", "Points vs expected", "Team statistics", "Compare teams"]
    )
    with table_tab:
        _table(standings, form)
    with outlook_tab:
        _outlook(simulation)
    with performance_tab:
        _performance(standings)
    team_matches = table(settings.data_dir / "processed/advanced/team_matches.parquet")
    if not team_matches.empty:
        team_matches = team_matches.loc[team_matches.competition_id.eq(competition)]
    season = str(standings.season.iloc[0])
    with stats_tab:
        render_team_table(team_matches, season=season, key="league_team_table")
    with compare_tab:
        leaders = standings.sort_values("position").get("club_short_name", pd.Series(dtype=str))
        render_team_compare(
            team_matches,
            season=season,
            default_clubs=tuple(leaders.dropna().head(2)),
            key="league_team_compare",
        )
