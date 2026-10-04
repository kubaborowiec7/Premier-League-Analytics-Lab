"""FPL Decision Lab: gameweek decisions built only from refresh-time artifacts.

Four tabs follow the weekly decision: what matters this gameweek, the player pool,
fixture runs, and the manager's own squad with a transfer planner. One planning horizon
in the header drives every tab. Nothing is fitted here; the only network call is the
explicit public-team lookup.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard import theme
from pl_analytics.dashboard.data import ArtifactError, report, table
from pl_analytics.data.fpl import (
    MAX_LEAGUE_TEAMS,
    fetch_league_picks,
    fetch_league_standings,
    fetch_public_team,
    manager_summary,
    map_public_squad,
)
from pl_analytics.features.fpl import league_ownership, price_watch
from pl_analytics.features.fpl_optimizer import (
    best_single_transfers,
    optimize_chip_schedule,
    optimize_squad,
    plan_transfers,
    replacement_options,
    select_starting_xi,
)

UK = "Europe/London"
DEADLINE_LEAD = pd.Timedelta(minutes=90)
POSITIONS = ("GK", "DEF", "MID", "FWD")
DIFFICULTY_VIEWS = {
    "Overall": "difficulty",
    "Attack": "attack_difficulty",
    "Defence": "defence_difficulty",
}
COMPONENTS = {
    "Appearance": "xpts_appearance_1",
    "Goals and assists": "xpts_attack_1",
    "Clean sheet": "xpts_clean_sheet_1",
    "Saves and penalties": "xpts_saves_1",
    "Defensive contributions": "xpts_defensive_1",
    "Bonus": "xpts_bonus_1",
    "Cards and errors": "xpts_discipline_1",
    "Goals conceded": "xpts_goals_conceded_1",
}
# (label, column, format, True when higher is better / False when lower / None when neutral)
HEAD_TO_HEAD = (
    ("xPts", None, "{:.1f}", True),
    ("xMins", "expected_minutes", "{:.0f}", True),
    ("Price £m", "price", "{:.1f}", False),
    ("Form", "form", "{:.1f}", True),
    ("xG / 90", "projected_xg_per90", "{:.2f}", True),
    ("xA / 90", "projected_xa_per90", "{:.2f}", True),
    ("Def. actions / 90", "projected_defensive_contribution_per90", "{:.1f}", True),
    ("Owned %", "ownership_pct", "{:.1f}", None),
)


def _money(value: object) -> str:
    return f"£{float(value):.1f}m" if pd.notna(value) else "—"


def _name(row: object) -> str:
    short = getattr(row, "web_name", None)
    if isinstance(short, str) and short.strip():
        return short.strip()
    return str(row.player_name).split()[-1]


def _horizon_label(horizon: int) -> str:
    return "next gameweek" if horizon == 1 else f"next {horizon} gameweeks"


def _load(settings: Settings, competition: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = settings.data_dir / "processed/fpl"
    players = table(
        root / "players.parquet",
        ("competition_id", "season", "player_id", "player_name", "club_name"),
    )
    fixtures = table(
        root / "fixtures.parquet",
        ("club_id", "club_name", "gameweek", "opponent_short_name", "difficulty"),
    )
    metadata = report(root / "metadata.json")
    players = players.loc[players.competition_id.eq(competition)]
    if players.empty or fixtures.empty or not metadata:
        raise ArtifactError("FPL artifacts are not built yet. Run the current-data refresh.")
    return players, fixtures, metadata


def _schedule(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = fixtures.sort_values(["gameweek", "kickoff_time"])
    return {club: frame for club, frame in ordered.groupby("club_id")}


def _fixture_chips(
    schedule: dict[str, pd.DataFrame],
    club_id: object,
    gameweeks: list[int],
    column: str = "difficulty",
) -> str:
    frame = schedule.get(club_id)
    if frame is None:
        return '<span class="pl-muted">No fixtures</span>'
    frame = frame.loc[frame.gameweek.isin(gameweeks)]
    if frame.empty:
        return '<span class="pl-muted">Blank</span>'
    column = column if column in frame else "difficulty"
    return " ".join(
        theme.fixture_chip(f"{row.opponent_short_name} ({row.venue})", getattr(row, column))
        for row in frame.itertuples(index=False)
    )


def _fixture_text(schedule: dict[str, pd.DataFrame], club_id: object, gameweek: int) -> str:
    frame = schedule.get(club_id)
    if frame is None:
        return "—"
    frame = frame.loc[frame.gameweek.eq(gameweek)]
    if frame.empty:
        return "Blank"
    return " + ".join(
        f"{row.opponent_short_name} ({row.venue})" for row in frame.itertuples(index=False)
    )


# ---------------------------------------------------------------- team state


def _team_state(settings: Settings, metadata: dict) -> tuple[pd.DataFrame | None, dict]:
    """Session team first, then the refresh-time artifact for the configured entry."""
    squad = st.session_state.get("fpl_squad")
    if not isinstance(squad, pd.DataFrame):
        squad = table(settings.data_dir / "processed/fpl/squad.parquet")
    manager = st.session_state.get("fpl_manager") or metadata.get("manager") or {}
    if squad.empty or len(squad) != 15 or not manager:
        return None, {}
    return squad, manager


def _team_form(settings: Settings, players: pd.DataFrame, metadata: dict) -> None:
    with st.form("public-fpl-team", border=False):
        default_id = str(st.session_state.get("fpl_entry_id") or settings.fpl_entry_id or "")
        entered = st.text_input(
            "Public FPL team ID",
            value=default_id,
            placeholder="For example: 123456",
            help="The number in the URL of your FPL Points page. No login or password is used.",
        )
        submitted = st.form_submit_button("Load team", type="primary")
    if not submitted:
        return
    if not entered.strip().isdigit() or int(entered) < 1:
        st.error("Enter a positive numeric FPL team ID.")
        return
    try:
        entry_id = int(entered)
        entry, picks = fetch_public_team(
            entry_id=entry_id, event=int(metadata["current_gameweek"])
        )
        squad = map_public_squad(picks, players)
    except ValueError:
        st.error("This team could not be loaded for the current gameweek. Check the ID.")
        return
    st.session_state["fpl_entry_id"] = entry_id
    st.session_state["fpl_squad"] = squad
    st.session_state["fpl_manager"] = manager_summary(
        entry_id, entry, int(metadata["current_gameweek"])
    )
    st.session_state.pop("fpl_league", None)
    st.rerun()


def _header(
    settings: Settings,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    metadata: dict,
    squad: pd.DataFrame | None,
    manager: dict,
) -> int:
    next_gw = int(fixtures.gameweek.min())
    kickoffs = pd.to_datetime(
        fixtures.loc[fixtures.gameweek.eq(next_gw)].get("kickoff_time"), utc=True, errors="coerce"
    ).dropna()
    if kickoffs.empty:
        deadline_text = "Kick-off time unavailable"
    else:
        deadline = (kickoffs.min() - DEADLINE_LEAD).tz_convert(UK)
        passed = deadline <= pd.Timestamp.now(tz=UK)
        deadline_text = "Deadline passed" if passed else f"{deadline:%a %d %b, %H:%M} UK"
    first, second, third = st.columns([2, 2, 3], vertical_alignment="center")
    first.markdown(
        theme.stat_card(
            f"Gameweek {next_gw} deadline",
            deadline_text,
            f"Projections use data through gameweek {metadata['current_gameweek']}",
        ),
        unsafe_allow_html=True,
    )
    with second:
        horizon = st.segmented_control(
            "Planning horizon",
            [1, 3, 5],
            default=3,
            format_func=lambda value: "Next GW" if value == 1 else f"{value} GWs",
            key="fpl-horizon",
            help="Every tab ranks players and transfers over this many gameweeks.",
        )
    with third:
        if squad is None:
            st.markdown(
                theme.stat_card("My team", "Not loaded", "Add your public team ID for advice"),
                unsafe_allow_html=True,
            )
        else:
            rank = manager.get("overall_rank")
            details = [
                f"Bank {_money((manager.get('bank') or 0) / 10)}",
                f"value {_money((manager.get('team_value') or 0) / 10)}",
            ]
            if rank:
                details.append(f"rank {int(rank):,}")
            team_name = str(manager.get("team_name") or "—")
            st.markdown(
                theme.stat_card("My team", team_name, " · ".join(details)),
                unsafe_allow_html=True,
            )
        with st.popover("Load my team" if squad is None else "Change team"):
            _team_form(settings, players, metadata)
    return int(horizon or 3)


# ---------------------------------------------------------------- this gameweek


def _pick_card(
    row: object,
    schedule: dict[str, pd.DataFrame],
    gameweeks: list[int],
    horizon: int,
    scale: float,
) -> str:
    points = getattr(row, f"projected_points_{horizon}")
    low = getattr(row, f"projection_low_{horizon}")
    high = getattr(row, f"projection_high_{horizon}")
    return (
        '<div class="pl-pick"><div class="top">'
        f'{theme.club_chip(row.club_short_name)}<span class="name">{escape(_name(row))}</span>'
        f'<span class="pts">{points:.1f}</span></div>'
        f"<div>{_fixture_chips(schedule, row.club_id, gameweeks)}</div>"
        f"{theme.range_bar(low, points, high, scale)}"
        f'<div class="pl-muted">{escape(str(row.position_group))} · {_money(row.price)} · '
        f"{row.ownership_pct:.0f}% owned · range {low:.1f}–{high:.1f}</div></div>"
    )


def _position_list(frame: pd.DataFrame, projection: str, position: str) -> str:
    rows = "".join(
        '<div class="pl-list-row">'
        f"{theme.club_chip(row.club_short_name)}"
        f'<span class="grow"><b>{escape(_name(row))}</b>'
        f'<span class="pl-muted">{_money(row.price)} · {row.expected_minutes:.0f} xMins</span>'
        f'</span><span class="big">{getattr(row, projection):.1f}</span></div>'
        for row in frame.itertuples(index=False)
    )
    return f'<div class="pl-pick"><div class="pl-eyebrow">{position}</div>{rows}</div>'


def _availability(players: pd.DataFrame, squad: pd.DataFrame | None) -> str:
    pool = squad if squad is not None else players.loc[players.ownership_pct.ge(5)]
    chance = pool.chance_of_playing
    flagged = pool.loc[pool.status.ne("a") | chance.lt(100)].sort_values(
        "ownership_pct", ascending=False
    )
    if flagged.empty:
        scope = "your squad" if squad is not None else "widely owned players"
        return f'<span class="pl-muted">No flagged players among {scope}.</span>'
    rows = []
    for row in flagged.head(6).itertuples(index=False):
        value = row.chance_of_playing
        label = f"{value:.0f}%" if pd.notna(value) else "Doubt"
        tone = "bad" if pd.notna(value) and value < 50 else "warning"
        news = str(getattr(row, "news", "") or "No detail published")
        rows.append(
            '<div class="pl-list-row">'
            f"{theme.club_chip(row.club_short_name)}"
            f'<span class="grow"><b>{escape(_name(row))}</b>'
            f'<span class="pl-muted">{escape(news)}</span></span>'
            f"{theme.badge(label, tone)}</div>"
        )
    return "".join(rows)


def _squad_summary(
    players: pd.DataFrame,
    squad: pd.DataFrame,
    manager: dict,
    horizon: int,
    next_gw: int,
) -> str:
    lineup = select_starting_xi(squad, "projected_points_1")
    starters = lineup.loc[lineup.is_starter]
    captain = starters.loc[starters.is_recommended_captain].iloc[0]
    total = starters.projected_points_1.sum() + captain.projected_points_1
    rows = [
        '<div class="pl-list-row"><span class="grow"><b>Projected XI points</b>'
        f'<span class="pl-muted">Gameweek {next_gw}, captain counted twice</span></span>'
        f'<span class="big">{total:.1f}</span></div>',
        '<div class="pl-list-row">'
        f"{theme.club_chip(captain.club_short_name)}"
        f'<span class="grow"><b>Captain: {escape(_name(captain))}</b>'
        f'<span class="pl-muted">Highest projection in your XI</span></span>'
        f'<span class="big">{captain.projected_points_1:.1f}</span></div>',
    ]
    projection = f"projected_points_{horizon}"
    transfers = best_single_transfers(
        squad, players, projection=projection, bank=(manager.get("bank") or 0) / 10
    )
    if not transfers.empty and transfers.lineup_gain.iloc[0] > 0.05:
        best = transfers.iloc[0]
        lookup = players.set_index("source_player_id")
        outgoing, incoming = lookup.loc[int(best.out_id)], lookup.loc[int(best.in_id)]
        rows.append(
            '<div class="pl-list-row"><span class="grow">'
            f"<b>Best transfer: {escape(_name(outgoing))} → {escape(_name(incoming))}</b>"
            f'<span class="pl-muted">Added to your best XI over the {_horizon_label(horizon)}, '
            f'before any hit</span></span>{theme.badge(f"+{best.lineup_gain:.1f}", "good")}</div>'
        )
    else:
        rows.append(
            '<div class="pl-list-row"><span class="grow"><b>No single transfer improves the '
            'squad</b><span class="pl-muted">Within your bank and the club limit</span></span>'
            "</div>"
        )
    return "".join(rows)


def _price_watch(players: pd.DataFrame, squad: pd.DataFrame | None, metadata: dict) -> str:
    """Players most bought and sold this gameweek, the ones likeliest to change price."""
    if "net_transfers_event" not in players or players.net_transfers_event.isna().all():
        return '<span class="pl-muted">Transfer counts are not in the current data.</span>'
    watch = price_watch(players, top_n=5)
    owned = set() if squad is None else set(squad.source_player_id.astype(int))
    sections = []
    for direction, title, tone, sign in (
        ("rise", "Most bought · may rise", "good", "+"),
        ("fall", "Most sold · may fall", "bad", "−"),
    ):
        rows = "".join(
            '<div class="pl-list-row">'
            f"{theme.club_chip(row.club_short_name)}"
            f'<span class="grow"><b>{escape(_name(row))}</b>'
            f'<span class="pl-muted">{_money(row.price)} · {row.ownership_pct:.0f}% owned'
            f'{" · in your squad" if int(row.source_player_id) in owned else ""}</span></span>'
            f"{theme.badge(f'{sign}{abs(row.net_transfers_event) / 1000:,.0f}k', tone)}</div>"
            for row in watch.loc[watch.direction.eq(direction)].itertuples(index=False)
        )
        sections.append(f'<div class="pl-eyebrow" style="margin-top:6px">{title}</div>{rows}')
    note = "Ranked by net transfers this gameweek; the game's exact price rules are not public."
    backtest = metadata.get("price_watch") or {}
    if backtest:
        note += (
            f" Over {backtest['gameweeks']} gameweeks of last season, "
            f"{backtest['rise_hit_rate']:.0%} of the ten most-bought players rose by the next "
            f"gameweek and {backtest['fall_hit_rate']:.0%} of the ten most-sold fell "
            f"(against {backtest['rise_base_rate']:.0%} and {backtest['fall_base_rate']:.0%} "
            "of all players)."
        )
    return "".join(sections) + f'<div class="pl-muted" style="margin-top:10px">{note}</div>'


def _gameweek(
    players: pd.DataFrame,
    schedule: dict[str, pd.DataFrame],
    squad: pd.DataFrame | None,
    manager: dict,
    horizon: int,
    gameweeks: list[int],
    metadata: dict,
) -> None:
    next_gw = gameweeks[0]
    scale = float(players.projection_high_1.max())
    left, right = st.columns([3, 2], gap="medium")
    with left:
        captains = players.nlargest(3, "projected_points_1")
        cards = "".join(
            _pick_card(row, schedule, [next_gw], 1, scale)
            for row in captains.itertuples(index=False)
        )
        note = (
            '<div class="pl-muted" style="margin-top:10px">Bar: indicative 80% range, marker '
            "at the projection. A high projection is not a guarantee.</div>"
        )
        st.markdown(
            theme.card(
                f"Captain picks · gameweek {next_gw}", f'<div class="pl-grid">{cards}</div>{note}'
            ),
            unsafe_allow_html=True,
        )
        if squad is not None:
            own = squad.nlargest(3, "projected_points_1")
            cards = "".join(
                _pick_card(row, schedule, [next_gw], 1, scale)
                for row in own.itertuples(index=False)
            )
            st.markdown(
                theme.card("Best captains in your squad", f'<div class="pl-grid">{cards}</div>'),
                unsafe_allow_html=True,
            )
        projection = f"projected_points_{horizon}"
        lists = "".join(
            _position_list(
                players.loc[players.position_group.eq(position)].nlargest(3, projection),
                projection,
                position,
            )
            for position in POSITIONS
        )
        st.markdown(
            theme.card(
                f"Top picks by position · {_horizon_label(horizon)}",
                f'<div class="pl-grid">{lists}</div>',
            ),
            unsafe_allow_html=True,
        )
    with right:
        if squad is not None:
            st.markdown(
                theme.card(
                    str(manager.get("team_name") or "My team"),
                    _squad_summary(players, squad, manager, horizon, next_gw),
                ),
                unsafe_allow_html=True,
            )
        title = "Availability in your squad" if squad is not None else "Availability watch"
        st.markdown(theme.card(title, _availability(players, squad)), unsafe_allow_html=True)
        st.markdown(
            theme.card("Price watch", _price_watch(players, squad, metadata)),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- players


def _player_detail(row: pd.Series) -> None:
    st.markdown(
        f'<div class="pl-card-title" style="margin-top:8px">{theme.club_chip(row.club_short_name)} '
        f"{escape(str(row.player_name))} · next gameweek</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        values = [float(row[column]) for column in COMPONENTS.values()]
        figure = go.Figure(
            go.Bar(
                x=values,
                y=list(COMPONENTS),
                orientation="h",
                marker_color=[theme.ACCENT if value >= 0 else theme.BAD for value in values],
                text=[f"{value:+.2f}" for value in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
        # Padding keeps the value labels clear of the category names and the plot edge.
        figure.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 10, "b": 30},
            xaxis={
                "title": "Expected points",
                "range": [min(values + [0.0]) - 0.6, max(values + [0.0]) + 0.6],
            },
            yaxis={"autorange": "reversed", "ticksuffix": "  "},
        )
        st.plotly_chart(
            figure, width="stretch", key="fpl-components", config={"displayModeBar": False}
        )
    with right:
        original = float(row.expected_minutes)
        override = st.slider(
            "What if he plays this many minutes?",
            0,
            95,
            int(round(original)),
            key="fpl-xmins-scenario",
        )
        ratio = override / original if original > 0 else 0.0
        appearance = min(1.0, float(row.appearance_probability) * ratio)
        sixty = min(appearance, float(row.sixty_probability) * ratio)
        other = float(row.projected_points_1) - float(row.xpts_appearance_1)
        scenario = max(0.0, appearance + sixty + other * ratio)
        first, second = st.columns(2)
        first.markdown(
            theme.stat_card("Model", f"{row.projected_points_1:.1f}", f"{original:.0f} xMins"),
            unsafe_allow_html=True,
        )
        second.markdown(
            theme.stat_card("Your scenario", f"{scenario:.1f}", f"{override} minutes"),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Start {row.start_probability:.0%} · any appearance "
            f"{row.appearance_probability:.0%} · 60+ minutes {row.sixty_probability:.0%}. "
            "The scenario rescales every event with minutes and is not saved."
        )


def _players(players: pd.DataFrame, horizon: int) -> None:
    projection = f"projected_points_{horizon}"
    first, second, third, fourth = st.columns([3, 2, 2, 2])
    available = [position for position in POSITIONS if position in set(players.position_group)]
    positions = first.pills(
        "Position", available, selection_mode="multi", default=available, key="fpl-positions"
    )
    club = second.selectbox(
        "Club", ["All clubs", *sorted(players.club_name.unique())], key="fpl-club"
    )
    ceiling = float(players.price.max())
    max_price = third.slider("Maximum price (£m)", 3.5, max(4.0, ceiling), max(4.0, ceiling), 0.5)
    search = fourth.text_input(
        "Search", placeholder="Surname, for example Saka", key="fpl-player-search"
    ).strip()
    more = st.toggle("More stats", key="fpl-more-stats")
    view = players.loc[
        players.position_group.isin(positions or available) & players.price.le(max_price)
    ]
    if club != "All clubs":
        view = view.loc[view.club_name.eq(club)]
    if search:
        view = view.loc[view.player_name.str.contains(search, case=False, regex=False, na=False)]
    if view.empty:
        st.info("No players match these filters.")
        return
    view = view.sort_values(projection, ascending=False).reset_index(drop=True)
    display = pd.DataFrame(
        {
            "Player": view.player_name,
            "Club": view.club_short_name,
            "Pos": view.position_group,
            "Price": view.price,
            "xPts": view[projection],
            "Range": [
                f"{low:.1f} – {high:.1f}"
                for low, high in zip(
                    view[f"projection_low_{horizon}"],
                    view[f"projection_high_{horizon}"],
                    strict=True,
                )
            ],
            "xMins": view.expected_minutes,
            "Start": view.start_probability,
            "Fixtures": view[f"fixtures_{horizon}"].str.replace(r"GW\d+ ", "", regex=True),
            "Owned": view.ownership_pct,
        }
    )
    if more:
        display["60+"] = view.sixty_probability
        display["Form"] = view.form
        display["xG/90"] = view.projected_xg_per90
        display["xA/90"] = view.projected_xa_per90
        display["Def/90"] = view.projected_defensive_contribution_per90
        display["Points"] = view.total_points
    event = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="fpl-player-table",
        column_config={
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            "xPts": st.column_config.ProgressColumn(
                f"xPts · {horizon} GW",
                format="%.1f",
                min_value=0.0,
                max_value=float(players[projection].max()),
                help="Expected points over the planning horizon.",
            ),
            "Range": st.column_config.TextColumn(
                "Range", help="Indicative 80% range; not a calibrated interval."
            ),
            "xMins": st.column_config.NumberColumn(
                "xMins", format="%.0f", help="Expected minutes per fixture."
            ),
            "Start": st.column_config.NumberColumn("Start", format="percent"),
            "Owned": st.column_config.NumberColumn("Owned", format="%.1f%%"),
            "60+": st.column_config.NumberColumn("60+ min", format="percent"),
            "Form": st.column_config.NumberColumn(format="%.1f"),
            "xG/90": st.column_config.NumberColumn(format="%.2f"),
            "xA/90": st.column_config.NumberColumn(format="%.2f"),
            "Def/90": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    selected = list(getattr(getattr(event, "selection", None), "rows", None) or [])
    st.caption("Select a row to see where a player's points come from.")
    _player_detail(view.iloc[selected[0] if selected and selected[0] < len(view) else 0])


# ---------------------------------------------------------------- fixtures


def _ticker(fixtures: pd.DataFrame) -> None:
    first, second = st.columns(2)
    view = first.segmented_control(
        "Difficulty for",
        list(DIFFICULTY_VIEWS),
        default="Overall",
        key="fpl-fdr-view",
        help="Attack rates the goals a club should score; Defence rates the goals it should "
        "concede. Use Attack for forwards and midfielders, Defence for defenders and keepers.",
    )
    order = second.segmented_control(
        "Sort by", ["Easiest run", "Club"], default="Easiest run", key="fpl-fdr-sort"
    )
    column = DIFFICULTY_VIEWS.get(view or "Overall", "difficulty")
    column = column if column in fixtures else "difficulty"
    weeks = sorted(int(value) for value in fixtures.gameweek.unique())[:5]
    shown = fixtures.loc[fixtures.gameweek.isin(weeks)]
    average = shown.groupby("club_id")[column].mean()
    clubs = shown.drop_duplicates("club_id").set_index("club_id")[["club_name", "club_short_name"]]
    clubs = clubs.assign(average=average)
    clubs = clubs.sort_values(["average", "club_name"] if order != "Club" else ["club_name"])
    template = f"minmax(150px,1.6fr) repeat({len(weeks)}, minmax(76px,1fr)) 52px"
    cells = ['<span class="head" style="text-align:left">CLUB</span>']
    cells += [f'<span class="head">GW{week}</span>' for week in weeks]
    cells.append('<span class="head">AVG</span>')
    by_club = {club: frame for club, frame in shown.groupby("club_id")}
    for club_id, club in clubs.iterrows():
        cells.append(
            f'<span class="club">{theme.club_chip(club.club_short_name)}'
            f"{escape(str(club.club_name))}</span>"
        )
        for week in weeks:
            games = by_club[club_id].loc[by_club[club_id].gameweek.eq(week)]
            if games.empty:
                cells.append('<span class="cell pl-muted">Blank</span>')
                continue
            blocks = []
            for game in games.itertuples(index=False):
                level = int(getattr(game, column))
                background, foreground = theme.FDR_COLOURS[level]
                blocks.append(
                    f'<span class="cell" style="background:{background};color:{foreground}">'
                    f"{escape(str(game.opponent_short_name))} ({game.venue}) · {level}</span>"
                )
            cells.append(
                blocks[0]
                if len(blocks) == 1
                else f'<span style="display:grid;gap:2px">{"".join(blocks)}</span>'
            )
        cells.append(f'<span class="avg">{club.average:.1f}</span>')
    legend = " ".join(theme.fixture_chip(level, level) for level in theme.FDR_COLOURS)
    source = (
        "the live Dixon–Coles goal model"
        if fixtures.get("rate_source", pd.Series(dtype=str)).eq("goal-model").all()
        else "this season's expected goals"
    )
    body = (
        f'<div class="pl-ticker"><div class="grid" style="grid-template-columns:{template}">'
        f'{"".join(cells)}</div></div>'
        f'<div class="pl-muted" style="margin-top:12px">Easiest {legend} hardest · the number '
        f"repeats the colour. Ratings come from each fixture's expected goals in {source}, so "
        "they reflect the opponent and the venue.</div>"
    )
    st.markdown(theme.card("Fixture ticker", body), unsafe_allow_html=True)


# ---------------------------------------------------------------- my team & transfers


def _tile(row: object, schedule: dict, next_gw: int, projection: str, captain: bool) -> str:
    doubt = " doubt" if str(getattr(row, "status", "a")) != "a" else ""
    badge = '<span class="cap" title="Recommended captain">C</span>' if captain else ""
    return (
        f'<div class="pl-tile{doubt}">{badge}{theme.club_chip(row.club_short_name)}'
        f'<span class="nm">{escape(_name(row))}</span>'
        f'<span class="fx">{escape(_fixture_text(schedule, row.club_id, next_gw))}</span>'
        f'<span class="pts">{getattr(row, projection):.1f}</span></div>'
    )


def _pitch(squad: pd.DataFrame, schedule: dict, horizon: int, next_gw: int) -> None:
    projection = f"projected_points_{horizon}"
    lineup = select_starting_xi(squad, projection)
    starters = lineup.loc[lineup.is_starter]
    captain = starters.loc[starters.is_recommended_captain].iloc[0]
    total = starters[projection].sum() + captain[projection]
    lines = "".join(
        '<div class="line">'
        + "".join(
            _tile(row, schedule, next_gw, projection, bool(row.is_recommended_captain))
            for row in starters.loc[starters.position_group.eq(position)]
            .sort_values(projection, ascending=False)
            .itertuples(index=False)
        )
        + "</div>"
        for position in POSITIONS
    )
    bench = lineup.loc[~lineup.is_starter].assign(
        keeper=lambda frame: frame.position_group.ne("GK")
    )
    bench_tiles = "".join(
        _tile(row, schedule, next_gw, projection, False)
        for row in bench.sort_values(["keeper", projection], ascending=[True, False]).itertuples(
            index=False
        )
    )
    shape = starters.position_group.value_counts()
    title = (
        f"Recommended XI · {shape.get('DEF', 0)}-{shape.get('MID', 0)}-{shape.get('FWD', 0)}"
    )
    body = (
        f'<div class="pl-muted" style="margin:-6px 0 12px">{total:.1f} expected points over the '
        f"{_horizon_label(horizon)}, captain counted twice. Each tile shows the gameweek "
        f"{next_gw} fixture.</div>"
        f'<div class="pl-pitch">{lines}</div>'
        f'<div class="pl-eyebrow" style="margin-top:14px">Bench</div>'
        f'<div class="pl-bench">{bench_tiles}</div>'
        '<div class="pl-muted" style="margin-top:10px">An amber outline marks a player with an '
        "injury or availability flag.</div>"
    )
    st.markdown(theme.card(title, body), unsafe_allow_html=True)


def _head_to_head(outgoing: pd.Series, incoming: pd.Series, projection: str) -> str:
    rows = [
        '<div style="display:flex;justify-content:space-between;gap:12px;margin-top:6px">'
        f'<span><span class="pl-muted">OUT</span><br><b>{escape(_name(outgoing))}</b></span>'
        f'<span style="text-align:right"><span class="pl-muted">IN</span><br>'
        f"<b>{escape(_name(incoming))}</b></span></div>"
    ]
    for label, column, fmt, higher in HEAD_TO_HEAD:
        column = column or projection
        if column not in outgoing or pd.isna(outgoing[column]) or pd.isna(incoming[column]):
            continue
        left, right = float(outgoing[column]), float(incoming[column])
        top = max(abs(left), abs(right), 1e-9)
        left_better = higher is not None and (left > right) == higher and left != right
        right_better = higher is not None and (right > left) == higher and left != right
        rows.append(
            '<div class="pl-h2h">'
            f'<span style="text-align:right">{fmt.format(left)}</span>'
            f'<span class="bar left{" better" if left_better else ""}">'
            f'<i style="width:{100 * abs(left) / top:.0f}%"></i></span>'
            f'<span class="label">{label}</span>'
            f'<span class="bar{" better" if right_better else ""}">'
            f'<i style="width:{100 * abs(right) / top:.0f}%"></i></span>'
            f"<span>{fmt.format(right)}</span></div>"
        )
    return "".join(rows)


def _planner(
    players: pd.DataFrame,
    schedule: dict[str, pd.DataFrame],
    squad: pd.DataFrame | None,
    manager: dict,
    horizon: int,
    gameweeks: list[int],
) -> None:
    projection = f"projected_points_{horizon}"
    lookup = players.set_index("source_player_id")
    default_bank = float((manager.get("bank") or 0) / 10)
    with st.container(border=True):
        st.markdown('<div class="pl-card-title">Transfer planner</div>', unsafe_allow_html=True)
        if squad is not None:
            ranked = best_single_transfers(squad, players, projection=projection, bank=default_bank)
            ordered = list(dict.fromkeys(ranked.out_id.tolist()))
            ordered += [
                int(value) for value in squad.source_player_id if int(value) not in ordered
            ]
        else:
            ordered = (
                players.loc[players.expected_minutes.gt(0)]
                .sort_values("player_name")
                .source_player_id.astype(int)
                .tolist()
            )

        def label(player_id: int) -> str:
            row = lookup.loc[player_id]
            return f"{row.player_name} · {row.position_group} · {_money(row.price)}"

        first, second = st.columns([3, 2])
        out_id = first.selectbox(
            "Player out",
            ordered,
            format_func=label,
            key="fpl-transfer-out",
            help="Your squad, ordered by the size of the best available upgrade."
            if squad is not None
            else "Load your team to limit this list to your squad.",
        )
        bank = second.number_input(
            "Bank (£m)", min_value=0.0, max_value=20.0, value=default_bank, step=0.1
        )
        outgoing = lookup.loc[out_id].copy()
        outgoing["source_player_id"] = out_id
        options = replacement_options(
            players, outgoing, projection=projection, bank=bank, squad=squad, limit=6
        )
        if options.empty:
            st.info("No likely replacement fits this position, budget and club limit.")
            return
        rows = []
        in_squad = "lineup_gain" in options
        for row in options.itertuples(index=False):
            gain = row.lineup_gain if in_squad else row.projected_gain
            tone = "good" if gain > 0.05 else "warning" if gain > -0.05 else "bad"
            change = f"{row.price_change:+.1f}m" if abs(row.price_change) >= 0.05 else "same price"
            rows.append(
                '<div class="pl-list-row">'
                f"{theme.club_chip(row.club_short_name)}"
                f'<span class="grow"><b>{escape(_name(row))}</b>'
                f'<span class="pl-muted">{_money(row.price)} ({change}) · '
                f"{row.expected_minutes:.0f} xMins</span>"
                f"<span>{_fixture_chips(schedule, row.club_id, gameweeks[:horizon])}</span></span>"
                f'{theme.badge(f"{gain:+.1f}", tone)}</div>'
            )
        measure = "gain for your best XI" if in_squad else "gain"
        st.markdown(
            f'<div class="pl-eyebrow">Replacements · {measure} over the '
            f"{_horizon_label(horizon)}</div>" + "".join(rows),
            unsafe_allow_html=True,
        )
        in_id = st.selectbox(
            "Compare with",
            options.source_player_id.astype(int).tolist(),
            format_func=label,
            key="fpl-transfer-in",
        )
        st.markdown(
            _head_to_head(outgoing, lookup.loc[in_id], projection), unsafe_allow_html=True
        )
        st.caption(
            "Same position, affordable from the sale plus your bank, at least 30 expected "
            "minutes and at most three players per club. Uses current prices, not your "
            "selling prices. Green bars mark the better value."
        )


def _optimizer(players: pd.DataFrame, squad: pd.DataFrame, manager: dict, horizon: int) -> None:
    projection = f"projected_points_{horizon}"
    with st.container(border=True):
        st.markdown(
            '<div class="pl-card-title">Squad optimizer and chip plan</div>',
            unsafe_allow_html=True,
        )
        first, second, third = st.columns([2, 2, 1], vertical_alignment="bottom")
        max_transfers = first.slider("Maximum transfers", 0, 6, 2)
        free_transfers = second.slider("Free transfers available", 0, 5, 1)
        if not third.button("Optimize", type="primary", width="stretch"):
            st.caption(
                "Finds the best legal squad within these transfers, charging 4 points for each "
                "one beyond your free transfers. Uses current prices as an approximation."
            )
            return
        current_ids = set(squad.source_player_id.astype(int))
        budget = float(squad.price.sum() + (manager.get("bank") or 0) / 10)
        current = select_starting_xi(squad, projection)
        current_points = (
            current.loc[current.is_starter, projection].sum()
            + current.loc[current.is_recommended_captain, projection].sum()
        )
        candidates = []
        for transfers in range(max_transfers + 1):
            optimized = optimize_squad(
                players,
                budget=budget,
                projection=projection,
                current_ids=current_ids,
                max_transfers=transfers,
            )
            made = int((~optimized.is_current).sum())
            hits = max(0, made - free_transfers) * 4
            points = (
                optimized.loc[optimized.is_starter, projection].sum()
                + optimized.loc[optimized.is_recommended_captain, projection].sum()
            )
            candidates.append((points - hits, made, hits, optimized, points))
        net_points, made, hits, optimized, raw_points = max(candidates, key=lambda item: item[0])
        theme.stat_row(
            (
                ("Recommended transfers", str(made), f"within {max_transfers} allowed"),
                ("Hit cost", f"-{hits} pts", f"{free_transfers} free"),
                (
                    "Net projected gain",
                    f"{net_points - current_points:+.1f}",
                    f"over the {_horizon_label(horizon)}",
                ),
            )
        )
        incoming = optimized.loc[~optimized.is_current].sort_values(
            ["position_group", projection], ascending=[True, False]
        )
        kept = set(optimized.source_player_id.astype(int))
        outgoing = squad.loc[~squad.source_player_id.astype(int).isin(kept)].sort_values(
            ["position_group", projection], ascending=[True, False]
        )
        if incoming.empty:
            st.success("The optimizer prefers holding the current squad for this horizon.")
        else:
            swaps = "".join(
                '<div class="pl-swap">'
                f'<span class="who">{theme.club_chip(sold.club_short_name)}'
                f'<span class="n">{escape(_name(sold))} · {_money(sold.price)}</span></span>'
                "<span>→</span>"
                f'<span class="who">{theme.club_chip(bought.club_short_name)}'
                f'<span class="n"><b>{escape(_name(bought))}</b> · {_money(bought.price)}</span>'
                "</span>"
                + theme.badge(
                    f"{getattr(bought, projection) - getattr(sold, projection):+.1f}", "good"
                )
                + "</div>"
                for sold, bought in zip(
                    outgoing.itertuples(index=False), incoming.itertuples(index=False), strict=True
                )
            )
            st.markdown(
                '<div class="pl-eyebrow" style="margin-top:14px">Sell → buy</div>' + swaps,
                unsafe_allow_html=True,
            )
        gameweeks = sorted(
            int(column.removeprefix("projected_points_gw_"))
            for column in players
            if column.startswith("projected_points_gw_")
        )
        advice = optimize_chip_schedule(squad, players, budget=budget, gameweeks=gameweeks[:5])
        st.markdown(
            '<div class="pl-eyebrow" style="margin:14px 0 6px">Chip plan · next five '
            "gameweeks</div>",
            unsafe_allow_html=True,
        )
        theme.stat_row(
            (
                chip.chip,
                f"GW{chip.recommended_gameweek} · {chip.signal}",
                f"+{chip.incremental_points:.1f} xPts · {chip.reason}",
            )
            for chip in advice.itertuples(index=False)
        )
        gain = raw_points - current_points
        wildcard = "Consider" if made >= 4 and gain >= 20 else "Hold"
        st.caption(
            f"Wildcard: {wildcard}. The optimized squad gains {gain:.1f} points before hits. "
            "Chips are placed in different gameweeks; Hold means the best week is still weak."
        )


def _plan(players: pd.DataFrame, squad: pd.DataFrame, manager: dict) -> None:
    """Week-by-week transfer plan over the prepared gameweeks."""
    gameweeks = sorted(
        int(column.removeprefix("projected_points_gw_"))
        for column in players
        if column.startswith("projected_points_gw_")
    )
    lookup = players.set_index("source_player_id")
    with st.container(border=True):
        st.markdown(
            f'<div class="pl-card-title">Transfer plan · gameweeks {gameweeks[0]}–'
            f"{gameweeks[-1]}</div>",
            unsafe_allow_html=True,
        )
        first, second, third = st.columns([2, 2, 1], vertical_alignment="bottom")
        free = first.number_input(
            "Free transfers now", min_value=0, max_value=5, value=1, step=1, key="fpl-plan-free"
        )
        hits = second.toggle(
            "Allow points hits",
            key="fpl-plan-hits",
            help="Without hits the plan only ever uses free transfers.",
        )
        if not third.button("Plan", type="primary", width="stretch", key="fpl-plan-run"):
            st.caption(
                "Suggests which transfer to make in each gameweek, or when to save it. One "
                "free transfer is added every week, up to five."
            )
            return
        plan = plan_transfers(
            squad,
            players,
            gameweeks=gameweeks,
            bank=float((manager.get("bank") or 0) / 10),
            free_transfers=int(free),
            allow_hits=bool(hits),
        )
        moves, weeks = plan["moves"], plan["weeks"]
        cards = (
            ("Transfers planned", str(len(moves)), f"over {len(weeks)} gameweeks"),
            ("Points hits", f"-{weeks.hit_points.sum():.0f}", "included in the gain"),
            (
                "Projected gain",
                f"{plan['total_gain']:+.1f}",
                "best XI, against making no transfers",
            ),
        )
        theme.stat_row(cards)
        rows = []
        for week in weeks.itertuples(index=False):
            made = moves.loc[moves.gameweek.eq(week.gameweek)]
            if made.empty:
                detail = (
                    '<span class="pl-muted">Save the transfer · '
                    f"{int(week.free_transfers_left) + 1} available next week</span>"
                )
            else:
                detail = "".join(
                    '<div class="pl-swap" style="border-top:none;padding:2px 0">'
                    f'<span class="who">{theme.club_chip(lookup.loc[move.out_id].club_short_name)}'
                    f'<span class="n">{escape(_name(lookup.loc[move.out_id]))}</span></span>'
                    "<span>→</span>"
                    f'<span class="who">{theme.club_chip(lookup.loc[move.in_id].club_short_name)}'
                    f'<span class="n"><b>{escape(_name(lookup.loc[move.in_id]))}</b>'
                    f'{" · −4 hit" if move.hit else ""}</span></span>'
                    f'{theme.badge(f"+{move.gain:.1f}", "good")}</div>'
                    for move in made.itertuples(index=False)
                )
            rows.append(
                '<div class="pl-list-row" style="align-items:flex-start">'
                f'<span style="width:64px"><b>GW{int(week.gameweek)}</b></span>'
                f'<span class="grow">{detail}</span>'
                f'<span class="pl-muted" style="width:130px;text-align:right">'
                f"XI {week.plan_points:.1f} · hold {week.hold_points:.1f}</span></div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(
            "Each badge is the gain to your best XI over the remaining planned gameweeks. This "
            "is a step-by-step heuristic, not an optimal plan: it takes the best single swap "
            "each week, saves a transfer when the swap would not help that week, and uses "
            "current prices rather than your selling prices."
        )


def _league_page(rank: object, size: int, around: bool) -> int:
    """Standings page (50 teams each) holding the top of the league or the user's rank."""
    if not around or rank is None or pd.isna(rank):
        return 1
    return max(1, (int(rank) - 1) // 50 + 1)


def _mini_league(
    players: pd.DataFrame, squad: pd.DataFrame | None, manager: dict, metadata: dict
) -> None:
    leagues = manager.get("leagues") or []
    if squad is None or not leagues:
        st.info(
            "Load your team with the button at the top of the page to compare it with a "
            "mini-league. Only your private classic leagues are listed."
        )
        return
    my_entry = int(manager["entry_id"])
    names = {int(league["id"]): league for league in leagues}
    first, second, third, fourth = st.columns([3, 2, 2, 1], vertical_alignment="bottom")
    league_id = first.selectbox(
        "League",
        list(names),
        format_func=lambda value: (
            f"{names[value]['name']} · rank {names[value].get('entry_rank') or '—'}"
            f" of {names[value].get('rank_count') or '—'}"
        ),
        key="fpl-league",
    )
    mode = second.segmented_control(
        "Compare with", ["Top of the league", "Teams near me"], default="Teams near me",
        key="fpl-league-mode",
    )
    count = third.slider("Rival teams", 5, MAX_LEAGUE_TEAMS - 1, 10, key="fpl-league-count")
    around = mode != "Top of the league"
    if fourth.button("Load", type="primary", width="stretch", key="fpl-league-load"):
        rank = names[league_id].get("entry_rank")
        try:
            name, standings = fetch_league_standings(
                league_id=league_id, page=_league_page(rank, 50, around)
            )
            rivals = standings.loc[standings.entry.ne(my_entry)].copy()
            if around and rank is not None:
                rivals["distance"] = (rivals["rank"] - int(rank)).abs()
                rivals = rivals.sort_values("distance")
            rivals = rivals.head(count)
            picks = fetch_league_picks(
                entries=rivals.entry.astype(int).tolist(),
                event=int(metadata["current_gameweek"]),
            )
        except ValueError as error:
            st.error(f"{error}. Try again, or choose another league.")
            return
        st.session_state["fpl_league"] = {
            "league_id": league_id,
            "name": name or names[league_id]["name"],
            "standings": rivals.drop(columns="distance", errors="ignore"),
            "picks": picks,
            "around": around,
        }
    loaded = st.session_state.get("fpl_league")
    if not loaded or loaded["league_id"] != league_id:
        st.caption(
            f"Loading reads the league table and up to {MAX_LEAGUE_TEAMS - 1} rival squads "
            "from the official public FPL endpoints. Nothing is saved, no login is used and "
            "manager names are not kept."
        )
        return
    mine = squad[["element", "multiplier", "is_captain"]].assign(entry=my_entry)
    picks = pd.concat([loaded["picks"].loc[loaded["picks"].entry.ne(my_entry)], mine])
    ownership = league_ownership(picks, my_entry).merge(
        players[
            [
                "source_player_id",
                "player_name",
                "web_name",
                "club_short_name",
                "position_group",
                "price",
                "projected_points_1",
            ]
        ],
        on="source_player_id",
        how="inner",
    )
    rivals = int(ownership.rivals.iloc[0])

    def rows(frame: pd.DataFrame, share: str) -> str:
        if frame.empty:
            return '<span class="pl-muted">None.</span>'
        return "".join(
            '<div class="pl-list-row">'
            f"{theme.club_chip(row.club_short_name)}"
            f'<span class="grow"><b>{escape(_name(row))}</b>'
            f'<span class="pl-muted">{row.position_group} · {_money(row.price)} · '
            f"{row.projected_points_1:.1f} xPts next gameweek</span></span>"
            f'<span class="big">{getattr(row, share):.0%}</span></div>'
            for row in frame.itertuples(index=False)
        )

    differentials = ownership.loc[ownership.mine].nsmallest(5, "owned_share")
    threats = ownership.loc[~ownership.mine].nlargest(5, "owned_share")
    captains = ownership.loc[ownership.captains.gt(0)].nlargest(5, "captain_share")
    scope = "nearest to your rank" if loaded["around"] else "at the top"
    st.markdown(
        f'<div class="pl-muted" style="margin:6px 0 12px">{escape(str(loaded["name"]))} · '
        f"{rivals} rival squads {scope} · gameweek {metadata['current_gameweek']} picks</div>",
        unsafe_allow_html=True,
    )
    for column, (title, frame, share) in zip(
        st.columns(3),
        (
            ("Your differentials", differentials, "owned_share"),
            ("Rivals have, you don't", threats, "owned_share"),
            ("Rivals' captains", captains, "captain_share"),
        ),
        strict=True,
    ):
        column.markdown(theme.card(title, rows(frame, share)), unsafe_allow_html=True)
    st.caption(
        "Percentages are the share of these rival squads owning (or captaining) the player. "
        "A differential gains you ground when it scores; a player most rivals own costs you "
        "ground when it scores and you do not have it."
    )
    table_tab, standings_tab = st.tabs(["All players", "Rival teams"])
    with table_tab:
        st.dataframe(
            pd.DataFrame(
                {
                    "Player": ownership.player_name,
                    "Club": ownership.club_short_name,
                    "Pos": ownership.position_group,
                    "Mine": ownership.mine,
                    "Rivals owning": ownership.owned_share,
                    "Rivals starting": ownership.started_share,
                    "Rivals captaining": ownership.captain_share,
                    "xPts next GW": ownership.projected_points_1,
                }
            ).sort_values("Rivals owning", ascending=False),
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "Rivals owning": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0
                ),
                "Rivals starting": st.column_config.NumberColumn(format="percent"),
                "Rivals captaining": st.column_config.NumberColumn(format="percent"),
                "xPts next GW": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    with standings_tab:
        st.dataframe(
            loaded["standings"]
            .sort_values("rank")
            .rename(
                columns={
                    "rank": "Rank",
                    "entry_name": "Team",
                    "total": "Points",
                    "event_total": "Gameweek points",
                }
            )
            .drop(columns="entry"),
            hide_index=True,
            width="stretch",
        )


def _my_team(
    players: pd.DataFrame,
    schedule: dict[str, pd.DataFrame],
    squad: pd.DataFrame | None,
    manager: dict,
    horizon: int,
    gameweeks: list[int],
) -> None:
    if squad is None:
        st.info(
            "Load your team with the button at the top of the page to see your pitch, "
            "optimizer and chip plan. The transfer planner below works for any player."
        )
        _planner(players, schedule, None, manager, horizon, gameweeks)
        return
    left, right = st.columns([3, 2], gap="medium")
    with left:
        _pitch(squad, schedule, horizon, gameweeks[0])
    with right:
        _planner(players, schedule, squad, manager, horizon, gameweeks)
    _plan(players, squad, manager)
    _optimizer(players, squad, manager, horizon)


# ---------------------------------------------------------------- model lab


def render_backtest(settings: Settings) -> None:
    """Model Lab view: how the FPL projection scored on gameweeks it had not seen."""
    root = settings.data_dir / "processed/fpl"
    frame = table(
        root / "backtest.parquet",
        ("target_gw", "predicted_points", "baseline_points", "actual_points"),
    )
    summary = report(root / "backtest.json")
    if frame.empty or not summary:
        st.info("Analytics artifacts are not built yet for this view.")
        st.caption("Run `python scripts/refresh_current_data.py` to build the FPL backtest.")
        return
    weeks = summary["gameweeks"]
    st.write(
        f"Each of gameweeks {weeks[0]}–{weeks[-1]} was projected using only data from earlier "
        f"gameweeks, then compared with the points players actually scored "
        f"({summary['observations']:,} player-gameweeks)."
    )
    series = {
        "Event model (used in the app)": "predicted_points",
        "Points-per-90 rate model": "rate_model_points",
        "FPL's own estimate (ep_next)": "provider_points",
        "Position average": "baseline_points",
    }
    series = {
        name: column
        for name, column in series.items()
        if column in frame and frame[column].notna().any()
    }
    cards = [
        ("Event model MAE", f"{summary['mae']:.2f}", "points per player-gameweek"),
        ("Position average MAE", f"{summary['baseline_mae']:.2f}", "naive reference"),
    ]
    if "provider_mae" in summary:
        cards.append(("FPL estimate MAE", f"{summary['provider_mae']:.2f}", "official ep_next"))
    cards += [
        ("Rank correlation", f"{summary['mean_spearman']:.2f}", "mean Spearman per gameweek"),
        ("Minutes MAE", f"{summary['minutes_mae']:.1f}", "expected vs played minutes"),
    ]
    theme.stat_row(cards)
    by_week = pd.concat(
        [
            frame.assign(error=(frame[column] - frame.actual_points).abs())
            .groupby("target_gw", as_index=False)
            .error.mean()
            .assign(Model=name)
            for name, column in series.items()
        ],
        ignore_index=True,
    )
    figure = go.Figure(
        [
            go.Scatter(
                x=part.target_gw,
                y=part.error,
                name=name,
                mode="lines+markers",
            )
            for name, part in by_week.groupby("Model", sort=False)
        ]
    )
    figure.update_layout(
        height=360,
        xaxis={"title": "Target gameweek", "dtick": 1},
        yaxis_title="Mean absolute error (points)",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(figure, width="stretch", key="fpl-backtest-chart")
    st.caption(
        "Lower error is better; rank correlation shows whether higher projections went to "
        "higher scorers. The model averages "
        f"{summary['mean_predicted_points']:.2f} projected points against "
        f"{summary['mean_actual_points']:.2f} scored. Players with no minutes before the "
        "forecast are excluded so reserve zeroes do not flatter the error. The backtest uses "
        "season-xG fixture strengths rebuilt for each week, not the live goal model. A handful "
        "of early-season gameweeks is too few to call any difference between models settled."
    )
    st.warning(
        "Projections are planning estimates. They do not include predicted line-ups, bookmaker "
        "odds, press conferences or late injury news."
    )


def render_fpl(settings: Settings, competition: str) -> None:
    """Render FPL planning tools from refresh-time artifacts only."""
    players, fixtures, metadata = _load(settings, competition)
    squad, manager = _team_state(settings, metadata)
    horizon = _header(settings, players, fixtures, metadata, squad, manager)
    schedule = _schedule(fixtures)
    gameweeks = sorted(int(value) for value in fixtures.gameweek.unique())
    tabs = st.tabs(
        ["This gameweek", "Players", "Fixtures", "My team & transfers", "Mini-league"]
    )
    with tabs[4]:
        _mini_league(players, squad, manager, metadata)
    with tabs[0]:
        _gameweek(players, schedule, squad, manager, horizon, gameweeks, metadata)
    with tabs[1]:
        _players(players, horizon)
    with tabs[2]:
        _ticker(fixtures)
    with tabs[3]:
        _my_team(players, schedule, squad, manager, horizon, gameweeks)
    with st.expander("How these projections work and what they cannot tell you"):
        st.markdown(
            "**Minutes.** Start, appearance and 60-minute probabilities come from each "
            "player's last six matches, weighting recent ones more, capped by official "
            "availability.\n\n"
            "**Events.** Expected goals, assists, saves, defensive contributions, bonus and "
            "cards per 90 blend this season, the player's previous season and the position "
            "average, each weighted by minutes.\n\n"
            "**Fixtures.** Each match's expected goals come from the live Dixon–Coles model, "
            "so attacking returns, clean sheets and goals conceded reflect the opponent and "
            "venue. Fixture difficulty uses the same numbers.\n\n"
            "**Limits.** The range is indicative, not a calibrated interval. Predicted "
            "line-ups, bookmaker odds, press conferences and late injuries are not included, "
            "so a projection can be stale by the deadline."
        )
        st.caption(
            f"Season {metadata['season']} · data through gameweek "
            f"{metadata['current_gameweek']} · source commit {metadata['source_commit'][:8]}"
        )
        st.page_link(
            "pages/5_Model_Lab.py",
            label="How accurate is this? See the backtest in Model Lab",
            icon=":material/query_stats:",
        )
