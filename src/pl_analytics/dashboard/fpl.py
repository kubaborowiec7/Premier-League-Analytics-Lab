"""Read-only Fantasy Premier League decision views."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import ArtifactError, report, table
from pl_analytics.data.fpl import fetch_public_team, map_public_squad
from pl_analytics.features.fpl_optimizer import (
    optimize_chip_schedule,
    optimize_squad,
    select_starting_xi,
)


def _money(value: object) -> str:
    return f"£{float(value):.1f}m" if pd.notna(value) else "—"


def _pitch(lineup: pd.DataFrame, projection: str) -> go.Figure:
    """Build an interactive football-pitch view for a legal expected-points XI."""
    starters = lineup.loc[lineup.is_starter].copy()
    coordinates = {"GK": 8, "DEF": 31, "MID": 59, "FWD": 86}
    frames = []
    for position in ("GK", "DEF", "MID", "FWD"):
        group = starters.loc[starters.position_group.eq(position)].sort_values(
            projection, ascending=False
        )
        if group.empty:
            continue
        group = group.copy()
        group["x"] = coordinates[position]
        group["y"] = [100 * (index + 1) / (len(group) + 1) for index in range(len(group))]
        frames.append(group)
    plotted = pd.concat(frames, ignore_index=True)
    plotted["label"] = plotted.player_name.str.split().str[-1]
    plotted.loc[plotted.is_recommended_captain, "label"] += " ©"
    figure = go.Figure()
    figure.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line_color="white")
    figure.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line_color="white")
    figure.add_shape(
        type="circle", x0=42, y0=35, x1=58, y1=65, line_color="white"
    )
    for x0, x1 in ((0, 17), (83, 100)):
        figure.add_shape(type="rect", x0=x0, y0=22, x1=x1, y1=78, line_color="white")
    fixture_column = "fixtures_" + projection.rsplit("_", 1)[-1]
    figure.add_trace(
        go.Scatter(
            x=plotted.x,
            y=plotted.y,
            mode="markers+text",
            text=plotted.label,
            textposition="top center",
            marker={
                "size": 28,
                "color": plotted[projection],
                "colorscale": "YlOrRd",
                "showscale": True,
                "colorbar": {"title": "xPts"},
                "line": {"color": "white", "width": 2},
            },
            customdata=plotted[
                ["player_name", "club_short_name", projection, fixture_column]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "Expected points: %{customdata[2]:.1f}<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=620,
        margin={"l": 15, "r": 15, "t": 25, "b": 15},
        paper_bgcolor="#123524",
        plot_bgcolor="#1f7a4d",
        xaxis={"range": [-4, 104], "visible": False, "fixedrange": True},
        yaxis={"range": [-5, 105], "visible": False, "fixedrange": True},
        showlegend=False,
    )
    return figure


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


def _picks(players: pd.DataFrame) -> None:
    st.subheader("Player picks")
    horizon = st.segmented_control("Projection horizon", [1, 3, 5], default=3)
    positions = sorted(players.position_group.dropna().unique())
    position = st.multiselect("Positions", positions, default=positions)
    max_price = st.slider(
        "Maximum price (£m)",
        3.5,
        float(max(4, players.price.max())),
        float(players.price.max()),
        0.5,
    )
    search = st.text_input(
        "Search player",
        placeholder="Type a surname, for example Konsa",
        key="fpl-player-search",
    ).strip()
    view = players.loc[players.position_group.isin(position) & players.price.le(max_price)].copy()
    if search:
        view = view.loc[view.player_name.str.contains(search, case=False, regex=False, na=False)]
    if view.empty:
        st.info("No players match the selected positions, price and name search.")
        return
    projection = f"projected_points_{horizon}"
    columns = {
        "player_name": "Player",
        "club_short_name": "Club",
        "position_group": "Pos",
        "price": "Price",
        projection: f"Projected {horizon} GW",
        f"projection_low_{horizon}": "Low (80%)",
        f"projection_high_{horizon}": "High (80%)",
        "expected_minutes": "xMins / fixture",
        "start_probability": "Start %",
        "sixty_probability": "60+ %",
        "ownership_pct": "Owned %",
        "form": "FPL form",
        "xg_per90": "xG/90",
        "xa_per90": "xA/90",
        "defensive_contribution_per90": "Def. actions/90",
        f"fixtures_{horizon}": "Fixtures",
    }
    st.dataframe(
        view.sort_values(projection, ascending=False)[list(columns)].rename(columns=columns),
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            f"Projected {horizon} GW": st.column_config.NumberColumn(format="%.1f"),
            "Low (80%)": st.column_config.NumberColumn(format="%.1f"),
            "High (80%)": st.column_config.NumberColumn(format="%.1f"),
            "xMins / fixture": st.column_config.NumberColumn(format="%.0f"),
            "Start %": st.column_config.NumberColumn(format="percent"),
            "60+ %": st.column_config.NumberColumn(format="percent"),
            "Owned %": st.column_config.NumberColumn(format="%.1f%%"),
            "FPL form": st.column_config.NumberColumn(format="%.1f"),
            "xG/90": st.column_config.NumberColumn(format="%.2f"),
            "xA/90": st.column_config.NumberColumn(format="%.2f"),
            "Def. actions/90": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    with st.expander("Player projection and xMins scenario"):
        selected = st.selectbox(
            "Player",
            view.sort_values("player_name").player_name,
            key="fpl-projection-player",
        )
        row = view.loc[view.player_name.eq(selected)].iloc[0]
        override = st.slider(
            "Scenario xMins",
            0,
            95,
            int(round(row.expected_minutes)),
            key="fpl-xmins-scenario",
        )
        original = float(row.expected_minutes)
        ratio = override / original if original > 0 else 0.0
        appearance_probability = min(1.0, float(row.appearance_probability) * ratio)
        sixty_probability = min(
            appearance_probability, float(row.sixty_probability) * ratio
        )
        original_appearance = float(row.xpts_appearance_1)
        other_points = float(row.projected_points_1) - original_appearance
        scenario_points = appearance_probability + sixty_probability + other_points * ratio
        first, second, third = st.columns(3)
        first.metric("Model xMins", f"{original:.1f}")
        second.metric("Scenario xMins", f"{override}")
        third.metric("Scenario GW xPts", f"{max(0, scenario_points):.2f}")
        components = pd.DataFrame(
            {
                "Component": [
                    "Appearance",
                    "Attack",
                    "Clean sheet",
                    "Saves / penalties",
                    "Defensive contributions",
                    "Bonus",
                    "Discipline",
                    "Goals conceded",
                ],
                "xPts": [
                    row.xpts_appearance_1,
                    row.xpts_attack_1,
                    row.xpts_clean_sheet_1,
                    row.xpts_saves_1,
                    row.xpts_defensive_1,
                    row.xpts_bonus_1,
                    row.xpts_discipline_1,
                    row.xpts_goals_conceded_1,
                ],
            }
        )
        st.dataframe(components, hide_index=True, width="stretch")
        st.caption(
            "The scenario is session-only and proportionally rescales event exposure. "
            "It helps test your own line-up view; it does not overwrite the model artifact."
        )


def _ticker(fixtures: pd.DataFrame) -> None:
    st.subheader("Fixture ticker")
    next_weeks = sorted(fixtures.gameweek.unique())[:5]
    view = fixtures.loc[fixtures.gameweek.isin(next_weeks)].copy()
    view["fixture"] = view.apply(
        lambda row: f"{row.opponent_short_name} ({row.venue}) · {row.difficulty}", axis=1
    )
    ticker = view.pivot_table(
        index="club_name",
        columns="gameweek",
        values="fixture",
        aggfunc=lambda values: " / ".join(values),
    )
    ticker.columns = [f"GW{value}" for value in ticker.columns]
    colors = {1: "#15803d", 2: "#65a30d", 3: "#475569", 4: "#c2410c", 5: "#b91c1c"}

    def fixture_style(value: object) -> str:
        if pd.isna(value):
            return ""
        levels = []
        for fixture in str(value).split(" / "):
            try:
                levels.append(int(fixture.rsplit("·", maxsplit=1)[-1].strip()))
            except ValueError:
                continue
        level = max(levels) if levels else 3
        return f"background-color: {colors[level]}; color: white; font-weight: 600"

    st.dataframe(ticker.style.map(fixture_style), width="stretch")
    badges = " ".join(
        f"<span style='background:{color};color:white;padding:4px 9px;border-radius:6px'>"
        f"{level}</span>"
        for level, color in colors.items()
    )
    st.markdown(f"**Difficulty:** easiest&nbsp; {badges} &nbsp;hardest", unsafe_allow_html=True)
    st.caption("A slash marks a double gameweek; its cell uses the harder fixture color.")


def _comparison(players: pd.DataFrame) -> None:
    st.subheader("Transfer comparison")
    horizon = st.segmented_control(
        "Transfer horizon", [1, 3, 5], default=3, key="transfer-horizon"
    )
    projection = f"projected_points_{horizon}"
    squad = st.session_state.get("fpl_squad")
    if isinstance(squad, pd.DataFrame) and not squad.empty:
        names = sorted(squad.player_name.unique())
        st.caption("Player-out choices use the public squad loaded in My team.")
    else:
        names = (
            players.loc[players.expected_minutes.gt(0)]
            .sort_values("player_name")
            .player_name.tolist()
        )
    left, middle = st.columns(2)
    outgoing = left.selectbox("Player out", names, index=0)
    bank = middle.number_input(
        "Money available in the bank (£m)", min_value=0.0, max_value=20.0, value=0.0, step=0.1
    )
    outgoing_row = players.loc[players.player_name.eq(outgoing)].iloc[0]
    budget = float(outgoing_row.price + bank)
    suggestions = players.loc[
        players.position_group.eq(outgoing_row.position_group)
        & players.price.le(budget)
        & players.source_player_id.ne(outgoing_row.source_player_id)
        & players.expected_minutes.ge(30)
    ].copy()
    suggestions["projected_gain"] = suggestions[projection] - float(outgoing_row[projection])
    suggestions["value_score"] = suggestions[projection] / suggestions.price
    suggestions = suggestions.sort_values(
        ["projected_gain", "expected_minutes"], ascending=False
    ).head(10)
    if suggestions.empty:
        st.info("No likely replacement fits this position and budget.")
        return
    incoming = st.selectbox("Player in", suggestions.player_name.tolist(), index=0)
    st.markdown("**Suggested replacements**")
    st.dataframe(
        suggestions[
            [
                "player_name",
                "club_short_name",
                "price",
                projection,
                "projected_gain",
                "expected_minutes",
                "value_score",
                f"fixtures_{horizon}",
            ]
        ].rename(
            columns={
                "player_name": "Player",
                "club_short_name": "Club",
                "price": "Price",
                projection: f"xPts · {horizon} GW",
                "projected_gain": "Gain vs out",
                "expected_minutes": "xMins / fixture",
                "value_score": "xPts / £m",
                f"fixtures_{horizon}": "Fixtures",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            f"xPts · {horizon} GW": st.column_config.NumberColumn(format="%.2f"),
            "Gain vs out": st.column_config.NumberColumn(format="%+.2f"),
            "xMins / fixture": st.column_config.NumberColumn(format="%.0f"),
            "xPts / £m": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Suggestions keep the same position, respect the selected purchase budget and require "
        "at least 30 modelled minutes per fixture. Club-limit and selling-price constraints are "
        "applied only by the full My team optimizer."
    )
    chosen = players.loc[players.player_name.isin([outgoing, incoming])]
    metrics = [
        projection,
        "expected_minutes",
        "price",
        "form",
        "xg_per90",
        "xa_per90",
        "defensive_contribution_per90",
    ]
    chart = chosen.melt(id_vars="player_name", value_vars=metrics, var_name="Metric")
    st.plotly_chart(
        px.bar(chart, x="Metric", y="value", color="player_name", barmode="group"),
        width="stretch",
    )
    st.caption(
        "Metrics use different units; compare each pair of bars rather than their "
        "heights across metrics."
    )


def _captaincy(players: pd.DataFrame) -> None:
    st.subheader("Captaincy shortlist")
    view = players.nlargest(10, "projected_points_1").copy()
    st.dataframe(
        view[[
            "player_name", "club_short_name", "projected_points_1", "projection_low_1",
            "projection_high_1", "expected_minutes", "fixtures_1", "ownership_pct",
        ]].rename(columns={
            "player_name": "Player", "club_short_name": "Club",
            "projected_points_1": "Projected", "projection_low_1": "Low (80%)",
            "projection_high_1": "High (80%)", "expected_minutes": "xMins",
            "fixtures_1": "Fixture", "ownership_pct": "Owned %",
        }),
        hide_index=True,
        width="stretch",
    )


def _optimizer(
    players: pd.DataFrame,
    squad: pd.DataFrame,
    manager: dict,
    *,
    horizon: int,
) -> None:
    projection = f"projected_points_{horizon}"
    with st.expander("Squad, transfer and chip optimizer", expanded=True):
        left, right = st.columns(2)
        max_transfers = left.slider("Maximum transfers", 0, 6, 2)
        free_transfers = right.slider("Free transfers", 0, 5, 1)
        if not st.button("Optimize my squad", type="primary"):
            st.caption(
                "Uses current prices as an approximation. Selling-price history and future "
                "price changes are unavailable."
            )
            return
        current_ids = set(squad.source_player_id.astype(int))
        bank = float((manager.get("bank") or 0) / 10)
        budget = float(squad.price.sum() + bank)
        current_lineup = select_starting_xi(squad, projection)
        current_points = (
            current_lineup.loc[current_lineup.is_starter, projection].sum()
            + current_lineup.loc[current_lineup.is_recommended_captain, projection].sum()
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
            actual_transfers = int((~optimized.is_current).sum())
            hits = max(0, actual_transfers - free_transfers) * 4
            optimized_points = (
                optimized.loc[optimized.is_starter, projection].sum()
                + optimized.loc[optimized.is_recommended_captain, projection].sum()
            )
            candidates.append(
                (optimized_points - hits, actual_transfers, hits, optimized, optimized_points)
            )
        net_points, transfers, hits, optimized, raw_points = max(
            candidates, key=lambda item: item[0]
        )
        incoming = optimized.loc[~optimized.is_current]
        retained = set(optimized.source_player_id.astype(int))
        outgoing = squad.loc[~squad.source_player_id.astype(int).isin(retained)]
        first, second, third = st.columns(3)
        first.metric("Recommended transfers", transfers)
        second.metric("Hit cost", f"-{hits} pts")
        third.metric("Net projected gain", f"{net_points - current_points:+.1f} pts")
        transfer_rows = []
        for direction, frame in (("OUT", outgoing), ("IN", incoming)):
            for row in frame.itertuples(index=False):
                transfer_rows.append(
                    {
                        "Action": direction,
                        "Player": row.player_name,
                        "Club": row.club_short_name,
                        "Position": row.position_group,
                        "Price": row.price,
                        "Projected": getattr(row, projection),
                    }
                )
        if transfer_rows:
            st.dataframe(pd.DataFrame(transfer_rows), hide_index=True, width="stretch")
        else:
            st.success("The optimizer prefers holding the current squad for this horizon.")

        gameweeks = sorted(
            int(column.removeprefix("projected_points_gw_"))
            for column in players
            if column.startswith("projected_points_gw_")
        )
        advice = optimize_chip_schedule(
            squad, players, budget=budget, gameweeks=gameweeks[:5]
        )
        st.markdown("**Five-gameweek chip schedule**")
        st.dataframe(
            advice.rename(
                columns={
                    "chip": "Chip",
                    "signal": "Signal",
                    "recommended_gameweek": "Gameweek",
                    "incremental_points": "Incremental xPts",
                    "reason": "Reason",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        wildcard_gain = raw_points - current_points
        wildcard_signal = "Consider" if transfers >= 4 and wildcard_gain >= 20 else "Hold"
        st.caption(
            f"Wildcard: **{wildcard_signal}** — optimized {horizon}-GW gain before hits is "
            f"{wildcard_gain:.1f} points. TC, BB and FH are allocated to different weeks; "
            "the schedule is limited to the five prepared gameweeks."
        )


def _my_team(settings: Settings, players: pd.DataFrame, metadata: dict) -> None:
    st.subheader("My team")
    with st.form("public-fpl-team"):
        default_id = str(settings.fpl_entry_id or st.session_state.get("fpl_entry_id", ""))
        entered = st.text_input(
            "Public FPL team ID",
            value=default_id,
            placeholder="For example: 123456",
            help="Find this number in the URL of your FPL Points page.",
        )
        submitted = st.form_submit_button("Load team")
    if submitted:
        if not entered.strip().isdigit() or int(entered) < 1:
            st.error("Enter a positive numeric FPL team ID.")
        else:
            try:
                entry_id = int(entered)
                entry, picks = fetch_public_team(
                    entry_id=entry_id,
                    event=int(metadata["current_gameweek"]),
                )
                loaded = map_public_squad(picks, players)
                st.session_state["fpl_entry_id"] = entry_id
                st.session_state["fpl_squad"] = loaded
                st.session_state["fpl_manager"] = {
                    "entry_id": entry_id,
                    "team_name": entry.get("name"),
                    "event": metadata["current_gameweek"],
                    "bank": entry.get("last_deadline_bank"),
                    "team_value": entry.get("last_deadline_value"),
                    "overall_points": entry.get("summary_overall_points"),
                    "overall_rank": entry.get("summary_overall_rank"),
                }
            except ValueError:
                st.error(
                    "This team could not be loaded for the current gameweek. "
                    "Check the ID and try again."
                )
    path = settings.data_dir / "processed/fpl/squad.parquet"
    squad = st.session_state.get("fpl_squad")
    if not isinstance(squad, pd.DataFrame):
        squad = table(path)
    manager = st.session_state.get("fpl_manager") or metadata.get("manager")
    if squad.empty or not manager:
        st.info(
            "Enter your public team ID above, or set FPL_ENTRY_ID for automatic refreshes. "
            "No FPL login or password is used."
        )
        return
    first, second, third = st.columns(3)
    first.metric("Team", manager.get("team_name") or "—")
    second.metric("Squad value", _money((manager.get("team_value") or 0) / 10))
    third.metric("Bank", _money((manager.get("bank") or 0) / 10))
    horizon = st.segmented_control(
        "Line-up projection horizon", [1, 3, 5], default=1, key="lineup-horizon"
    )
    projection = f"projected_points_{horizon}"
    lineup = select_starting_xi(squad, projection)
    formation = lineup.loc[lineup.is_starter].position_group.value_counts()
    captain = lineup.loc[lineup.is_recommended_captain].player_name.iloc[0]
    st.markdown(
        f"**Recommended XI · {formation.get('DEF', 0)}-"
        f"{formation.get('MID', 0)}-{formation.get('FWD', 0)} · Captain: {captain}**"
    )
    st.plotly_chart(_pitch(lineup, projection), width="stretch", key="recommended-xi-pitch")
    st.dataframe(
        lineup.sort_values(["is_starter", projection], ascending=[False, False])[
            [
                "is_starter",
                "is_recommended_captain",
                "player_name",
                "club_short_name",
                "position_group",
                "price",
                projection,
                f"fixtures_{horizon}",
            ]
        ].rename(
            columns={
                "is_starter": "Start",
                "is_recommended_captain": "Captain",
                "player_name": "Player",
                "club_short_name": "Club",
                "position_group": "Position",
                "price": "Price",
                projection: "Projected",
                f"fixtures_{horizon}": "Fixtures",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            "Projected": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    _optimizer(players, squad, manager, horizon=horizon)
    weakest = squad.nsmallest(1, projection).iloc[0]
    budget = float(weakest.price) + float((manager.get("bank") or 0) / 10)
    replacements = players.loc[
        players.price.le(budget)
        & players.position_group.eq(weakest.position_group)
        & ~players.source_player_id.isin(squad.source_player_id)
    ].nlargest(5, projection)
    st.caption(f"Candidates for {weakest.player_name} within a {_money(budget)} budget")
    st.dataframe(
        replacements[
            ["player_name", "club_short_name", "price", projection, f"fixtures_{horizon}"]
        ],
        hide_index=True,
        width="stretch",
    )


def _backtest(settings: Settings) -> None:
    st.subheader("Historical expected-points backtest")
    root = settings.data_dir / "processed/fpl"
    frame = table(
        root / "backtest.parquet",
        ("target_gw", "predicted_points", "baseline_points", "actual_points"),
    )
    summary = report(root / "backtest.json")
    if frame.empty or not summary:
        st.info("Backtest artifacts are not built yet. Run the current-data refresh.")
        return
    first, second, third, fourth, fifth = st.columns(5)
    first.metric("Observations", summary["observations"])
    second.metric("Model MAE", f"{summary['mae']:.2f}")
    third.metric("Position baseline MAE", f"{summary['baseline_mae']:.2f}")
    fourth.metric("Mean Spearman", f"{summary['mean_spearman']:.2f}")
    fifth.metric("xMins MAE", f"{summary['minutes_mae']:.1f}")
    by_week = (
        frame.assign(
            model_error=(frame.predicted_points - frame.actual_points).abs(),
            baseline_error=(frame.baseline_points - frame.actual_points).abs(),
        )
        .groupby("target_gw", as_index=False)[["model_error", "baseline_error"]]
        .mean()
        .melt(id_vars="target_gw", var_name="Model", value_name="MAE")
    )
    st.plotly_chart(
        px.line(by_week, x="target_gw", y="MAE", color="Model", markers=True),
        width="stretch",
    )
    st.caption(
        "Every target gameweek is predicted from the latest snapshot strictly before it. "
        "Players with no prior senior minutes are excluded before scoring, preventing hundreds "
        "of predictable reserve zeroes from flattering MAE. The pilot still covers only the "
        "available early-season gameweeks, so results remain unstable."
    )


def render_fpl(settings: Settings, competition: str) -> None:
    """Render FPL planning tools from refresh-time artifacts only."""
    players, fixtures, metadata = _load(settings, competition)
    st.caption(
        f"Season {metadata['season']} · through GW{metadata['current_gameweek']} · "
        f"next GW{metadata['next_gameweek']} · source commit {metadata['source_commit'][:8]}"
    )
    tabs = st.tabs(
        ["Player picks", "Fixtures", "Transfers", "Captaincy", "My team", "Backtest"]
    )
    with tabs[0]:
        _picks(players)
    with tabs[1]:
        _ticker(fixtures)
    with tabs[2]:
        _comparison(players)
    with tabs[3]:
        _captaincy(players)
    with tabs[4]:
        _my_team(settings, players, metadata)
    with tabs[5]:
        _backtest(settings)
    with st.expander("How these projections work and what they cannot tell you"):
        st.markdown(
            "The model estimates start, appearance and 60-minute probabilities from up to six "
            "recent matches with recency weighting and position priors. Expected points apply "
            "the official scoring structure to appearance, xG/xA, clean sheets, saves, penalty "
            "saves, defensive contributions, bonus, discipline and goals conceded. Event rates "
            "are shrunk toward active positional peers when minutes are scarce. Fixture strength "
            "comes from Elo. The displayed range is indicative rather than fully calibrated, and "
            "late injuries, press conferences and tactical changes can still make it stale."
        )
