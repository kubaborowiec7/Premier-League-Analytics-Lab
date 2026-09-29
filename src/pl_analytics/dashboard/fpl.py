"""Read-only Fantasy Premier League decision views."""

import pandas as pd
import plotly.express as px
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import ArtifactError, report, table


def _money(value: object) -> str:
    return f"£{float(value):.1f}m" if pd.notna(value) else "—"


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
    view = players.loc[players.position_group.isin(position) & players.price.le(max_price)].copy()
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
            "Owned %": st.column_config.NumberColumn(format="%.1f%%"),
            "FPL form": st.column_config.NumberColumn(format="%.1f"),
            "xG/90": st.column_config.NumberColumn(format="%.2f"),
            "xA/90": st.column_config.NumberColumn(format="%.2f"),
            "Def. actions/90": st.column_config.NumberColumn(format="%.1f"),
        },
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
    st.dataframe(ticker, width="stretch")
    st.caption("Difficulty: 1 is easiest and 5 is hardest. A slash marks a double gameweek.")


def _comparison(players: pd.DataFrame) -> None:
    st.subheader("Transfer comparison")
    names = players.sort_values("player_name").player_name.tolist()
    left, right = st.columns(2)
    outgoing = left.selectbox("Player out", names, index=0)
    incoming = right.selectbox("Player in", names, index=min(1, len(names) - 1))
    chosen = players.loc[players.player_name.isin([outgoing, incoming])]
    metrics = [
        "projected_points_3",
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


def _my_team(settings: Settings, players: pd.DataFrame, metadata: dict) -> None:
    st.subheader("My team")
    path = settings.data_dir / "processed/fpl/squad.parquet"
    squad = table(path)
    manager = metadata.get("manager")
    if squad.empty or not manager:
        st.info(
            "Optional: set FPL_ENTRY_ID to your public team ID and run the refresh. "
            "No FPL login or password is used."
        )
        return
    first, second, third = st.columns(3)
    first.metric("Team", manager.get("team_name") or "—")
    second.metric("Squad value", _money((manager.get("team_value") or 0) / 10))
    third.metric("Bank", _money((manager.get("bank") or 0) / 10))
    st.dataframe(
        squad.sort_values("position")[[
            "position", "player_name", "club_short_name", "price", "projected_points_3",
            "fixtures_3", "is_captain", "is_vice_captain",
        ]],
        hide_index=True,
        width="stretch",
    )
    weakest = squad.nsmallest(1, "projected_points_3").iloc[0]
    budget = float(weakest.price) + float((manager.get("bank") or 0) / 10)
    replacements = players.loc[
        players.price.le(budget)
        & players.position_group.eq(weakest.position_group)
        & ~players.source_player_id.isin(squad.source_player_id)
    ].nlargest(5, "projected_points_3")
    st.caption(f"Candidates for {weakest.player_name} within a {_money(budget)} budget")
    st.dataframe(
        replacements[
            ["player_name", "club_short_name", "price", "projected_points_3", "fixtures_3"]
        ],
        hide_index=True,
        width="stretch",
    )


def render_fpl(settings: Settings, competition: str) -> None:
    """Render FPL planning tools from refresh-time artifacts only."""
    players, fixtures, metadata = _load(settings, competition)
    st.caption(
        f"Season {metadata['season']} · through GW{metadata['current_gameweek']} · "
        f"next GW{metadata['next_gameweek']} · source commit {metadata['source_commit'][:8]}"
    )
    tabs = st.tabs(["Player picks", "Fixtures", "Transfers", "Captaincy", "My team"])
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
    with st.expander("How these projections work and what they cannot tell you"):
        st.markdown(
            "The baseline blends season points per 90 with xG, xA, clean-sheet probability, "
            "defensive contributions, expected minutes and Elo fixture strength. Rates are shrunk "
            "toward the position average when minutes are scarce. The displayed 80% range uses "
            "past gameweek volatility and is indicative rather than fully calibrated. Injuries, "
            "line-ups and late team news can make the projection stale; it is decision support, "
            "not a guarantee."
        )
