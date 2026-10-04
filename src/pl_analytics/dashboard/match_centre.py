"""Match Centre: one upcoming fixture with its forecast, both clubs' form and key players.

Everything is read from refresh-time artifacts. A fixture the live match model cannot
price still shows form, standings and players.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard import components, theme
from pl_analytics.dashboard.data import ArtifactError, table
from pl_analytics.dashboard.hub import UK, fixture_predictions
from pl_analytics.dashboard.standings import form_guide
from pl_analytics.features.fpl import scorer_markets

FIXTURE_COLUMNS = (
    "match_id",
    "gameweek",
    "kickoff_time",
    "club_id",
    "club_name",
    "club_short_name",
    "opponent_id",
    "opponent_name",
    "opponent_short_name",
    "venue",
)


def _games(fixtures: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    games = fixtures.loc[fixtures.gameweek.eq(gameweek) & fixtures.venue.eq("H")].copy()
    games["kickoff_uk"] = pd.to_datetime(games.kickoff_time, utc=True).dt.tz_convert(UK)
    return games.sort_values(["kickoff_uk", "club_name"]).reset_index(drop=True)


def club_matches(results: pd.DataFrame, short_name: str) -> pd.DataFrame:
    """One row per completed match from a club's point of view, oldest first."""
    columns = ["gameweek", "opponent", "venue", "scored", "conceded", "xg", "xga", "result"]
    required = {"home_short_name", "away_short_name", "home_score", "away_score", "gameweek"}
    if results.empty or not required <= set(results):
        return pd.DataFrame(columns=columns)
    rows = []
    for match in results.sort_values("kickoff_time").itertuples(index=False):
        if short_name == match.home_short_name:
            side, other, venue = "home", "away", "H"
        elif short_name == match.away_short_name:
            side, other, venue = "away", "home", "A"
        else:
            continue
        scored, conceded = getattr(match, f"{side}_score"), getattr(match, f"{other}_score")
        rows.append(
            {
                "gameweek": int(match.gameweek),
                "opponent": getattr(match, f"{other}_short_name"),
                "venue": venue,
                "scored": int(scored),
                "conceded": int(conceded),
                "xg": getattr(match, f"{side}_xg", float("nan")),
                "xga": getattr(match, f"{other}_xg", float("nan")),
                "result": "W" if scored > conceded else "D" if scored == conceded else "L",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _club_card(
    name: str,
    short: str,
    matches: pd.DataFrame,
    standing: pd.Series | None,
    outlook: pd.Series | None,
    form: str,
) -> str:
    facts = []
    if standing is not None:
        facts += [
            ("League position", f"{int(standing.position)} · {int(standing.points)} pts"),
            (
                "Record",
                f"{int(standing.wins)}W {int(standing.draws)}D {int(standing.losses)}L · "
                f"{int(standing.goals_for)}–{int(standing.goals_against)}",
            ),
            (
                "xG for – against per match",
                f"{standing.expected_goals / standing.played:.2f} – "
                f"{standing.expected_goals_against / standing.played:.2f}",
            ),
        ]
    if outlook is not None:
        facts += [
            ("Projected points", f"{outlook.mean_points:.0f}"),
            (
                f"Top {int(outlook.top_places)} · relegation",
                f"{outlook.top_probability:.0%} · {outlook.relegation_probability:.0%}",
            ),
        ]
    body = (
        f'<div style="margin:-4px 0 8px">{theme.form_pills(form)}</div>'
        + "".join(
            f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
            f"<b>{value}</b></div>"
            for label, value in facts
        )
    )
    recent = matches.tail(5).iloc[::-1]
    if not recent.empty:
        tone = {"W": "good", "D": "warning", "L": "bad"}

        def chances(row: object) -> str:
            known = pd.notna(row.xg) and pd.notna(row.xga)
            return f" · xG {row.xg:.1f}–{row.xga:.1f}" if known else ""

        body += '<div class="pl-eyebrow" style="margin-top:12px">Last matches</div>' + "".join(
            '<div class="pl-list-row">'
            f"{theme.club_chip(row.opponent)}"
            f'<span class="grow"><b>{row.scored}–{row.conceded}</b>'
            f'<span class="pl-muted">GW{row.gameweek} · {"home" if row.venue == "H" else "away"}'
            f"{chances(row)}</span></span>{theme.badge(row.result, tone[row.result])}</div>"
            for row in recent.itertuples(index=False)
        )
    return theme.card(f"{name}", theme.club_chip(short) + body)


def _xg_trend(matches: pd.DataFrame, name: str) -> go.Figure:
    figure = go.Figure(
        [
            go.Bar(
                x=[f"GW{value}" for value in matches.gameweek],
                y=matches.xg,
                name="xG for",
                marker_color=theme.ACCENT,
                hovertemplate="%{x}: %{y:.2f} xG for<extra></extra>",
            ),
            go.Bar(
                x=[f"GW{value}" for value in matches.gameweek],
                y=matches.xga,
                name="xG against",
                marker_color=theme.BAD,
                hovertemplate="%{x}: %{y:.2f} xG against<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        barmode="group",
        height=240,
        margin={"l": 10, "r": 10, "t": 30, "b": 30},
        title={"text": name, "font": {"size": 16}},
        legend={"orientation": "h", "y": 1.2, "x": 1, "xanchor": "right"},
        yaxis_title="Expected goals",
    )
    return figure


def _players(markets: pd.DataFrame, club_id: object) -> str:
    squad = markets.loc[markets.club_id.eq(club_id)]
    if squad.empty:
        return '<span class="pl-muted">No player projections for this fixture.</span>'
    rows = "".join(
        '<div class="pl-list-row">'
        f'<span class="grow"><b>{escape(str(row.player_name))}</b>'
        f'<span class="pl-muted">{row.position_group} · {row.start_probability:.0%} to start · '
        f"{row.expected_minutes:.0f} xMins</span></span>"
        f'<span class="big">{row.scorer_probability:.0%}</span></div>'
        for row in squad.nlargest(5, "scorer_probability").itertuples(index=False)
    )
    flagged = squad.loc[squad.status.ne("a") & squad.minutes.gt(0)] if "status" in squad else squad
    doubts = ""
    if "status" in squad and not flagged.empty:
        doubts = '<div class="pl-eyebrow" style="margin-top:12px">Availability</div>' + "".join(
            '<div class="pl-list-row">'
            f'<span class="grow"><b>{escape(str(row.player_name))}</b>'
            f'<span class="pl-muted">{escape(str(getattr(row, "news", "") or "Flagged"))}</span>'
            "</span></div>"
            for row in flagged.nlargest(4, "minutes").itertuples(index=False)
        )
    return rows + doubts


def render_match_centre(settings: Settings, competition: str) -> None:
    """Render one upcoming fixture from prepared artifacts only."""
    root = settings.data_dir / "processed/fpl"
    fixtures = table(root / "fixtures.parquet", FIXTURE_COLUMNS)
    if fixtures.empty:
        raise ArtifactError(
            "Match Centre artifacts are not built yet. Run the current-data refresh."
        )
    results = table(root / "results.parquet")
    standings = table(root / "standings.parquet")
    simulation = table(root / "season_simulation.parquet")
    players = table(root / "players.parquet")
    if not standings.empty:
        standings = standings.loc[standings.competition_id.eq(competition)]
    if not players.empty:
        players = players.loc[players.competition_id.eq(competition)]

    weeks = sorted(int(value) for value in fixtures.gameweek.unique())[:5]
    first, second = st.columns([1, 3])
    gameweek = first.selectbox(
        "Gameweek", weeks, format_func=lambda value: f"Gameweek {value}", key="centre-gameweek"
    )
    games = _games(fixtures, gameweek)
    labels = {
        game.match_id: f"{game.club_name} v {game.opponent_name} · {game.kickoff_uk:%a %d %b %H:%M}"
        for game in games.itertuples(index=False)
    }
    match_id = second.selectbox(
        "Fixture", list(labels), format_func=labels.get, key=f"centre-fixture-{gameweek}"
    )
    game = games.loc[games.match_id.eq(match_id)].iloc[0]
    home, away = str(game.club_name), str(game.opponent_name)
    st.markdown(
        '<div class="pl-card" style="text-align:center">'
        f'<div class="pl-eyebrow">Gameweek {gameweek} · {game.kickoff_uk:%A %d %B, %H:%M} UK</div>'
        '<div class="pl-title" style="font-size:36px;margin:8px 0 0">'
        f"{theme.club_chip(game.club_short_name)} {escape(home)} "
        f'<span style="color:{theme.MUTED}">v</span> {escape(away)} '
        f"{theme.club_chip(game.opponent_short_name)}</div></div>",
        unsafe_allow_html=True,
    )

    predictions, _ = fixture_predictions(settings, competition, games)
    prediction = predictions.get(match_id)
    with st.container(border=True):
        st.markdown('<div class="pl-card-title">Forecast</div>', unsafe_allow_html=True)
        if prediction is None:
            st.info(
                "The live match model has no forecast for this fixture. Form and players "
                "are shown below."
            )
        else:
            probabilities = [float(value) for value in prediction["probabilities"]]
            matrix = prediction["matrix"]
            left, right = st.columns([2, 3], gap="medium")
            with left:
                st.markdown(
                    components.outcome_cards(*probabilities)
                    + f'<div style="margin-top:12px">{theme.probability_bar(*probabilities)}</div>'
                    + (
                        '<div style="margin-top:10px">'
                        + components.score_facts(matrix, prediction["rates"])
                        + "</div>"
                        if matrix is not None
                        else ""
                    ),
                    unsafe_allow_html=True,
                )
            with right:
                if matrix is not None:
                    st.plotly_chart(
                        components.score_heatmap(matrix, home, away),
                        width="stretch",
                        key="centre-heatmap",
                        config={"displayModeBar": False},
                    )
            st.caption(
                "From the live Dixon–Coles goal model. Bookmakers have been more accurate "
                "than this model; see Betting Lab for prices and the benchmark."
            )

    form = form_guide(results) if not results.empty else {}
    left, right = st.columns(2, gap="medium")
    histories = {}
    for column, name, short, club_id in (
        (left, home, game.club_short_name, game.club_id),
        (right, away, game.opponent_short_name, game.opponent_id),
    ):
        histories[name] = club_matches(results, short)
        standing = standings.loc[standings.club_id.eq(club_id)] if not standings.empty else None
        outlook = simulation.loc[simulation.club_id.eq(club_id)] if not simulation.empty else None
        column.markdown(
            _club_card(
                name,
                short,
                histories[name],
                None if standing is None or standing.empty else standing.iloc[0],
                None if outlook is None or outlook.empty else outlook.iloc[0],
                form.get(short, ""),
            ),
            unsafe_allow_html=True,
        )
    if any(frame.xg.notna().any() for frame in histories.values()):
        with st.container(border=True):
            st.markdown(
                '<div class="pl-card-title">Expected goals by match</div>', unsafe_allow_html=True
            )
            for column, (name, frame) in zip(st.columns(2), histories.items(), strict=True):
                if frame.xg.notna().any():
                    column.plotly_chart(
                        _xg_trend(frame, name),
                        width="stretch",
                        key=f"centre-xg-{name}",
                        config={"displayModeBar": False},
                    )
            st.caption("Green: chances created. Coral: chances conceded. This season only.")

    if not players.empty and "projected_xg_per90" in players:
        markets = scorer_markets(players, fixtures, gameweek)
        markets = markets.loc[markets.match_id.eq(match_id)]
        left, right = st.columns(2, gap="medium")
        for column, name, club_id in (
            (left, home, game.club_id),
            (right, away, game.opponent_id),
        ):
            column.markdown(
                theme.card(f"{name} · chance to score", _players(markets, club_id)),
                unsafe_allow_html=True,
            )
        st.caption(
            "Each figure is the model's probability that the player scores in this match. "
            "Line-ups and late news are not included."
        )
    first, second = st.columns(2)
    first.page_link(
        "pages/7_Betting_Lab.py", label="Fair odds for this gameweek", icon=":material/paid:"
    )
    second.page_link(
        "pages/4_Match_Predictor.py",
        label="Try other models in Match Predictor",
        icon=":material/query_stats:",
    )
