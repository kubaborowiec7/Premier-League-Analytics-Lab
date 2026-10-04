"""Matchday Hub: the landing page built only from refresh-time artifacts.

It combines the next gameweek's fixtures with model probabilities, the current table and
form, last gameweek's results, FPL projections and availability. Nothing is fitted or
downloaded here; missing optional inputs degrade to explanatory empty states.
"""

from __future__ import annotations

import re
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard import theme
from pl_analytics.dashboard.data import (
    ArtifactError,
    model_clubs,
    prepared_models,
    report,
    scenario,
    table,
)
from pl_analytics.dashboard.standings import form_guide
from pl_analytics.data.fpl import normalise_club

PREFERRED_MODEL = "dixon_coles_half365"
UK = "Europe/London"
DEADLINE_LEAD = pd.Timedelta(minutes=90)

def _load(settings: Settings, competition: str) -> dict:
    root = settings.data_dir / "processed/fpl"
    standings = table(
        root / "standings.parquet",
        ("competition_id", "position", "club_name", "club_short_name", "played", "points"),
    )
    fixtures = table(
        root / "fixtures.parquet",
        (
            "match_id",
            "gameweek",
            "kickoff_time",
            "club_name",
            "club_short_name",
            "opponent_name",
            "opponent_short_name",
            "venue",
        ),
    )
    results = table(
        root / "results.parquet",
        (
            "gameweek",
            "kickoff_time",
            "home_name",
            "home_short_name",
            "away_name",
            "away_short_name",
            "home_score",
            "away_score",
        ),
    )
    players = table(
        root / "players.parquet",
        ("competition_id", "player_name", "club_short_name", "total_points"),
    )
    standings = standings.loc[standings.competition_id.eq(competition)]
    players = players.loc[players.competition_id.eq(competition)]
    if standings.empty and fixtures.empty:
        raise ArtifactError(
            "Matchday artifacts are not built yet. Run `python scripts/refresh_current_data.py`."
        )
    return {
        "standings": standings,
        "fixtures": fixtures,
        "results": results,
        "players": players,
        "metadata": report(root / "metadata.json"),
    }


def next_games(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    gameweek = fixtures.gameweek.min()
    games = fixtures.loc[fixtures.gameweek.eq(gameweek) & fixtures.venue.eq("H")].copy()
    games["kickoff_uk"] = pd.to_datetime(games.kickoff_time, utc=True).dt.tz_convert(UK)
    return games.sort_values(["kickoff_uk", "club_name"]).reset_index(drop=True)


def fixture_predictions(
    settings: Settings, competition: str, games: pd.DataFrame
) -> tuple[dict, dict]:
    """Score each fixture with the preferred goal model; unknown clubs are skipped."""
    try:
        statistics, ml, states, metadata = prepared_models(settings)
        names = model_clubs(settings, metadata)
    except ArtifactError:
        return {}, {}
    models = statistics.get("models", {})
    model = (
        PREFERRED_MODEL
        if PREFERRED_MODEL in models
        else next((name for name, item in models.items() if hasattr(item, "rates")), None)
    )
    if model is None:
        return {}, metadata
    lookup = {
        normalise_club(name): club
        for club, name in names.loc[
            names.competition_id.eq(competition), ["club_id", "name"]
        ].itertuples(index=False)
    }
    predictions = {}
    for game in games.itertuples(index=False):
        home, away = (
            lookup.get(normalise_club(game.club_name)),
            lookup.get(normalise_club(game.opponent_name)),
        )
        if home is None or away is None:
            continue
        try:
            predictions[game.match_id] = scenario(
                home, away, competition, model, statistics, ml, states
            )
        except ArtifactError:
            continue
    return predictions, {**metadata, "model": model}


_form = form_guide


def _likely_score(matrix: np.ndarray) -> tuple[str, float]:
    home, away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return f"{home}–{away}", float(matrix[home, away])


# ---------------------------------------------------------------- sections


def _header(games: pd.DataFrame, metadata: dict) -> None:
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    if games.empty:
        eyebrow = f"Season {metadata.get('season', '')}".strip()
    else:
        first, last = games.kickoff_uk.min(), games.kickoff_uk.max()
        eyebrow = f"Gameweek {int(games.gameweek.iloc[0])} · {first:%a %d %b} – {last:%a %d %b}"
    with left:
        theme.page_header("Matchday Hub", eyebrow)
    if not games.empty:
        deadline = games.kickoff_uk.min() - DEADLINE_LEAD
        passed = deadline <= pd.Timestamp.now(tz=UK)
        text = "Deadline passed" if passed else f"{deadline:%a %d %b, %H:%M} UK"
        with right:
            a, b = st.columns([3, 2], vertical_alignment="center")
            a.markdown(theme.stat_card("FPL deadline", text), unsafe_allow_html=True)
            b.page_link(
                "pages/4_FPL_Decision_Lab.py",
                label="Plan my FPL team",
                icon=":material/sports_soccer:",
            )


def _kpis(standings: pd.DataFrame, results: pd.DataFrame, players: pd.DataFrame) -> None:
    cards = []
    if not standings.empty:
        ordered = standings.sort_values("position")
        top = ordered.iloc[0]
        lead = int(top.points - ordered.iloc[1].points) if len(ordered) > 1 else 0
        wins = int(top.wins) if "wins" in top else 0
        cards.append(
            theme.stat_card(
                "Leaders",
                str(top.club_name),
                f"{int(top.points)} pts · {wins} wins from {int(top.played)} · +{lead} clear",
            )
        )
        if "goals_for" in standings:
            matches = standings.played.sum() / 2
            goals = int(standings.goals_for.sum())
            if matches:
                cards.append(
                    theme.stat_card(
                        "Goals per match",
                        f"{goals / matches:.2f}",
                        f"{goals} goals in {matches:.0f} matches",
                    )
                )
    if not results.empty:
        home = (results.home_score > results.away_score).mean()
        draw = (results.home_score == results.away_score).mean()
        cards.append(
            theme.stat_card(
                "Results split",
                f"{home:.0%} home wins",
                f"Draws {draw:.0%} · away wins {1 - home - draw:.0%}",
            )
        )
    if not players.empty:
        best = players.sort_values("total_points", ascending=False).iloc[0]
        price = f" · £{best.price:.1f}m" if "price" in best and pd.notna(best.price) else ""
        cards.append(
            theme.stat_card(
                "Top FPL scorer",
                f"{best.player_name.split()[-1]} · {int(best.total_points)}",
                f"{best.player_name} · {best.club_short_name}{price}",
            )
        )
    if cards:
        st.markdown(f'<div class="pl-stats">{"".join(cards)}</div>', unsafe_allow_html=True)


def _fixtures(games: pd.DataFrame, predictions: dict) -> None:
    rows = [
        '<div class="grid head"><span>KO</span><span style="text-align:right">HOME</span>'
        '<span style="text-align:center">WIN · DRAW · WIN</span><span>AWAY</span>'
        '<span style="text-align:right">SCORE</span></div>'
    ]
    for game in games.itertuples(index=False):
        prediction = predictions.get(game.match_id)
        if prediction is None:
            bar, likely = (
                '<span class="pl-muted" style="text-align:center">No model estimate</span>',
                "",
            )
        else:
            bar = theme.probability_bar(*prediction["probabilities"])
            likely = (
                _likely_score(prediction["matrix"])[0] if prediction["matrix"] is not None else ""
            )
        rows.append(
            '<div class="grid row">'
            f'<span><span class="pl-muted">{game.kickoff_uk:%a}</span><br>'
            f"<b>{game.kickoff_uk:%H:%M}</b></span>"
            f'<span class="team" style="justify-content:flex-end"><span class="name">'
            f"{escape(str(game.club_name))}</span>{theme.club_chip(game.club_short_name)}</span>"
            f"{bar}"
            f'<span class="team">{theme.club_chip(game.opponent_short_name)}'
            f'<span class="name">{escape(str(game.opponent_name))}</span></span>'
            f'<span class="pl-muted" style="text-align:right">{likely}</span></div>'
        )
    legend = (
        f'<span class="pl-muted">Home team on the left. Bar: home win '
        f'<b style="color:#7AA7FF">■</b>, draw <b style="color:{theme.DRAW}">■</b>, '
        f'away win <b style="color:{theme.AWAY}">■</b>. '
        "Score: most likely result. Times are UK.</span>"
    )
    title = f"Gameweek {int(games.gameweek.iloc[0])} fixtures"
    st.markdown(
        theme.card(title, f'<div class="pl-fixtures">{"".join(rows)}</div>{legend}'),
        unsafe_allow_html=True,
    )


def _preview(games: pd.DataFrame, predictions: dict) -> None:
    scored = [game for game in games.itertuples(index=False) if game.match_id in predictions]
    if not scored:
        return
    labels = {
        game.match_id: f"{game.club_name} v {game.opponent_name} · {game.kickoff_uk:%a %H:%M}"
        for game in scored
    }
    with st.container(border=True):
        match_id = st.selectbox(
            "Match preview", list(labels), format_func=labels.get, key="hub_preview"
        )
        game = next(item for item in scored if item.match_id == match_id)
        result = predictions[match_id]
        home, draw, away = (float(value) for value in result["probabilities"])
        left, right = st.columns([2, 3])
        with left:
            # One HTML grid, not st.columns: the hub is already two column levels deep.
            st.markdown(
                '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">'
                + "".join(
                    f'<div class="pl-stat"><span class="label">{label}</span>'
                    f'<span class="value" style="color:{colour}">{value:.0%}</span></div>'
                    for label, value, colour in zip(
                        ("Home win", "Draw", "Away win"),
                        (home, draw, away),
                        ("#7AA7FF", theme.TEXT_SOFT, theme.WARNING),
                        strict=True,
                    )
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            matrix = result["matrix"]
            if matrix is not None:
                score, chance = _likely_score(matrix)
                both = float(matrix[1:, 1:].sum())
                totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
                over = float(matrix[totals >= 3].sum())
                rates = result.get("rates")
                rates = (np.nan, np.nan) if rates is None else tuple(float(v) for v in rates)
                st.markdown(
                    '<div style="margin-top:14px">'
                    + "".join(
                        f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span>'
                        f"<b>{value}</b></div>"
                        for label, value in (
                            ("Expected goals (model)", f"{rates[0]:.2f} – {rates[1]:.2f}"),
                            ("Most likely score", f"{score} ({chance:.0%})"),
                            ("Both teams score", f"{both:.0%}"),
                            ("Over 2.5 goals", f"{over:.0%}"),
                        )
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
        with right:
            matrix = result["matrix"]
            if matrix is None:
                st.info("This model estimates outcomes only; it does not produce scorelines.")
            else:
                grid = matrix[:6, :6]
                figure = go.Figure(
                    go.Heatmap(
                        z=grid,
                        x=[str(value) for value in range(6)],
                        y=[str(value) for value in range(6)],
                        colorscale=theme.SEQUENTIAL,
                        text=[
                            [f"{value:.0%}" if value >= 0.095 else f"{value:.1%}" for value in row]
                            for row in grid
                        ],
                        texttemplate="%{text}",
                        showscale=False,
                        xgap=3,
                        ygap=3,
                        hovertemplate=(
                            f"{game.club_short_name} %{{y}} – %{{x}} {game.opponent_short_name}: "
                            "%{z:.1%}<extra></extra>"
                        ),
                    )
                )
                figure.update_layout(
                    height=330,
                    margin={"l": 40, "r": 10, "t": 10, "b": 40},
                    xaxis={"title": f"{game.opponent_name} goals", "showgrid": False},
                    yaxis={"title": f"{game.club_name} goals", "showgrid": False},
                )
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _table(standings: pd.DataFrame, form: dict[str, str]) -> None:
    if standings.empty:
        return
    has_xg = {"expected_goals", "expected_goals_against"} <= set(standings)
    head = (
        "<tr><th>#</th><th>CLUB</th><th>P</th><th>GD</th>"
        + ("<th>xGD</th>" if has_xg else "")
        + "<th>PTS</th><th>FORM</th></tr>"
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
        goal_difference = int(getattr(club, "goal_difference", 0))
        xg_cell = ""
        if has_xg:
            xgd = float(club.expected_goals - club.expected_goals_against)
            colour = (
                theme.ACCENT_TEXT
                if xgd > 0.05
                else theme.BAD_TEXT
                if xgd < -0.05
                else theme.TEXT_SOFT
            )
            xg_cell = f'<td style="color:{colour}">{xgd:+.1f}</td>'
        rows.append(
            "<tr>"
            f'<td><span class="pl-badge" style="{zone};border-radius:6px;padding:2px 7px">'
            f"{position}</span></td>"
            f'<td style="white-space:nowrap;color:{theme.TEXT}">'
            f"{theme.club_chip(club.club_short_name)} "
            f"{escape(str(club.club_name))}</td>"
            f"<td>{int(club.played)}</td><td>{goal_difference:+d}</td>{xg_cell}"
            f'<td style="color:{theme.TEXT};font-weight:700;font-size:14px">{int(club.points)}</td>'
            f'<td style="white-space:nowrap">'
            f"{theme.form_pills(form.get(club.club_short_name, ''))}</td>"
            "</tr>"
        )
    legend = (
        f'<div class="pl-muted" style="margin-top:10px">'
        f'<b style="color:{theme.ACCENT}">■</b> Champions League places · '
        f'<b style="color:{theme.BAD}">■</b> Relegation · xGD = expected goal difference</div>'
    )
    st.markdown(
        theme.card(
            "Table",
            f'<div style="overflow-x:auto"><table class="pl-table">{head}{"".join(rows)}'
            f"</table></div>{legend}",
        ),
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/3_League_Table.py", label="Full table with xPts", icon=":material/table_chart:"
    )


def _results(results: pd.DataFrame) -> None:
    if results.empty:
        st.markdown(
            theme.card(
                "Latest results",
                '<span class="pl-muted">Results appear after the next current-data refresh.</span>',
            ),
            unsafe_allow_html=True,
        )
        return
    gameweek = int(results.gameweek.max())
    rows = []
    for match in results.loc[results.gameweek.eq(gameweek)].itertuples(index=False):
        xg = ""
        if "home_xg" in results and pd.notna(match.home_xg) and pd.notna(match.away_xg):
            xg = f'<br><span class="pl-muted">xG {match.home_xg:.1f}–{match.away_xg:.1f}</span>'
        rows.append(
            '<div class="pl-list-row" style="display:grid;'
            'grid-template-columns:minmax(0,1fr) 70px minmax(0,1fr)">'
            f'<span style="text-align:right;font-weight:600">{escape(str(match.home_name))} '
            f"{theme.club_chip(match.home_short_name)}</span>"
            f'<span style="text-align:center"><span class="big">{int(match.home_score)} – '
            f"{int(match.away_score)}</span>{xg}</span>"
            f'<span style="font-weight:600">{theme.club_chip(match.away_short_name)} '
            f"{escape(str(match.away_name))}</span></div>"
        )
    st.markdown(theme.card(f"Gameweek {gameweek} results", "".join(rows)), unsafe_allow_html=True)


def _picks(players: pd.DataFrame) -> None:
    column = "projected_points_1"
    if column not in players or players[column].isna().all():
        return
    available = players.loc[players.get("status", pd.Series("a", index=players.index)).eq("a")]
    top = available.sort_values(column, ascending=False).head(7)
    best = float(top[column].max()) or 1.0
    rows = []
    for player in top.itertuples(index=False):
        fixture = escape(str(getattr(player, "fixtures_1", "") or ""))
        details = " · ".join(
            item
            for item in (
                str(getattr(player, "position_group", "") or ""),
                f"£{player.price:.1f}m" if pd.notna(getattr(player, "price", np.nan)) else "",
                fixture,
                f"{player.ownership_pct:.1f}% owned"
                if pd.notna(getattr(player, "ownership_pct", np.nan))
                else "",
            )
            if item
        )
        value = float(getattr(player, column))
        rows.append(
            f'<div class="pl-list-row">{theme.club_chip(player.club_short_name)}'
            f'<span class="grow"><b>{escape(str(player.player_name))}</b>'
            f'<span class="pl-muted">{details}</span></span>'
            '<span style="width:64px;height:6px;border-radius:3px;'
            f'background:{theme.GRIDLINE};overflow:hidden">'
            f'<span style="display:block;height:100%;width:{100 * value / best:.0f}%;'
            f'background:{theme.ACCENT}"></span></span>'
            f'<span class="big" style="width:40px;text-align:right">{value:.1f}</span></div>'
        )
    st.markdown(theme.card("FPL picks · next gameweek xPts", "".join(rows)), unsafe_allow_html=True)


_STATUS = {
    "d": "Doubtful",
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
    "n": "Not in squad",
}


def _availability(players: pd.DataFrame) -> None:
    if "status" not in players:
        return
    doubts = players.loc[players.status.isin(list(_STATUS))].copy()
    if doubts.empty:
        return
    sort = "ownership_pct" if "ownership_pct" in doubts else "total_points"
    rows = []
    for player in doubts.sort_values(sort, ascending=False).head(6).itertuples(index=False):
        news = str(getattr(player, "news", "") or _STATUS[player.status])
        news = re.sub(r"\s*-\s*\d+% chance of playing", "", news)
        chance = getattr(player, "chance_of_playing", np.nan)
        label, tone = (
            (f"{chance:.0f}%", "warning") if pd.notna(chance) and chance > 0 else ("Out", "bad")
        )
        owned = (
            f" · {player.ownership_pct:.1f}% owned"
            if pd.notna(getattr(player, "ownership_pct", np.nan))
            else ""
        )
        rows.append(
            f'<div class="pl-list-row">{theme.club_chip(player.club_short_name)}'
            f'<span class="grow"><b>{escape(str(player.player_name))}</b>'
            f'<span class="pl-muted">{escape(news)}{owned}</span></span>'
            f"{theme.badge(label, tone)}</div>"
        )
    st.markdown(theme.card("Availability watch", "".join(rows)), unsafe_allow_html=True)


def _about(metadata: dict, model_metadata: dict) -> None:
    with st.expander("About this data"):
        lines = [
            "Table, fixtures, results, xG and FPL figures come from the FPL-Core-Insights snapshot"
            + (f" of {metadata['reference_date']}" if metadata.get("reference_date") else "")
            + ".",
        ]
        if model_metadata:
            last_result = model_metadata.get("last_result_date", model_metadata.get("origin", "?"))
            lines.append(
                f"Match probabilities use the {model_metadata.get('model', 'goal')} model with "
                f"results through {last_result}. "
                "They describe uncertainty, not certainty, and ignore late team news."
            )
        else:
            lines.append("Match probabilities appear once live match models are prepared.")
        lines.append(
            "FPL xPts are one-gameweek projections from the Decision Lab model; availability uses "
            "the official FPL status at refresh time. Clubs are shown by colour codes, not crests."
        )
        for line in lines:
            st.markdown(f"- {line}")


def render_hub(settings: Settings, competition: str) -> None:
    """Render the Matchday Hub for one competition."""
    data = _load(settings, competition)
    games = next_games(data["fixtures"])
    predictions, model_metadata = (
        fixture_predictions(settings, competition, games) if not games.empty else ({}, {})
    )
    _header(games, data["metadata"])
    _kpis(data["standings"], data["results"], data["players"])
    left, right = st.columns([3, 2], gap="medium")
    with left:
        if not games.empty:
            _fixtures(games, predictions)
            _preview(games, predictions)
            st.page_link(
                "pages/8_Match_Centre.py",
                label="Form, xG and players for each fixture",
                icon=":material/sports:",
            )
        else:
            st.info("No upcoming fixtures are prepared.")
    with right:
        _table(data["standings"], _form(data["results"]))
    first, second, third = st.columns(3, gap="medium")
    with first:
        _results(data["results"])
    with second:
        _picks(data["players"])
    with third:
        _availability(data["players"])
    _about(data["metadata"], model_metadata)
