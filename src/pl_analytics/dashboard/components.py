"""Match-forecast display pieces shared by the Matchday Hub and Match Predictor."""

import numpy as np
import plotly.graph_objects as go

from pl_analytics.dashboard import theme

OUTCOME_COLOURS = ("#7AA7FF", theme.TEXT_SOFT, theme.WARNING)


def outcome_cards(home: float, draw: float, away: float) -> str:
    """Home/draw/away probabilities as three stat tiles in one HTML grid."""
    tiles = "".join(
        f'<div class="pl-stat tight"><span class="label">{label}</span>'
        f'<span class="value" style="color:{colour}">{value:.1%}</span></div>'
        for label, value, colour in zip(
            ("Home win", "Draw", "Away win"), (home, draw, away), OUTCOME_COLOURS, strict=True
        )
    )
    return (
        '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">'
        f"{tiles}</div>"
    )


def likely_score(matrix: np.ndarray) -> tuple[str, float]:
    home, away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return f"{home}–{away}", float(matrix[home, away])


def score_facts(matrix: np.ndarray, rates: tuple[float, float] | None) -> str:
    """Derived markets from a score matrix: expected goals, likeliest score, BTTS, over 2.5."""
    score, chance = likely_score(matrix)
    both = float(matrix[1:, 1:].sum())
    totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    over = float(matrix[totals >= 3].sum())
    clean_home = float(matrix[:, 0].sum())
    clean_away = float(matrix[0, :].sum())
    facts = []
    if rates is not None:
        facts.append(("Expected goals (model)", f"{float(rates[0]):.2f} – {float(rates[1]):.2f}"))
    facts += [
        ("Most likely score", f"{score} ({chance:.0%})"),
        ("Both teams score", f"{both:.0%}"),
        ("Over 2.5 goals", f"{over:.0%}"),
        ("Clean sheet, home – away", f"{clean_home:.0%} – {clean_away:.0%}"),
    ]
    return "".join(
        f'<div class="pl-list-row"><span class="grow pl-muted">{label}</span><b>{value}</b></div>'
        for label, value in facts
    )


def score_heatmap(
    matrix: np.ndarray, home_name: str, away_name: str, *, goals: int = 6
) -> go.Figure:
    """Scoreline probability grid for 0 to ``goals - 1`` goals each."""
    grid = matrix[:goals, :goals]
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=[str(value) for value in range(grid.shape[1])],
            y=[str(value) for value in range(grid.shape[0])],
            colorscale=theme.SEQUENTIAL,
            text=[
                [f"{value:.0%}" if value >= 0.095 else f"{value:.1%}" for value in row]
                for row in grid
            ],
            texttemplate="%{text}",
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate=f"{home_name} %{{y}} – %{{x}} {away_name}: %{{z:.1%}}<extra></extra>",
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 40, "r": 10, "t": 10, "b": 40},
        xaxis={"title": f"{away_name} goals", "showgrid": False},
        yaxis={"title": f"{home_name} goals", "showgrid": False},
    )
    return figure
