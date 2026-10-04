"""Monte Carlo season outcomes from the current table and remaining fixtures' goal rates."""

import numpy as np
import pandas as pd

from pl_analytics.statistics.matches import score_matrix

MAX_GOALS = 20


def simulate_season(
    standings: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    rho: float = 0.0,
    simulations: int = 10_000,
    seed: int = 42,
    top_places: int = 4,
    relegation_places: int = 3,
) -> pd.DataFrame:
    """Play out every remaining match ``simulations`` times and summarise final positions.

    ``standings`` holds ``club_id``, ``points``, ``goals_for`` and ``goals_against`` so far;
    ``matches`` holds ``home_club_id``, ``away_club_id``, ``home_rate`` and ``away_rate``
    for unplayed fixtures. Scores are drawn from the Dixon–Coles score matrix of each
    fixture. Ties are split by goal difference, goals scored, then at random.

    Goal rates are held fixed for the whole season: form changes, injuries and transfers
    are not simulated, so the probabilities are more confident than they should be.
    """
    required = {"club_id", "points", "goals_for", "goals_against"}
    fixture_columns = {"home_club_id", "away_club_id", "home_rate", "away_rate"}
    if not required <= set(standings) or not fixture_columns <= set(matches):
        raise ValueError("Season simulation inputs are incomplete")
    if standings.club_id.duplicated().any() or simulations < 100:
        raise ValueError("Unique clubs and at least 100 simulations are required")
    clubs = list(
        dict.fromkeys([*standings.club_id, *matches.home_club_id, *matches.away_club_id])
    )
    count = len(clubs)
    if not 0 <= top_places <= count or not 0 <= relegation_places <= count:
        raise ValueError("Zone sizes exceed the number of clubs")
    index = {club: position for position, club in enumerate(clubs)}
    start = standings.set_index("club_id").reindex(clubs).fillna(0)
    points = np.tile(start.points.to_numpy(float), (simulations, 1))
    scored = np.tile(start.goals_for.to_numpy(float), (simulations, 1))
    conceded = np.tile(start.goals_against.to_numpy(float), (simulations, 1))
    rng = np.random.default_rng(seed)
    for match in matches.itertuples(index=False):
        matrix, _ = score_matrix(
            float(match.home_rate), float(match.away_rate), rho, max_goals=MAX_GOALS
        )
        cells = rng.choice(matrix.size, size=simulations, p=matrix.ravel())
        home_goals, away_goals = np.divmod(cells, MAX_GOALS + 1)
        home, away = index[match.home_club_id], index[match.away_club_id]
        points[:, home] += np.where(
            home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0)
        )
        points[:, away] += np.where(
            away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0)
        )
        scored[:, home] += home_goals
        conceded[:, home] += away_goals
        scored[:, away] += away_goals
        conceded[:, away] += home_goals
    # One sortable key per club: points, then goal difference, goals scored, random tie-break.
    key = (
        points * 1e7
        + (scored - conceded + 1_000) * 1e3
        + scored
        + rng.random((simulations, count)) * 0.5
    )
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    position[np.arange(simulations)[:, None], order] = np.arange(1, count + 1)
    result = pd.DataFrame(
        {
            "club_id": clubs,
            "current_points": start.points.to_numpy(float),
            "mean_points": points.mean(axis=0),
            "points_low": np.quantile(points, 0.05, axis=0),
            "points_high": np.quantile(points, 0.95, axis=0),
            "expected_position": position.mean(axis=0),
            "title_probability": (position == 1).mean(axis=0),
            "top_probability": (position <= top_places).mean(axis=0),
            "relegation_probability": (position > count - relegation_places).mean(axis=0),
        }
    )
    for place in range(1, count + 1):
        result[f"position_{place}"] = (position == place).mean(axis=0)
    result["simulations"] = simulations
    result["remaining_matches"] = len(matches)
    result["top_places"] = top_places
    result["relegation_places"] = relegation_places
    return result.sort_values(
        ["expected_position", "club_id"], ignore_index=True
    )
