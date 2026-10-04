"""Season simulation: conservation, ordering, determinism and tie-breaking."""

import pandas as pd
import pytest

from pl_analytics.statistics.season import simulate_season


def _league() -> tuple[pd.DataFrame, pd.DataFrame]:
    clubs = ["strong", "middle", "weak", "bottom"]
    strength = {"strong": 2.4, "middle": 1.4, "weak": 1.0, "bottom": 0.6}
    standings = pd.DataFrame(
        {"club_id": clubs, "points": [6, 4, 3, 1], "goals_for": [5, 3, 2, 1], "goals_against": 2}
    )
    matches = pd.DataFrame(
        [
            {
                "home_club_id": home,
                "away_club_id": away,
                "home_rate": strength[home] * 1.1,
                "away_rate": strength[away] * 0.9,
            }
            for home in clubs
            for away in clubs
            if home != away
        ]
    )
    return standings, matches


def test_simulation_conserves_probability_and_ranks_by_strength():
    standings, matches = _league()
    result = simulate_season(
        standings, matches, simulations=2000, seed=1, top_places=2, relegation_places=1
    ).set_index("club_id")
    assert result.title_probability.sum() == pytest.approx(1)
    assert result.top_probability.sum() == pytest.approx(2)
    assert result.relegation_probability.sum() == pytest.approx(1)
    places = [column for column in result if column.startswith("position_")]
    assert len(places) == 4
    assert result[places].sum(axis=1).tolist() == pytest.approx([1, 1, 1, 1])
    assert result[places].sum(axis=0).tolist() == pytest.approx([1, 1, 1, 1])
    assert list(result.index) == ["strong", "middle", "weak", "bottom"]
    assert result.title_probability.is_monotonic_decreasing
    assert result.relegation_probability.is_monotonic_increasing
    assert (result.points_low <= result.mean_points).all()
    assert (result.mean_points <= result.points_high).all()
    assert (result.mean_points > result.current_points).all()


def test_simulation_is_deterministic_and_respects_a_finished_season():
    standings, matches = _league()
    first = simulate_season(standings, matches, simulations=500, seed=9)
    second = simulate_season(standings, matches, simulations=500, seed=9)
    pd.testing.assert_frame_equal(first, second)
    finished = simulate_season(
        standings, matches.iloc[0:0], simulations=200, top_places=1, relegation_places=1
    ).set_index("club_id")
    # With nothing left to play, the current table is final.
    assert finished.title_probability.to_dict() == {
        "strong": 1.0,
        "middle": 0.0,
        "weak": 0.0,
        "bottom": 0.0,
    }
    assert finished.loc["bottom", "relegation_probability"] == 1.0
    assert (finished.mean_points == finished.current_points).all()


def test_level_clubs_are_split_by_goal_difference_then_at_random():
    level = pd.DataFrame(
        {
            "club_id": ["a", "b", "c"],
            "points": [10, 10, 10],
            "goals_for": [9, 5, 5],
            "goals_against": [3, 5, 5],
        }
    )
    result = simulate_season(
        level, pd.DataFrame(columns=["home_club_id", "away_club_id", "home_rate", "away_rate"]),
        simulations=4000, seed=3, top_places=1, relegation_places=1,
    ).set_index("club_id")
    assert result.loc["a", "title_probability"] == 1.0
    # b and c are identical, so the last place is a coin flip between them.
    assert result.loc["b", "relegation_probability"] == pytest.approx(0.5, abs=0.04)
    with pytest.raises(ValueError):
        simulate_season(level.drop(columns="points"), _league()[1])
