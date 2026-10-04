"""Matchday Hub helpers: club-name matching, results, form and probability bars."""

import re

import pandas as pd
import pytest

from pl_analytics.dashboard.hub import _form, normalise_club
from pl_analytics.dashboard.theme import club_chip, form_pills, probability_bar
from pl_analytics.features.fpl import match_results


@pytest.mark.parametrize(
    ("fpl", "football_data"),
    [
        ("Man Utd", "Man United"),
        ("Spurs", "Tottenham"),
        ("Coventry City", "Coventry"),
        ("Hull City", "Hull"),
        ("Ipswich Town", "Ipswich"),
        ("Nott'm Forest", "Nott'm Forest"),
        ("Man City", "Man City"),
    ],
)
def test_fpl_club_names_match_model_names(fpl, football_data):
    assert normalise_club(fpl) == normalise_club(football_data)


def test_man_city_and_man_united_stay_distinct():
    assert normalise_club("Man City") != normalise_club("Man Utd")


def test_probability_bar_labels_sum_to_100_and_escape_nothing_unsafe():
    bar = probability_bar(0.333, 0.333, 0.334)
    values = [int(value) for value in re.findall(r"width:(\d+)%", bar)]
    assert sum(values) == 100
    assert probability_bar(0, 0, 0) == ""
    assert "<script>" not in club_chip("<script>")


def test_match_results_and_form():
    teams = pd.DataFrame(
        {"code": [1, 2], "name": ["Alpha City", "Beta Town"], "short_name": ["ALP", "BET"]}
    )
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "gameweek": [1, 2, 3],
            "kickoff_time": [
                "2026-08-01T14:00:00Z",
                "2026-08-08T14:00:00Z",
                "2026-08-15T14:00:00Z",
            ],
            "home_team": [1, 2, 1],
            "away_team": [2, 1, 2],
            "home_score": [2, 1, None],
            "away_score": [0, 1, None],
            "home_expected_goals_xg": [1.4, 0.9, None],
            "away_expected_goals_xg": [0.3, 1.1, None],
            "finished": [True, True, False],
        }
    )
    results = match_results(matches, teams)
    assert list(results.match_id) == ["m1", "m2"]
    assert results.loc[0, "home_short_name"] == "ALP" and results.loc[0, "home_xg"] == 1.4
    assert _form(results) == {"ALP": "WD", "BET": "LD"}
    assert form_pills("WDLX").count("pl-pill") == 3
