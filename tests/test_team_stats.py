import pandas as pd
import pytest

from pl_analytics.features.team_stats import STATS, summarize_teams, team_match_rows


def _match(match_id, date, home, away, home_stats, away_stats, finished=True):
    row = dict(
        match_id=match_id, kickoff_time=date, home_team=home, away_team=away, finished=finished
    )
    row |= {f"home_{name}": value for name, value in home_stats.items()}
    row |= {f"away_{name}": value for name, value in away_stats.items()}
    return row


@pytest.fixture
def rows():
    teams = pd.DataFrame(
        [
            dict(code=1, name="Alpha", short_name="ALP"),
            dict(code=2, name="Beta", short_name="BET"),
            dict(code=3, name="Gamma", short_name="GAM"),
        ]
    )
    strong = dict(score=3, expected_goals_xg=2.0, total_shots=20, shots_on_target=10,
                  passes=500, accurate_passes=450, possession=60, fouls_committed=8)  # fmt: skip
    weak = dict(score=0, expected_goals_xg=0.5, total_shots=5, shots_on_target=1,
                passes=300, accurate_passes=210, possession=40, fouls_committed=14)  # fmt: skip
    matches = pd.DataFrame(
        [
            _match("m1", "2026-08-15", 1, 2, strong, weak),
            _match("m2", "2026-08-22", 2, 3, strong, weak),
            _match("m3", "2026-08-29", 3, 1, weak, strong),
            _match("m4", "2026-09-05", 1, 3, {}, {}, finished=False),
        ]
    )
    return team_match_rows(matches, teams, competition_id="TEST", season="2026/27")


def test_rows_hold_both_sides_and_skip_unfinished_matches(rows):
    assert len(rows) == 6 and set(rows.match_id) == {"m1", "m2", "m3"}
    assert rows.competition_id.eq("TEST").all() and rows.season.eq("2026/27").all()
    beta_away = rows.loc[rows.club_id.eq("fpl:club:2") & rows.venue.eq("Away")].iloc[0]
    assert beta_away.score == 0 and beta_away.score_against == 3
    assert beta_away.opponent_short_name == "ALP"
    # A statistic the source does not record stays missing rather than becoming zero.
    assert rows.corners.isna().all()


def test_summary_averages_ranks_and_flips_lower_is_better(rows):
    summary = summarize_teams(rows).set_index("club_short_name")
    assert summary.loc["ALP", "matches"] == 2
    assert summary.loc["ALP", "goals"] == 3 and summary.loc["ALP", "goals_against"] == 0
    assert summary.loc["ALP", "pass_accuracy"] == pytest.approx(90.0)
    assert summary.loc["BET", "shot_accuracy"] == pytest.approx(100 * 11 / 25)
    assert summary.loc["ALP", "goals_rank"] == 1 and summary.loc["ALP", "goals_percentile"] == 100
    # Conceding least and fouling least rank first.
    assert summary.loc["ALP", "goals_against_rank"] == 1
    assert summary.loc["GAM", "goals_against_percentile"] == 0
    assert summary.loc["ALP", "fouls_rank"] == 1
    assert summary.loc["ALP", "fouls_won"] == 14
    assert summary.corners.isna().all() and summary.corners_percentile.isna().all()
    assert {f"{name}_percentile" for name in STATS} <= set(summary.columns)


def test_periods_filter_matches(rows):
    home = summarize_teams(rows, "Home").set_index("club_short_name")
    assert home.loc["ALP", "matches"] == 1 and home.loc["GAM", "goals"] == 0
    away = summarize_teams(rows, "Away").set_index("club_short_name")
    assert away.loc["ALP", "goals"] == 3 and away.loc["BET", "goals"] == 0
    longer = pd.concat(
        [rows.assign(match_id=rows.match_id + str(n), match_date=rows.match_date
                     + pd.Timedelta(days=30 * n)) for n in range(4)]
    )  # fmt: skip
    assert summarize_teams(longer, "Last 5 matches").matches.eq(5).all()
    with pytest.raises(ValueError, match="Unknown period"):
        summarize_teams(rows, "Decade")


def test_unknown_team_is_rejected():
    teams = pd.DataFrame([dict(code=1, name="Alpha", short_name="ALP")])
    matches = pd.DataFrame([_match("m1", "2026-08-15", 1, 9, {}, {})])
    with pytest.raises(ValueError, match="unknown team"):
        team_match_rows(matches, teams, competition_id="TEST", season="2026/27")


def test_compare_teams_view_renders(rows, tmp_path):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "rows.parquet"
    rows.to_parquet(path)

    def view(path):
        import pandas as pd

        from pl_analytics.dashboard import theme
        from pl_analytics.dashboard.team_compare import render_team_compare

        theme.apply()
        render_team_compare(pd.read_parquet(path), season="2026/27", club_id="fpl:club:2")

    app = AppTest.from_function(view, args=(str(path),)).run(timeout=30)
    assert not app.error and not app.exception
    picker = next(item for item in app.multiselect if item.label.startswith("Compare teams"))
    assert picker.value[0] == "fpl:club:2" and len(picker.value) == 2
    assert any("Goals conceded" in item.value for item in app.markdown)
    app.radio[0].set_value("Home").run(timeout=30)
    assert not app.error and not app.exception
    picker = next(item for item in app.multiselect if item.label.startswith("Compare teams"))
    picker.set_value([]).run(timeout=30)
    assert any("Select at least one team" in item.value for item in app.info)

    def ranking(path):
        import pandas as pd

        from pl_analytics.dashboard.team_compare import render_team_compare, render_team_table

        frame = pd.read_parquet(path)
        render_team_table(frame, season="2026/27")
        render_team_compare(frame, season="2026/27", default_clubs=("GAM", "ALP"), key="other")

    listed = AppTest.from_function(ranking, args=(str(path),)).run(timeout=30)
    assert not listed.exception
    # Sorted by the first statistic of the group, best club first.
    assert list(listed.dataframe[0].value.Club) == ["Alpha", "Beta", "Gamma"]
    assert listed.dataframe[0].value.Goals.iloc[0] == 3
    assert listed.multiselect[0].value == ["fpl:club:3", "fpl:club:1"]
    next(item for item in listed.radio if item.label == "Show").set_value("League rank").run(
        timeout=30
    )
    assert list(listed.dataframe[0].value.Goals) == [1, 2, 3]
    listed.selectbox[0].select("Defence").run(timeout=30)
    assert not listed.exception and "Goals conceded ↓" in listed.dataframe[0].value

    def missing():
        from pl_analytics.dashboard.team_compare import render_team_compare

        render_team_compare(None, season="2026/27", club_id="x")

    empty = AppTest.from_function(missing).run(timeout=30)
    assert not empty.exception and any("not prepared" in item.value for item in empty.info)
