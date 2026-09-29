"""Fantasy planning projections and fixture context."""

import pandas as pd

from pl_analytics.features.fpl import fixture_rows, latest_players, project_players


def test_fpl_projection_preserves_scope_and_fixture_horizons():
    stats = pd.DataFrame(
        [
            {
                "id": player,
                "gw": gw,
                "now_cost": 50 + player,
                "total_points": 10 * player,
                "event_points": player + gw,
                "minutes": 360,
                "starts": 4,
                "points_per_game": 4 + player,
                "expected_goals_per_90": 0.1 * player,
                "expected_assists_per_90": 0.05,
                "defensive_contribution_per_90": 11 if player == 1 else 2,
                "status": "a",
            }
            for player in (1, 2)
            for gw in (3, 4)
        ]
    )
    identities = pd.DataFrame(
        [
            dict(
                player_id=player,
                player_code=100 + player,
                first_name="Test",
                second_name=str(player),
                position="Defender" if player == 1 else "Forward",
                team_code=player,
            )
            for player in (1, 2)
        ]
    )
    teams = pd.DataFrame(
        [
            dict(code=1, name="Alpha", short_name="ALP", elo=1600),
            dict(code=2, name="Beta", short_name="BET", elo=1400),
        ]
    )
    matches = pd.DataFrame(
        [
            dict(
                gameweek=5,
                kickoff_time="2026-10-01T12:00:00Z",
                home_team=1,
                away_team=2,
                finished=False,
                match_id="m5",
            ),
            dict(
                gameweek=6,
                kickoff_time="2026-10-08T12:00:00Z",
                home_team=2,
                away_team=1,
                finished=False,
                match_id="m6",
            ),
        ]
    )
    players, history = latest_players(
        stats,
        identities,
        teams,
        competition_id="TEST",
        season="2026/27",
        position_map={"Defender": "DEF", "Forward": "FWD"},
    )
    fixtures = fixture_rows(matches, teams)
    result = project_players(players, history, fixtures)
    assert set(result.competition_id) == {"TEST"}
    assert result.player_id.is_unique and fixtures.club_id.nunique() == 2
    assert result.projected_points_3.ge(result.projected_points_1).all()
    assert result.projection_low_1.le(result.projected_points_1).all()
    assert result.projection_high_1.ge(result.projected_points_1).all()
    assert fixtures.loc[fixtures.club_name.eq("Alpha"), "difficulty"].iloc[0] == 1
