"""Fantasy planning projections and fixture context."""

import httpx
import pandas as pd

from pl_analytics.data.fpl import fetch_public_team, map_public_squad
from pl_analytics.features.fpl import fixture_rows, latest_players, project_players
from pl_analytics.features.fpl_optimizer import (
    backtest_summary,
    optimize_chip_schedule,
    optimize_squad,
    rolling_backtest,
    select_starting_xi,
)


def test_public_team_fetch_requires_complete_unique_squad():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/entry/123/"):
            return httpx.Response(200, json={"name": "Test XI"})
        return httpx.Response(
            200,
            json={
                "picks": [
                    {
                        "element": player,
                        "position": player,
                        "multiplier": 1,
                        "is_captain": player == 1,
                        "is_vice_captain": player == 2,
                    }
                    for player in range(1, 16)
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        entry, picks = fetch_public_team(entry_id=123, event=5, client=client)
    assert entry["name"] == "Test XI"
    assert len(picks) == 15 and picks.element.is_unique
    assert "position" in picks
    players = pd.DataFrame(
        {"source_player_id": range(1, 16), "position": ["Defender"] * 15}
    )
    squad = map_public_squad(picks, players)
    assert len(squad) == 15
    assert {"squad_position", "position"} <= set(squad)


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


def _optimizer_pool() -> pd.DataFrame:
    rows = []
    player_id = 1
    for position, count in {"GK": 4, "DEF": 10, "MID": 10, "FWD": 6}.items():
        for index in range(count):
            rows.append(
                {
                    "source_player_id": player_id,
                    "player_name": f"{position} {index}",
                    "position_group": position,
                    "club_id": f"club-{index % 10}",
                    "club_short_name": f"C{index % 10}",
                    "price": 4 + (index % 4) * 0.5,
                    "projected_points_1": 2 + index / 2,
                    "fixtures_1": "TEST (H)",
                }
            )
            for gameweek in range(6, 11):
                rows[-1][f"projected_points_gw_{gameweek}"] = (
                    rows[-1]["projected_points_1"] * (1 + (gameweek - 6) * 0.03)
                )
            player_id += 1
    return pd.DataFrame(rows)


def test_optimizer_returns_legal_squad_lineup_and_captain():
    pool = _optimizer_pool()
    optimized = optimize_squad(pool, budget=100, projection="projected_points_1")
    assert len(optimized) == 15 and optimized.is_starter.sum() == 11
    assert optimized.is_recommended_captain.sum() == 1
    assert optimized.groupby("position_group").size().to_dict() == {
        "DEF": 5,
        "FWD": 3,
        "GK": 2,
        "MID": 5,
    }
    assert optimized.groupby("club_id").size().max() <= 3
    assert optimized.price.sum() <= 100
    lineup = select_starting_xi(optimized, "projected_points_1")
    starters = lineup.loc[lineup.is_starter].position_group.value_counts()
    assert starters.GK == 1 and 3 <= starters.DEF <= 5
    assert 2 <= starters.MID <= 5 and 1 <= starters.FWD <= 3
    schedule = optimize_chip_schedule(
        optimized,
        pool,
        budget=100,
        gameweeks=range(6, 11),
    )
    assert set(schedule.chip) == {"Triple captain", "Bench boost", "Free hit"}
    assert schedule.recommended_gameweek.is_unique


def test_rolling_backtest_is_invariant_to_later_gameweek_changes():
    identities = pd.DataFrame(
        [
            {"player_id": player, "position": "Defender" if player == 1 else "Forward"}
            for player in (1, 2)
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "id": player,
                "gw": gw,
                "event_points": player + gw,
                "points_per_game": player + gw / 2,
                "minutes": 90 * gw,
                "status": "a",
            }
            for gw in (1, 2, 3)
            for player in (1, 2)
        ]
    )
    kwargs = {
        "position_map": {"Defender": "DEF", "Forward": "FWD"},
        "competition_id": "TEST",
        "season": "2026/27",
    }
    original = rolling_backtest(stats, identities, **kwargs)
    changed = stats.copy()
    changed.loc[changed.gw.eq(3), ["event_points", "points_per_game"]] = 999
    mutated = rolling_backtest(changed, identities, **kwargs)
    columns = ["source_player_id", "predicted_points"]
    pd.testing.assert_frame_equal(
        original.loc[original.target_gw.eq(2), columns].reset_index(drop=True),
        mutated.loc[mutated.target_gw.eq(2), columns].reset_index(drop=True),
    )
    summary = backtest_summary(original)
    assert summary["observations"] == 4 and summary["gameweeks"] == [2, 3]
