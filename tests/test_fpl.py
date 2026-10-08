"""Fantasy planning projections and fixture context."""

import httpx
import numpy as np
import pandas as pd
import pytest

from pl_analytics.data.fpl import (
    fetch_league_picks,
    fetch_league_standings,
    fetch_live_players,
    fetch_public_team,
    manager_summary,
    map_public_squad,
    model_goal_rates,
)
from pl_analytics.features.fpl import (
    apply_availability,
    attach_fit_projections,
    expected_minutes_features,
    fixture_rows,
    latest_players,
    league_ownership,
    league_table,
    price_watch,
    price_watch_backtest,
    prior_season_rates,
    project_players,
    scorer_markets,
    season_goal_rates,
)
from pl_analytics.features.fpl_optimizer import (
    backtest_summary,
    best_single_transfers,
    lineup_points,
    optimize_chip_schedule,
    optimize_squad,
    plan_transfers,
    replacement_options,
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
    rates = pd.DataFrame(
        {"match_id": ["m5", "m6"], "home_rate": [2.4, 0.7], "away_rate": [0.6, 2.2]}
    )
    fixtures = fixture_rows(matches, teams, rates)
    priors = pd.DataFrame(
        {"player_id": ["fpl:102"], "prior_minutes": [3000.0], "prior_xg_per90": [0.9]}
    )
    result = project_players(players, history, fixtures, priors=priors)
    without_prior = project_players(players, history, fixtures)
    forward = result.position_group.eq("FWD")
    assert (
        result.loc[forward, "projected_xg_per90"].iloc[0]
        > without_prior.loc[forward, "projected_xg_per90"].iloc[0]
    )
    alpha = fixtures.loc[fixtures.club_name.eq("Alpha")].sort_values("gameweek")
    assert alpha.rate_source.eq("goal-model").all()
    assert alpha.expected_goals_for.tolist() == [2.4, 2.2]
    assert alpha.attack_difficulty.tolist() == [1, 1] and alpha.defence_difficulty.iloc[0] == 1
    beta = fixtures.loc[fixtures.club_name.eq("Beta")]
    assert beta.difficulty.eq(5).all()
    assert alpha.clean_sheet_probability.iloc[0] > beta.clean_sheet_probability.max()
    assert set(result.competition_id) == {"TEST"}
    assert result.player_id.is_unique and fixtures.club_id.nunique() == 2
    assert result.projected_points_3.ge(result.projected_points_1).all()
    assert result.projection_low_1.le(result.projected_points_1).all()
    assert result.projection_high_1.ge(result.projected_points_1).all()
    assert result.appearance_probability.ge(result.sixty_probability).all()
    assert {
        "xpts_appearance_1",
        "xpts_attack_1",
        "xpts_clean_sheet_1",
        "xpts_saves_1",
    } <= set(result)
    assert alpha.difficulty.iloc[0] == 1
    markets = scorer_markets(result, fixtures).set_index("position_group")
    assert set(markets.gameweek) == {5}
    assert markets.scorer_probability.between(0, 1).all()
    # The forward has the higher expected goals and the kinder fixture multiplier is his own.
    assert markets.loc["FWD", "scorer_probability"] > markets.loc["DEF", "scorer_probability"]
    # A mixture over minutes can never exceed the chance from playing the average minutes.
    upper = 1 - np.exp(-markets.expected_scorer_events)
    assert (markets.scorer_probability <= upper + 1e-12).all()
    assert (markets.scorer_probability <= markets.appearance_probability).all()


def test_availability_overlay_reprices_only_the_players_that_change():
    stats = pd.DataFrame(
        [
            {
                "id": player,
                "gw": gw,
                "now_cost": 60,
                "total_points": 20,
                "event_points": 5,
                "minutes": 360,
                "starts": 4,
                "expected_goals_per_90": 0.4,
                "expected_assists_per_90": 0.2,
                "selected_by_percent": 10.0,
                "status": "i" if player == 2 else "a",
                "chance_of_playing_next_round": 0 if player == 2 else None,
            }
            for player in (1, 2, 3)
            for gw in (3, 4)
        ]
    )
    identities = pd.DataFrame(
        [
            dict(player_id=p, player_code=100 + p, first_name="T", second_name=str(p),
                 position="Forward", team_code=1)
            for p in (1, 2, 3)
        ]
    )
    teams = pd.DataFrame(
        [dict(code=1, name="Alpha", short_name="ALP", elo=None),
         dict(code=2, name="Beta", short_name="BET", elo=None)]
    )
    matches = pd.DataFrame(
        [
            dict(gameweek=gw, kickoff_time=f"2026-10-{gw:02d}T12:00:00Z", home_team=1,
                 away_team=2, finished=False, match_id=f"m{gw}")
            for gw in (5, 6, 7)
        ]
    )
    players, history = latest_players(
        stats, identities, teams, competition_id="TEST", season="2026/27",
        position_map={"Forward": "FWD"},
    )
    fixtures = fixture_rows(matches, teams)
    built = attach_fit_projections(
        project_players(players, history, fixtures),
        project_players(players.assign(status="a", chance_of_playing=100.0), history, fixtures),
    ).set_index("source_player_id")
    assert built.loc[2, "projected_points_1"] == 0 and built.loc[2, "fit_expected_minutes"] > 0
    assert built.loc[1, "fit_projected_points_gw_5"] == pytest.approx(
        built.loc[1, "projected_points_gw_5"]
    )
    untouched = apply_availability(built.reset_index()).set_index("source_player_id")
    pd.testing.assert_series_equal(untouched.projected_points_3, built.projected_points_3)
    assert untouched.availability_source.eq("build").all()

    live = pd.DataFrame(
        {
            "source_player_id": [1, 2],
            "status": ["d", "a"],
            "news": ["Knock - 50% chance of playing", ""],
            "chance_of_playing": [50.0, None],
            "price": [6.1, 6.0],
        }
    )
    updated = apply_availability(
        built.reset_index(), live, overrides={3: 0.25}
    ).set_index("source_player_id")
    # Player 1 is now a doubt, player 2's flag has cleared, player 3 is set by hand.
    assert updated.availability_probability.to_dict() == {1: 0.5, 2: 1.0, 3: 0.25}
    assert updated.availability_source.to_dict() == {1: "news", 2: "news", 3: "manual"}
    assert updated.loc[1, "price"] == 6.1 and updated.loc[1, "news"].startswith("Knock")
    assert updated.loc[1, "projected_points_gw_5"] == pytest.approx(
        0.5 * built.loc[1, "fit_projected_points_gw_5"]
    )
    # A player who was ruled out is revived from the fully-fit projection, not left at zero.
    assert updated.loc[2, "projected_points_3"] == pytest.approx(
        built.loc[2, [f"fit_projected_points_gw_{gw}" for gw in (5, 6, 7)]].sum()
    )
    assert updated.loc[2, "projected_points_3"] > 0 and updated.loc[2, "expected_minutes"] > 0
    assert updated.loc[3, "projected_points_1"] == pytest.approx(
        0.25 * built.loc[3, "fit_projected_points_gw_5"]
    )
    for horizon in (1, 2, 3):
        points = updated[f"projected_points_{horizon}"]
        assert (updated[f"projection_low_{horizon}"] <= points).all()
        assert (points <= updated[f"projection_high_{horizon}"]).all()
    with pytest.raises(ValueError):
        apply_availability(built.reset_index(), overrides={1: 1.5})


def test_live_player_feed_maps_the_public_fields():
    payload = {
        "elements": [
            dict(id=7, status="d", news="Knock", chance_of_playing_next_round=75, now_cost=61,
                 selected_by_percent="12.5", transfers_in_event=900, transfers_out_event=100),
            dict(id=8, status="a", news="", chance_of_playing_next_round=None, now_cost=45,
                 selected_by_percent="0.4", transfers_in_event=0, transfers_out_event=10),
        ]
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        live = fetch_live_players(client).set_index("source_player_id")
    assert live.loc[7, ["status", "chance_of_playing", "price", "ownership_pct"]].tolist() == [
        "d", 75.0, 6.1, 12.5,
    ]
    assert live.loc[7, "net_transfers_event"] == 800 and pd.isna(live.loc[8, "chance_of_playing"])
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"elements": []}))
    ) as client, pytest.raises(ValueError):
        fetch_live_players(client)


def _season_matches() -> pd.DataFrame:
    rows = []
    for gameweek, (home, away, home_xg, away_xg) in enumerate(
        [(1, 2, 3.0, 0.4), (2, 1, 0.5, 2.6), (1, 2, 2.8, 0.3), (2, 1, 0.6, 2.9)], start=1
    ):
        rows.append(
            dict(
                match_id=f"m{gameweek}",
                gameweek=gameweek,
                kickoff_time=f"2026-08-{gameweek:02d}T12:00:00Z",
                home_team=home,
                away_team=away,
                home_score=round(home_xg),
                away_score=round(away_xg),
                home_expected_goals_xg=home_xg,
                away_expected_goals_xg=away_xg,
                finished=gameweek <= 2,
            )
        )
    return pd.DataFrame(rows)


def test_season_goal_rates_use_only_finished_matches():
    matches = _season_matches()
    rates = season_goal_rates(matches).set_index("match_id")
    # The stronger club (1) is expected to outscore the weaker one home and away.
    assert rates.loc["m3", "home_rate"] > rates.loc["m3", "away_rate"]
    assert rates.loc["m4", "away_rate"] > rates.loc["m4", "home_rate"]
    changed = matches.copy()
    changed.loc[~changed.finished, ["home_expected_goals_xg", "away_expected_goals_xg"]] = 9.0
    pd.testing.assert_frame_equal(season_goal_rates(changed).set_index("match_id"), rates)
    teams = pd.DataFrame(
        [dict(code=1, name="Alpha", short_name="ALP"), dict(code=2, name="Beta", short_name="BET")]
    )
    fixtures = fixture_rows(matches, teams)
    assert fixtures.rate_source.eq("season-xg").all()
    assert set(fixtures.gameweek) == {3, 4}
    alpha = fixtures.loc[fixtures.club_name.eq("Alpha")]
    assert alpha.difficulty.max() < fixtures.loc[fixtures.club_name.eq("Beta")].difficulty.min()


def test_goal_model_rates_match_clubs_by_normalised_name():
    class Model:
        teams = ("man-united", "spurs")

        def rates(self, home: str, away: str) -> tuple[float, float]:
            return (2.0, 1.0) if home == "man-united" else (1.1, 1.9)

    teams = pd.DataFrame(
        [
            dict(code=1, name="Man Utd", short_name="MUN"),
            dict(code=2, name="Spurs", short_name="TOT"),
            dict(code=3, name="Newcomers", short_name="NEW"),
        ]
    )
    clubs = pd.DataFrame(
        {"club_id": ["man-united", "spurs"], "name": ["Man United", "Tottenham"]}
    )
    matches = pd.DataFrame(
        [
            dict(match_id="a", home_team=1, away_team=2),
            dict(match_id="b", home_team=2, away_team=1),
            dict(match_id="c", home_team=3, away_team=1),
        ]
    )
    rates = model_goal_rates(matches, teams, Model(), clubs).set_index("match_id")
    assert rates.loc["a"].tolist() == [2.0, 1.0] and rates.loc["b"].tolist() == [1.1, 1.9]
    assert "c" not in rates.index


def test_prior_season_rates_use_final_gameweek_and_stable_identity():
    stats = pd.DataFrame(
        [
            dict(id=7, gw=1, minutes=90, expected_goals_per_90=0.1, bonus=0),
            dict(id=7, gw=38, minutes=1800, expected_goals_per_90=0.5, bonus=20),
            dict(id=8, gw=38, minutes=0, expected_goals_per_90=0.0, bonus=0),
        ]
    )
    players = pd.DataFrame({"player_id": [7, 8], "player_code": [500, 501]})
    priors = prior_season_rates(stats, players)
    assert priors.player_id.tolist() == ["fpl:500"]
    assert priors.prior_xg_per90.iloc[0] == 0.5 and priors.prior_minutes.iloc[0] == 1800
    assert priors.prior_raw_projected_bonus_per90.iloc[0] == 1.0


def test_expected_minutes_favors_recent_starts_over_old_absences():
    players = pd.DataFrame(
        [
            {
                "source_player_id": 1,
                "position_group": "DEF",
                "minutes": 270,
                "starts": 3,
                "gw": 5,
                "status": "a",
                "chance_of_playing": None,
            },
            {
                "source_player_id": 2,
                "position_group": "DEF",
                "minutes": 180,
                "starts": 2,
                "gw": 5,
                "status": "a",
                "chance_of_playing": None,
            },
        ]
    )
    rows = []
    patterns = {1: [0, 0, 90, 90, 90], 2: [90, 90, 0, 0, 0]}
    for player, minutes in patterns.items():
        for gameweek, played in enumerate(minutes, start=1):
            rows.append(
                {
                    "player_id": player,
                    "match_id": f"{player}:{gameweek}",
                    "gameweek": gameweek,
                    "minutes_played": played,
                    "start_min": 0 if played else None,
                }
            )
    result = expected_minutes_features(players, pd.DataFrame(rows)).set_index(
        "source_player_id"
    )
    assert result.loc[1, "expected_minutes"] > 65
    assert result.loc[1, "expected_minutes"] > result.loc[2, "expected_minutes"]
    assert result.loc[1, "start_probability"] > result.loc[2, "start_probability"]


def test_league_table_adds_xg_expected_points_without_changing_observed_order():
    teams = pd.DataFrame(
        [
            {"code": 1, "name": "Alpha", "short_name": "ALP"},
            {"code": 2, "name": "Beta", "short_name": "BET"},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_team": 1,
                "away_team": 2,
                "home_score": 1,
                "away_score": 0,
                "home_expected_goals_xg": 0.4,
                "away_expected_goals_xg": 1.8,
                "finished": True,
            }
        ]
    )
    result = league_table(matches, teams, competition_id="TEST", season="2026/27")
    alpha, beta = result.set_index("club_name").loc[["Alpha", "Beta"]].itertuples()
    assert alpha.position == 1 and alpha.points == 3 and alpha.goal_difference == 1
    assert alpha.expected_goals == 0.4 and alpha.expected_goals_against == 1.8
    assert alpha.expected_points < beta.expected_points
    assert result.competition_id.eq("TEST").all()


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


def test_transfer_options_value_the_best_xi_and_respect_squad_rules():
    pool = _optimizer_pool().assign(expected_minutes=80.0)
    squad = optimize_squad(pool, budget=100, projection="projected_points_1")
    lineup = select_starting_xi(squad, "projected_points_1")
    starters = lineup.loc[lineup.is_starter]
    captain = starters.loc[starters.is_recommended_captain, "projected_points_1"].iloc[0]
    assert lineup_points(squad.position_group, squad.projected_points_1) == pytest.approx(
        starters.projected_points_1.sum() + captain
    )
    # A cheap squad leaves upgrades available, so the helpers have something to rank.
    weak = pd.concat(
        [
            pool.loc[pool.position_group.eq(position)].nsmallest(count, "projected_points_1")
            for position, count in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items()
        ]
    )
    outgoing = weak.loc[weak.position_group.eq("MID")].nlargest(1, "projected_points_1").iloc[0]
    options = replacement_options(
        pool, outgoing, projection="projected_points_1", bank=5.0, squad=weak
    )
    assert not options.source_player_id.isin(weak.source_player_id).any()
    assert options.position_group.eq("MID").all()
    assert options.price.le(outgoing.price + 5.0).all()
    assert options.lineup_gain.is_monotonic_decreasing and options.lineup_gain.iloc[0] > 0
    clubs_after = weak.loc[weak.source_player_id.ne(outgoing.source_player_id)].club_id
    assert all(clubs_after.eq(club).sum() < 3 for club in options.club_id)
    ranked = best_single_transfers(weak, pool, projection="projected_points_1", bank=5.0)
    assert ranked.lineup_gain.is_monotonic_decreasing and ranked.out_id.is_unique
    free = replacement_options(pool, outgoing, projection="projected_points_1", bank=5.0)
    assert "lineup_gain" not in free and free.projected_gain.is_monotonic_decreasing


def test_transfer_plan_uses_free_transfers_and_never_loses_points():
    pool = _optimizer_pool().assign(expected_minutes=80.0)
    weak = pd.concat(
        [
            pool.loc[pool.position_group.eq(position)].nsmallest(count, "projected_points_1")
            for position, count in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items()
        ]
    )
    weeks = range(6, 11)
    plan = plan_transfers(weak, pool, gameweeks=weeks, bank=5.0, free_transfers=1)
    moves, summary = plan["moves"], plan["weeks"]
    assert summary.gameweek.tolist() == list(weeks) and not moves.empty
    assert not moves.hit.any() and summary.hit_points.eq(0).all()
    # One free transfer a week: never more moves than transfers accrued so far.
    assert (moves.groupby("gameweek").size().reindex(weeks, fill_value=0).cumsum()
            <= pd.Series(range(1, 6), index=weeks)).all()
    assert (summary.plan_points >= summary.hold_points - 1e-9).all() and plan["total_gain"] > 0
    assert moves.in_id.is_unique and not moves.in_id.isin(weak.source_player_id).any()
    final = plan["final_squad"]
    assert len(final) == 15 and final.groupby("club_id").size().max() <= 3
    assert final.groupby("position_group").size().to_dict() == {
        "DEF": 5,
        "FWD": 3,
        "GK": 2,
        "MID": 5,
    }
    assert final.price.sum() <= weak.price.sum() + 5.0 + 1e-9
    with_hits = plan_transfers(
        weak, pool, gameweeks=weeks, bank=5.0, free_transfers=0, allow_hits=True
    )
    assert with_hits["moves"].hit.any()
    held = plan_transfers(weak, pool, gameweeks=weeks, free_transfers=0)
    assert held["moves"].loc[held["moves"].gameweek.eq(6)].empty
    with pytest.raises(ValueError):
        plan_transfers(weak.head(14), pool, gameweeks=weeks)


def test_price_watch_ranks_by_net_transfers_and_scores_itself():
    players = pd.DataFrame(
        {
            "player_name": list("ABCD"),
            "price": [5.0, 6.0, 7.0, 8.0],
            "net_transfers_event": [900.0, -400.0, 50.0, None],
        }
    )
    watch = price_watch(players, top_n=1)
    assert watch.player_name.tolist() == ["A", "B"] and watch.direction.tolist() == ["rise", "fall"]
    stats = pd.DataFrame(
        [
            dict(id=player, gw=gw, now_cost=50 + step * (gw - 1),
                 transfers_in_event=bought, transfers_out_event=sold)
            for player, (bought, sold, step) in enumerate(
                [(900, 100, 1), (100, 900, -1), (300, 300, 0), (310, 300, 0)], start=1
            )
            for gw in (1, 2, 3, 4)
        ]
    )
    report = price_watch_backtest(stats, top_n=1)
    # The most-bought player rises and the most-sold falls in every scored gameweek.
    assert report["gameweeks"] == 2 and report["rise_hit_rate"] == 1.0
    assert report["fall_hit_rate"] == 1.0 and report["rise_base_rate"] == 0.25


def test_league_ownership_separates_my_picks_from_rival_shares():
    picks = pd.DataFrame(
        [
            dict(entry=1, element=10, multiplier=2, is_captain=True),
            dict(entry=1, element=11, multiplier=0, is_captain=False),
            dict(entry=2, element=10, multiplier=1, is_captain=False),
            dict(entry=2, element=12, multiplier=2, is_captain=True),
            dict(entry=3, element=12, multiplier=1, is_captain=False),
            dict(entry=3, element=13, multiplier=0, is_captain=False),
        ]
    )
    result = league_ownership(picks, my_entry=1).set_index("source_player_id")
    assert result.rivals.eq(2).all()
    assert result.loc[10, ["owned_share", "mine", "my_captain"]].tolist() == [0.5, True, True]
    assert result.loc[11, ["owned_share", "mine"]].tolist() == [0.0, True]
    assert result.loc[12, ["owned_share", "captain_share", "mine"]].tolist() == [1.0, 0.5, False]
    assert result.loc[13, "started_share"] == 0.0
    with pytest.raises(ValueError):
        league_ownership(picks.loc[picks.entry.eq(1)], my_entry=1)


def test_league_fetch_drops_manager_names_and_skips_unreadable_squads():
    def handler(request: httpx.Request) -> httpx.Response:
        if "leagues-classic/77/standings" in str(request.url):
            assert request.url.params["page_standings"] == "3"
            return httpx.Response(
                200,
                json={
                    "league": {"name": "Rivals"},
                    "standings": {
                        "results": [
                            dict(entry=5, entry_name="Five", player_name="Real Name",
                                 rank=101, total=300, event_total=40),
                            dict(entry=6, entry_name="Six", player_name="Other Name",
                                 rank=102, total=290, event_total=55),
                        ]
                    },
                },
            )
        if "/entry/6/" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={
                "picks": [
                    dict(element=index, multiplier=1, is_captain=index == 1)
                    for index in range(1, 16)
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        name, standings = fetch_league_standings(league_id=77, page=3, client=client)
        picks = fetch_league_picks(entries=[5, 6], event=4, client=client)
        with pytest.raises(ValueError):
            fetch_league_picks(entries=[6], event=4, client=client)
    assert name == "Rivals" and "player_name" not in standings
    assert standings.entry_name.tolist() == ["Five", "Six"]
    assert set(picks.entry) == {5} and len(picks) == 15
    summary = manager_summary(
        9,
        {
            "name": "Mine",
            "player_first_name": "Private",
            "leagues": {
                "classic": [
                    dict(id=314, name="Overall", league_type="s", entry_rank=1, rank_count=9),
                    dict(id=77, name="Rivals", league_type="x", entry_rank=3, rank_count=20),
                ]
            },
        },
        5,
    )
    assert [league["id"] for league in summary["leagues"]] == [77]
    assert "Private" not in str(summary)


def _backtest_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(
        [
            {
                "id": player,
                "gw": gw,
                "now_cost": 50,
                "total_points": (player + 2) * gw,
                "event_points": player + gw,
                "points_per_game": player + 2,
                "minutes": 90 * gw,
                "starts": gw,
                "expected_goals_per_90": 0.2 * player,
                "expected_assists_per_90": 0.1,
                "ep_next": 3.0,
                "goals_scored": gw if player == 2 else 0,
                "status": "a",
            }
            for gw in (1, 2, 3)
            for player in (1, 2)
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
            dict(code=1, name="Alpha", short_name="ALP", elo=None),
            dict(code=2, name="Beta", short_name="BET", elo=None),
        ]
    )
    matches = _season_matches().loc[lambda frame: frame.gameweek.le(3)].assign(finished=True)
    return stats, identities, teams, matches


def test_rolling_backtest_is_invariant_to_later_gameweek_changes():
    stats, identities, teams, matches = _backtest_inputs()
    kwargs = {
        "position_map": {"Defender": "DEF", "Forward": "FWD"},
        "competition_id": "TEST",
        "season": "2026/27",
    }
    original = rolling_backtest(stats, identities, teams, matches, **kwargs)
    changed = stats.copy()
    changed.loc[changed.gw.ge(2), ["event_points", "total_points", "expected_goals_per_90"]] = 999
    later = matches.copy()
    later.loc[later.gameweek.ge(2), ["home_expected_goals_xg", "away_expected_goals_xg"]] = 9.0
    mutated = rolling_backtest(changed, identities, teams, later, **kwargs)
    columns = ["source_player_id", "predicted_points", "rate_model_points", "provider_points"]
    pd.testing.assert_frame_equal(
        original.loc[original.target_gw.eq(2), columns].reset_index(drop=True),
        mutated.loc[mutated.target_gw.eq(2), columns].reset_index(drop=True),
    )
    summary = backtest_summary(original)
    assert summary["observations"] == 4 and summary["gameweeks"] == [2, 3]
    assert {"rate_model_mae", "provider_mae"} <= set(summary)
    # The forward scores every gameweek and the defender never does.
    scorer = summary["scorer"]
    assert scorer["observations"] == 4 and scorer["scoring_rate"] == 0.5
    assert original.groupby("position_group").scored.mean().to_dict() == {"DEF": 0.0, "FWD": 1.0}
    assert original.scorer_probability.between(0, 1).all()
    assert sum(row["n"] for row in scorer["calibration"]) == 4
