"""Current-source discovery, season totals and leakage-safe live artifact tests."""

import json

import httpx
import joblib
import pandas as pd

from pl_analytics.data.advanced_players import validate_season
from pl_analytics.data.current_sources import discover_current_player_config
from pl_analytics.features.advanced_players import summarize_players
from pl_analytics.models.live_match import build_live_match_artifacts


def test_current_player_discovery_resolves_one_commit_and_archives_files(tmp_path):
    commit = "a" * 40
    paths = [
        "data/2026-2027/players.csv",
        "data/2026-2027/playerstats.csv",
        "data/2026-2027/By Tournament/Premier League/GW1/matches.csv",
        "data/2026-2027/By Tournament/Premier League/GW1/playermatchstats.csv",
    ]
    payloads = {
        paths[0]: b"player_id,player_code\n1,10\n",
        paths[1]: b"id,yellow_cards,red_cards\n1,2,0\n",
        paths[2]: b"match_id,kickoff_time,finished,tournament\nm,2026-08-01,true,prem\n",
        paths[3]: b"player_id,match_id,minutes_played,duels_lost\n1,m,90,0\n",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/commits/main"):
            return httpx.Response(200, json={"sha": commit})
        if "/git/trees/" in request.url.path:
            return httpx.Response(
                200, json={"truncated": False, "tree": [{"path": p, "type": "blob"} for p in paths]}
            )
        marker = f"/{commit}/"
        path = httpx.URL(str(request.url)).path.split(marker, 1)[1]
        return httpx.Response(200, content=payloads[path])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    base = {
        "repository": "owner/repository",
        "license_note": "test terms",
        "source_competition": "Premier League",
        "seasons": [
            {
                "season": "2026/27",
                "files": [{"path": "data/2026-2027/players.csv"}],
                "expected_matches": 1,
                "expected_players": 1,
            }
        ],
    }
    result = discover_current_player_config(
        base, tmp_path / "raw", reference_date="2026-09-29", client=client
    )
    assert result["commit"] == commit and result["reference_date"] == "2026-09-29"
    assert result["seasons"][-1]["completed_matches_only"] is True
    assert {item["kind"] for item in result["seasons"][-1]["files"]} == {
        "players",
        "playerstats",
        "matches",
        "appearances",
    }
    assert "expected_matches" not in result["seasons"][-1]


def test_player_season_totals_fill_cards_without_repeating_per_match():
    players = pd.DataFrame(
        [dict(player_id=1, player_code=42, first_name="A", second_name="B", position="Defender")]
    )
    matches = pd.DataFrame(
        [
            dict(match_id="m1", kickoff_time="2026-08-01", finished=True, tournament="prem"),
            dict(match_id="m2", kickoff_time="2026-08-08", finished=True, tournament="prem"),
            dict(match_id="m3", kickoff_time="2026-08-15", finished=False, tournament="prem"),
        ]
    )
    appearances = pd.DataFrame(
        [
            dict(player_id=1, match_id="m1", minutes_played=90, duels_lost=1),
            dict(player_id=1, match_id="m2", minutes_played=90, duels_lost=1),
            dict(player_id=1, match_id="m3", minutes_played=0, duels_lost=0),
        ]
    )
    totals = pd.DataFrame(
        [
            dict(id=1, gw=1, yellow_cards=1, red_cards=0),
            dict(id=1, gw=2, yellow_cards=3, red_cards=1),
        ]
    )
    config = {
        "competition_id": "TEST",
        "source_tournament_codes": ["prem"],
        "position_map": {"Defender": "DEF"},
    }
    season = {
        "season": "2026/27",
        "start_date": "2026-07-01",
        "end_date": "2027-06-30",
        "completed_matches_only": True,
    }
    observations = validate_season(
        players,
        matches,
        appearances,
        config=config,
        season=season,
        playerstats=totals,
    )
    profile = summarize_players(observations, "2026-09-29").iloc[0]
    assert profile.yellow_cards == 3 and profile.red_cards == 1
    assert profile.minutes == 180 and observations.match_id.nunique() == 2
    assert profile.yellow_cards_coverage == 1 and profile.yellow_cards_per90 == 1.5


def test_live_bundle_uses_only_pre_origin_history_and_versioned_paths(tmp_path):
    clubs = ["a", "b", "c", "d"]
    rows = []
    for index in range(24):
        home, away = clubs[index % 4], clubs[(index + 1) % 4]
        rows.append(
            {
                "match_id": f"m{index}",
                "competition_id": "TEST",
                "season": "2026/27",
                "match_day": pd.Timestamp("2026-07-01", tz="UTC") + pd.Timedelta(days=index),
                "home_club_id": home,
                "away_club_id": away,
                "home_name": home.upper(),
                "away_name": away.upper(),
                "home_goals": index % 3,
                "away_goals": (index + 1) % 2,
            }
        )
    history = pd.DataFrame(rows)
    config = {
        "competition_id": "TEST",
        "lookback_days": 1095,
        "min_train_matches": 5,
        "elo_home_advantage": 60,
        "elo_season_retention": 0.75,
        "elo_k": [20, 40],
        "feature_elo_k": 20,
        "half_lives": [None, 365],
        "goal_penalty": 0.01,
        "max_rate": 8,
        "rho_bounds": [-0.12, 0.015],
        "max_goals": 30,
        "rolling_matches": 5,
    }
    ml = tmp_path / "m7/classifiers.joblib"
    ml.parent.mkdir()
    joblib.dump({"chosen": "none", "models": []}, ml)
    artifacts = tmp_path / "artifacts"
    result = build_live_match_artifacts(
        history,
        config,
        ml,
        artifacts,
        origin=pd.Timestamp("2026-08-01", tz="UTC"),
        source_sha256="b" * 64,
        refreshed_at="2026-08-01T06:00:00+00:00",
    )
    assert result["mode"] == "operational" and result["training_matches"] == 24
    states = pd.read_parquet(artifacts / result["paths"]["states"])
    assert set(states.club_id) == set(clubs)
    catalog = json.loads((artifacts / "dashboard/live_catalog.json").read_text())
    assert catalog["last_result_date"] == "2026-07-24"
