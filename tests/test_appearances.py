"""Source identity/date integrity for the published appearance adapter."""

from datetime import date
from pathlib import Path

import pytest

from pl_analytics.data.appearances import parse_appearances
from pl_analytics.data.contracts import Scope
from pl_analytics.data.snapshots import SnapshotStore


@pytest.mark.parametrize("bad", [None, "club", "date", "profile", "duplicate", "missing_game"])
def test_appearance_source_integrity(tmp_path: Path, bad: str | None) -> None:
    games = (
        "game_id,competition_id,season,date,home_club_id,away_club_id,home_club_goals,away_club_goals\n"
        "1,CUSTOM,2023,2023-08-01,10,11,1,0\n"
    )
    if bad == "missing_game":
        games += "2,CUSTOM,2023,2023-08-02,10,11,1,0\n"
    club = "99" if bad == "club" else "10"
    played = "2023-08-02" if bad == "date" else "2023-08-01"
    appearances = (
        "game_id,player_id,player_club_id,date,competition_id,minutes_played,goals,assists,"
        "yellow_cards,red_cards\n"
        f"1,1,{club},{played},CUSTOM,90,1,0,0,0\n"
    )
    if bad == "duplicate":
        appearances += "1,1,10,2023-08-01,CUSTOM,90,2,0,0,0\n"
    players = "player_id,name,sub_position\n" + (
        "2,Other,Goalkeeper\n" if bad == "profile" else "1,Example,Centre-Forward\n"
    )
    snapshots = []
    for name, content in (("games", games), ("appearances", appearances), ("players", players)):
        path = tmp_path / f"{name}.csv"
        path.write_text(content, encoding="utf-8")
        snapshots.append(
            SnapshotStore(tmp_path / "raw").import_file(
                path,
                source_name=name,
                source_url="https://example.test/fixture",
                dataset_version="synthetic",
                license_note="Generated fixture",
            )
        )
    scope = Scope(
        "OTHER", "Other league", "Country", "CUSTOM", "2023/24", date(2023, 7, 1), date(2024, 6, 30)
    )
    if bad:
        with pytest.raises(ValueError):
            parse_appearances(*snapshots, scope=scope, source_season="2023")
    else:
        result = parse_appearances(*snapshots, scope=scope, source_season="2023")
        assert result.iloc[0].competition_id == "OTHER"
        assert result.iloc[0].player_id == "tm:player:1"
        assert result.iloc[0].club_id == "tm:club:10"
        assert result.iloc[0].position_context == "snapshot_unverified"
        assert result.iloc[0].position_group == "ST"
