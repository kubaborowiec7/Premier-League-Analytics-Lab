"""Published appearance snapshots joined by source IDs, without inferred club aliases."""

from datetime import date

import pandas as pd

from pl_analytics.data.contracts import Scope, add_unique, csv_rows, integer, source_id
from pl_analytics.data.snapshots import Snapshot

POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB/WB",
    "Right-Back": "FB/WB",
    "Defensive Midfield": "DM/CM",
    "Central Midfield": "DM/CM",
    "Attacking Midfield": "AM/W",
    "Left Midfield": "AM/W",
    "Right Midfield": "AM/W",
    "Left Winger": "AM/W",
    "Right Winger": "AM/W",
    "Second Striker": "ST",
    "Centre-Forward": "ST",
}
COUNTS = ("goals", "assists", "yellow_cards", "red_cards")


def parse_appearances(
    games: Snapshot,
    appearances: Snapshot,
    players: Snapshot,
    *,
    scope: Scope,
    source_season: str,
) -> pd.DataFrame:
    """Return one row per selected match/player, preserving unknown counts and positions.

    Match dates are date-only UTC markers, not known kickoffs. Position comes from
    the supplied profile snapshot and is explicitly unsuitable as a historical fact.
    Source IDs never join Football-Data IDs, and canonical database tables are untouched.
    """
    for snapshot in (games, appearances, players):
        snapshot.verify()
    selected_games: dict[str, dict] = {}
    for _, row in csv_rows(
        games.path,
        {
            "game_id",
            "competition_id",
            "season",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        },
    ):
        if row["competition_id"] != scope.source_code or row["season"] != source_season:
            continue
        played = date.fromisoformat(row["date"])
        if not scope.start_date <= played <= scope.end_date:
            continue
        if not row["home_club_goals"] or not row["away_club_goals"]:
            continue
        integer(row["home_club_goals"], "home goals")
        integer(row["away_club_goals"], "away goals")
        if row["home_club_id"] == row["away_club_id"]:
            raise ValueError("A match cannot have identical home and away clubs")
        add_unique(selected_games, row["game_id"], row)
    profiles: dict[str, dict] = {}
    for _, row in csv_rows(players.path, {"player_id", "name", "sub_position"}):
        add_unique(
            profiles,
            row["player_id"],
            {
                "player_name": row["name"],
                "position_group": POSITION_GROUPS.get(row["sub_position"], "UNKNOWN"),
            },
        )
    records: dict[tuple[str, str], dict] = {}
    for _, row in csv_rows(
        appearances.path,
        {
            "game_id",
            "player_id",
            "player_club_id",
            "date",
            "competition_id",
            "minutes_played",
            *COUNTS,
        },
    ):
        game = selected_games.get(row["game_id"])
        if game is None:
            continue
        if row["date"] != game["date"] or row["competition_id"] != scope.source_code:
            raise ValueError("Appearance date/competition conflicts with its match")
        if row["player_club_id"] not in {game["home_club_id"], game["away_club_id"]}:
            raise ValueError("Appearance club is not a participant in its match")
        if row["player_id"] not in profiles:
            raise ValueError("Appearance player has no profile in the pinned snapshot")
        record = {
            "match_id": source_id(row["game_id"], "match"),
            "player_id": source_id(row["player_id"], "player"),
            "club_id": source_id(row["player_club_id"], "club"),
            "competition_id": scope.competition_id,
            "season": scope.season,
            "match_date": game["date"],
            "minutes": integer(row["minutes_played"], "minutes", optional=True),
            **{metric: integer(row[metric], metric, optional=True) for metric in COUNTS},
            **profiles[row["player_id"]],
            "position_context": "snapshot_unverified",
            "source": "transfermarkt-datasets",
        }
        add_unique(records, (record["match_id"], record["player_id"]), record)
    if not records:
        raise ValueError("No appearances match the requested competition/season")
    covered = {record["match_id"] for record in records.values()}
    expected = {source_id(game, "match") for game in selected_games}
    if covered != expected:
        raise ValueError("Selected games have missing appearance coverage")
    return pd.DataFrame(records.values()).sort_values(["match_date", "match_id", "player_id"])
