"""Import published player/value snapshots; this module never scrapes Transfermarkt."""

from datetime import datetime
from decimal import Decimal, InvalidOperation

from pl_analytics.data.contracts import (
    Batch,
    DataValidationError,
    Scope,
    add_unique,
    csv_rows,
    source_id,
)
from pl_analytics.data.snapshots import Snapshot

SOURCE = "transfermarkt-datasets"
TERMS = (
    "Published dcaribou/transfermarkt-datasets snapshot; upstream repository declares CC0-1.0. "
    "Attribution retained to upstream and Transfermarkt; raw files are not redistributed. "
    "https://github.com/dcaribou/transfermarkt-datasets"
)
PUBLISHED_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"


def parse_player_values(
    players: Snapshot,
    valuations: Snapshot,
    clubs: Snapshot,
    scope: Scope,
) -> Batch:
    """Select source-reported scope without treating current clubs as historical."""
    for snapshot in (players, valuations, clubs):
        snapshot.verify()
    values: dict[tuple[str, object], dict] = {}
    skipped = 0
    required = {
        "player_id",
        "date",
        "market_value_in_eur",
        "current_club_id",
        "player_club_domestic_competition_id",
    }
    for line, row in csv_rows(valuations.path, required):
        try:
            if row["player_club_domestic_competition_id"] != scope.source_code:
                skipped += 1
                continue
            valuation_date = datetime.fromisoformat(row["date"]).date()
            if not scope.start_date <= valuation_date <= scope.end_date:
                skipped += 1
                continue
            amount = Decimal(row["market_value_in_eur"])
            if not amount.is_finite() or amount < 0 or amount >= Decimal("100000000000000"):
                raise DataValidationError(
                    "Market value must be finite, non-negative and fit NUMERIC"
                )
            if amount != amount.quantize(Decimal("0.01")):
                raise DataValidationError("Market value has more than two decimal places")
            player_id = source_id(row["player_id"], "player")
            reported_club = source_id(row["current_club_id"], "club")
            record = {
                "player_id": player_id,
                "valuation_date": valuation_date,
                "market_value_eur": amount,
                "club_id": None,
                "source_reported_club_id": reported_club,
                "source": SOURCE,
                "competition_id": scope.competition_id,
                "season": scope.season,
                "competition_context": "source_reported_unverified",
            }
            add_unique(values, (player_id, valuation_date), record)
        except (ValueError, InvalidOperation) as error:
            raise DataValidationError(f"Valuations line {line}: {error}") from error
    if not values:
        raise DataValidationError("No valuations match the requested competition and date window")
    player_ids = {row["player_id"] for row in values.values()}
    club_ids = {row["source_reported_club_id"] for row in values.values()}
    player_records: dict[str, dict] = {}
    for line, row in csv_rows(players.path, {"player_id", "name", "date_of_birth", "position"}):
        player_id = source_id(row["player_id"], "player")
        if player_id not in player_ids:
            continue
        try:
            if not row["name"]:
                raise DataValidationError("Player name is empty")
            birth = (
                datetime.fromisoformat(row["date_of_birth"]).date()
                if row["date_of_birth"]
                else None
            )
            add_unique(
                player_records,
                player_id,
                {
                    "player_id": player_id,
                    "name": row["name"],
                    "date_of_birth": birth,
                    "position": row["position"] or None,
                    "nationality": row.get("country_of_citizenship") or None,
                },
            )
        except ValueError as error:
            raise DataValidationError(f"Players line {line}: {error}") from error
    club_records: dict[str, dict] = {}
    for line, row in csv_rows(clubs.path, {"club_id", "name"}):
        club_id = source_id(row["club_id"], "club")
        if club_id in club_ids:
            if not row["name"]:
                raise DataValidationError(f"Clubs line {line}: club name is empty")
            add_unique(
                club_records,
                club_id,
                {
                    "club_id": club_id,
                    "name": row["name"],
                    "country": None,
                },
            )
    if player_ids != set(player_records) or club_ids != set(club_records):
        raise DataValidationError(
            "Valuations reference players/clubs missing from the provided files"
        )
    for row in values.values():
        birth = player_records[row["player_id"]]["date_of_birth"]
        if birth and row["valuation_date"] < birth:
            raise DataValidationError("A valuation precedes the player's birth date")
    return Batch(
        SOURCE,
        scope,
        {
            "clubs": list(club_records.values()),
            "players": list(player_records.values()),
            "player_market_values": sorted(
                values.values(),
                key=lambda row: (
                    row["valuation_date"],
                    row["player_id"],
                ),
            ),
        },
        skipped_rows=skipped,
        warnings=[
            "Competition and club are source-reported, not verified historical membership. "
            "Canonical club_id is left NULL. Do not use these fields as pre-valuation features.",
            "Player position/nationality are snapshot attributes, not historical observations. "
            "Current/highest values and current clubs are excluded from player records.",
        ],
    )
