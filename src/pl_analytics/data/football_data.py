"""Football-Data.co.uk results adapter; bookmaker columns are deliberately excluded."""

import hashlib
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from pl_analytics.data.contracts import (
    Batch,
    DataValidationError,
    Scope,
    add_unique,
    csv_rows,
    integer,
)
from pl_analytics.data.snapshots import Snapshot, SnapshotStore

SOURCE = "football-data.co.uk"
TERMS = (
    "Public downloadable results CSV; source attribution: Football-Data.co.uk. "
    "No blanket redistribution licence assumed; raw files remain local. "
    "See https://www.football-data.co.uk/ and https://www.football-data.co.uk/notes.txt."
)


def download_season(
    store: SnapshotStore,
    *,
    source_code: str,
    source_season: str,
    expected_sha256: str | None = None,
) -> Snapshot:
    """Download one source division/season; canonical competition IDs are mapped separately."""
    if not re.fullmatch(r"[A-Za-z0-9]+", source_code) or not re.fullmatch(r"\d{4}", source_season):
        raise DataValidationError("Expected an alphanumeric division and four-digit source season")
    url = f"https://www.football-data.co.uk/mmz4281/{source_season}/{source_code}.csv"
    return store.download(
        url,
        source_name=SOURCE,
        dataset_version=f"{source_code}-{source_season}",
        license_note=TERMS,
        expected_sha256=expected_sha256,
    )


def _club_id(name: str, country: str) -> str:
    digest = hashlib.sha256(f"{country}\0{name}".encode()).hexdigest()[:24]
    return f"fd:club:{digest}"


def parse_results(snapshot: Snapshot, scope: Scope, *, timezone: str) -> Batch:
    """Validate a league season, retain UTC timestamps and flag unknown kickoff times."""
    snapshot.verify()
    zone = ZoneInfo(timezone)
    required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    clubs: dict[str, dict] = {}
    matches: dict[str, dict] = {}
    for line, row in csv_rows(snapshot.path, required):
        try:
            if row["Div"] != scope.source_code:
                raise DataValidationError("Division does not match the requested source mapping")
            parsed_date = None
            for pattern in ("%d/%m/%Y", "%d/%m/%y"):
                try:
                    parsed_date = datetime.strptime(row["Date"], pattern).date()
                    break
                except ValueError:
                    continue
            if parsed_date is None or not scope.start_date <= parsed_date <= scope.end_date:
                raise DataValidationError("Invalid match date or date outside the requested season")
            known_time = bool(row.get("Time"))
            kickoff = datetime.strptime(row.get("Time") or "00:00", "%H:%M").time()
            match_date = datetime.combine(parsed_date, kickoff, tzinfo=zone).astimezone(UTC)
            home, away = row["HomeTeam"], row["AwayTeam"]
            if not home or not away or home == away:
                raise DataValidationError("Home and away clubs must be distinct and non-empty")
            home_id, away_id = _club_id(home, scope.country), _club_id(away, scope.country)
            for club_id, name in ((home_id, home), (away_id, away)):
                clubs[club_id] = {"club_id": club_id, "name": name, "country": scope.country}
            hg, ag = integer(row["FTHG"], "FTHG"), integer(row["FTAG"], "FTAG")
            result = "H" if hg > ag else "A" if hg < ag else "D"
            if row["FTR"] != result:
                raise DataValidationError("Full-time result contradicts the goal counts")
            identity = f"{scope.competition_id}\0{scope.season}\0{home_id}\0{away_id}"
            match_id = "fd:match:" + hashlib.sha256(identity.encode()).hexdigest()[:32]
            record = {
                "match_id": match_id,
                "competition_id": scope.competition_id,
                "season": scope.season,
                "match_date": match_date,
                "match_time_known": known_time,
                "home_club_id": home_id,
                "away_club_id": away_id,
                "home_goals": hg,
                "away_goals": ag,
                "status": "finished",
                "source": SOURCE,
            }
            for source_column, target in (
                ("HS", "home_shots"),
                ("AS", "away_shots"),
                ("HST", "home_shots_on_target"),
                ("AST", "away_shots_on_target"),
            ):
                record[target] = integer(row.get(source_column, ""), source_column, optional=True)
            for side in ("home", "away"):
                shots, on_target = record[f"{side}_shots"], record[f"{side}_shots_on_target"]
                if shots is not None and on_target is not None and on_target > shots:
                    raise DataValidationError("Shots on target exceed total shots")
            add_unique(matches, match_id, record)
        except (ValueError, TypeError) as error:
            raise DataValidationError(f"Results line {line}: {error}") from error
    if not matches:
        raise DataValidationError("Results file contains no completed matches")
    return Batch(
        SOURCE,
        scope,
        {
            "clubs": list(clubs.values()),
            "matches": sorted(
                matches.values(), key=lambda row: (row["match_date"], row["match_id"])
            ),
        },
    )
