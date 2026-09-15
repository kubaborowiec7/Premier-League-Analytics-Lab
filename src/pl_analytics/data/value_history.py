"""Load only dated valuation/appearance facts from pinned published archives."""

import re
from pathlib import Path

import pandas as pd

from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.data.transfermarkt import PUBLISHED_BASE_URL, TERMS


def read_value_history(
    config: dict,
    raw_dir: Path,
    *,
    end_date: str,
    download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Filter outcomes by stage before returning them; never use current club/position.

    Competition scope comes from dated game participation. Valuation source league
    fields are deliberately not read. Recent participation does not prove membership
    at valuation time, so the cohort is labelled as recent participants.
    """
    sources = {}
    store = SnapshotStore(raw_dir)
    for table in ("games", "appearances", "players", "player_valuations"):
        digest = config["sha256"][table]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid SHA-256 source pin")
        source = f"transfermarkt-datasets.{table}"
        sidecar = raw_dir / source / digest / "metadata.json"
        if sidecar.exists():
            snapshot = Snapshot.read(sidecar)
        elif download:
            snapshot = store.download(
                f"{PUBLISHED_BASE_URL}/{table}.csv.gz",
                source_name=source,
                dataset_version="sha256:" + digest,
                license_note=TERMS,
                expected_sha256=digest,
                max_bytes=256 * 1024 * 1024,
            )
        else:
            raise FileNotFoundError(f"Missing {table}; use --download to acquire pinned input")
        if snapshot.source_name != source or snapshot.sha256 != digest:
            raise ValueError("Snapshot source identity mismatch")
        sources[table] = snapshot
    games = pd.read_csv(
        sources["games"].path,
        dtype=str,
        usecols=[
            "game_id",
            "competition_id",
            "date",
            "home_club_id",
            "away_club_id",
        ],
    )
    games = games.loc[
        games.competition_id.eq(config["source_competition_code"])
        & games.date.ge(config["history_start"])
        & games.date.le(end_date)
    ]
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate game identities")
    chunks = []
    columns = [
        "game_id",
        "player_id",
        "player_club_id",
        "date",
        "minutes_played",
        "goals",
        "assists",
    ]
    for chunk in pd.read_csv(
        sources["appearances"].path, dtype=str, usecols=columns, chunksize=100000
    ):
        selected = chunk.loc[chunk.game_id.isin(games.game_id)]
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise ValueError("No appearance history for requested scope")
    appearances = pd.concat(chunks, ignore_index=True).merge(
        games.rename(columns={"date": "game_date"}),
        on="game_id",
        validate="many_to_one",
    )
    if (
        not appearances.date.eq(appearances.game_date).all()
        or not (
            appearances.player_club_id.eq(appearances.home_club_id)
            | appearances.player_club_id.eq(appearances.away_club_id)
        ).all()
    ):
        raise ValueError("Appearance date/club conflicts with dated game")
    appearances = appearances.rename(columns={"minutes_played": "minutes"})
    appearances["competition_id"] = config["competition_id"]
    if appearances.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate appearance observations")
    values = []
    for chunk in pd.read_csv(
        sources["player_valuations"].path,
        dtype=str,
        usecols=["player_id", "date", "market_value_in_eur"],
        chunksize=100000,
    ):
        # Returning stage-limited targets prevents validation code from seeing test outcomes.
        selected = chunk.loc[chunk.date.le(end_date) & chunk.player_id.isin(appearances.player_id)]
        if len(selected):
            values.append(selected)
    if not values:
        raise ValueError("No valuation history for requested scope")
    valuations = pd.concat(values, ignore_index=True).rename(
        columns={"market_value_in_eur": "market_value_eur"}
    )
    valuations = valuations.drop_duplicates()
    if valuations.duplicated(["player_id", "date"]).any():
        raise ValueError("Conflicting same-day valuations")
    profiles = pd.read_csv(
        sources["players"].path, dtype=str, usecols=["player_id", "name", "date_of_birth"]
    )
    if profiles.player_id.duplicated().any():
        raise ValueError("Duplicate profile identities")
    for frame in (appearances, valuations, profiles):
        frame["player_id"] = "tm:player:" + frame.player_id
    provenance = [
        {
            "source": s.source_name,
            "sha256": s.sha256,
            "url": s.source_url,
            "retrieved_at": s.retrieved_at,
        }
        for s in sources.values()
    ]
    return appearances, valuations, profiles, provenance
