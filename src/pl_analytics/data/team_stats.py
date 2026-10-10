"""Club match statistics from the same pinned FPL Core match and team snapshots."""

from pathlib import Path

import pandas as pd

from pl_analytics.data.snapshots import Snapshot
from pl_analytics.features.team_stats import team_match_rows


def load_team_matches(config: dict, season: dict, raw_dir: Path) -> pd.DataFrame:
    """Read one season's archived match and team files into club match rows.

    Nothing is downloaded: the files are the ones the player profiles already archived.
    """
    tables: dict[str, list[pd.DataFrame]] = {"matches": [], "teams": []}
    for entry in season["files"]:
        if entry["kind"] not in tables:
            continue
        source = "fpl-core." + entry["kind"]
        sidecar = raw_dir / source / entry["sha256"] / "metadata.json"
        if not sidecar.exists():
            raise FileNotFoundError("Missing match archive; rebuild the advanced player profiles")
        snapshot = Snapshot.read(sidecar)
        if snapshot.sha256 != entry["sha256"] or snapshot.source_name != source:
            raise ValueError("Advanced source identity mismatch")
        tables[entry["kind"]].append(pd.read_csv(snapshot.path))
    if not tables["matches"] or not tables["teams"]:
        raise ValueError("Season has no match or team files")
    matches = pd.concat(tables["matches"], ignore_index=True)
    teams = pd.concat(tables["teams"], ignore_index=True)
    if matches.match_id.duplicated().any() or teams.code.duplicated().any():
        raise ValueError("Duplicate source identities")
    if (
        "tournament" in matches
        and not matches.tournament.dropna().isin(config["source_tournament_codes"]).all()
    ):
        raise ValueError("Unexpected source competition")
    rows = team_match_rows(
        matches, teams, competition_id=config["competition_id"], season=season["season"]
    )
    inside = rows.match_date.between(
        pd.Timestamp(season["start_date"], tz="UTC"),
        pd.Timestamp(season["end_date"], tz="UTC") + pd.Timedelta(days=1),
        inclusive="left",
    )
    if not inside.all():
        raise ValueError("Match falls outside declared season")
    return rows
