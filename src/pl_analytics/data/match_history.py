"""Pinned multi-season match history through the existing canonical results adapter."""

from datetime import date
from pathlib import Path

import pandas as pd

from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import SOURCE, download_season, parse_results
from pl_analytics.data.snapshots import Snapshot, SnapshotStore


def read_match_history(
    config: dict, raw_dir: Path, *, end_date: str, download: bool = False
) -> tuple[pd.DataFrame, list[dict]]:
    """Skip future season archives entirely; retain source IDs and local calendar days."""
    frames, sources = [], []
    for season in config["seasons"]:
        if season["start_date"] > end_date:
            continue
        digest = season["sha256"]
        sidecar = raw_dir / SOURCE / digest / "metadata.json"
        if sidecar.exists():
            snapshot = Snapshot.read(sidecar)
        elif download:
            snapshot = download_season(
                SnapshotStore(raw_dir),
                source_code=config["source_code"],
                source_season=season["source_season"],
                expected_sha256=digest,
            )
        else:
            raise FileNotFoundError(f"Missing season {season['season']}; acquire with --download")
        if snapshot.source_name != SOURCE or snapshot.sha256 != digest:
            raise ValueError("Match source identity mismatch")
        scope = Scope(
            config["competition_id"],
            config["competition_name"],
            config["country"],
            config["source_code"],
            season["season"],
            date.fromisoformat(season["start_date"]),
            date.fromisoformat(season["end_date"]),
        )
        batch = parse_results(snapshot, scope, timezone=config["timezone"], include_shots=False)
        frame = pd.DataFrame(batch.tables["matches"])
        if len(frame) != season["expected_matches"]:
            raise ValueError(f"Unexpected match coverage in {season['season']}")
        names = {row["club_id"]: row["name"] for row in batch.tables["clubs"]}
        for side in ("home", "away"):
            frame[f"{side}_name"] = frame[f"{side}_club_id"].map(names)
        local_days = (
            pd.to_datetime(frame.match_date, utc=True)
            .dt.tz_convert(config["timezone"])
            .dt.strftime("%Y-%m-%d")
        )
        frame["match_day"] = pd.to_datetime(local_days, utc=True)
        frames.append(frame.loc[frame.match_day <= pd.Timestamp(end_date, tz="UTC")])
        sources.append(
            {
                "season": season["season"],
                "sha256": digest,
                "url": snapshot.source_url,
                "retrieved_at": snapshot.retrieved_at,
            }
        )
    if not frames:
        raise ValueError("No match history in requested period")
    result = pd.concat(frames, ignore_index=True)
    if result.match_id.duplicated().any():
        raise ValueError("Duplicate match identities across seasons")
    return result.sort_values(["match_day", "match_id"]).reset_index(drop=True), sources
