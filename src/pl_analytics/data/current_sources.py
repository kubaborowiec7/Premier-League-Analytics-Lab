"""Acquire immutable snapshots from mutable current-season publication endpoints."""

import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd

from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import download_season, parse_results
from pl_analytics.data.snapshots import Snapshot, SnapshotStore


def refresh_match_config(config: dict, raw_dir: Path) -> tuple[dict, Snapshot]:
    """Archive the current results CSV and return a new in-memory pinned configuration."""
    current = deepcopy(config)
    season = current["seasons"][-1]
    snapshot = download_season(
        SnapshotStore(raw_dir),
        source_code=current["source_code"],
        source_season=season["source_season"],
    )
    scope = Scope(
        current["competition_id"],
        current["competition_name"],
        current["country"],
        current["source_code"],
        season["season"],
        date.fromisoformat(season["start_date"]),
        date.fromisoformat(season["end_date"]),
    )
    parsed = parse_results(snapshot, scope, timezone=current["timezone"], include_shots=False)
    season["sha256"] = snapshot.sha256
    season["expected_matches"] = len(parsed.tables["matches"])
    season.pop("mutable_current", None)
    season.pop("expected_canonical_sha256", None)
    return current, snapshot


def _github_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("GitHub metadata response is not an object")
    return value


def discover_current_player_config(
    base: dict,
    raw_dir: Path,
    *,
    reference_date: str,
    client: httpx.Client | None = None,
) -> dict:
    """Pin the latest published current-season FPL files to one resolved Git commit."""
    if client is None:
        with httpx.Client(timeout=60, follow_redirects=True) as owned:
            return discover_current_player_config(
                base, raw_dir, reference_date=reference_date, client=owned
            )
    repository = base["repository"]
    commit = _github_json(client, f"https://api.github.com/repos/{repository}/commits/main")[
        "sha"
    ]
    tree = _github_json(
        client, f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"
    )
    if tree.get("truncated"):
        raise ValueError("GitHub tree is truncated; refusing an incomplete player snapshot")
    season = deepcopy(base["seasons"][-1])
    match = re.search(r"data/(\d{4}-\d{4})", season["files"][0]["path"])
    if match is None:
        raise ValueError("Cannot determine the current player source season folder")
    folder = match.group(1)
    tournament = re.escape(base["source_competition"])
    per_gameweek = re.compile(
        rf"^data/{folder}/By Tournament/{tournament}/GW\d+/(matches|playermatchstats)\.csv$"
    )
    root_files = {
        f"data/{folder}/players.csv": "players",
        f"data/{folder}/playerstats.csv": "playerstats",
    }
    candidates: dict[str, str] = {}
    for item in tree.get("tree", []):
        path = item.get("path", "")
        if item.get("type") != "blob":
            continue
        if path in root_files:
            candidates[path] = root_files[path]
        elif found := per_gameweek.match(path):
            candidates[path] = "matches" if found.group(1) == "matches" else "appearances"
    if not root_files.keys() <= candidates.keys():
        raise ValueError("Current player identity or season-total file is missing")
    if "matches" not in candidates.values() or "appearances" not in candidates.values():
        raise ValueError("Current player match files are missing")

    store = SnapshotStore(raw_dir)
    files: list[dict[str, str]] = []

    def archive(path: str, kind: str) -> Snapshot:
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{quote(path, safe='/')}"
        snapshot = store.download(
            url,
            source_name=f"fpl-core.{kind}",
            dataset_version=commit,
            license_note=base["license_note"],
            client=client,
        )
        files.append({"kind": kind, "path": path, "url": url, "sha256": snapshot.sha256})
        return snapshot

    for path in sorted(root_files):
        archive(path, candidates[path])
    for path in sorted(path for path, kind in candidates.items() if kind == "matches"):
        snapshot = archive(path, "matches")
        match_rows = pd.read_csv(snapshot.path)
        if "finished" not in match_rows:
            raise ValueError(f"Current match file has no finished flag: {path}")
        finished = match_rows.finished.astype(str).str.lower().isin(["true", "1"])
        if not finished.any():
            files.pop()
            continue
        appearance_path = path.replace("/matches.csv", "/playermatchstats.csv")
        if candidates.get(appearance_path) != "appearances":
            raise ValueError(f"Completed gameweek has no player appearances: {path}")
        archive(appearance_path, "appearances")
    season["files"] = files
    # The publisher creates all gameweek files before their fixtures are played.
    # The loader may ignore those scheduled matches while historical snapshots
    # continue to require every referenced match to be complete.
    season["completed_matches_only"] = True
    season.pop("expected_matches", None)
    season.pop("expected_players", None)
    current = deepcopy(base)
    current["commit"] = commit
    current["reference_date"] = reference_date
    current["seasons"][-1] = season
    return current


def latest_completed_day(frame: pd.DataFrame) -> str:
    """Return the newest completed source day in ISO format."""
    if frame.empty:
        raise ValueError("Cannot describe an empty current-season snapshot")
    return pd.to_datetime(frame.match_day, utc=True).max().date().isoformat()
