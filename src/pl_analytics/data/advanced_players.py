"""Pinned published FPL Core CSV adapter; never contacts an underlying sports website."""

from pathlib import Path

import numpy as np
import pandas as pd

from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.features.advanced_players import METRICS


def load_season(
    config: dict, season: dict, raw_dir: Path, *, download: bool = False
) -> pd.DataFrame:
    """Validate source identity, season-local IDs and dated match joins before aggregation."""
    tables: dict[str, list[pd.DataFrame]] = {
        kind: [] for kind in ("players", "teams", "matches", "appearances", "playerstats")
    }
    store = SnapshotStore(raw_dir)
    for entry in season["files"]:
        source = "fpl-core." + entry["kind"]
        sidecar = raw_dir / source / entry["sha256"] / "metadata.json"
        if sidecar.exists():
            snapshot = Snapshot.read(sidecar)
        elif download:
            snapshot = store.download(
                entry["url"],
                source_name=source,
                dataset_version=config["commit"],
                license_note=config["license_note"],
                expected_sha256=entry["sha256"],
            )
        else:
            raise FileNotFoundError("Missing advanced player archive; run with --download")
        if snapshot.sha256 != entry["sha256"] or snapshot.source_name != source:
            raise ValueError("Advanced source identity mismatch")
        tables[entry["kind"]].append(pd.read_csv(snapshot.path))
    return validate_season(
        *(pd.concat(tables[k], ignore_index=True) for k in ("players", "matches", "appearances")),
        config=config,
        season=season,
        teams=pd.concat(tables["teams"], ignore_index=True) if tables["teams"] else None,
        playerstats=(
            pd.concat(tables["playerstats"], ignore_index=True)
            if tables["playerstats"]
            else None
        ),
    )


def validate_season(
    players: pd.DataFrame,
    matches: pd.DataFrame,
    appearances: pd.DataFrame,
    *,
    config: dict,
    season: dict,
    teams: pd.DataFrame | None = None,
    playerstats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fail on ambiguous joins and invalid values; retain absent metrics as NaN."""
    players, matches, appearances = players.copy(), matches.copy(), appearances.copy()
    for frame, key in (
        (players, ["player_id"]),
        (matches, ["match_id"]),
        (appearances, ["player_id", "match_id"]),
    ):
        if frame.duplicated(key).any() or frame[key].isna().any().any():
            raise ValueError("Duplicate or missing source identities")
    if not set(appearances.player_id) <= set(players.player_id):
        raise ValueError("Appearance has no player profile")
    if not set(appearances.match_id) <= set(matches.match_id):
        raise ValueError("Appearance has no match")
    matches["match_date"] = pd.to_datetime(matches.kickoff_time, utc=True, format="mixed")
    if not matches.match_date.between(
        pd.Timestamp(season["start_date"], tz="UTC"),
        pd.Timestamp(season["end_date"], tz="UTC") + pd.Timedelta(days=1),
        inclusive="left",
    ).all():
        raise ValueError("Match falls outside declared season")
    if (
        "tournament" in matches
        and not matches.tournament.dropna().isin(config["source_tournament_codes"]).all()
    ):
        raise ValueError("Unexpected source competition")
    finished = matches.finished.astype(str).str.lower().isin(["true", "1"])
    if season.get("completed_matches_only"):
        matches = matches.loc[finished].copy()
        appearances = appearances.loc[appearances.match_id.isin(matches.match_id)].copy()
        if matches.empty or appearances.empty:
            raise ValueError("Current player snapshot has no completed appearances")
    elif not finished.all():
        raise ValueError("Advanced appearances must reference completed matches")
    appearances["minutes"] = pd.to_numeric(appearances.minutes_played, errors="raise")
    if not appearances.minutes.between(0, 130).all():
        raise ValueError("Invalid appearance minutes")
    if playerstats is not None:
        totals = playerstats.copy()
        required = {"id", "yellow_cards", "red_cards"}
        if not required <= set(totals) or totals.id.isna().any():
            raise ValueError("Invalid player season totals")
        if totals.id.duplicated().any():
            if "gw" not in totals or pd.to_numeric(totals.gw, errors="coerce").isna().any():
                raise ValueError("Repeated player season totals have no valid gameweek")
            totals["gw"] = pd.to_numeric(totals.gw, errors="raise")
            totals = totals.sort_values("gw").drop_duplicates("id", keep="last")
        season_totals = [
            name
            for name, metric in METRICS.items()
            if metric.aggregation == "season_total" and name in totals
        ]
        totals = totals[["id", *season_totals]].rename(columns={"id": "player_id"})
        for name in season_totals:
            totals[name] = pd.to_numeric(totals[name], errors="raise")
            if totals[name].dropna().lt(0).any():
                raise ValueError(f"Invalid metric: {name}")
        appearances = appearances.drop(
            columns=season_totals, errors="ignore"
        ).merge(totals, on="player_id", how="left", validate="many_to_one")
    for name, metric in METRICS.items():
        if metric.kind == "ratio":
            continue
        values = pd.to_numeric(
            appearances.get(name, pd.Series(np.nan, index=appearances.index)), errors="raise"
        )
        if np.isinf(values).any() or (name != "goals_prevented" and values.dropna().lt(0).any()):
            raise ValueError(f"Invalid metric: {name}")
        if metric.kind == "match_pct" and values.dropna().gt(100).any():
            raise ValueError(f"Percentage outside 0–100: {name}")
        appearances[name] = values
    appearances["duels_lost"] = pd.to_numeric(appearances.duels_lost, errors="raise")
    if appearances.duels_lost.dropna().lt(0).any():
        raise ValueError("Negative lost duels")
    profile_columns = ["player_id", "first_name", "second_name", "position", "player_code"]
    if teams is not None:
        required_team_columns = {"code", "name", "short_name"}
        if not required_team_columns <= set(teams) or teams["code"].isna().any():
            raise ValueError("Invalid team snapshot")
        if teams["code"].duplicated().any():
            raise ValueError("Duplicate team codes")
        if "team_code" not in players or players["team_code"].isna().any():
            raise ValueError("Player team identity is missing")
        team_lookup = teams[["code", "name", "short_name"]].rename(
            columns={"code": "team_code", "name": "club_name", "short_name": "club_short_name"}
        )
        profiles = players[[*profile_columns, "team_code"]].merge(
            team_lookup, on="team_code", how="left", validate="many_to_one"
        )
        if profiles.club_name.isna().any():
            raise ValueError("Player references an unknown team")
        profiles["club_id"] = "fpl:club:" + profiles.team_code.astype(int).astype(str)
    else:
        profiles = players[profile_columns].copy()
        profiles["club_id"] = pd.NA
        profiles["club_name"] = pd.NA
        profiles["club_short_name"] = pd.NA
    result = appearances.merge(
        matches[["match_id", "match_date"]], on="match_id", validate="many_to_one"
    )
    result = result.merge(profiles, on="player_id", validate="many_to_one")
    result["player_name"] = result.first_name.fillna("") + " " + result.second_name.fillna("")
    result["position_group"] = result.position.map(config["position_map"]).fillna("UNKNOWN")
    # player_code is stable across FPL seasons; player_id alone is not.
    if result.player_code.isna().any():
        raise ValueError("Stable player code missing")
    result["player_id"] = "fpl:" + result.player_code.astype(int).astype(str)
    result["competition_id"] = config["competition_id"]
    result["season"] = season["season"]
    return result
