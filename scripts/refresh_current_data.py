"""Refresh current match/player snapshots and publish versioned local dashboard artifacts."""

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from pl_analytics.config import get_settings
from pl_analytics.data.advanced_players import load_season
from pl_analytics.data.current_sources import (
    discover_current_player_config,
    refresh_match_config,
)
from pl_analytics.data.match_history import read_match_history
from pl_analytics.features.advanced_players import METRICS, summarize_players
from pl_analytics.models.live_match import build_live_match_artifacts


def _profiles(config: dict, raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames, coverage = [], []
    for season in config["seasons"]:
        observations = load_season(config, season, raw_dir, download=True)
        frame = summarize_players(observations, config["reference_date"])
        expected_matches = season.get("expected_matches")
        expected_players = season.get("expected_players")
        if expected_matches is not None and observations.match_id.nunique() != expected_matches:
            raise ValueError("Historical player match coverage changed")
        if expected_players is not None and len(frame) != expected_players:
            raise ValueError("Historical player coverage changed")
        frames.append(frame)
        coverage.append(
            {
                "competition_id": config["competition_id"],
                "season": season["season"],
                "players": len(frame),
                "matches": observations.match_id.nunique(),
                "last_match": str(observations.match_date.max()),
                "metrics": {name: int(frame[name].notna().sum()) for name in METRICS},
            }
        )
    return pd.concat(frames, ignore_index=True), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="UTC information date (YYYY-MM-DD); defaults to today")
    parser.add_argument(
        "--match-manifest", type=Path, default=Path("data/manifests/m7_experiment.json")
    )
    parser.add_argument(
        "--player-manifest", type=Path, default=Path("data/manifests/advanced_players.json")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    now = datetime.now(UTC)
    origin = pd.Timestamp(args.as_of or now.date().isoformat(), tz="UTC")
    if origin > pd.Timestamp(now.date() + timedelta(days=1), tz="UTC"):
        raise ValueError("Refresh origin cannot be in the future")

    match_base = json.loads(args.match_manifest.read_text(encoding="utf-8"))
    player_base = json.loads(args.player_manifest.read_text(encoding="utf-8"))
    if match_base["competition_id"] not in settings.active_competitions:
        raise ValueError("Match competition is not enabled")
    if player_base["competition_id"] not in settings.active_competitions:
        raise ValueError("Player competition is not enabled")

    match_config, match_snapshot = refresh_match_config(match_base, settings.data_dir / "raw")
    end_date = (origin - pd.Timedelta(days=1)).date().isoformat()
    history, match_sources = read_match_history(
        match_config, settings.data_dir / "raw", end_date=end_date
    )
    ml_bundle = settings.artifact_dir / "m7/classifiers.joblib"
    if not ml_bundle.exists():
        raise FileNotFoundError(
            "Frozen M7 classifiers are missing; reproduce M7 before refreshing live inference"
        )

    player_config = discover_current_player_config(
        player_base,
        settings.data_dir / "raw",
        reference_date=origin.date().isoformat(),
    )
    profiles, coverage = _profiles(player_config, settings.data_dir / "raw")

    refreshed_at = now.isoformat()
    live = build_live_match_artifacts(
        history,
        match_config,
        ml_bundle,
        settings.artifact_dir,
        origin=origin,
        source_sha256=match_snapshot.sha256,
        refreshed_at=refreshed_at,
    )
    destination = settings.data_dir / "processed/advanced"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "player_profiles.parquet.tmp"
    profiles.to_parquet(temporary, index=False)
    temporary.replace(destination / "player_profiles.parquet")
    player_coverage = {
        "commit": player_config["commit"],
        "reference_date": player_config["reference_date"],
        "refreshed_at": refreshed_at,
        "seasons": coverage,
    }
    coverage_temp = destination / "coverage.json.tmp"
    coverage_temp.write_text(json.dumps(player_coverage, indent=2) + "\n", encoding="utf-8")
    coverage_temp.replace(destination / "coverage.json")
    manifest = {
        "refreshed_at": refreshed_at,
        "origin": live["origin"],
        "last_result_date": live["last_result_date"],
        "match_source_sha256": match_snapshot.sha256,
        "match_sources": match_sources,
        "player_source_commit": player_config["commit"],
        "player_reference_date": player_config["reference_date"],
    }
    source_path = settings.artifact_dir / "live/source_manifest.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Current data refreshed: origin=%s, last_result=%s, profiles=%d",
        live["origin"],
        live["last_result_date"],
        len(profiles),
    )


if __name__ == "__main__":
    main()
