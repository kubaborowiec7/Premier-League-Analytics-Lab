"""Prepare display names, player uncertainty and frozen team states; never run at app startup."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from pl_analytics.config import get_settings
from pl_analytics.data.match_history import read_match_history
from pl_analytics.statistics.players import bootstrap_per90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m7_experiment.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("The manifest competition must be enabled in ACTIVE_COMPETITIONS")
    history, _ = read_match_history(
        config, settings.data_dir / "raw", end_date=config["test_end"], download=args.download
    )
    root = settings.artifact_dir / "dashboard"
    root.mkdir(parents=True, exist_ok=True)
    clubs = pd.concat(
        [
            history[["competition_id", f"{side}_club_id", f"{side}_name"]].rename(
                columns={f"{side}_club_id": "club_id", f"{side}_name": "name"}
            )
            for side in ("home", "away")
        ]
    ).drop_duplicates()
    clubs.to_parquet(root / "clubs.parquet", index=False)
    paths = {
        "statistics": settings.artifact_dir
        / "m7/statistical_benchmarks/latest_origin_models.joblib",
        "ml": settings.artifact_dir / "m7/classifiers.joblib",
    }
    statistics = joblib.load(paths["statistics"])
    features = pd.read_parquet(settings.artifact_dir / "m7/test_features.parquet")
    features = features.loc[features.origin.eq(statistics["origin"])]
    elo = statistics["models"][f"elo_k{config['feature_elo_k']}"]
    states = []
    for row in features.itertuples():
        for side, other in (("home", "away"), ("away", "home")):
            club, opponent = getattr(row, f"{side}_club_id"), getattr(row, f"{other}_club_id")
            _, rating, _ = elo.predict(club, opponent, row.season)
            state = {
                "club_id": club,
                "competition_id": row.competition_id,
                "season": row.season,
                "origin": row.origin,
                "elo": rating,
            }
            for suffix in (
                "points_recent",
                "goals_for_recent",
                "goals_against_recent",
                "recent_matches",
                "days_since_result",
            ):
                state[suffix] = getattr(row, f"{side}_{suffix}")
            states.append(state)
    prepared = pd.DataFrame(states).drop_duplicates()
    if prepared.duplicated(["club_id", "competition_id", "season", "origin"]).any():
        raise ValueError("Conflicting prepared club states")
    prepared.to_parquet(root / "team_states.parquet", index=False)
    player_path = settings.data_dir / "processed/m4/player_features.parquet"
    appearances_path = settings.data_dir / "processed/m4/appearances.parquet"
    if player_path.exists() and appearances_path.exists():
        players, appearances = pd.read_parquet(player_path), pd.read_parquet(appearances_path)
        intervals = []
        grouped = appearances.groupby(["player_id", "competition_id", "season"])
        for player in players.itertuples():
            key = (player.player_id, player.competition_id, player.season)
            if key not in grouped.groups:
                continue
            for metric in ("goals", "assists"):
                intervals.append(
                    {
                        "player_id": player.player_id,
                        "competition_id": player.competition_id,
                        "season": player.season,
                        "metric": metric,
                        **bootstrap_per90(
                            grouped.get_group(key), metric, reference_date=player.reference_date
                        ),
                    }
                )
        pd.DataFrame(intervals).to_parquet(root / "player_intervals.parquet", index=False)
    metadata = {
        "competition_id": config["competition_id"],
        "season": prepared.iloc[0].season,
        "origin": statistics["origin"].isoformat(),
        "available_clubs": len(prepared),
        "models": {
            key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
        },
        "source_manifest": args.manifest.as_posix(),
    }
    (root / "catalog.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Prepared dashboard context: %s clubs, origin %s", len(prepared), metadata["origin"]
    )


if __name__ == "__main__":
    main()
