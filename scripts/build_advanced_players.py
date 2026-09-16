"""Build multi-season advanced profiles from immutable published CSV archives."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from pl_analytics.config import get_settings
from pl_analytics.data.advanced_players import load_season
from pl_analytics.features.advanced_players import METRICS, summarize_players


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/advanced_players.json")
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    settings = get_settings()
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("Competition must be enabled")
    frames, coverage = [], []
    for season in config["seasons"]:
        matches = load_season(config, season, settings.data_dir / "raw", download=args.download)
        frame = summarize_players(matches, config["reference_date"])
        if (
            matches.match_id.nunique() != season["expected_matches"]
            or len(frame) != season["expected_players"]
        ):
            raise ValueError("Advanced season coverage differs from the pinned manifest")
        frames.append(frame)
        coverage.append(
            {
                "competition_id": config["competition_id"],
                "season": season["season"],
                "players": len(frame),
                "matches": matches.match_id.nunique(),
                "last_match": str(matches.match_date.max()),
                "metrics": {name: int(frame[name].notna().sum()) for name in METRICS},
            }
        )
    destination = settings.data_dir / "processed/advanced"
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        destination / "player_profiles.parquet", index=False
    )
    (destination / "coverage.json").write_text(
        json.dumps(
            {
                "commit": config["commit"],
                "reference_date": config["reference_date"],
                "seasons": coverage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Built advanced profiles: %s", [(r["season"], r["players"], r["matches"]) for r in coverage]
    )


if __name__ == "__main__":
    main()
