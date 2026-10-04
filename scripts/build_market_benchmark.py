"""Compare the frozen match forecasts with bookmaker odds already archived locally.

Reads the rolling-origin predictions written by the M6 and M7 pipelines and the odds
columns of the same Football-Data.co.uk archives. Nothing is fitted or downloaded.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pl_analytics.config import get_settings
from pl_analytics.data.market_odds import read_market_odds
from pl_analytics.statistics.betting import market_benchmark, market_blend

# Prediction files in priority order: a match forecast twice keeps the first occurrence.
PREDICTIONS = (
    ("m6/final/predictions.parquet", "final test"),
    ("m7/statistical_benchmarks/predictions.parquet", "final test"),
    ("m6/select/predictions.parquet", "model selection"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m7_experiment.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("Match competition is not enabled")

    frames, periods = [], {}
    for relative, role in PREDICTIONS:
        path = settings.artifact_dir / relative
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.competition_id.eq(config["competition_id"])]
        frames.append(frame)
        for season in frame.season.unique():
            periods.setdefault(str(season), role)
    if not frames:
        raise FileNotFoundError("No match predictions found; reproduce M6/M7 first")
    predictions = pd.concat(frames, ignore_index=True).drop_duplicates(["match_id", "model"])
    odds = read_market_odds(
        config, settings.data_dir / "raw", end_date=datetime.now(UTC).date().isoformat()
    )
    report = market_benchmark(predictions, odds)
    report.update(
        config={"competition_id": config["competition_id"]},
        periods={season: periods[season] for season in report["seasons"]},
        # The blend weight is fitted on the model-selection seasons and tested on the rest.
        blend=market_blend(
            predictions,
            odds,
            train_seasons=tuple(
                season for season in report["seasons"] if periods[season] == "model selection"
            ),
        ),
        built_at=datetime.now(UTC).isoformat(),
    )
    destination = settings.artifact_dir / "market"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    model, market = report["metrics"][report["value_model"]], report["metrics"]["market_avg_close"]
    logging.info(
        "Market benchmark: %d matches, model log loss %.4f, market %.4f",
        report["matches"],
        model["log_loss"],
        market["log_loss"],
    )


if __name__ == "__main__":
    main()
