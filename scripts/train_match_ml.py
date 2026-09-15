"""Build M7 match classifiers and evaluate the frozen calibrated model."""

import argparse
import json
import logging
from pathlib import Path

from pl_analytics.config import get_settings
from pl_analytics.models.match_ml_experiment import run_match_ml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "final"), required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m7_experiment.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("Competition is not enabled")
    logging.basicConfig(level=logging.INFO)
    report = run_match_ml(
        config,
        settings.data_dir / "raw",
        args.output or settings.artifact_dir / "m7",
        stage=args.stage,
        download=args.download,
    )
    logging.info("M7 %s: chosen=%s", args.stage, report["chosen"])


if __name__ == "__main__":
    main()
