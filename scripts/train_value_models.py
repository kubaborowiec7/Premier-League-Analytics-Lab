"""Run the predeclared M5 selection, then a separate frozen final evaluation."""

import argparse
import json
import logging
from pathlib import Path

from pl_analytics.config import get_settings
from pl_analytics.models.value_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "final"), required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m5_experiment.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config["competition_id"] not in settings.active_competitions:
        raise ValueError("Competition is not enabled")
    report = run_experiment(
        config,
        settings.data_dir / "raw",
        args.output or settings.artifact_dir / "m5",
        stage=args.stage,
        download=args.download,
    )
    logging.basicConfig(level=logging.INFO)
    logging.info("M5 %s: chosen=%s", args.stage, report["chosen"])


if __name__ == "__main__":
    main()
