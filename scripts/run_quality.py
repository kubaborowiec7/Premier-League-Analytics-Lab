"""Generate aggregate M3 diagnostics from local pinned M1 snapshots."""

import argparse
import json
import logging
from pathlib import Path

from pl_analytics.config import get_settings
from pl_analytics.data.quality import build_quality_report, load_quality_sample


def main() -> None:
    """Explicit offline report generation; no database connection or downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m1_sources.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    sample = load_quality_sample(args.manifest, settings.data_dir / "raw")
    if sample.scope.competition_id not in settings.active_competitions:
        raise ValueError("Manifest competition is not enabled in ACTIVE_COMPETITIONS")
    report = build_quality_report(sample)
    output = args.output or settings.artifact_dir / "m3-quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    logging.info("M3 data-quality report written to %s", output)


if __name__ == "__main__":
    main()
