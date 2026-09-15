"""Plot the aligned M7 test comparison and calibrated model reliability."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pl_analytics.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or get_settings().artifact_dir / "m7"
    report = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    chosen = report["chosen"]
    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    names = sorted(report["metrics"], key=lambda name: report["metrics"][name]["log_loss"])
    axes[0].barh(
        names,
        [report["metrics"][name]["log_loss"] for name in names],
        color=["#b05c24" if name == chosen else "#236b8e" for name in names],
    )
    axes[0].set(xlabel="Log loss · lower is better", title="Identical final fixtures")
    for label, color in zip(("H", "D", "A"), ("#236b8e", "#b05c24", "#377e4b"), strict=True):
        rows = [
            row for row in report["calibration"][chosen] if row["outcome"] == label and row["n"]
        ]
        axes[1].scatter(
            [row["predicted"] for row in rows],
            [row["observed"] for row in rows],
            s=[15 + row["n"] * 3 for row in rows],
            label=label,
            color=color,
            alpha=0.8,
        )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability · marker size reflects bin count",
    )
    axes[1].legend()
    fig.suptitle(f"M7 · {report['test_matches']} held-out matches · small-sample pilot")
    fig.savefig(root / "ml_diagnostics.png", dpi=150)


if __name__ == "__main__":
    main()
