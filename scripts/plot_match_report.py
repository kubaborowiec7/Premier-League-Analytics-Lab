"""Render model comparison, reliability and an example score grid from M6 artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pl_analytics.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or get_settings().artifact_dir / "m6"
    report = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    chosen = report["chosen"]
    fig = Figure(figsize=(14, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    names = sorted(report["metrics"], key=lambda name: report["metrics"][name]["log_loss"])
    axes[0].barh(
        names,
        [report["metrics"][name]["log_loss"] for name in names],
        color=["#b05c24" if name == chosen else "#236b8e" for name in names],
    )
    axes[0].set(xlabel="Final log loss (lower is better)", title="Selected policy highlighted")
    for label, color in zip(("H", "D", "A"), ("#236b8e", "#b05c24", "#377e4b"), strict=True):
        rows = [
            row
            for row in report["calibration"][chosen]
            if row["outcome"] == label and row["n"] >= 5
        ]
        axes[1].plot(
            [row["predicted"] for row in rows],
            [row["observed"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability · bins with at least 5 games",
    )
    axes[1].legend()
    index = pd.read_parquet(root / "final/score_matrix_index.parquet")
    eligible = index.loc[index.model.eq(chosen)]
    if eligible.empty:
        axes[2].text(0.5, 0.5, "Selected model has no score distribution", ha="center", wrap=True)
        axes[2].axis("off")
    else:
        i = eligible.index[0]
        with np.load(root / "final/score_matrices.npz") as saved:
            matrix = saved["probabilities"][i, :7, :7]
        plot = axes[2].imshow(matrix, origin="lower", cmap="Blues")
        fig.colorbar(plot, ax=axes[2], label="Score probability")
        axes[2].set(
            xlabel="Away goals", ylabel="Home goals", title="First test fixture · display 0–6 goals"
        )
    fig.suptitle(f"M6 · final chronological evaluation · {chosen}")
    fig.savefig(root / "match_diagnostics.png", dpi=150)


if __name__ == "__main__":
    main()
