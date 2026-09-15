"""Render static M5 diagnostics from existing artifacts without fitting models."""

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
    root = args.output or get_settings().artifact_dir / "m5"
    report = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(root / "predictions.parquet")
    figure = Figure(figsize=(13, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    names = ["median", "persistence", report["chosen"], "xgb_depth5_log1p"]
    names = list(dict.fromkeys(names))
    axes[0].barh(
        names, [report["metrics"][name]["mae_eur"] / 1e6 for name in names], color="#236b8e"
    )
    axes[0].set(xlabel="Final test MAE (million EUR)", title="Frozen model comparison")
    axes[1].scatter(
        predictions.predicted_value_eur / 1e6,
        predictions.residual_eur / 1e6,
        s=9,
        alpha=0.35,
        color="#236b8e",
    )
    axes[1].axhline(0, color="#bd6036")
    axes[1].set(
        xlabel="Predicted value (million EUR)",
        ylabel="Observed minus predicted (million EUR)",
        title=f"{report['chosen']}: all test residuals",
    )
    explanation = report["explanations"][report["chosen"]]
    values = explanation["mean_absolute_shap"]
    scale = 1e6 if explanation["units"] == "eur" else 1
    units = "million EUR" if explanation["units"] == "eur" else "log1p EUR"
    order = sorted(values, key=values.get)
    axes[2].barh(np.arange(len(order)), [values[key] / scale for key in order], color="#bd6036")
    axes[2].set(
        yticks=np.arange(len(order)),
        yticklabels=order,
        xlabel=f"Mean |SHAP| ({units})",
        title=f"{report['chosen']}: first 200 test rows",
    )
    figure.suptitle(
        f"M5 · {report['test_start']} to {report['test_end']} · historical editorial valuations",
        fontsize=12,
    )
    figure.savefig(root / "model_diagnostics.png", dpi=150)


if __name__ == "__main__":
    main()
