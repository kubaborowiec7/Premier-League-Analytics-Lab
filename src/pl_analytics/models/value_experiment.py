"""Separate selection and final evaluation with a verifiable frozen model bundle."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pl_analytics.data.value_history import read_value_history
from pl_analytics.features.value import FEATURES, build_value_features, chronological_parts
from pl_analytics.models.value import (
    calibrate_interval,
    candidate_models,
    evaluate_model,
    regression_metrics,
)


def write_json(path: Path, payload: dict) -> None:
    """Reject undefined floating-point values in machine-readable reports."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(
    config: dict, raw_dir: Path, output: Path, *, stage: str, download: bool = False
) -> dict:
    """Select without test targets; final evaluation loads only a frozen local bundle.

    The bundle is trusted local output, never an externally supplied pickle. Starting
    another selection requires a new output directory; final outcomes cannot silently
    overwrite the predeclared selection record.
    """
    if stage not in {"select", "final"}:
        raise ValueError("Stage must be select or final")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    selection_path = output / "selection.json"
    bundle_path = output / "models.joblib"
    if stage == "select" and selection_path.exists():
        raise ValueError("Selection is already frozen; use final or a new experiment directory")
    if stage == "final":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["config_sha256"] != digest:
            raise ValueError("Configuration changed since selection")
        if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != selection["bundle_sha256"]:
            raise ValueError("Frozen model bundle changed")
    end = (pd.Timestamp(config["test_start"]) - pd.Timedelta(days=1)).date().isoformat()
    apps, values, profiles, sources = read_value_history(
        config,
        raw_dir,
        end_date=end if stage == "select" else config["test_end"],
        download=download,
    )
    features = build_value_features(apps, values, profiles, config)
    parts = chronological_parts(features, config)
    output.mkdir(parents=True, exist_ok=True)
    floor = config["interval_scale_floor_eur"]
    if stage == "select":
        if any(parts[name].empty for name in ("train", "validation", "calibration")):
            raise ValueError("Each development window needs observations")
        models = candidate_models(parts["train"], seed=config["seed"])
        metrics = {
            model.name: regression_metrics(
                parts["validation"].market_value_eur, model.predict(parts["validation"])
            )
            for model in models
        }
        chosen = min(models, key=lambda model: metrics[model.name][config["primary_metric"]])
        quantile = calibrate_interval(
            chosen,
            parts["calibration"],
            alpha=config["interval_alpha"],
            floor=floor,
        )
        joblib.dump(
            {
                "models": models,
                "chosen": chosen.name,
                "quantile": quantile,
                "training_players": set(parts["train"].player_id),
                "value_cuts": parts["train"].market_value_eur.quantile([1 / 3, 2 / 3]).tolist(),
            },
            bundle_path,
        )
        report = {
            "config_sha256": digest,
            "config": config,
            "sources": sources,
            "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "chosen": chosen.name,
            "target_mode": chosen.target_mode,
            "counts": {key: len(parts[key]) for key in ("train", "validation", "calibration")},
            "validation_metrics": metrics,
            "interval_quantile": quantile,
            "feature_names": FEATURES,
        }
        write_json(selection_path, report)
        return report
    bundle = joblib.load(bundle_path)
    test = parts["test"]
    if test.empty:
        raise ValueError("No final test observations in the declared period")
    models = bundle["models"]
    chosen = next(model for model in models if model.name == bundle["chosen"])
    predictions = evaluate_model(chosen, test, bundle["quantile"], floor)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    test.to_parquet(output / "test_features.parquet", index=False)
    latest = predictions.sort_values("valuation_date").drop_duplicates(
        ["competition_id", "player_id"], keep="last"
    )
    latest.sort_values("uncertainty_scaled_gap", ascending=False).to_parquet(
        output / "value_ranking.parquet", index=False
    )
    groups = {
        "season": test.season,
        "age": pd.cut(
            test.age_years,
            [-np.inf, 23, 30, np.inf],
            labels=["under_23", "23_to_29", "30_plus"],
            right=False,
        )
        .astype("string")
        .fillna("unknown"),
        "value_band": pd.cut(
            test.market_value_eur,
            [-np.inf, *bundle["value_cuts"], np.inf],
            labels=["low", "middle", "high"],
            duplicates="raise",
        ),
        "player_history": test.player_id.isin(bundle["training_players"]).map(
            {True: "seen_in_training", False: "new_since_training"}
        ),
    }
    segments = {
        name: {
            str(label): regression_metrics(
                predictions.loc[labels.eq(label), "market_value_eur"],
                predictions.loc[labels.eq(label), "predicted_value_eur"],
            )
            for label in labels.dropna().unique()
        }
        for name, labels in groups.items()
    }
    explanations = {}
    # Explain the selected model and the best validation XGBoost even if a baseline wins.
    tree = min(
        (model for model in models if model.name.startswith("xgb")),
        key=lambda model: selection["validation_metrics"][model.name]["mae_eur"],
    )
    sample = test.iloc[:200]
    for model in {chosen.name: chosen, tree.name: tree}.values():
        contributions, base = model.explain(sample)
        detail = sample[["player_id", "competition_id", "season", "valuation_date"]].copy()
        for index, feature in enumerate(FEATURES):
            detail[f"shap_{feature}"] = contributions[:, index]
        detail["base_value"] = base
        detail.to_parquet(output / f"shap_{model.name}.parquet", index=False)
        explanations[model.name] = {
            "units": model.target_mode,
            "n": len(sample),
            "mean_absolute_shap": dict(
                zip(FEATURES, np.abs(contributions).mean(axis=0).tolist(), strict=True)
            ),
        }
    report = {
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "config_sha256": digest,
        "chosen": chosen.name,
        "test_start": config["test_start"],
        "test_end": config["test_end"],
        "metrics": {
            model.name: regression_metrics(test.market_value_eur, model.predict(test))
            for model in models
        },
        "interval_coverage": float(
            (
                (predictions.market_value_eur >= predictions.lower_eur)
                & (predictions.market_value_eur <= predictions.upper_eur)
            ).mean()
        ),
        "mean_interval_width_eur": float((predictions.upper_eur - predictions.lower_eur).mean()),
        "segments": segments,
        "explanations": explanations,
        "ranking_players": len(latest),
        "residual_mean_eur": float(predictions.residual_eur.mean()),
        "residual_quantiles_eur": predictions.residual_eur.quantile([0.05, 0.5, 0.95]).to_dict(),
    }
    write_json(output / "evaluation.json", report)
    return report
