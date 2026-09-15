"""Chronological classifier selection, disjoint calibration and new-season comparison."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from pl_analytics.data.match_history import read_match_history
from pl_analytics.features.matches import MATCH_FEATURES, build_match_features
from pl_analytics.models.match_experiment import labels, match_report, rolling_backtest
from pl_analytics.models.match_ml import fit_classifiers, fit_temperature
from pl_analytics.statistics.matches import calibration_bins, probability_metrics


def _code_hash() -> str:
    root = Path(__file__).parents[1]
    names = [
        "features/matches.py",
        "models/match_ml.py",
        "models/match_ml_experiment.py",
        "models/match_experiment.py",
        "models/match_statistical.py",
        "statistics/matches.py",
        "data/match_history.py",
        "data/football_data.py",
    ]
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


def run_match_ml(
    config: dict, raw_dir: Path, output: Path, *, stage: str, download: bool = False
) -> dict:
    """Freeze classifiers and calibration before opening the new final-season snapshot."""
    if stage not in {"select", "final"}:
        raise ValueError("Stage must be select or final")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    decision, model_path = output / "selection.json", output / "classifiers.joblib"
    if stage == "select" and decision.exists():
        raise ValueError("Selection already frozen; use a new directory for reproduction")
    if stage == "final":
        selected = json.loads(decision.read_text(encoding="utf-8"))
        if selected["config_sha256"] != digest or selected["code_sha256"] != _code_hash():
            raise ValueError("Classifier experiment changed after selection")
        if selected["bundle_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("Fitted classifier bundle changed after selection")
    development_end = (pd.Timestamp(config["test_start"]) - pd.Timedelta(days=1)).date().isoformat()
    history, sources = read_match_history(
        config,
        raw_dir,
        end_date=development_end if stage == "select" else config["test_end"],
        download=download,
    )
    features = build_match_features(history, config)
    val_start, cal_start, test_start = [
        pd.Timestamp(config[key], tz="UTC")
        for key in ("validation_start", "calibration_start", "test_start")
    ]
    if not pd.Timestamp(config["train_start"], tz="UTC") < val_start < cal_start < test_start:
        raise ValueError("Training, validation, calibration and test windows must be ordered")
    output.mkdir(parents=True, exist_ok=True)
    if stage == "select":
        train = features.loc[features.match_day < val_start]
        validation = features.loc[
            (features.match_day >= val_start) & (features.match_day < cal_start)
        ]
        calibration = features.loc[
            (features.match_day >= cal_start) & (features.match_day < test_start)
        ]
        models = fit_classifiers(train, config)
        metrics = {
            model.name: probability_metrics(labels(validation), model.predict(validation))
            for model in models
        }
        chosen = min(models, key=lambda model: metrics[model.name][config["primary_metric"]])
        raw = chosen.predict(calibration, calibrated=False)
        chosen.temperature = fit_temperature(
            raw, labels(calibration), tuple(config["temperature_bounds"])
        )
        joblib.dump({"models": models, "chosen": chosen.name}, model_path)
        report = {
            "chosen": chosen.name,
            "temperature": chosen.temperature,
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "calibration": len(calibration),
            },
            "validation_metrics": metrics,
            "calibration_raw": probability_metrics(labels(calibration), raw),
            "calibration_scaled": probability_metrics(
                labels(calibration), chosen.predict(calibration)
            ),
            "bundle_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "feature_names": MATCH_FEATURES,
        }
        path = decision
    else:
        bundle = joblib.load(model_path)
        models = bundle["models"]
        test = features.loc[features.match_day >= test_start].copy()
        if test.empty:
            raise ValueError("No final-test fixtures")
        test.to_parquet(output / "test_features.parquet", index=False)
        metrics, calibration_report, rows = {}, {}, []
        for model in models:
            variants = [False, True] if model.name == bundle["chosen"] else [False]
            for calibrated in variants:
                name = model.name + ("_calibrated" if calibrated else "")
                p = model.predict(test, calibrated=calibrated)
                metrics[name] = probability_metrics(labels(test), p)
                calibration_report[name] = calibration_bins(labels(test), p)
                predicted = test[
                    [
                        "match_id",
                        "competition_id",
                        "season",
                        "match_day",
                        "origin",
                        "feature_history_end",
                        "home_club_id",
                        "away_club_id",
                        "home_goals",
                        "away_goals",
                    ]
                ].copy()
                predicted["model"] = name
                predicted[["p_home", "p_draw", "p_away"]] = p
                rows.append(predicted)
        statistical = rolling_backtest(
            history,
            config,
            start=config["test_start"],
            end=config["test_end"],
            output=output / "statistical_benchmarks",
        )
        if set(statistical.match_id) != set(test.match_id):
            raise ValueError("Statistical and ML benchmarks must cover identical fixtures")
        stats_report = match_report(statistical)
        metrics.update(stats_report["metrics"])
        calibration_report.update(stats_report["calibration"])
        pd.concat([*rows, statistical], ignore_index=True).to_parquet(
            output / "predictions.parquet", index=False
        )
        report = {
            "chosen": bundle["chosen"] + "_calibrated",
            "metrics": metrics,
            "calibration": calibration_report,
            "test_matches": len(test),
            "temperature": selected["temperature"],
            "selection_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
            "first_match_day": test.match_day.min().isoformat(),
            "last_match_day": test.match_day.max().isoformat(),
            "forecast_months": int(test.origin.nunique()),
        }
        path = output / "evaluation.json"
    report.update(config_sha256=digest, code_sha256=_code_hash(), config=config, sources=sources)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
