"""Monthly rolling-origin forecasts and frozen-policy match-model evaluation."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from pl_analytics.data.match_history import read_match_history
from pl_analytics.models.match_statistical import fit_elo, fit_goals, validate_history
from pl_analytics.statistics.matches import (
    OUTCOMES,
    calibration_bins,
    outcome_probabilities,
    probability_metrics,
    score_matrix,
    tau,
)


def labels(frame: pd.DataFrame) -> np.ndarray:
    """Ordered outcome encoding: home, draw, away."""
    return np.where(
        frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2)
    )


def forecast_origin(
    history: pd.DataFrame, fixtures: pd.DataFrame, origin: pd.Timestamp, config: dict
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Fit once before origin; fixture outcomes are neither required nor accessed."""
    train = validate_history(history, origin)
    if fixtures.empty or not fixtures.competition_id.eq(train.iloc[0].competition_id).all():
        raise ValueError("Fixtures must match the training competition")
    if (fixtures.match_day < origin).any():
        raise ValueError("Fixtures precede the forecast origin")
    counts = np.bincount(labels(train), minlength=3) + 1
    base = counts / counts.sum()
    elos = {
        f"elo_k{k}": fit_elo(
            train,
            origin,
            k=k,
            home_advantage=config["elo_home_advantage"],
            season_retention=config["elo_season_retention"],
        )
        for k in config["elo_k"]
    }
    goals = {}
    for dc in (False, True):
        for half in config["half_lives"]:
            name = ("dixon_coles" if dc else "poisson") + (f"_half{half}" if half else "_flat")
            goals[name] = fit_goals(
                train,
                origin,
                dixon_coles=dc,
                half_life_days=half,
                penalty=config["goal_penalty"],
                max_rate=config["max_rate"],
                rho_bounds=tuple(config["rho_bounds"]),
            )
    rows, matrices = [], []
    known = set(train.home_club_id) | set(train.away_club_id)
    for fixture in fixtures.itertuples():
        metadata = {
            key: getattr(fixture, key)
            for key in (
                "match_id",
                "competition_id",
                "season",
                "match_day",
                "home_club_id",
                "away_club_id",
            )
        }
        metadata.update(
            origin=origin,
            training_end_day=train.match_day.max(),
            training_matches=len(train),
            unseen_team=fixture.home_club_id not in known or fixture.away_club_id not in known,
        )

        def append(
            name: str, probabilities: np.ndarray, *, metadata: dict = metadata, **extra: float
        ) -> None:
            rows.append(
                {
                    **metadata,
                    "model": name,
                    **dict(zip(("p_home", "p_draw", "p_away"), probabilities, strict=True)),
                    **extra,
                }
            )

        append("base_rate", base)
        for name, elo in elos.items():
            p, rh, ra = elo.predict(fixture.home_club_id, fixture.away_club_id, fixture.season)
            append(name, p, home_elo=rh, away_elo=ra)
        for name, model in goals.items():
            lam, mu = model.rates(fixture.home_club_id, fixture.away_club_id)
            matrix, tail = score_matrix(lam, mu, model.rho, max_goals=config["max_goals"])
            p = outcome_probabilities(matrix)
            best = np.unravel_index(matrix.argmax(), matrix.shape)
            append(
                name,
                p,
                expected_home_goals=lam,
                expected_away_goals=mu,
                rho=model.rho,
                omitted_tail=tail,
                most_likely_home_goals=int(best[0]),
                most_likely_away_goals=int(best[1]),
            )
            matrices.append(
                {
                    "match_id": fixture.match_id,
                    "competition_id": fixture.competition_id,
                    "season": fixture.season,
                    "origin": origin.isoformat(),
                    "model": name,
                    "max_goals": config["max_goals"],
                    "omitted_tail": tail,
                    "probabilities": matrix.ravel().tolist(),
                }
            )
    return pd.DataFrame(rows), matrices, {"base_rate": base, **elos, **goals}


def rolling_backtest(
    frame: pd.DataFrame, config: dict, *, start: str, end: str, output: Path
) -> pd.DataFrame:
    """Every origin uses a fixed trailing history; results join only after prediction."""
    batches, fits, matrices = [], [], []
    fixture_columns = [
        "match_id",
        "competition_id",
        "season",
        "match_day",
        "home_club_id",
        "away_club_id",
    ]
    end_time = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    origins = pd.date_range(start, end, freq="MS", tz="UTC")
    output.mkdir(parents=True, exist_ok=True)
    for origin in origins:
        finish = min(origin + pd.offsets.MonthBegin(), end_time)
        fixtures = frame.loc[
            (frame.match_day >= origin) & (frame.match_day < finish), fixture_columns
        ]
        if fixtures.empty:
            continue
        history = frame.loc[
            (frame.match_day < origin)
            & (frame.match_day >= origin - pd.Timedelta(days=config["lookback_days"]))
        ]
        if len(history) < config["min_train_matches"]:
            raise ValueError("Insufficient pre-origin history")
        predicted, scores, models = forecast_origin(history, fixtures, origin, config)
        batches.append(predicted)
        matrices.extend(scores)
        fits.extend(
            {"origin": origin.isoformat(), "model": name, **model.diagnostics}
            for name, model in models.items()
            if hasattr(model, "diagnostics")
        )
        logging.info(
            "Match origin %s: %d prior matches, %d fixtures",
            origin.date(),
            len(history),
            len(fixtures),
        )
        joblib.dump(
            {"origin": origin, "competition_id": config["competition_id"], "models": models},
            output / "latest_origin_models.joblib",
        )
    if not batches:
        raise ValueError("No forecast fixtures in the requested window")
    predictions = pd.concat(batches, ignore_index=True).merge(
        frame[["match_id", "home_goals", "away_goals"]], on="match_id", validate="many_to_one"
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    probabilities = np.asarray([row.pop("probabilities") for row in matrices]).reshape(
        -1, config["max_goals"] + 1, config["max_goals"] + 1
    )
    np.savez_compressed(output / "score_matrices.npz", probabilities=probabilities)
    pd.DataFrame(matrices).to_parquet(output / "score_matrix_index.parquet", index=False)
    (output / "fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False), encoding="utf-8")
    return predictions


def match_report(predictions: pd.DataFrame) -> dict:
    """Outcome/score metrics, seasonal diagnostics and fixed-bin calibration."""
    metrics, seasonal, calibration = {}, {}, {}
    for name, group in predictions.groupby("model", sort=True):
        p = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        actual = labels(group)
        metrics[name] = probability_metrics(actual, p)
        if group.expected_home_goals.notna().all():
            lam, mu = group.expected_home_goals.to_numpy(), group.expected_away_goals.to_numpy()
            x, y = group.home_goals.to_numpy(), group.away_goals.to_numpy()
            correction = tau(x, y, lam, mu, group.rho.to_numpy())
            metrics[name].update(
                score_nll=float(
                    -(poisson.logpmf(x, lam) + poisson.logpmf(y, mu) + np.log(correction)).mean()
                ),
                goal_mae=float((np.abs(x - lam) + np.abs(y - mu)).mean() / 2),
                max_omitted_tail=float(group.omitted_tail.max()),
            )
        calibration[name] = calibration_bins(actual, p)
        seasonal[name] = {
            str(season): probability_metrics(
                labels(part), part[["p_home", "p_draw", "p_away"]].to_numpy()
            )
            for season, part in group.groupby("season")
        }
    return {"metrics": metrics, "seasonal": seasonal, "calibration": calibration}


def _policy_hash() -> str:
    root = Path(__file__).parents[1]
    files = [
        "data/football_data.py",
        "data/match_history.py",
        "statistics/matches.py",
        "models/match_statistical.py",
        "models/match_experiment.py",
    ]
    return hashlib.sha256(b"".join((root / file).read_bytes() for file in files)).hexdigest()


def run_match_experiment(
    config: dict, raw_dir: Path, output: Path, *, stage: str, download: bool = False
) -> dict:
    """Freeze the rolling forecasting policy, not a model fitted on future outcomes."""
    if stage not in {"select", "final"}:
        raise ValueError("Stage must be select or final")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    decision = output / "selection.json"
    code_hash = _policy_hash()
    if stage == "select" and decision.exists():
        raise ValueError("Selection already frozen; use a fresh output directory for reproduction")
    if stage == "final":
        selected = json.loads(decision.read_text(encoding="utf-8"))
        if selected["config_sha256"] != digest or selected["policy_sha256"] != code_hash:
            raise ValueError("Forecast policy changed after selection")
    development_end = (pd.Timestamp(config["test_start"]) - pd.Timedelta(days=1)).date().isoformat()
    finish = development_end if stage == "select" else config["test_end"]
    frame, sources = read_match_history(config, raw_dir, end_date=finish, download=download)
    predictions = rolling_backtest(
        frame,
        config,
        start=config["validation_start"] if stage == "select" else config["test_start"],
        end=finish,
        output=output / stage,
    )
    report = match_report(predictions)
    report.update(
        config_sha256=digest,
        policy_sha256=code_hash,
        sources=sources,
        outcome_order=OUTCOMES,
        config=config,
    )
    report["chosen"] = (
        min(report["metrics"], key=lambda name: report["metrics"][name][config["primary_metric"]])
        if stage == "select"
        else selected["chosen"]
    )
    if stage == "final":
        # Paired log-loss differences resample whole forecast months, not individual matches.
        chosen = predictions.loc[predictions.model.eq(report["chosen"])].set_index("match_id")
        baseline = (
            predictions.loc[predictions.model.eq("base_rate")]
            .set_index("match_id")
            .loc[chosen.index]
        )
        actual = labels(chosen)
        cols = ["p_home", "p_draw", "p_away"]
        delta = -np.log(chosen[cols].to_numpy()[np.arange(len(chosen)), actual]) + np.log(
            baseline[cols].to_numpy()[np.arange(len(baseline)), actual]
        )
        clusters = (
            pd.DataFrame({"origin": chosen.origin.to_numpy(), "delta": delta})
            .groupby("origin")
            .delta.agg(["sum", "count"])
        )
        draws = np.random.default_rng(config["seed"]).integers(
            0, len(clusters), (2000, len(clusters))
        )
        differences = clusters["sum"].to_numpy()[draws].sum(axis=1) / clusters["count"].to_numpy()[
            draws
        ].sum(axis=1)
        report["paired_log_loss_vs_base"] = {
            "difference": float(delta.mean()),
            "lower_95": float(np.quantile(differences, 0.025)),
            "upper_95": float(np.quantile(differences, 0.975)),
            "origin_clusters": len(clusters),
            "resamples": 2000,
        }
    path = decision if stage == "select" else output / "evaluation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
