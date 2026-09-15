"""Chronological market-value benchmarks and calibrated residual intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBRegressor

from pl_analytics.features.value import FEATURES


@dataclass
class ValueModel:
    """A fitted model with a fixed feature allowlist and explicit target units."""

    name: str
    target_mode: str
    pipeline: Pipeline | None
    fallback: float
    background: np.ndarray | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Currency predictions, with identical nonnegative clipping for all candidates."""
        if self.name == "median":
            return np.full(len(frame), self.fallback)
        if self.name == "persistence":
            return frame.previous_value_eur.fillna(self.fallback).to_numpy(dtype=float)
        prediction = self.pipeline.predict(frame.loc[:, FEATURES])
        if self.target_mode == "log1p":
            prediction = np.expm1(np.clip(prediction, -50, 30))
        return np.maximum(prediction, 0)

    def explain(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """TreeSHAP for XGBoost, independent-feature linear SHAP otherwise.

        Additivity is in fitted target units before inverse transformation/clipping.
        For median/persistence, the decomposition is constant/one-feature exact.
        """
        if self.pipeline is None:
            contributions = np.zeros((len(frame), len(FEATURES)))
            contributions[:, FEATURES.index("previous_value_eur")] = (
                self.predict(frame) - self.fallback
            )
            return contributions, np.full(len(frame), self.fallback)
        transformed = self.pipeline[:-1].transform(frame.loc[:, FEATURES])
        estimator = self.pipeline[-1]
        if isinstance(estimator, XGBRegressor):
            values = estimator.get_booster().predict(DMatrix(transformed), pred_contribs=True)
            return values[:, :-1], values[:, -1]
        contributions = (transformed - self.background) * estimator.coef_
        base = estimator.intercept_ + np.dot(self.background, estimator.coef_)
        return contributions, np.full(len(frame), base)


def candidate_models(training: pd.DataFrame, seed: int = 42) -> list[ValueModel]:
    """Fixed small benchmark grid; preprocessing fits only the supplied training rows."""
    target = training.market_value_eur.to_numpy(dtype=float)
    fallback = float(np.median(target))
    candidates = [ValueModel(name, "eur", None, fallback) for name in ("median", "persistence")]
    for mode in ("eur", "log1p"):
        y = np.log1p(target) if mode == "log1p" else target
        estimators = {
            "ols": LinearRegression(),
            "ridge": Ridge(alpha=10),
            "lasso": Lasso(alpha=0.001 if mode == "log1p" else 10000, max_iter=20000),
            **{
                f"xgb_depth{depth}": XGBRegressor(
                    n_estimators=250,
                    max_depth=depth,
                    learning_rate=0.05,
                    min_child_weight=10,
                    objective="reg:absoluteerror",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=seed,
                    subsample=1,
                    colsample_bytree=1,
                )
                for depth in (3, 5)
            },
        }
        for name, estimator in estimators.items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            ).fit(training.loc[:, FEATURES], y)
            background = pipeline[:-1].transform(training.loc[:, FEATURES]).mean(axis=0)
            candidates.append(ValueModel(f"{name}_{mode}", mode, pipeline, fallback, background))
    return candidates


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """All model selection/evaluation metrics are computed in observed EUR."""
    if not len(actual) or not np.isfinite(prediction).all():
        raise ValueError("Metrics require nonempty observations and finite predictions")
    return {
        "n": len(actual),
        "mae_eur": float(mean_absolute_error(actual, prediction)),
        "rmse_eur": float(np.sqrt(mean_squared_error(actual, prediction))),
        "median_ae_eur": float(median_absolute_error(actual, prediction)),
        "r2": float(r2_score(actual, prediction)) if len(actual) > 1 else None,
    }


def interval_scale(frame: pd.DataFrame, fallback: float, floor: float) -> np.ndarray:
    """Predetermined scale depends on the prior value, never the current target."""
    return np.maximum(frame.previous_value_eur.fillna(fallback).to_numpy(dtype=float), floor)


def calibrate_interval(
    model: ValueModel,
    calibration: pd.DataFrame,
    *,
    alpha: float,
    floor: float,
) -> float:
    """Finite-sample residual quantile; temporal dependence limits coverage guarantees."""
    if not 0 < alpha < 1 or floor <= 0 or len(calibration) < 20:
        raise ValueError("Intervals need alpha in (0,1), positive scale and >=20 calibration rows")
    scores = np.abs(
        calibration.market_value_eur.to_numpy() - model.predict(calibration)
    ) / interval_scale(calibration, model.fallback, floor)
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError("Calibration sample is too small for the requested interval level")
    return float(np.sort(scores)[rank - 1])


def evaluate_model(
    model: ValueModel, test: pd.DataFrame, quantile: float, floor: float
) -> pd.DataFrame:
    """Return dated observations, predictions, intervals and model-relative residuals."""
    result = test[
        [
            "player_id",
            "player_name",
            "competition_id",
            "season",
            "valuation_date",
            "cohort_context",
            "age_years",
            "market_value_eur",
        ]
    ].copy()
    result["predicted_value_eur"] = model.predict(test)
    width = quantile * interval_scale(test, model.fallback, floor)
    result["lower_eur"] = np.maximum(0, result.predicted_value_eur - width)
    result["upper_eur"] = result.predicted_value_eur + width
    result["residual_eur"] = result.market_value_eur - result.predicted_value_eur
    result["model_undervaluation_eur"] = -result.residual_eur
    result["relative_gap"] = -result.residual_eur / np.maximum(result.market_value_eur, floor)
    result["uncertainty_scaled_gap"] = -result.residual_eur / np.maximum(width, 1)
    result["assessment"] = np.select(
        [result.market_value_eur < result.lower_eur, result.market_value_eur > result.upper_eur],
        ["below_model_interval", "above_model_interval"],
        default="within_model_interval",
    )
    return result
