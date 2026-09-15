"""Training-only match classifiers with disjoint chronological temperature calibration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pl_analytics.features.matches import MATCH_FEATURES
from pl_analytics.models.match_experiment import labels
from pl_analytics.statistics.matches import probability_metrics


@dataclass
class MatchClassifier:
    """Predict H/D/A in canonical order; temperature never changes fitted feature transforms."""

    name: str
    pipeline: Pipeline
    temperature: float = 1.0

    def predict(self, features: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        probabilities = self.pipeline.predict_proba(features.loc[:, MATCH_FEATURES])
        order = [list(self.pipeline.classes_).index(index) for index in (0, 1, 2)]
        probabilities = probabilities[:, order]
        if calibrated:
            probabilities = scale_temperature(probabilities, self.temperature)
        return probabilities


def scale_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Renormalized softmax(log p / T), with a numerical floor for zero probabilities."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    probability_metrics(np.zeros(len(probabilities), dtype=int), probabilities)
    return softmax(np.log(np.maximum(probabilities, 1e-15)) / temperature, axis=1)


def fit_classifiers(training: pd.DataFrame, config: dict) -> list[MatchClassifier]:
    """Fixed logistic/XGBoost grid; fit all imputation and scaling on training only."""
    y = labels(training)
    if set(y) != {0, 1, 2}:
        raise ValueError("Classifier training requires all three outcome classes")
    estimators = {
        f"logistic_c{c:g}": LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        for c in config["logistic_c"]
    }
    estimators.update(
        {
            f"xgb_depth{depth}": XGBClassifier(
                n_estimators=config["xgb_estimators"],
                max_depth=depth,
                learning_rate=config["xgb_learning_rate"],
                min_child_weight=10,
                reg_lambda=1,
                objective="multi:softprob",
                num_class=3,
                tree_method="hist",
                n_jobs=1,
                subsample=1,
                colsample_bytree=1,
                random_state=config["seed"],
                eval_metric="mlogloss",
            )
            for depth in config["xgb_depths"]
        }
    )
    result = []
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        ).fit(training.loc[:, MATCH_FEATURES], y)
        result.append(MatchClassifier(name, pipeline))
    return result


def fit_temperature(
    probabilities: np.ndarray, actual: np.ndarray, bounds: tuple[float, float] = (0.5, 5.0)
) -> float:
    """Fit a single scalar on a separate calibration period; do not refit the classifier."""
    if len(actual) < 30 or not 0 < bounds[0] < bounds[1]:
        raise ValueError("Calibration needs at least 30 rows and positive ordered bounds")
    fitted = minimize_scalar(
        lambda t: probability_metrics(actual, scale_temperature(probabilities, t))["log_loss"],
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError("Temperature calibration did not converge")
    return float(fitted.x)
