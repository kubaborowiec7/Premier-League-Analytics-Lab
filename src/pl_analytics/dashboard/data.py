"""Cached, schema-checked access to explicitly prepared dashboard artifacts."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.features.matches import MATCH_FEATURES
from pl_analytics.statistics.matches import outcome_probabilities, score_matrix


class ArtifactError(ValueError):
    """An optional local artifact is missing, inconsistent or unreadable."""


@st.cache_data(show_spinner=False)
def _table(path: str, digest: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _json(path: str, digest: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError("Expected a report object")
    return value


def table(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return an empty schema on missing data; reject corrupt/present artifacts clearly."""
    if not path.exists():
        return pd.DataFrame(columns=list(required))
    try:
        frame = _table(str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
        if not set(required) <= set(frame.columns):
            raise ArtifactError("Required columns are missing")
        return frame
    except Exception as error:
        logging.warning("Cannot read dashboard table %s: %s", path.name, type(error).__name__)
        raise ArtifactError(
            f"The prepared {path.stem} table cannot be read. Rebuild its artifacts."
        ) from None


def report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return _json(str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as error:
        logging.warning("Cannot read dashboard report %s: %s", path.name, type(error).__name__)
        raise ArtifactError("An evaluation report cannot be read. Rebuild its artifacts.") from None


def catalog(settings: Settings) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """One broken module does not prevent independent pages from opening."""
    scope = ("competition_id", "season")
    specs = {
        "players": (
            settings.data_dir / "processed/m4/player_features.parquet",
            (*scope, "player_id", "player_name", "minutes", "position_group"),
        ),
        "values": (
            settings.artifact_dir / "m5/value_ranking.parquet",
            (
                *scope,
                "player_id",
                "player_name",
                "valuation_date",
                "predicted_value_eur",
                "market_value_eur",
            ),
        ),
        "matches": (
            settings.artifact_dir / "m7/predictions.parquet",
            (
                *scope,
                "match_id",
                "model",
                "match_day",
                "home_club_id",
                "away_club_id",
                "p_home",
                "p_draw",
                "p_away",
            ),
        ),
    }
    if not specs["matches"][0].exists():
        specs["matches"] = (
            settings.artifact_dir / "m6/final/predictions.parquet",
            specs["matches"][1],
        )
    tables, errors = {}, []
    for name, (path, required) in specs.items():
        try:
            tables[name] = table(path, required)
            tables[name] = tables[name].loc[
                tables[name].competition_id.isin(settings.active_competitions)
            ]
        except ArtifactError as error:
            errors.append(str(error))
            tables[name] = pd.DataFrame(columns=required)
    return tables, errors


def available_competitions(tables: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(
        {str(code) for frame in tables.values() for code in frame.competition_id.dropna().unique()}
    )


@st.cache_resource(show_spinner=False)
def _bundle(path: str, digest: str) -> dict:
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ArtifactError("Model bundle no longer matches prepared dashboard metadata")
    return joblib.load(source)


def prepared_models(settings: Settings) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Load only trusted local, checksum-pinned bundles; the UI accepts no uploaded models."""
    root = settings.artifact_dir
    metadata = report(root / "dashboard/catalog.json")
    if not metadata or not metadata.get("models"):
        raise ArtifactError("Match models are not prepared for the dashboard yet.")
    states = table(
        root / "dashboard/team_states.parquet",
        ("club_id", "competition_id", "season", "origin", "elo"),
    )
    bundles = {}
    paths = {
        "statistics": root / "m7/statistical_benchmarks/latest_origin_models.joblib",
        "ml": root / "m7/classifiers.joblib",
    }
    try:
        for name, path in paths.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != metadata["models"][name]:
                raise ArtifactError("Model checksum mismatch")
            bundles[name] = _bundle(str(path.resolve()), digest)
        origin = pd.Timestamp(bundles["statistics"]["origin"])
        if (
            states.empty
            or not pd.to_datetime(states.origin, utc=True).eq(origin).all()
            or pd.Timestamp(metadata["origin"]) != origin
            or not states.competition_id.eq(bundles["statistics"]["competition_id"]).all()
        ):
            raise ArtifactError("Team features and models have different origins or competitions")
    except Exception as error:
        logging.warning("Dashboard model loading failed: %s", type(error).__name__)
        raise ArtifactError(
            "Prepared match models are missing or incompatible; rebuild the dashboard data."
        ) from error
    return bundles["statistics"], bundles["ml"], states, metadata


def scenario(
    home: str,
    away: str,
    competition: str,
    model_name: str,
    statistics: dict,
    ml: dict,
    states: pd.DataFrame,
) -> dict:
    """Inference only from the prepared origin; no raw data or fit call is involved."""
    if home == away:
        raise ArtifactError("Choose two different clubs.")
    if statistics["competition_id"] != competition:
        raise ArtifactError("No fitted model is available for this competition.")
    context = states.loc[states.competition_id.eq(competition)]
    if context.club_id.duplicated().any():
        raise ArtifactError("Prepared club states are not unique.")
    context = context.set_index("club_id")
    if home not in context.index or away not in context.index:
        raise ArtifactError("Prepared features are missing for the selected clubs.")
    if context.loc[home, "season"] != context.loc[away, "season"]:
        raise ArtifactError("Club feature seasons do not match.")
    if (
        "origin" in context
        and not pd.to_datetime(context.origin, utc=True)
        .eq(pd.Timestamp(statistics["origin"]))
        .all()
    ):
        raise ArtifactError("Club features and model origins do not match.")
    matrix, rates = None, None
    if model_name == "calibrated_ml":
        row = {"elo_difference": context.loc[home, "elo"] - context.loc[away, "elo"]}
        for side, club in (("home", home), ("away", away)):
            for suffix in (
                "points_recent",
                "goals_for_recent",
                "goals_against_recent",
                "recent_matches",
                "days_since_result",
            ):
                row[f"{side}_{suffix}"] = context.loc[club, suffix]
        model = next(item for item in ml["models"] if item.name == ml["chosen"])
        probabilities = model.predict(pd.DataFrame([row], columns=MATCH_FEATURES))[0]
    else:
        model = statistics["models"].get(model_name)
        if model is None:
            raise ArtifactError("This model is not available.")
        if hasattr(model, "rates"):
            rates = model.rates(home, away)
            matrix, _ = score_matrix(*rates, model.rho)
            probabilities = outcome_probabilities(matrix)
        elif hasattr(model, "predict"):
            probabilities, _, _ = model.predict(home, away, context.loc[home, "season"])
        else:
            probabilities = np.asarray(model)
    return {
        "probabilities": probabilities,
        "matrix": matrix,
        "rates": rates,
        "origin": statistics["origin"],
        "model": model_name,
    }
