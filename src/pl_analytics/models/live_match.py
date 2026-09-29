"""Fit current operational match artifacts while preserving frozen evaluation bundles."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from pl_analytics.features.matches import match_features_at_origin
from pl_analytics.models.match_experiment import forecast_origin


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_ring(history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    season = history.sort_values("match_day").season.iloc[-1]
    season_rows = history.loc[history.season.eq(season)]
    clubs = sorted(set(season_rows.home_club_id) | set(season_rows.away_club_id))
    if len(clubs) < 2:
        raise ValueError("Current season needs at least two clubs")
    return pd.DataFrame(
        [
            {
                "match_id": f"live:context:{index}",
                "competition_id": history.competition_id.iloc[0],
                "season": season,
                "match_day": origin,
                "home_club_id": club,
                "away_club_id": clubs[(index + 1) % len(clubs)],
            }
            for index, club in enumerate(clubs)
        ]
    )


def build_live_match_artifacts(
    history: pd.DataFrame,
    config: dict,
    ml_bundle_path: Path,
    artifact_dir: Path,
    *,
    origin: pd.Timestamp,
    source_sha256: str,
    refreshed_at: str,
) -> dict:
    """Build a versioned live bundle from results strictly before ``origin``."""
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    history = history.loc[
        pd.to_datetime(history.match_day, utc=True).lt(origin)
        & pd.to_datetime(history.match_day, utc=True).ge(
            origin - pd.Timedelta(days=config["lookback_days"])
        )
    ].copy()
    if len(history) < config["min_train_matches"]:
        raise ValueError("Insufficient live history before the requested origin")
    fixtures = _fixture_ring(history, origin)
    _, _, models = forecast_origin(history, fixtures, origin, config)
    features = match_features_at_origin(history, fixtures, origin, config)
    elo = models[f"elo_k{config['feature_elo_k']}"]
    rows = []
    for feature in features.itertuples():
        _, home_elo, away_elo = elo.predict(
            feature.home_club_id, feature.away_club_id, feature.season
        )
        for side, rating in (("home", home_elo), ("away", away_elo)):
            state = {
                "club_id": getattr(feature, f"{side}_club_id"),
                "competition_id": feature.competition_id,
                "season": feature.season,
                "origin": feature.origin,
                "elo": rating,
            }
            for suffix in (
                "points_recent",
                "goals_for_recent",
                "goals_against_recent",
                "recent_matches",
                "days_since_result",
            ):
                state[suffix] = getattr(feature, f"{side}_{suffix}")
            rows.append(state)
    states = pd.DataFrame(rows).drop_duplicates("club_id")
    expected = set(fixtures.home_club_id) | set(fixtures.away_club_id)
    if set(states.club_id) != expected:
        raise ValueError("Live team states do not cover the current competition")

    version = f"{origin.date().isoformat()}-{source_sha256[:12]}"
    root = artifact_dir / "live" / "snapshots" / version
    root.mkdir(parents=True, exist_ok=True)
    statistics_path = root / "statistical_models.joblib"
    ml_path = root / "classifiers.joblib"
    states_path = root / "team_states.parquet"
    clubs_path = root / "clubs.parquet"
    joblib.dump(
        {"origin": origin, "competition_id": config["competition_id"], "models": models},
        statistics_path,
    )
    ml_path.write_bytes(ml_bundle_path.read_bytes())
    states.to_parquet(states_path, index=False)
    clubs = pd.concat(
        [
            history[["competition_id", f"{side}_club_id", f"{side}_name"]].rename(
                columns={f"{side}_club_id": "club_id", f"{side}_name": "name"}
            )
            for side in ("home", "away")
        ]
    ).drop_duplicates(["competition_id", "club_id"], keep="last")
    clubs.to_parquet(clubs_path, index=False)
    relative = root.relative_to(artifact_dir)
    metadata = {
        "mode": "operational",
        "competition_id": config["competition_id"],
        "season": fixtures.season.iloc[0],
        "origin": origin.isoformat(),
        "last_result_date": pd.to_datetime(history.match_day, utc=True).max().date().isoformat(),
        "refreshed_at": refreshed_at,
        "training_matches": len(history),
        "source_sha256": source_sha256,
        "available_clubs": len(states),
        "paths": {
            "statistics": (relative / statistics_path.name).as_posix(),
            "ml": (relative / ml_path.name).as_posix(),
            "states": (relative / states_path.name).as_posix(),
            "clubs": (relative / clubs_path.name).as_posix(),
        },
        "models": {"statistics": _sha256(statistics_path), "ml": _sha256(ml_path)},
    }
    dashboard = artifact_dir / "dashboard"
    dashboard.mkdir(parents=True, exist_ok=True)
    temporary = dashboard / "live_catalog.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(dashboard / "live_catalog.json")
    return metadata
