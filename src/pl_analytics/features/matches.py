"""Pre-origin Elo and recent form features, with no current-month result access."""

import numpy as np
import pandas as pd

from pl_analytics.models.match_statistical import fit_elo, validate_history

MATCH_FEATURES = (
    "elo_difference",
    "home_points_recent",
    "away_points_recent",
    "home_goals_for_recent",
    "away_goals_for_recent",
    "home_goals_against_recent",
    "away_goals_against_recent",
    "home_recent_matches",
    "away_recent_matches",
    "home_days_since_result",
    "away_days_since_result",
)


def match_features_at_origin(
    history: pd.DataFrame, fixtures: pd.DataFrame, origin: pd.Timestamp, config: dict
) -> pd.DataFrame:
    """Freeze all features before the month, matching the M6 information horizon.

    Counts expose small samples; missing history remains missing until training-only
    imputation. Days since result is measured at origin, not an invented rest estimate.
    """
    train = validate_history(history, origin)
    if not fixtures.competition_id.eq(train.iloc[0].competition_id).all():
        raise ValueError("Fixture competition differs from training")
    if (fixtures.match_day < origin).any():
        raise ValueError("Fixture precedes feature origin")
    elo = fit_elo(
        train,
        origin,
        k=config["feature_elo_k"],
        home_advantage=config["elo_home_advantage"],
        season_retention=config["elo_season_retention"],
    )
    team_rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows = train[["match_id", "match_day"]].copy()
        rows["club_id"] = train[f"{side}_club_id"]
        rows["goals_for"] = train[f"{side}_goals"]
        rows["goals_against"] = train[f"{other}_goals"]
        rows["points"] = np.where(
            rows.goals_for > rows.goals_against,
            3,
            np.where(rows.goals_for == rows.goals_against, 1, 0),
        )
        team_rows.append(rows)
    recent = (
        pd.concat(team_rows)
        .sort_values(["match_day", "match_id"])
        .groupby("club_id")
        .tail(config["rolling_matches"])
    )
    grouped = {club: group for club, group in recent.groupby("club_id")}
    output = []
    for fixture in fixtures.itertuples():
        row = {
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
        row.update(origin=origin, feature_history_end=train.match_day.max())
        _, rh, ra = elo.predict(fixture.home_club_id, fixture.away_club_id, fixture.season)
        row["elo_difference"] = rh - ra
        for side in ("home", "away"):
            group = grouped.get(getattr(fixture, f"{side}_club_id"))
            row[f"{side}_recent_matches"] = len(group) if group is not None else 0
            row[f"{side}_days_since_result"] = (
                (origin - group.match_day.max()).days if group is not None else np.nan
            )
            for column in ("points", "goals_for", "goals_against"):
                row[f"{side}_{column}_recent"] = (
                    group[column].mean() if group is not None else np.nan
                )
        output.append(row)
    return pd.DataFrame(output)


def build_match_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate monthly rolling-origin rows, joining labels after feature construction."""
    batches = []
    end = frame.match_day.max()
    columns = ["match_id", "competition_id", "season", "match_day", "home_club_id", "away_club_id"]
    for origin in pd.date_range(pd.Timestamp(config["train_start"], tz="UTC"), end, freq="MS"):
        fixtures = frame.loc[
            (frame.match_day >= origin) & (frame.match_day < origin + pd.offsets.MonthBegin()),
            columns,
        ]
        if fixtures.empty:
            continue
        history = frame.loc[
            (frame.match_day < origin)
            & (frame.match_day >= origin - pd.Timedelta(days=config["lookback_days"]))
        ]
        if len(history) < config["min_feature_matches"]:
            raise ValueError("Insufficient feature history before origin")
        batches.append(match_features_at_origin(history, fixtures, origin, config))
    if not batches:
        raise ValueError("No match features in requested period")
    return pd.concat(batches, ignore_index=True).merge(
        frame[["match_id", "home_goals", "away_goals"]], on="match_id", validate="one_to_one"
    )
