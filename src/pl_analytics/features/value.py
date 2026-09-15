"""Point-in-time value features from prior observations, never snapshot roles."""

import numpy as np
import pandas as pd

FEATURES = (
    "age_years",
    "previous_value_eur",
    "days_since_valuation",
    "days_since_appearance",
    "appearances_365",
    "minutes_365",
    "goals_per90_365",
    "assists_per90_365",
)


def build_value_features(
    appearances: pd.DataFrame,
    valuations: pd.DataFrame,
    profiles: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Select recent competition participants and strictly prior 365-day exposure.

    Past valuations from this provider are legitimate predictors in sequential updating;
    same-day valuations/appearances never enter features. All target-date source clubs
    and profile roles are excluded. This does not assert current league membership.
    """
    apps, values = appearances.copy(), valuations.copy()
    for frame in (apps, values):
        frame["date"] = pd.to_datetime(frame.date, utc=True, errors="raise")
        if frame.date.isna().any():
            raise ValueError("Observation dates are required")
    if not apps.competition_id.eq(config["competition_id"]).all():
        raise ValueError("Appearance scope differs from requested competition")
    if (
        apps.duplicated(["player_id", "game_id"]).any()
        or values.duplicated(["player_id", "date"]).any()
    ):
        raise ValueError("Duplicate historical observation")
    for frame, columns in ((apps, ["minutes", "goals", "assists"]), (values, ["market_value_eur"])):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            observed = frame[column].dropna()
            if not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError("Counts and values must be finite and nonnegative")
    if values.market_value_eur.isna().any():
        raise ValueError("Target valuations cannot be missing")
    if profiles.player_id.isna().any() or profiles.player_id.duplicated().any():
        raise ValueError("Profile identities must be present and unique")
    people = profiles.set_index("player_id")
    histories = {key: group.sort_values("date") for key, group in apps.groupby("player_id")}
    rows = []
    for player, history in values.groupby("player_id", sort=True):
        if player not in histories or player not in people.index:
            continue
        games = histories[player]
        birth = pd.to_datetime(people.loc[player, "date_of_birth"], utc=True)
        previous = None
        for target in history.sort_values("date").itertuples():
            before = games.loc[
                (games.date < target.date)
                & (games.date >= target.date - pd.Timedelta(days=config["lookback_days"]))
            ]
            eligible = (
                not before.empty
                and (target.date - before.date.max()).days <= config["activity_days"]
            )
            if eligible and target.date >= pd.to_datetime(config["train_start"], utc=True):
                year = target.date.year - (target.date.month < config["season_start_month"])
                minutes = before.minutes.sum() if before.minutes.notna().all() else np.nan
                row = {
                    "player_id": player,
                    "player_name": people.loc[player, "name"],
                    "competition_id": config["competition_id"],
                    "season": f"{year}/{(year + 1) % 100:02}",
                    "valuation_date": target.date,
                    "market_value_eur": target.market_value_eur,
                    "cohort_context": "recent_prior_competition_participant",
                    "age_years": (target.date - birth).days / 365.25 if pd.notna(birth) else np.nan,
                    "previous_value_eur": previous.market_value_eur if previous else np.nan,
                    "previous_valuation_date": previous.date if previous else pd.NaT,
                    "days_since_valuation": (target.date - previous.date).days
                    if previous
                    else np.nan,
                    "last_appearance_date": before.date.max(),
                    "days_since_appearance": (target.date - before.date.max()).days,
                    "appearances_365": len(before),
                    "minutes_365": minutes,
                }
                for metric in ("goals", "assists"):
                    row[f"{metric}_per90_365"] = (
                        90 * before[metric].sum() / minutes
                        if minutes > 0 and before[metric].notna().all()
                        else np.nan
                    )
                if row["age_years"] < 0:
                    raise ValueError("Valuation precedes birth date")
                rows.append(row)
            previous = target
    if not rows:
        raise ValueError("No targets meet the declared prior-activity cohort rule")
    return pd.DataFrame(rows).sort_values(["valuation_date", "player_id"]).reset_index(drop=True)


def chronological_parts(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Four non-overlapping time windows; boundary dates belong to the newer window."""
    cuts = pd.to_datetime(
        [
            config[key]
            for key in (
                "train_start",
                "validation_start",
                "calibration_start",
                "test_start",
                "test_end",
            )
        ],
        utc=True,
    )
    if not all(left < right for left, right in zip(cuts[:-1], cuts[1:], strict=True)):
        raise ValueError("Split dates must be strictly increasing")
    dates = pd.to_datetime(frame.valuation_date, utc=True)
    return {
        name: frame.loc[(dates >= start) & (dates < end)].copy()
        for name, start, end in zip(
            ("train", "validation", "calibration", "test"),
            cuts[:-1],
            [*cuts[1:-1], cuts[-1] + pd.Timedelta(days=1)],
            strict=True,
        )
    }
