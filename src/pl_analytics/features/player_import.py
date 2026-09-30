"""Validate session-only player inputs and align them with analytical schemas."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pl_analytics.features.advanced_players import METRICS
from pl_analytics.features.value import FEATURES as VALUE_FEATURES

IDENTITY_COLUMNS = (
    "player_name",
    "competition_id",
    "season",
    "position_group",
    "club_name",
    "minutes",
    "appearances",
)
POSITION_ALIASES = {
    "GK": "GK",
    "GOALKEEPER": "GK",
    "DEF": "DEF",
    "DEFENDER": "DEF",
    "MID": "MID",
    "MIDFIELDER": "MID",
    "FWD": "FWD",
    "FORWARD": "FWD",
    "ATTACKER": "FWD",
    "STRIKER": "FWD",
}
MAX_IMPORT_ROWS = 20


def performance_template() -> pd.DataFrame:
    """Return an empty CSV template accepting totals or per-90 count metrics."""
    columns = [*IDENTITY_COLUMNS]
    for name, metric in METRICS.items():
        columns.append(name)
        if metric.kind == "count":
            columns.append(f"{name}_per90")
        columns.append(f"{name}_coverage")
    return pd.DataFrame(columns=columns)


def _text(value: object, field: str, *, maximum: int = 120) -> str:
    result = str(value).strip() if pd.notna(value) else ""
    if not result or len(result) > maximum:
        raise ValueError(f"{field} must contain 1–{maximum} characters")
    return result


def _number(
    value: object,
    field: str,
    *,
    minimum: float = 0,
    maximum: float | None = None,
    required: bool = True,
) -> float:
    if pd.isna(value) or value == "":
        if required:
            raise ValueError(f"{field} is required")
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        suffix = f" and at most {maximum:g}" if maximum is not None else ""
        raise ValueError(f"{field} must be finite, at least {minimum:g}{suffix}")
    return result


def _identifier(name: str, competition: str, season: str, position: str) -> str:
    digest = hashlib.sha256(
        f"{name.casefold()}|{competition}|{season}|{position}".encode()
    ).hexdigest()[:16]
    return f"user:{digest}"


def _defaulted(source: Mapping[str, object], field: str, default: object) -> object:
    value = source.get(field, default)
    return default if pd.isna(value) or value == "" else value


def performance_rows(
    input_frame: pd.DataFrame,
    *,
    competition_id: str,
    season: str,
    reference_date: str,
) -> pd.DataFrame:
    """Validate up to 20 player rows and create complete advanced-profile records.

    Count metrics may be supplied either as totals (``goals``) or per-90 values
    (``goals_per90``). Percentage/ratio metrics use their base column. A supplied
    metric defaults to complete coverage; callers can provide a 0–1 coverage column.
    """
    if input_frame.empty or len(input_frame) > MAX_IMPORT_ROWS:
        raise ValueError(f"Import must contain between 1 and {MAX_IMPORT_ROWS} players")
    required = {"player_name", "position_group", "minutes"}
    missing = required - set(input_frame)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    allowed = set(performance_template().columns) | {"reference_date", "last_match_date"}
    unknown = set(input_frame) - allowed
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")

    rows: list[dict[str, object]] = []
    for source in input_frame.to_dict("records"):
        name = _text(source.get("player_name"), "player_name")
        row_competition = _text(
            _defaulted(source, "competition_id", competition_id),
            "competition_id",
            maximum=32,
        )
        row_season = _text(_defaulted(source, "season", season), "season", maximum=16)
        if row_competition != competition_id or row_season != season:
            raise ValueError("Imported competition_id and season must match the selected cohort")
        raw_position = _text(source.get("position_group"), "position_group", maximum=24).upper()
        if raw_position not in POSITION_ALIASES:
            raise ValueError("position_group must be GK, DEF, MID or FWD")
        position = POSITION_ALIASES[raw_position]
        minutes = _number(source.get("minutes"), "minutes", minimum=1, maximum=6000)
        appearances = _number(
            source.get("appearances", np.nan),
            "appearances",
            minimum=1,
            maximum=80,
            required=False,
        )
        if pd.isna(appearances):
            appearances = max(1, int(np.ceil(minutes / 90)))
        if not float(appearances).is_integer():
            raise ValueError("appearances must be a whole number")
        club = str(_defaulted(source, "club_name", "User supplied")).strip()
        player_id = _identifier(name, row_competition, row_season, position)
        row: dict[str, object] = {
            "competition_id": row_competition,
            "season": row_season,
            "player_id": player_id,
            "player_name": name,
            "position_group": position,
            "club_id": pd.NA,
            "club_name": club,
            "club_short_name": club[:3].upper(),
            "club_context": "User-supplied session record; club is not independently verified",
            "position_context": "User-supplied broad role",
            "minutes": minutes,
            "appearances": int(appearances),
            "reference_date": str(_defaulted(source, "reference_date", reference_date)),
            "last_match_date": str(_defaulted(source, "last_match_date", reference_date)),
            "source": "User import (session)",
            "comparison_only": True,
        }
        supplied = 0
        for metric_name, metric in METRICS.items():
            total = source.get(metric_name, np.nan)
            per90 = source.get(f"{metric_name}_per90", np.nan)
            has_total, has_per90 = pd.notna(total) and total != "", pd.notna(per90) and per90 != ""
            if has_total and has_per90 and metric.kind == "count":
                raise ValueError(f"Supply either {metric_name} or {metric_name}_per90, not both")
            if not has_total and not has_per90:
                row[metric_name] = np.nan
                row[f"{metric_name}_per90"] = np.nan
                row[f"{metric_name}_coverage"] = 0.0
                continue
            supplied += 1
            coverage = _number(
                source.get(f"{metric_name}_coverage", 1.0),
                f"{metric_name}_coverage",
                minimum=0.01,
                maximum=1,
            )
            observed_minutes = minutes * coverage
            limit = 100 if metric.kind != "count" else None
            value = _number(
                per90 if has_per90 and metric.kind == "count" else total,
                f"{metric_name}{'_per90' if has_per90 and metric.kind == 'count' else ''}",
                maximum=limit,
            )
            if metric.kind == "count":
                metric_per90 = value if has_per90 else value * 90 / observed_minutes
                metric_total = metric_per90 * observed_minutes / 90 if has_per90 else value
            else:
                metric_total = metric_per90 = value
            row[metric_name] = metric_total
            row[f"{metric_name}_per90"] = metric_per90
            row[f"{metric_name}_coverage"] = coverage
        if supplied == 0:
            raise ValueError(f"{name} must include at least one supported statistic")
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.player_id.duplicated().any():
        raise ValueError("Each imported player must have a unique name and position")
    return result


def value_feature_rows(
    input_frame: pd.DataFrame, *, competition_id: str, default_season: str
) -> pd.DataFrame:
    """Validate user-supplied rows for frozen market-value scenario inference."""
    required = {"player_name", "market_value_eur", *VALUE_FEATURES}
    missing = required - set(input_frame)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if input_frame.empty or len(input_frame) > MAX_IMPORT_ROWS:
        raise ValueError(f"Import must contain between 1 and {MAX_IMPORT_ROWS} players")
    allowed = required | {"competition_id", "season", "valuation_date"}
    unknown = set(input_frame) - allowed
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")
    rows = []
    for source in input_frame.to_dict("records"):
        name = _text(source.get("player_name"), "player_name")
        row_competition = _text(
            _defaulted(source, "competition_id", competition_id),
            "competition_id",
            maximum=32,
        )
        if row_competition != competition_id:
            raise ValueError("Imported competition_id must match the selected competition")
        season = _text(_defaulted(source, "season", default_season), "season", maximum=16)
        row: dict[str, object] = {
            "player_id": _identifier(name, row_competition, season, "VALUE"),
            "player_name": name,
            "competition_id": row_competition,
            "season": season,
            "valuation_date": str(
                _defaulted(source, "valuation_date", pd.Timestamp.now().date())
            ),
            "market_value_eur": _number(source.get("market_value_eur"), "market_value_eur"),
            "cohort_context": "user_supplied_scenario",
        }
        limits = {
            "age_years": (15, 50),
            "previous_value_eur": (0, 500_000_000),
            "days_since_valuation": (0, 10_000),
            "days_since_appearance": (0, 3650),
            "appearances_365": (0, 50),
            "minutes_365": (0, 3420),
            "goals_per90_365": (0, 10),
            "assists_per90_365": (0, 10),
        }
        for field in VALUE_FEATURES:
            minimum, maximum = limits[field]
            row[field] = _number(source.get(field), field, minimum=minimum, maximum=maximum)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.player_id.duplicated().any():
        raise ValueError("Each imported valuation player must have a unique name and season")
    return result


def manual_performance_frame(
    identity: Mapping[str, object], metrics: Mapping[str, float], *, basis: str
) -> pd.DataFrame:
    """Build a one-row raw frame from manual form fields."""
    if basis not in {"Total", "Per 90"}:
        raise ValueError("Metric basis must be Total or Per 90")
    row = dict(identity)
    for name, value in metrics.items():
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}")
        column = f"{name}_per90" if basis == "Per 90" and METRICS[name].kind == "count" else name
        row[column] = value
    return pd.DataFrame([row])
