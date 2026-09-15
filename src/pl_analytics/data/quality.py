"""Descriptive, scope-preserving EDA over canonical records; never fits a model."""

import json
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import parse_results
from pl_analytics.data.snapshots import Snapshot
from pl_analytics.data.transfermarkt import SOURCE, parse_player_values


@dataclass
class QualitySample:
    """Selected canonical frames with source provenance and explicit scope."""

    scope: Scope
    matches: pd.DataFrame
    players: pd.DataFrame
    values: pd.DataFrame
    provenance: list[dict[str, str]]


def load_quality_sample(manifest_path: Path, raw_dir: Path) -> QualitySample:
    """Read M1's pinned local archives only; missing/changed input fails closed.

    Acquisition remains the explicit M1 command. M3 never fetches a replacement
    dataset or changes raw files, so a clean notebook kernel cannot trigger downloads.
    """
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = Scope(
        config["competition_id"],
        config["competition_name"],
        config["country"],
        config["match_source_code"],
        config["season"],
        date.fromisoformat(config["start_date"]),
        date.fromisoformat(config["end_date"]),
    )
    snapshots = []
    pins = [("football-data.co.uk", config["match_sha256"])] + [
        (f"{SOURCE}.{table}", config["player_sha256"][f"{table}.csv.gz"])
        for table in ("players", "player_valuations", "clubs")
    ]
    for source, digest in pins:
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Quality manifest requires a lowercase SHA-256 pin for each source")
        sidecar = raw_dir / source / digest / "metadata.json"
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"Missing pinned archive: {sidecar}. Run scripts/verify_ingestion.py first."
            )
        snapshot = Snapshot.read(sidecar)
        if snapshot.source_name != source or snapshot.sha256 != digest:
            raise ValueError("Archived source identity does not match the manifest")
        snapshots.append(snapshot)
    matches = parse_results(snapshots[0], scope, timezone=config["match_timezone"])
    players = parse_player_values(
        *snapshots[1:], scope=replace(scope, source_code=config["player_source_code"])
    )
    tables = {**matches.tables, **players.tables}
    for table, expected in config["expected_counts"].items():
        if len(tables[table]) != expected:
            raise ValueError(f"Pinned sample count mismatch for {table}")
    return QualitySample(
        scope,
        pd.DataFrame(tables["matches"]),
        pd.DataFrame(tables["players"]),
        pd.DataFrame(tables["player_market_values"]),
        [
            {
                "source": s.source_name,
                "sha256": s.sha256,
                "dataset_version": s.dataset_version,
                "retrieved_at": s.retrieved_at,
                "source_url": s.source_url,
                "license_or_terms_note": s.license_or_terms_note,
            }
            for s in snapshots
        ],
    )


def missingness(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Column denominators are explicit; an empty table has undefined missing fraction."""
    return [
        {
            "column": str(column),
            "rows": len(frame),
            "missing": int(frame[column].isna().sum()),
            "missing_fraction": float(frame[column].isna().mean()) if len(frame) else None,
        }
        for column in frame.columns
    ]


def numeric_summary(series: pd.Series) -> dict[str, int | float | None]:
    """Describe finite observations; skew is undefined for constant/tiny samples."""
    values = pd.to_numeric(series, errors="raise").dropna().astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values are not valid EDA observations")
    result: dict[str, int | float | None] = {
        "n": len(values),
        "minimum": None,
        "median": None,
        "mean": None,
        "p95": None,
        "maximum": None,
        "sample_skew": None,
    }
    if len(values):
        result.update(
            minimum=float(values.min()),
            median=float(values.median()),
            mean=float(values.mean()),
            p95=float(values.quantile(0.95)),
            maximum=float(values.max()),
        )
        if len(values) >= 3 and values.nunique() > 1:
            result["sample_skew"] = float(values.skew())
    return result


def latest_player_values(values: pd.DataFrame) -> pd.DataFrame:
    """Retrospective latest observation per player/scope/provider, never a past feature."""
    keys = ["player_id", "competition_id", "season", "source"]
    result = values.copy()
    result["valuation_date"] = pd.to_datetime(result["valuation_date"], utc=True)
    if result.duplicated([*keys, "valuation_date"]).any():
        raise ValueError("Ambiguous duplicate valuation date within player/scope/source")
    return result.sort_values("valuation_date").drop_duplicates(keys, keep="last")


def temporal_coverage(dates: pd.Series, scope: Scope) -> dict[str, Any]:
    """Report zero-count months as well as observed months in the requested window."""
    timestamps = pd.to_datetime(dates, utc=True, errors="raise")
    months = pd.period_range(scope.start_date, scope.end_date, freq="M").astype(str)
    counts = timestamps.dt.strftime("%Y-%m").value_counts().reindex(months, fill_value=0)
    return {
        "minimum": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "maximum": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "missing_dates": int(timestamps.isna().sum()),
        "outside_requested_dates": int(
            (
                timestamps.notna() & ~timestamps.dt.date.between(scope.start_date, scope.end_date)
            ).sum()
        ),
        "monthly_counts_utc": {str(k): int(v) for k, v in counts.items()},
    }


def build_quality_report(sample: QualitySample) -> dict[str, Any]:
    """Describe one scope, rejecting accidental cross-competition/season mixtures.

    The input is canonical post-ingestion data, not a report about every raw source
    row. No imputation, outlier removal or learned preprocessing is performed.
    """
    matches, players, values = sample.matches, sample.players, sample.values
    scope = sample.scope
    for frame in (matches, values):
        if not (
            (frame["competition_id"] == scope.competition_id) & (frame["season"] == scope.season)
        ).all():
            raise ValueError("Quality analysis requires one explicit competition/season scope")
    targets = pd.to_numeric(values["market_value_eur"], errors="raise").astype(float)
    if (targets.dropna() < 0).any():
        raise ValueError("Market values must be nonnegative")
    latest = latest_player_values(values)
    latest_targets = pd.to_numeric(latest["market_value_eur"]).astype(float)
    finished = matches.loc[
        matches["status"].eq("finished") & matches[["home_goals", "away_goals"]].notna().all(axis=1)
    ]
    outcome = np.select(
        [
            finished["home_goals"] > finished["away_goals"],
            finished["home_goals"] == finished["away_goals"],
        ],
        ["H", "D"],
        default="A",
    )
    counts = pd.Series(outcome).value_counts().reindex(["H", "D", "A"], fill_value=0)
    repeats = values.groupby(["player_id", "source"], dropna=False).size()
    return {
        "report_version": 1,
        "scope": {
            "competition_id": scope.competition_id,
            "season": scope.season,
            "start_date": scope.start_date.isoformat(),
            "end_date": scope.end_date.isoformat(),
        },
        "usage": "Descriptive development sample; not an untouched final evaluation set.",
        "provenance": sample.provenance,
        "counts": {
            "matches": len(matches),
            "players": len(players),
            "valuations": len(values),
            "player_source_pairs": len(repeats),
            "appearance_rows_loaded_by_m1": 0,
            "match_clubs": len(set(matches["home_club_id"]) | set(matches["away_club_id"])),
            "source_reported_valuation_clubs": int(values["source_reported_club_id"].nunique()),
        },
        "duplicates": {
            "match_ids": int(matches.duplicated(["match_id"]).sum()),
            "player_ids": int(players.duplicated(["player_id"]).sum()),
            "valuation_keys": int(
                values.duplicated(["player_id", "valuation_date", "source"]).sum()
            ),
        },
        "missingness": {
            name: missingness(frame)
            for name, frame in (("matches", matches), ("players", players), ("valuations", values))
        },
        "temporal": {
            "matches": temporal_coverage(matches["match_date"], scope),
            "valuations": temporal_coverage(values["valuation_date"], scope),
        },
        "match_targets": {
            "eligible_finished_matches": len(finished),
            "excluded_rows": len(matches) - len(finished),
            "outcomes": {
                key: {
                    "count": int(count),
                    "fraction": float(count / len(finished)) if len(finished) else None,
                }
                for key, count in counts.items()
            },
            "home_goals": numeric_summary(finished["home_goals"]),
            "away_goals": numeric_summary(finished["away_goals"]),
        },
        "valuation_targets": {
            "all_observations_eur": numeric_summary(targets),
            "all_observations_log1p_eur": numeric_summary(np.log1p(targets)),
            "latest_per_player_source_eur": numeric_summary(latest_targets),
            "latest_per_player_source_log1p_eur": numeric_summary(np.log1p(latest_targets)),
            "records_per_player_source": numeric_summary(repeats),
            "zero_values": int(targets.eq(0).sum()),
        },
        "leakage_and_coverage": {
            "unknown_kickoffs": int((~matches["match_time_known"].fillna(False)).sum()),
            "unverified_valuation_context": int(
                values["competition_context"].eq("source_reported_unverified").sum()
            ),
            "missing_historical_valuation_club": int(values["club_id"].isna().sum()),
            "valuations_without_profile": int(
                (~values["player_id"].isin(players["player_id"])).sum()
            ),
            "profile_forbidden_columns_present": sorted(
                set(players.columns)
                & {
                    "current_club_id",
                    "current_club_name",
                    "market_value_in_eur",
                    "highest_market_value_in_eur",
                }
            ),
            "publication_time_availability": "Not reconstructable from final historical snapshots",
            "player_performance_readiness": "Blocked: M1 does not load appearances",
            "final_holdout_readiness": "Not established; current sample has been explored",
        },
    }
