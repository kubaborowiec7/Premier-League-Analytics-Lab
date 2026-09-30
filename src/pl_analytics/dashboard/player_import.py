"""Session-only CSV and manual player import controls."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from pl_analytics.config import Settings
from pl_analytics.dashboard.data import report
from pl_analytics.features.advanced_players import METRICS, PROFILES
from pl_analytics.features.player_import import (
    manual_performance_frame,
    performance_rows,
    performance_template,
    value_feature_rows,
)
from pl_analytics.features.value import FEATURES as VALUE_FEATURES
from pl_analytics.models.value import evaluate_model

PERFORMANCE_STATE = "session_imported_performance_players"
VALUE_STATE = "session_imported_value_players"


def _stored(key: str) -> pd.DataFrame:
    value = st.session_state.get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _save(key: str, frame: pd.DataFrame) -> None:
    current = _stored(key)
    combined = pd.concat([current, frame], ignore_index=True)
    st.session_state[key] = combined.drop_duplicates("player_id", keep="last")


def _performance_manual(
    *, prefix: str, competition_id: str, season: str, reference_date: str
) -> None:
    role = st.selectbox(
        "Position group", ["GK", "DEF", "MID", "FWD"], index=1, key=f"{prefix}-role"
    )
    defaults = [name for name in PROFILES[role] if name in METRICS][:8]
    selected_metrics = st.multiselect(
        "Statistics to enter",
        list(METRICS),
        default=defaults,
        format_func=lambda name: METRICS[name].label,
        key=f"{prefix}-metrics",
    )
    basis = st.radio(
        "Count-stat basis",
        ["Per 90", "Total"],
        horizontal=True,
        key=f"{prefix}-basis",
        help="Percentage statistics always use their displayed 0–100 scale.",
    )
    with st.form(f"{prefix}-manual-form"):
        left, middle, right = st.columns(3)
        player_name = left.text_input("Player name")
        club_name = middle.text_input("Club (optional)")
        minutes = right.number_input("Minutes", min_value=1, max_value=6000, value=900, step=30)
        appearances = left.number_input(
            "Appearances", min_value=1, max_value=80, value=10, step=1
        )
        values: dict[str, float] = {}
        for name in selected_metrics:
            metric = METRICS[name]
            maximum = 100.0 if metric.kind != "count" else None
            label = metric.label + (" / 90" if basis == "Per 90" and metric.kind == "count" else "")
            values[name] = st.number_input(
                label,
                min_value=0.0,
                max_value=maximum,
                value=0.0,
                step=0.01,
                key=f"{prefix}-value-{name}",
            )
        submitted = st.form_submit_button("Add player to this session", type="primary")
    if submitted:
        try:
            raw = manual_performance_frame(
                {
                    "player_name": player_name,
                    "competition_id": competition_id,
                    "season": season,
                    "position_group": role,
                    "club_name": club_name,
                    "minutes": minutes,
                    "appearances": appearances,
                },
                values,
                basis=basis,
            )
            parsed = performance_rows(
                raw,
                competition_id=competition_id,
                season=season,
                reference_date=reference_date,
            )
            _save(PERFORMANCE_STATE, parsed)
            st.query_params["player"] = parsed.player_id.iloc[0]
            st.success(f"Added {parsed.player_name.iloc[0]} to the session comparison.")
        except ValueError as error:
            st.error(str(error))


def _performance_csv(
    *, prefix: str, competition_id: str, season: str, reference_date: str
) -> None:
    st.download_button(
        "Download CSV template",
        performance_template().to_csv(index=False),
        file_name="player-performance-template.csv",
        mime="text/csv",
        key=f"{prefix}-template",
    )
    uploaded = st.file_uploader("Player statistics CSV", type=["csv"], key=f"{prefix}-upload")
    if st.button("Import CSV", disabled=uploaded is None, key=f"{prefix}-csv-submit"):
        try:
            raw = pd.read_csv(uploaded)
            parsed = performance_rows(
                raw,
                competition_id=competition_id,
                season=season,
                reference_date=reference_date,
            )
            _save(PERFORMANCE_STATE, parsed)
            st.query_params["player"] = parsed.player_id.iloc[0]
            st.success(f"Imported {len(parsed)} player(s) into this session.")
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"CSV import failed: {error}")


def performance_importer(
    frame: pd.DataFrame, *, competition_id: str, season: str, prefix: str
) -> pd.DataFrame:
    """Render performance import controls and return applicable session rows."""
    reference_date = str(frame.reference_date.iloc[0])
    with st.expander("Import a player for league comparison"):
        st.caption(
            "CSV and manual inputs stay in this browser session. They are never written to raw "
            "data or used to retrain a model. Missing statistics remain unavailable; supplied "
            "statistics are treated as fully covered unless the CSV sets a coverage column."
        )
        method = st.segmented_control(
            "Input method", ["Manual", "CSV"], default="Manual", key=f"{prefix}-method"
        )
        if method == "CSV":
            _performance_csv(
                prefix=prefix,
                competition_id=competition_id,
                season=season,
                reference_date=reference_date,
            )
        else:
            _performance_manual(
                prefix=prefix,
                competition_id=competition_id,
                season=season,
                reference_date=reference_date,
            )
        imports = _stored(PERFORMANCE_STATE)
        if not imports.empty:
            st.dataframe(
                imports[["player_name", "competition_id", "season", "position_group", "minutes"]],
                hide_index=True,
                width="stretch",
            )
            if st.button("Clear imported performance players", key=f"{prefix}-clear"):
                st.session_state.pop(PERFORMANCE_STATE, None)
                st.rerun()
    imports = _stored(PERFORMANCE_STATE)
    if imports.empty:
        return imports
    return imports.loc[
        imports.competition_id.eq(competition_id) & imports.season.eq(season)
    ].copy()


def performance_session_rows() -> pd.DataFrame:
    """Return every performance player currently stored in the Streamlit session."""
    return _stored(PERFORMANCE_STATE)


@st.cache_resource(show_spinner=False)
def _trusted_value_bundle(path: str, expected_digest: str) -> dict:
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Frozen value-model checksum does not match its selection record")
    bundle = joblib.load(source)
    if not isinstance(bundle, dict) or not {"models", "chosen", "quantile"} <= set(bundle):
        raise ValueError("Frozen value-model bundle is incompatible")
    return bundle


def _predict_values(rows: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    selection = report(settings.artifact_dir / "m5/selection.json")
    model_path = settings.artifact_dir / "m5/models.joblib"
    if not selection or not model_path.exists():
        raise ValueError("Frozen M5 model artifacts are unavailable")
    if selection.get("config", {}).get("competition_id") != rows.competition_id.iloc[0]:
        raise ValueError("The frozen value model does not cover the selected competition")
    bundle = _trusted_value_bundle(str(model_path.resolve()), selection["bundle_sha256"])
    chosen = next(model for model in bundle["models"] if model.name == bundle["chosen"])
    floor = float(selection["config"]["interval_scale_floor_eur"])
    result = evaluate_model(chosen, rows, float(bundle["quantile"]), floor)
    result["is_user_import"] = True
    return result


def _value_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_name",
            "competition_id",
            "season",
            "valuation_date",
            "market_value_eur",
            *VALUE_FEATURES,
        ]
    )


def _value_manual(
    prefix: str, competition_id: str, default_season: str, settings: Settings
) -> None:
    with st.form(f"{prefix}-value-manual"):
        first, second, third = st.columns(3)
        player_name = first.text_input("Player name")
        season = second.text_input("Season", value=default_season)
        valuation_date = third.date_input("Valuation date")
        market_value = first.number_input(
            "Observed editorial value (€m)", min_value=0.01, value=10.0, step=0.5
        )
        age = second.number_input("Age", min_value=15.0, max_value=50.0, value=24.0, step=0.1)
        previous_value = third.number_input(
            "Previous editorial value (€m)", min_value=0.0, value=8.0, step=0.5
        )
        days_value = first.number_input(
            "Days since previous valuation", min_value=0, max_value=3000, value=180
        )
        days_appearance = second.number_input(
            "Days since last appearance", min_value=0, max_value=365, value=7
        )
        appearances = third.number_input(
            "Appearances in prior 365 days", min_value=0, max_value=50, value=20
        )
        minutes = first.number_input(
            "Minutes in prior 365 days", min_value=0, max_value=3420, value=1500, step=30
        )
        goals = second.number_input("Goals / 90", min_value=0.0, max_value=10.0, value=0.1)
        assists = third.number_input("Assists / 90", min_value=0.0, max_value=10.0, value=0.1)
        submitted = st.form_submit_button("Estimate and add player", type="primary")
    if submitted:
        raw = pd.DataFrame(
            [
                {
                    "player_name": player_name,
                    "competition_id": competition_id,
                    "season": season,
                    "valuation_date": valuation_date,
                    "market_value_eur": market_value * 1e6,
                    "age_years": age,
                    "previous_value_eur": previous_value * 1e6,
                    "days_since_valuation": days_value,
                    "days_since_appearance": days_appearance,
                    "appearances_365": appearances,
                    "minutes_365": minutes,
                    "goals_per90_365": goals,
                    "assists_per90_365": assists,
                }
            ]
        )
        try:
            parsed = value_feature_rows(
                raw, competition_id=competition_id, default_season=default_season
            )
            _save(VALUE_STATE, _predict_values(parsed, settings))
            st.success(f"Estimated and added {player_name} to this session.")
        except (KeyError, StopIteration, ValueError) as error:
            st.error(str(error))


def _value_csv(prefix: str, competition_id: str, default_season: str, settings: Settings) -> None:
    st.download_button(
        "Download valuation CSV template",
        _value_template().to_csv(index=False),
        file_name="player-value-template.csv",
        mime="text/csv",
        key=f"{prefix}-value-template",
    )
    uploaded = st.file_uploader(
        "Valuation feature CSV", type=["csv"], key=f"{prefix}-value-upload"
    )
    if st.button(
        "Estimate imported players", disabled=uploaded is None, key=f"{prefix}-value-submit"
    ):
        try:
            parsed = value_feature_rows(
                pd.read_csv(uploaded),
                competition_id=competition_id,
                default_season=default_season,
            )
            _save(VALUE_STATE, _predict_values(parsed, settings))
            st.success(f"Estimated and added {len(parsed)} player(s) to this session.")
        except (
            KeyError,
            OSError,
            StopIteration,
            UnicodeError,
            pd.errors.ParserError,
            ValueError,
        ) as error:
            st.error(f"CSV import failed: {error}")


def value_importer(
    *, settings: Settings, competition_id: str, default_season: str, prefix: str
) -> pd.DataFrame:
    """Render market-value scenario controls and return applicable session rows."""
    with st.expander("Import a player for market-value comparison"):
        st.warning(
            "This runs the frozen M5 Ridge benchmark on user-supplied inputs. It estimates the "
            "next editorial value in the historical experiment design; it is not a transfer fee "
            "or a current market quote. Inputs are not independently verified."
        )
        method = st.segmented_control(
            "Input method", ["Manual", "CSV"], default="Manual", key=f"{prefix}-value-method"
        )
        if method == "CSV":
            _value_csv(prefix, competition_id, default_season, settings)
        else:
            _value_manual(prefix, competition_id, default_season, settings)
        imports = _stored(VALUE_STATE)
        if not imports.empty:
            st.dataframe(
                imports[
                    [
                        "player_name",
                        "season",
                        "market_value_eur",
                        "predicted_value_eur",
                        "lower_eur",
                        "upper_eur",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            if st.button("Clear imported valuation players", key=f"{prefix}-value-clear"):
                st.session_state.pop(VALUE_STATE, None)
                st.rerun()
    imports = _stored(VALUE_STATE)
    if imports.empty:
        return imports
    return imports.loc[imports.competition_id.eq(competition_id)].copy()
