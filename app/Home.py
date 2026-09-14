"""M0 landing page: project context without data loading or model training."""

import streamlit as st
from pydantic import ValidationError

from pl_analytics.config import get_settings

st.set_page_config(
    page_title="Premier League Analytics Lab",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Premier League Analytics Lab")
st.write(
    """
    Portfolio-grade football analytics: player performance, market valuation,
    scouting and probabilistic match prediction.
    """
)

st.info(
    "M0 — Repository foundation. Data and model artifacts are not built yet. "
    "Analytics and predictions will become available in later milestones."
)

try:
    settings = get_settings()
except ValidationError:
    st.error(
        "Configuration is invalid. Check .env and environment variables, then restart the app."
    )
    st.stop()

st.caption("Configured competition scope: " + ", ".join(settings.active_competitions))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Player analytics", "Planned")
with col2:
    st.metric("Market value model", "Planned")
with col3:
    st.metric("Match predictor", "Planned")

st.subheader("Methodology")
st.markdown(
    """
    - Time-aware validation and leakage controls
    - Elo, Poisson and Dixon-Coles baselines
    - Regression and gradient boosting
    - Calibration and uncertainty
    - SHAP explanations
    - PostgreSQL analytical layer
    """
)

st.subheader("How to interpret future results")
st.write(
    "Observed performance describes recorded matches. Predicted market values and match "
    "probabilities are estimates with uncertainty, not guarantees. Market value is an "
    "editorial estimate and differs from an actual transfer fee."
)
st.caption(
    "The dashboard uses prepared artifacts. This foundation page does not download data, "
    "connect to PostgreSQL, or train models."
)
