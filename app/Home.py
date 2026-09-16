"""Analytics dashboard overview."""

import streamlit as st

from pl_analytics.dashboard.ui import PAGES, render

st.navigation(
    [
        st.Page(lambda: render("Overview"), title="Overview", default=True),
        *[st.Page(path, title=name) for name, path in PAGES.items() if name != "Overview"],
    ]
).run()
