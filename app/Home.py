"""Analytics dashboard entry point with grouped navigation."""

import streamlit as st

from pl_analytics.dashboard.ui import PAGES, SECTIONS, render


def matchday_hub() -> None:
    render("Matchday Hub")


st.navigation(
    {
        section: [
            st.Page(matchday_hub, title=name, default=True, icon=":material/stadium:")
            if PAGES[name] == "Home.py"
            else st.Page(PAGES[name], title=name)
            for name in names
        ]
        for section, names in SECTIONS.items()
    }
).run()
