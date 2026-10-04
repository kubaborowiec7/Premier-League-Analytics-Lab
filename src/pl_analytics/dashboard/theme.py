"""Shared visual system: palette, Plotly template, page CSS and small HTML components.

Every page calls :func:`apply` once. Colours carry one meaning across the app:
accent/teal = good or a win, coral = bad or a loss, amber = warning, blue/slate/orange =
home/draw/away. Components return HTML strings for ``st.markdown(..., unsafe_allow_html=True)``;
all interpolated text is escaped.
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BACKGROUND = "#0E1117"
SIDEBAR = "#0A0D14"
CARD = "#161B26"
CARD_RAISED = "#1B2130"
BORDER = "#262D3B"
GRIDLINE = "#222938"
TEXT = "#E6EAF2"
TEXT_SOFT = "#C9D0DD"
MUTED = "#8E98AD"
ACCENT = "#22D3A6"
ACCENT_TEXT = "#7FE9CC"
ON_ACCENT = "#0B1F18"
BAD = "#F2685A"
BAD_TEXT = "#F59A90"
ON_BAD = "#2A0D09"
WARNING = "#F5B544"
HOME = "#2563EB"
DRAW = "#4A5263"
AWAY = "#F59E0B"
ON_AWAY = "#1A1206"

# Ordered for multi-series charts: distinct in hue and lightness on a dark ground.
SERIES = ("#22D3A6", "#7AA7FF", "#F5B544", "#F2685A", "#C084FC", "#94A3B8")
# Diverging scale for "below expectation ← 0 → above expectation".
DIVERGING = [[0.0, BAD], [0.5, "#2A3142"], [1.0, ACCENT]]
SEQUENTIAL = [[0.0, "#12221F"], [1.0, ACCENT]]

# Fixture difficulty 1 (easiest) – 5 (hardest): (background, text). The number is always shown
# with the colour so the scale never relies on hue alone.
FDR_COLOURS: dict[int, tuple[str, str]] = {
    1: (ACCENT, ON_ACCENT),
    2: ("#17806A", "#FFFFFF"),
    3: ("#3A4254", TEXT),
    4: ("#B4553F", "#FFFFFF"),
    5: (BAD, ON_BAD),
}

HEADING_FONT = "'Barlow Condensed', 'Arial Narrow', sans-serif"
BODY_FONT = "'Archivo', system-ui, sans-serif"

# (background, text) by FPL short name. Approximate club colours, darkened where needed
# so the three-letter code keeps readable contrast. No crests or marks are used.
CLUB_COLOURS: dict[str, tuple[str, str]] = {
    "ARS": ("#D50A0A", "#FFFFFF"),
    "AVL": ("#670E36", "#FFFFFF"),
    "BOU": ("#B50E12", "#FFFFFF"),
    "BRE": ("#C8070F", "#FFFFFF"),
    "BHA": ("#0057B8", "#FFFFFF"),
    "BUR": ("#6C1D45", "#FFFFFF"),
    "CHE": ("#034694", "#FFFFFF"),
    "COV": ("#59CBE8", "#062C3B"),
    "CRY": ("#1B458F", "#FFFFFF"),
    "EVE": ("#003399", "#FFFFFF"),
    "FUL": ("#F2F2F2", "#111111"),
    "HUL": ("#F5A12D", "#1A1206"),
    "IPS": ("#0044A9", "#FFFFFF"),
    "LEE": ("#F2F2F2", "#1D428A"),
    "LEI": ("#003090", "#FFFFFF"),
    "LIV": ("#C8102E", "#FFFFFF"),
    "LUT": ("#F78F1E", "#1A1206"),
    "MCI": ("#6CABDD", "#08243A"),
    "MUN": ("#C70101", "#FFFFFF"),
    "NEW": ("#241F20", "#FFFFFF"),
    "NFO": ("#C10000", "#FFFFFF"),
    "SHU": ("#B30A0A", "#FFFFFF"),
    "SOU": ("#C8102E", "#FFFFFF"),
    "SUN": ("#D11124", "#FFFFFF"),
    "TOT": ("#132257", "#FFFFFF"),
    "WHU": ("#7A263A", "#FFFFFF"),
    "WOL": ("#FDB913", "#1A1206"),
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
  font-family: {BODY_FONT};
}}
h1, h2, h3, [data-testid="stHeading"] * {{
  font-family: {HEADING_FONT} !important; letter-spacing: .01em;
}}
.block-container {{ max-width: 1440px; padding-top: 2.5rem; padding-bottom: 3rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
[data-testid="stMetric"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.1rem;
}}
[data-testid="stMetricLabel"] p {{ white-space: normal; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2rem; }}
[data-testid="stMetricValue"] div {{ white-space: normal; overflow-wrap: anywhere; }}
[data-testid="stExpander"] details {{ background: {CARD}; border-radius: 14px; }}
.pl-eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .12em; color: {ACCENT_TEXT};
  text-transform: uppercase; }}
.pl-title {{ margin: 2px 0 12px; font-family: {HEADING_FONT}; font-weight: 700; font-size: 44px;
  line-height: 1; color: {TEXT}; }}
.pl-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 20px;
  margin-bottom: 16px; color: {TEXT}; }}
.pl-card-title {{ margin: 0 0 12px; font-family: {HEADING_FONT}; font-size: 24px; font-weight: 700;
  color: {TEXT}; }}
.pl-muted {{ color: {MUTED}; font-size: 12px; }}
.pl-stat {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px;
  display: flex; flex-direction: column; gap: 6px; height: 100%; box-sizing: border-box;
  container-type: inline-size; }}
.pl-stat .label {{ font-size: 12px; color: {MUTED}; }}
/* The value shrinks with its card so a long club name stays on one line. */
.pl-stat .value {{ font-family: {HEADING_FONT}; font-weight: 700;
  font-size: clamp(20px, 15cqi, 34px); line-height: 1; color: {TEXT}; }}
.pl-narrow {{ display: none; }}
/* The Plotly toolbar overlaps chart content on phones and adds nothing these charts need. */
.js-plotly-plot .modebar {{ display: none !important; }}
/* Stat cards share one grid so a phone shows them two across instead of one long column. */
.pl-stats {{ display: grid; gap: 12px; margin-bottom: 14px;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); }}
@media (max-width: 640px) {{
  .pl-stats {{ gap: 8px; }}
  .pl-stat {{ padding: 12px 14px; }}
  .pl-card {{ padding: 14px 14px; }}
  .pl-title {{ font-size: 34px; }}
  /* Secondary table columns are dropped on phones; the key ones stay without scrolling. */
  .pl-wide {{ display: none; }}
  .pl-narrow {{ display: inline !important; }}
  .block-container {{ padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 2rem; }}
}}
.pl-stat.tight {{ padding: 12px; }}
.pl-stat.tight .value {{ font-size: clamp(17px, 26cqi, 34px); }}
.pl-stat .sub {{ font-size: 12px; color: {TEXT_SOFT}; }}
.pl-chip {{ display: inline-flex; align-items: center; justify-content: center; flex: none;
  width: 38px; height: 22px; border-radius: 6px; font-family: {HEADING_FONT}; font-weight: 700;
  font-size: 13px; letter-spacing: .04em; box-shadow: inset 0 0 0 1px rgba(255,255,255,.16); }}
.pl-pill {{ display: inline-flex; width: 18px; height: 18px; border-radius: 4px;
  align-items: center; justify-content: center; font-size: 10px; font-weight: 700;
  margin-left: 3px; }}
.pl-bar {{ display: flex; height: 24px; border-radius: 6px; overflow: hidden;
  font-family: {HEADING_FONT}; font-weight: 700; font-size: 14px; }}
.pl-bar span {{ display: flex; align-items: center; justify-content: center; min-width: 0; }}
.pl-badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px;
  font-weight: 700; }}
.pl-fixtures {{ overflow-x: auto; container-type: inline-size; }}
.pl-fixtures .grid {{ display: grid; min-width: 540px;
  grid-template-columns: 56px minmax(0,1fr) minmax(170px,240px) minmax(0,1fr) 44px;
  align-items: center; gap: 12px; }}
.pl-fixtures .head {{ padding: 0 12px 6px; font-size: 11px; font-weight: 600; letter-spacing: .1em;
  color: {MUTED}; }}
.pl-fixtures .row {{ padding: 8px 12px; margin-bottom: 6px; border-radius: 10px;
  background: {CARD_RAISED}; min-height: 36px; }}
.pl-fixtures .team {{ display: flex; align-items: center; gap: 10px; min-width: 0;
  font-weight: 600; }}
.pl-fixtures .team span.name {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
/* Narrow cards keep the club chips and drop names rather than truncating them to a letter. */
@container (max-width: 640px) {{
  .pl-fixtures .team span.name {{ display: none; }}
  .pl-fixtures .grid {{ min-width: 380px;
    grid-template-columns: 56px 38px minmax(170px,1fr) 38px 44px; }}
}}
/* Phones: no sideways scroll; the likely-score column gives way to the probability bar. */
@container (max-width: 400px) {{
  .pl-fixtures .grid {{ min-width: 0; gap: 8px;
    grid-template-columns: 44px 38px minmax(110px,1fr) 38px; }}
  .pl-fixtures .grid > :last-child {{ display: none; }}
  .pl-fixtures .row, .pl-fixtures .head {{ padding-left: 8px; padding-right: 8px; }}
}}
.pl-table {{ width: 100%; border-collapse: collapse; font-size: 13px;
  font-variant-numeric: tabular-nums; }}
.pl-table th {{ color: {MUTED}; font-size: 11px; letter-spacing: .06em; font-weight: 600;
  padding: 0 4px 8px; text-align: right; border: none; }}
.pl-table td {{ padding: 6px 4px; border: none; border-top: 1px solid {GRIDLINE}; text-align: right;
  color: {TEXT_SOFT}; }}
.pl-table th:nth-child(-n+2), .pl-table td:nth-child(-n+2) {{ text-align: left; }}
.pl-list-row {{ display: flex; align-items: center; gap: 10px; padding: 7px 0;
  border-top: 1px solid {GRIDLINE}; }}
.pl-list-row .grow {{ flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }}
.pl-list-row .big {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 18px; }}
.pl-fdr {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 6px;
  padding: 2px 6px; font-size: 11.5px; font-weight: 700; white-space: nowrap; }}
.pl-range {{ position: relative; height: 6px; border-radius: 3px; background: {GRIDLINE};
  min-width: 60px; }}
.pl-range i {{ position: absolute; top: 0; bottom: 0; border-radius: 3px;
  background: rgba(34,211,166,.45); }}
.pl-range b {{ position: absolute; top: -3px; width: 3px; height: 12px; border-radius: 2px;
  background: {TEXT}; }}
.pl-grid {{ display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
.pl-pick {{ background: {CARD_RAISED}; border-radius: 12px; padding: 12px 14px; display: flex;
  flex-direction: column; gap: 8px; box-sizing: border-box; min-width: 0; }}
.pl-pick .top {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
.pl-pick .name {{ font-weight: 600; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.pl-pick .pts {{ margin-left: auto; font-family: {HEADING_FONT}; font-weight: 700; font-size: 26px;
  line-height: 1; }}
.pl-pitch {{ background: linear-gradient(180deg, #133B2D, #0F2E24); border: 1px solid #1F4A3A;
  border-radius: 14px; padding: 18px 6px; display: flex; flex-direction: column; gap: 18px; }}
.pl-pitch .line {{ display: flex; justify-content: space-evenly; gap: 4px; }}
.pl-tile {{ position: relative; width: 96px; min-width: 0; display: flex; flex-direction: column;
  align-items: center; gap: 3px; text-align: center; }}
.pl-tile .nm {{ max-width: 96px; font-size: 12.5px; font-weight: 600; color: {TEXT};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pl-tile .fx {{ font-size: 11px; color: {TEXT_SOFT}; white-space: nowrap; }}
.pl-tile .pts {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 17px; line-height: 1.2;
  padding: 0 8px; border-radius: 6px; background: rgba(0,0,0,.38); color: {TEXT}; }}
.pl-tile .cap {{ position: absolute; top: -7px; right: 12px; width: 18px; height: 18px;
  border-radius: 50%; background: {ACCENT}; color: {ON_ACCENT}; font-size: 11px; font-weight: 800;
  display: flex; align-items: center; justify-content: center; }}
.pl-tile.doubt .pts {{ box-shadow: inset 0 0 0 1px {WARNING}; }}
.pl-bench {{ margin-top: 10px; padding: 12px 6px; border-radius: 12px; background: {CARD_RAISED};
  display: flex; justify-content: space-evenly; gap: 4px; }}
.pl-ticker {{ overflow-x: auto; }}
.pl-ticker .grid {{ display: grid; gap: 4px; align-items: stretch; min-width: 520px; }}
.pl-ticker .club {{ display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 13px;
  min-width: 0; white-space: nowrap; }}
.pl-ticker .head {{ font-size: 11px; font-weight: 600; letter-spacing: .08em; color: {MUTED};
  text-align: center; padding-bottom: 4px; }}
.pl-ticker .cell {{ display: flex; flex-direction: column; justify-content: center; gap: 2px;
  border-radius: 6px; padding: 5px 4px; text-align: center; font-size: 12px; font-weight: 700;
  min-height: 30px; }}
.pl-ticker .avg {{ display: flex; align-items: center; justify-content: center;
  font-family: {HEADING_FONT}; font-weight: 700; font-size: 16px; color: {TEXT_SOFT}; }}
.pl-h2h {{ display: grid;
  grid-template-columns: 40px minmax(28px,1fr) minmax(64px,auto) minmax(28px,1fr) 40px;
  gap: 8px; align-items: center; padding: 6px 0; border-top: 1px solid {GRIDLINE};
  font-size: 13px; font-variant-numeric: tabular-nums; }}
.pl-h2h .label {{ text-align: center; color: {MUTED}; font-size: 12px; }}
.pl-h2h .bar {{ height: 8px; border-radius: 4px; background: {GRIDLINE}; display: flex; }}
.pl-h2h .bar i {{ height: 100%; border-radius: 4px; background: {DRAW}; }}
.pl-h2h .bar.better i {{ background: {ACCENT}; }}
.pl-h2h .bar.left {{ justify-content: flex-end; }}
.pl-swap {{ display: grid; grid-template-columns: minmax(0,1fr) 22px minmax(0,1fr) auto; gap: 8px;
  align-items: center; padding: 8px 0; border-top: 1px solid {GRIDLINE}; }}
.pl-swap .who {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
.pl-swap .who span.n {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style>
"""


def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Archivo, system-ui, sans-serif", "color": TEXT_SOFT, "size": 13},
        title={"font": {"family": "Barlow Condensed, sans-serif", "size": 22, "color": TEXT}},
        colorway=list(SERIES),
        xaxis={"gridcolor": GRIDLINE, "zerolinecolor": BORDER, "linecolor": BORDER},
        yaxis={"gridcolor": GRIDLINE, "zerolinecolor": BORDER, "linecolor": BORDER},
        polar={
            "bgcolor": CARD,
            "radialaxis": {"gridcolor": BORDER, "linecolor": BORDER},
            "angularaxis": {"gridcolor": BORDER, "linecolor": BORDER},
        },
        legend={"bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": CARD_RAISED, "bordercolor": BORDER, "font": {"color": TEXT}},
        colorscale={"sequential": SEQUENTIAL, "diverging": DIVERGING},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    pio.templates["analytics_lab"] = template


TEMPLATE = "plotly_dark+analytics_lab"
_register_plotly_template()


def apply() -> None:
    """Inject page CSS and make the shared Plotly template the default."""
    pio.templates.default = TEMPLATE
    st.markdown(_CSS, unsafe_allow_html=True)


def club_chip(short_name: object) -> str:
    # Missing codes arrive as None/NaN/NA, whose truth value is ambiguous.
    code = short_name.upper()[:3] if isinstance(short_name, str) and short_name else "?"
    background, foreground = CLUB_COLOURS.get(code, (DRAW, "#FFFFFF"))
    return (
        f'<span class="pl-chip" style="background:{background};color:{foreground}">'
        f"{escape(code)}</span>"
    )


_FORM = {"W": (ACCENT, ON_ACCENT), "D": ("#3A4254", TEXT), "L": (BAD, ON_BAD)}


def form_pills(form: object) -> str:
    """Render a W/D/L string, oldest first; letters keep the meaning without colour."""
    pills = []
    for letter in str(form or "")[-5:]:
        if letter not in _FORM:
            continue
        background, foreground = _FORM[letter]
        pills.append(
            f'<span class="pl-pill" style="background:{background};color:{foreground}">'
            f"{letter}</span>"
        )
    return "".join(pills)


def probability_bar(home: float, draw: float, away: float) -> str:
    """Three-segment home/draw/away bar with whole-percent labels summing to 100."""
    total = home + draw + away
    if total <= 0:
        return ""
    h = round(100 * home / total)
    a = round(100 * away / total)
    d = 100 - h - a
    segments = ((h, HOME, "#FFFFFF"), (d, DRAW, "#FFFFFF"), (a, AWAY, ON_AWAY))
    inner = "".join(
        f'<span style="width:{value}%;background:{background};color:{foreground}">'
        f"{value if value >= 9 else ''}{'%' if value >= 9 else ''}</span>"
        for value, background, foreground in segments
    )
    return (
        f'<div class="pl-bar" role="img" aria-label="Home {h}%, draw {d}%, away {a}%">{inner}</div>'
    )


def stat_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="pl-stat"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span>'
        f'<span class="sub">{escape(sub)}</span></div>'
    )


def stat_row(cards: object) -> None:
    """Render ``(label, value, sub)`` cards in one responsive grid."""
    st.markdown(
        '<div class="pl-stats">' + "".join(stat_card(*card) for card in cards) + "</div>",
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "warning") -> str:
    colours = {
        "warning": ("#3A2A0B", WARNING),
        "bad": ("#3A1512", BAD_TEXT),
        "good": ("#10302A", ACCENT_TEXT),
    }
    background, foreground = colours.get(tone, colours["warning"])
    return (
        f'<span class="pl-badge" style="background:{background};color:{foreground}">'
        f"{escape(text)}</span>"
    )


def page_header(title: str, eyebrow: str = "") -> None:
    """Consistent page heading; replaces st.title so every page shares one look."""
    eyebrow_html = f'<div class="pl-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f'{eyebrow_html}<div class="pl-title" role="heading" aria-level="1">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str) -> str:
    return (
        '<div class="pl-card"><div class="pl-card-title" role="heading" aria-level="2">'
        f"{escape(title)}</div>{body_html}</div>"
    )


def fixture_chip(label: object, level: object) -> str:
    """Opponent label on the five-step fixture-difficulty colour."""
    background, foreground = FDR_COLOURS.get(int(level), FDR_COLOURS[3])
    return (
        f'<span class="pl-fdr" style="background:{background};color:{foreground}">'
        f"{escape(str(label))}</span>"
    )


def range_bar(low: float, value: float, high: float, scale: float) -> str:
    """Indicative low–high range with a marker at the projection, on a shared 0–scale axis."""
    scale = max(float(scale), 1e-9)
    left = 100 * min(max(low / scale, 0.0), 1.0)
    right = 100 * min(max(high / scale, 0.0), 1.0)
    marker = 100 * min(max(value / scale, 0.0), 1.0)
    return (
        f'<div class="pl-range" role="img" aria-label="Range {low:.1f} to {high:.1f}">'
        f'<i style="left:{left:.1f}%;width:{max(right - left, 1):.1f}%"></i>'
        f'<b style="left:calc({marker:.1f}% - 1px)"></b></div>'
    )
