"""
VitaTwin Dashboard — Streamlit UI  (v2 — Professional Edition)
Run: streamlit run ui/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from rag.rag_pipeline import _load_index, retrieve_by_query
from models.clinical_engine import screen_user
from models.assistant import MentalHealthAssistant

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VitaTwin — Mental Health Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  FULL PROFESSIONAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── 0. GOOGLE FONT IMPORT ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── 1. ROOT VARIABLES ── */
:root {
    --bg:         #0a0d13;
    --bg-card:    #111622;
    --bg-input:   #161d2e;
    --bg-hover:   #1a2235;
    --border:     #232d42;
    --border-hi:  #2e3d5c;
    --text-pri:   #e8edf5;
    --text-sec:   #b0bdd0;
    --text-muted: #6b7a96;
    --text-faint: #3a4560;
    --accent:     #4f8ef7;
    --accent-glow:rgba(79,142,247,0.18);
    --green:      #34d399;
    --yellow:     #fbbf24;
    --orange:     #f97316;
    --red:        #f43f5e;
    --purple:     #a78bfa;
    --font:       'Inter', -apple-system, sans-serif;
}

/* ── MATERIAL ICONS FONT OVERRIDE — prevents icon ligature names showing as text ── */
@font-face {
    font-family: 'Material Symbols Rounded';
    src: local(''), url('') format('woff2');
    font-display: block;
}
@font-face {
    font-family: 'Material Icons';
    src: local(''), url('') format('woff2');
    font-display: block;
}

/* ── 2. BASE WIPE ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container,
[class^="st-"], [class*=" st-"],
[class^="css-"], [class*=" css-"],
[class^="st-emotion-cache"],
[class*="st-emotion-cache"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background-color: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--text-sec) !important;
}

/* ── 3. SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section {
    background: linear-gradient(180deg, #0d1422 0%, #0a1020 100%) !important;
    border-right: 1px solid var(--border) !important;
    width: 220px !important;
}
[data-testid="stHeader"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── 4. TYPOGRAPHY ── */
h1,h2,h3,h4,h5,h6 { font-family: var(--font) !important; color: var(--text-pri) !important; font-weight: 600 !important; letter-spacing: -0.02em; }
p, li, label, small, caption, span { font-family: var(--font) !important; }
a { color: var(--accent) !important; text-decoration: none; }
code { background: var(--bg-hover) !important; color: #79c0ff !important; border-radius: 4px; padding: 1px 6px; font-size: 12px; }

/* ── 5. FORM CONTROLS ── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-sec) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: var(--text-faint) !important; }
[data-testid="stChatInputTextArea"] textarea,
[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-sec) !important;
    border-radius: 12px !important;
}

/* Selectbox */
[data-baseweb="select"] > div, [data-baseweb="select"] [role="combobox"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-sec) !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="menu"] ul {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 10px !important;
}
[data-baseweb="option"] { background: transparent !important; color: var(--text-sec) !important; padding: 9px 14px !important; }
[data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] {
    background: var(--bg-hover) !important; color: var(--text-pri) !important;
}

/* Slider */
[data-testid="stSlider"] p { color: var(--text-muted) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; }

/* Radio — not used for nav */
[data-testid="stRadio"] { display: none !important; }

/* ── NAV BUTTONS in sidebar — interactive hover ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    letter-spacing: 0.01em !important;
    transition: background 0.18s, color 0.18s, transform 0.15s, border-color 0.18s, box-shadow 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-hi) !important;
    color: var(--text-pri) !important;
    transform: translateX(4px) scale(1.02) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    font-size: 15px !important;
}

/* ── 6. MAIN CONTENT BUTTONS ── */
.stButton > button,
[data-testid="stButton"] button {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-sec) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    padding: 7px 15px !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover, [data-testid="stButton"] button:hover {
    background: var(--bg-input) !important;
    border-color: var(--accent) !important;
    color: var(--text-pri) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* ── 7. METRICS ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-hi) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetricValue"] { color: var(--text-pri) !important; font-weight: 700 !important; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }

/* ── 8. EXPANDER ── */
details, [data-testid="stExpander"], [data-testid="stExpander"] > details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
details:hover, [data-testid="stExpander"]:hover { border-color: var(--border-hi) !important; }
details > summary, [data-testid="stExpander"] summary {
    color: var(--text-sec) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
}
details > summary:hover { color: var(--text-pri) !important; }

/* ── 9. DATAFRAME ── */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] iframe {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── 10. ALERTS ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 10px !important;
}

/* ── 11. CHAT ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatMessage"]:hover { border-color: var(--border-hi) !important; }

/* ── 12. MISC ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
iframe { background: transparent !important; }
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }


/* Hide sidebar collapse button (shows "keyboard_double" tooltip) */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"] { display: none !important; }

/* Hide chat message avatars (face / smart_toy icon text) */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] [class*="avatarContainer"],
[data-testid="stChatMessage"] [class*="Avatar"] { display: none !important; }

/* Fix: "_arr" is the Material Symbols ligature "keyboard_arrow_right"
   rendered as text. Streamlit puts it in a <span> as the FIRST child
   of the <summary> element. We zero-size only that first span
   (the icon span), leaving the label text untouched. */

/* The icon span is always the first child of the summary div */
[data-testid="stExpander"] summary > div > span:first-child,
details > summary > div > span:first-child {
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    height: 0 !important;
    display: inline-block !important;
    overflow: hidden !important;
    opacity: 0 !important;
}

/* Also override the Material Symbols font itself for the private-use
   unicode range where all icon ligatures live (U+E000–U+F8FF) */
@font-face {
    font-family: 'Material Symbols Rounded';
    src: local('Arial');
    unicode-range: U+E000-U+F8FF;
}
@font-face {
    font-family: 'Material Icons';
    src: local('Arial');
    unicode-range: U+E000-U+F8FF;
}

/* ══════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════ */

/* ── SIDEBAR BRANDING ── */
.vt-brand {
    padding: 20px 16px 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.vt-brand-logo {
    font-size: 22px; font-weight: 800;
    color: var(--text-pri);
    letter-spacing: -0.03em;
    display: flex; align-items: center; gap: 8px;
}
.vt-brand-sub {
    font-size: 10px; font-weight: 500;
    color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-top: 4px; padding-left: 2px;
}
.vt-brand-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px rgba(52,211,153,0.6);
    display: inline-block; margin-left: 6px;
    animation: pulse 2.2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.6; transform:scale(0.85); }
}

/* ── NAV ITEMS ── */
.nav-section {
    padding: 6px 12px 2px;
    font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: var(--text-faint);
    margin-top: 12px;
}
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; margin: 2px 8px;
    border-radius: 8px; cursor: pointer;
    transition: background 0.18s, color 0.18s, transform 0.15s, box-shadow 0.2s;
    font-size: 14px; font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
    text-decoration: none;
}
.nav-item:hover {
    background: var(--bg-hover) !important;
    color: var(--text-pri) !important;
    transform: translateX(3px) scale(1.02) !important;
    border-color: var(--border) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}
.nav-item.active {
    background: linear-gradient(90deg, rgba(79,142,247,0.15), rgba(79,142,247,0.05)) !important;
    color: var(--accent) !important;
    border-color: rgba(79,142,247,0.3) !important;
    font-weight: 600 !important;
}
.nav-item .nav-icon {
    font-size: 16px; width: 20px; text-align: center;
    transition: transform 0.2s;
}
.nav-item:hover .nav-icon { transform: scale(1.2); }
.nav-badge {
    margin-left: auto; font-size: 10px; font-weight: 700;
    background: rgba(248,67,94,0.18); color: var(--red);
    border: 1px solid rgba(248,67,94,0.3);
    border-radius: 10px; padding: 1px 7px;
}

/* ── PAGE HEADER ── */
.page-hdr {
    margin-bottom: 20px; padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.page-hdr-title {
    font-size: 1.65rem; font-weight: 700;
    color: var(--text-pri); letter-spacing: -0.025em;
    display: flex; align-items: center; gap: 10px;
}
.page-hdr-sub {
    font-size: 13px; color: var(--text-muted);
    margin-top: 4px; font-weight: 400;
}
.page-hdr-badge {
    font-size: 11px; font-weight: 600;
    background: var(--accent-glow); color: var(--accent);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 20px; padding: 3px 10px;
    vertical-align: middle;
}

/* ── STAT CARDS ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 4px;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--card-accent, var(--accent));
    opacity: 0.7;
}
.stat-card:hover {
    border-color: var(--border-hi);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.sc-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 8px; }
.sc-value { font-size: 2rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.sc-sub   { font-size: 11px; color: var(--text-faint); margin-top: 6px; }

/* ── PILLS ── */
.pill {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.06em; margin-right: 5px; text-transform: uppercase;
    transition: transform 0.15s, box-shadow 0.15s;
}
.pill:hover { transform: scale(1.05); }
.pill-critical { background:rgba(244,63,94,0.12);  color:#f43f5e !important; border:1px solid rgba(244,63,94,0.3); }
.pill-high     { background:rgba(249,115,22,0.12); color:#f97316 !important; border:1px solid rgba(249,115,22,0.3); }
.pill-moderate { background:rgba(251,191,36,0.12); color:#fbbf24 !important; border:1px solid rgba(251,191,36,0.3); }
.pill-low      { background:rgba(52,211,153,0.12); color:#34d399 !important; border:1px solid rgba(52,211,153,0.3); }
.pill-blue     { background:rgba(79,142,247,0.12); color:#4f8ef7 !important; border:1px solid rgba(79,142,247,0.3); }
.pill-grey     { background:rgba(107,122,150,0.1); color:#6b7a96 !important; border:1px solid rgba(107,122,150,0.25); }
.pill-purple   { background:rgba(167,139,250,0.12);color:#a78bfa !important; border:1px solid rgba(167,139,250,0.3); }

/* ── SECTION HEADER ── */
.sec-hdr {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--accent);
    margin: 1.6rem 0 0.8rem; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.sec-hdr::before {
    content: ''; display: inline-block;
    width: 3px; height: 14px; border-radius: 2px;
    background: var(--accent); flex-shrink: 0;
}

/* ── METRIC BAR ── */
.mbar-wrap { background: var(--border); border-radius: 4px; height: 4px; margin-top: 8px; }
.mbar { height: 4px; border-radius: 4px; transition: width 0.6s ease; }

/* ── JOURNAL ENTRY ── */
.journal-entry {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 3px solid var(--border-hi);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    transition: border-color 0.2s, transform 0.15s;
}
.journal-entry:hover { border-left-color: var(--accent); transform: translateX(2px); }
.journal-date { font-size: 11px; color: var(--accent); font-weight: 700; margin-bottom: 6px; letter-spacing: 0.04em; }
.journal-text { color: var(--text-sec) !important; font-size: 13.5px; line-height: 1.65; }

/* ── SIGNAL ROW ── */
.signal-row {
    display: flex; align-items: center; gap: 10px;
    background: var(--bg-hover); border-radius: 8px;
    padding: 8px 12px; margin-bottom: 6px;
    border-left: 3px solid var(--border-hi);
    transition: border-color 0.2s;
}
.signal-name { font-size: 12px; font-family: monospace; color: #79c0ff; font-weight: 600; min-width: 140px; }
.signal-desc { font-size: 12px; color: var(--text-sec); flex: 1; }
.signal-sev  { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; padding: 2px 7px; border-radius: 10px; }
.sev-severe   { background:rgba(244,63,94,0.12);  color:#f43f5e; border:1px solid rgba(244,63,94,0.25); }
.sev-moderate { background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.25); }
.sev-mild     { background:rgba(52,211,153,0.12); color:#34d399; border:1px solid rgba(52,211,153,0.25); }

/* ── QUICK-Q BUTTON override ── */
.quick-q button {
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    height: auto !important;
}

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 70px 20px; }
.es-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
.es-title { font-size: 16px; color: var(--text-muted); font-weight: 600; margin-bottom: 6px; }
.es-hint  { font-size: 13px; color: var(--text-faint); }

/* ── SIDEBAR FOOTER ── */
.vt-footer {
    position: fixed; bottom: 0; left: 0; width: 220px;
    padding: 12px 16px; background: var(--bg);
    border-top: 1px solid var(--border);
    font-size: 10px; color: var(--text-faint);
    font-weight: 500; letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════
BG         = "rgba(0,0,0,0)"
GRID       = "#1a2235"
AXIS_LINE  = "#232d42"
TICK_C     = "#3a4560"
FONT_C     = "#6b7a96"
TITLE_C    = "#b0bdd0"

def base_layout(title="", height=280, yrange=None):
    l = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=FONT_C, size=12, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 16, b=8),
        legend=dict(bgcolor=BG, font=dict(color=FONT_C, size=11),
                    orientation="h", y=1.14, x=0),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE,
                   tickcolor=TICK_C, tickfont=dict(color=TICK_C, size=11)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE,
                   tickcolor=TICK_C, tickfont=dict(color=TICK_C, size=11),
                   **({"range": yrange} if yrange else {})),
    )
    if title:
        l["title"] = dict(text=title, font=dict(color=TITLE_C, size=13, family="Inter, sans-serif"), x=0, xanchor="left")
    return l

def rc(level):  # risk colour
    return {"critical":"#f43f5e","high":"#f97316","moderate":"#fbbf24","low":"#34d399"}.get(level,"#6b7a96")

def rp(level):  # risk pill css class
    return {"critical":"pill-critical","high":"pill-high","moderate":"pill-moderate","low":"pill-low"}.get(level,"pill-grey")

def bar_html(val, mx, color):
    pct = min(100, val/mx*100)
    return f'<div class="mbar-wrap"><div class="mbar" style="width:{pct:.0f}%;background:{color}"></div></div>'

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_data():
    _, meta = _load_index()
    return meta["users"]

@st.cache_resource
def get_assistant():
    return MentalHealthAssistant()

users     = load_data()
asst      = get_assistant()
user_map  = {u["user_id"]: u for u in users}
all_res   = [screen_user(u) for u in users]  # pre-compute

risk_counts = {"critical":0,"high":0,"moderate":0,"low":0}
for r in all_res:
    risk_counts[r.risk_level] += 1

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — session_state navigation (works reliably)
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "📊 Overview"

NAV_PAGES = [
    ("📊", "Overview",  "📊 Overview",  "Population metrics & risk overview"),
    ("👤", "Profile",   "👤 Profile",   "Individual user deep-dive"),
    ("🤖", "Assistant", "🤖 Assistant", "AI clinical Q&A chatbot"),
    ("🔍", "Search",    "🔍 Search",    "Semantic FAISS profile search"),
]

with st.sidebar:
    now = datetime.now().strftime("%H:%M")
    critical_count = risk_counts["critical"]

    # Brand block
    st.markdown(f"""
    <div class="vt-brand">
      <div class="vt-brand-logo">🧠 VitaTwin <span class="vt-brand-dot"></span></div>
      <div class="vt-brand-sub">Mental Health Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)

    for icon, label, key, desc in NAV_PAGES:
        is_active = st.session_state.page == key
        # Active page gets accent styling via inline style on the button
        btn_style = """
        <style>
        div[data-testid="stButton"]:has(button[aria-label="{key}"]) button {{
            background: linear-gradient(90deg,rgba(79,142,247,0.2),rgba(79,142,247,0.07)) !important;
            border-color: rgba(79,142,247,0.4) !important;
            color: #4f8ef7 !important;
        }}
        </style>""" if is_active else ""

        badge_txt = f" 🔴{critical_count}" if (label == "Overview" and critical_count > 0) else ""
        clicked = st.button(
            f"{icon}  {label}{badge_txt}",
            key=f"nav_btn_{label}",
            use_container_width=True,
            help=desc,
        )
        if clicked:
            st.session_state.page = key
            st.rerun()

        if is_active:
            st.markdown(f"""<style>
            div[data-testid="stButton"]:nth-of-type({NAV_PAGES.index((icon,label,key,desc))+2}) button {{
                background: linear-gradient(90deg,rgba(79,142,247,0.18),rgba(79,142,247,0.06)) !important;
                border-color: rgba(79,142,247,0.4) !important;
                color: #58a6ff !important;
                font-weight: 600 !important;
            }}</style>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:20px">
    <div class="nav-section">System</div>
    <div style="padding:8px 16px 60px;font-size:11px;color:var(--text-faint);line-height:1.9">
        <div>🟢 &nbsp;FAISS Index &nbsp;<span style="color:var(--green)">Online</span></div>
        <div>👥 &nbsp;{len(users)} profiles monitored</div>
        <div>⏰ &nbsp;Updated {now}</div>
        <div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--border)">{asst.llm_status}</div>
        <div style="margin-top:4px">{asst.memory_status}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="vt-footer">VitaTwin v2.0 &nbsp;·&nbsp; Clinical Prototype</div>""",
                unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — POPULATION OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-title">📊 Population Overview
        <span class="page-hdr-badge">LIVE</span>
      </div>
      <div class="page-hdr-sub">Monitoring {len(users)} users across {len(set(u['condition_label'] for u in users))} condition categories · VitaTwin Clinical Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    # KPI cards
    card_data = [
        ("Total Users",  len(users),              "#4f8ef7", "all cohorts",         "4f8ef7"),
        ("Critical",     risk_counts["critical"],  "#f43f5e", "immediate attention", "f43f5e"),
        ("High Risk",    risk_counts["high"],      "#f97316", "priority monitoring", "f97316"),
        ("Moderate",     risk_counts["moderate"],  "#fbbf24", "active monitoring",   "fbbf24"),
        ("Low Risk",     risk_counts["low"],       "#34d399", "routine check-in",    "34d399"),
    ]
    cols = st.columns(5)
    for col, (label, val, color, sub, hex_) in zip(cols, card_data):
        col.markdown(f"""
        <div class="stat-card" style="--card-accent:{color}">
            <div class="sc-label">{label}</div>
            <div class="sc-value" style="color:{color}">{val}</div>
            <div class="sc-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Risk Distribution</div>', unsafe_allow_html=True)
        pie_layout = base_layout(height=290)
        pie_layout.pop("xaxis",None); pie_layout.pop("yaxis",None)
        pie_layout["showlegend"] = False
        fig = go.Figure(go.Pie(
            labels=[k.capitalize() for k in risk_counts],
            values=list(risk_counts.values()),
            marker=dict(colors=[rc(k) for k in risk_counts],
                        line=dict(color=BG, width=3)),
            hole=0.6, textinfo="label+percent",
            textfont=dict(size=12, color="#e8edf5"),
            hovertemplate="<b>%{label}</b><br>%{value} users (%{percent})<extra></extra>",
        ))
        fig.update_layout(**pie_layout)
        fig.add_annotation(text=f"<b style='font-size:20px'>{len(users)}</b><br>users",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color="#e8edf5", family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-hdr">Condition Distribution</div>', unsafe_allow_html=True)
        cond_counts = {}
        for u in users:
            c = u["condition_label"].replace("_"," ").title()
            cond_counts[c] = cond_counts.get(c,0)+1
        palette = ["#f43f5e","#f97316","#fbbf24","#34d399","#4f8ef7","#a78bfa","#38bdf8","#fb7185"]
        bar_layout = base_layout(height=290)
        bar_layout["xaxis"]["tickangle"] = 28
        bar_layout["xaxis"]["tickfont"]["size"] = 10
        fig2 = go.Figure(go.Bar(
            x=list(cond_counts.keys()), y=list(cond_counts.values()),
            marker=dict(color=palette[:len(cond_counts)],
                        line=dict(color=BG, width=1.5)),
            hovertemplate="<b>%{x}</b><br>%{y} users<extra></extra>",
        ))
        fig2.update_layout(**bar_layout)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    st.markdown('<div class="sec-hdr">Stress vs Mood — All Users</div>', unsafe_allow_html=True)
    sdata = [{"User":u["name"],"ID":u["user_id"],
              "Avg Stress":u["aggregates"]["avg_stress_14d"],
              "Avg Mood":u["aggregates"]["avg_mood_14d"],
              "Risk":r.risk_level.capitalize(),"Score":r.overall_risk_score,
              "Condition":u["condition_label"].replace("_"," ").title()}
             for u,r in zip(users,all_res)]
    df = pd.DataFrame(sdata)
    cmap = {"Critical":"#f43f5e","High":"#f97316","Moderate":"#fbbf24","Low":"#34d399"}
    fig3 = px.scatter(df, x="Avg Stress", y="Avg Mood", color="Risk",
                      color_discrete_map=cmap, size="Score", size_max=24,
                      hover_data=["User","ID","Condition","Score"],
                      labels={"Avg Stress":"Avg Stress (14d)","Avg Mood":"Avg Mood (14d)"})
    fig3.update_layout(**base_layout(height=400))
    fig3.update_traces(marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)")))
    st.plotly_chart(fig3, use_container_width=True)

    # High-risk table
    st.markdown('<div class="sec-hdr">⚠️ Users Requiring Attention</div>', unsafe_allow_html=True)
    flagged = sorted([(u,r) for u,r in zip(users,all_res)
                      if r.risk_level in ("critical","high","moderate")],
                     key=lambda x: -x[1].overall_risk_score)
    if flagged:
        rows = [{"ID":u["user_id"],"Name":u["name"],"Occupation":u["occupation"],
                 "Risk":r.risk_level.upper(),"Score":r.overall_risk_score,
                 "Stress":round(u["aggregates"]["avg_stress_14d"],1),
                 "Mood":round(u["aggregates"]["avg_mood_14d"],1),
                 "Sleep":round(u["aggregates"]["avg_sleep_14d"],1)}
                for u,r in flagged[:15]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=340,
                     column_config={
                         "Score":  st.column_config.ProgressColumn("Score",  min_value=0, max_value=100, format="%d"),
                         "Stress": st.column_config.NumberColumn("Stress /10", format="%.1f"),
                         "Mood":   st.column_config.NumberColumn("Mood /10",   format="%.1f"),
                         "Sleep":  st.column_config.NumberColumn("Sleep h",    format="%.1f"),
                     })


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — INDIVIDUAL PROFILE
# ══════════════════════════════════════════════════════════════
elif page == "👤 Profile":
    user_opts = {f"{u['user_id']} — {u['name']}": u["user_id"] for u in users}
    uid  = user_opts[st.selectbox("Select user", list(user_opts.keys()))]
    user = user_map[uid]
    res  = screen_user(user)
    agg  = user["aggregates"]
    si   = user["social_indicators"]

    pcls = rp(res.risk_level)
    r_col = rc(res.risk_level)

    # Page header
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-title">👤 {user['name']}</div>
      <div style="margin-top:8px">
        <span class="pill pill-grey">{user['occupation']}</span>
        <span class="pill pill-grey">Age {user['age']}</span>
        <span class="pill pill-grey">{user['user_id']}</span>
        <span class="pill {pcls}">{res.risk_level.upper()} RISK</span>
        <span class="pill pill-blue">Score {res.overall_risk_score:.0f}/100</span>
        <span class="pill pill-purple">{user['condition_label'].replace('_',' ').title()}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # 5 quick metrics
    m_cols = st.columns(5)
    mdata = [
        ("Avg Mood",   agg["avg_mood_14d"],           10, "#4f8ef7"),
        ("Avg Stress", agg["avg_stress_14d"],         10, "#f43f5e"),
        ("Avg Sleep",  agg["avg_sleep_14d"],           9, "#fbbf24"),
        ("Avg Energy", agg["avg_energy_14d"],         10, "#34d399"),
        ("Work h/day", si["work_hours_daily"],        14, "#a78bfa"),
    ]
    for col, (lbl, val, mx, color) in zip(m_cols, mdata):
        col.markdown(f"""
        <div class="stat-card" style="--card-accent:{color}">
            <div class="sc-label">{lbl}</div>
            <div class="sc-value" style="color:{color};font-size:1.6rem">{val:.1f}</div>
            {bar_html(val, mx, color)}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # Trend charts
    st.markdown('<div class="sec-hdr">14-Day Trend Analysis</div>', unsafe_allow_html=True)
    dates       = [d["date"]  for d in user["mood_history"]]
    mood_vals   = [d["score"] for d in user["mood_history"]]
    stress_vals = [d["score"] for d in user["stress_scores"]]
    sleep_vals  = [d["hours"] for d in user["sleep_hours"]]
    energy_vals = [d["level"] for d in user["energy_levels"]]

    tc1, tc2 = st.columns(2)
    with tc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=mood_vals, name="Mood",
            line=dict(color="#4f8ef7", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.06)",
            hovertemplate="<b>%{x}</b><br>Mood: %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=dates, y=energy_vals, name="Energy",
            line=dict(color="#34d399", width=2),
            hovertemplate="<b>%{x}</b><br>Energy: %{y:.1f}<extra></extra>"))
        fig.update_layout(**base_layout("Mood & Energy", height=265, yrange=[0,10.5]))
        st.plotly_chart(fig, use_container_width=True)

    with tc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=stress_vals, name="Stress",
            line=dict(color="#f43f5e", width=2.5),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.06)",
            hovertemplate="<b>%{x}</b><br>Stress: %{y:.1f}<extra></extra>"))
        fig2.add_trace(go.Scatter(x=dates, y=sleep_vals, name="Sleep (h)",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="<b>%{x}</b><br>Sleep: %{y:.1f}h<extra></extra>"))
        fig2.update_layout(**base_layout("Stress & Sleep", height=265, yrange=[0,10.5]))
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart
    radar_vals = [
        agg["avg_mood_14d"],
        round(10 - agg["avg_stress_14d"], 2),
        round(agg["avg_sleep_14d"] / 9 * 10, 2),
        agg["avg_energy_14d"],
        round(si["social_interactions_per_week"] / 7 * 10, 2),
    ]
    cats = ["Mood","Low Stress","Sleep","Energy","Social"]
    fig_r = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(79,142,247,0.1)",
        line=dict(color="#4f8ef7", width=2.5),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig_r.update_layout(
        paper_bgcolor=BG, font=dict(color=FONT_C, family="Inter"), height=300,
        margin=dict(l=30,r=30,t=44,b=20), showlegend=False,
        title=dict(text="Wellness Radar", font=dict(color=TITLE_C, size=13, family="Inter"), x=0),
        polar=dict(bgcolor=BG,
                   radialaxis=dict(visible=True, range=[0,10], gridcolor=GRID,
                                   tickcolor=TICK_C, tickfont=dict(color=TICK_C, size=9)),
                   angularaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE,
                                    tickfont=dict(color=FONT_C, size=12))),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Clinical findings — pure HTML cards, no st.expander
    st.markdown('<div class="sec-hdr">Clinical Findings — Explainable AI</div>', unsafe_allow_html=True)
    if res.findings:
        for fi, f in enumerate(res.findings):
            icon = {"burnout_risk":"🔥","anxiety_trend":"⚡","depression_indicators":"🌧️","resilient":"✅"}.get(f.category,"⚠️")
            conf_color = "#34d399" if f.confidence>=0.8 else "#fbbf24" if f.confidence>=0.5 else "#f43f5e"
            conf_pct = int(f.confidence * 100)
            border_color = {"burnout_risk":"#f43f5e","anxiety_trend":"#f97316","depression_indicators":"#a78bfa","resilient":"#34d399"}.get(f.category,"#fbbf24")

            # Header card
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid {border_color};
                        border-radius:10px;padding:16px 20px;margin-bottom:4px;">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
                <div style="font-size:15px;font-weight:600;color:var(--text-pri)">
                  {icon} &nbsp;{f.label}
                </div>
                <div style="display:flex;gap:8px;align-items:center">
                  <span class="pill pill-grey">Risk {f.risk_score:.0f}/100</span>
                  <span style="font-size:11px;font-weight:700;background:rgba(52,211,153,0.1);
                               color:{conf_color};border:1px solid {conf_color}33;
                               border-radius:20px;padding:3px 10px">{conf_pct}% confidence</span>
                </div>
              </div>
              <p style="color:var(--text-sec);font-size:13px;line-height:1.6;margin:10px 0 8px">{f.explanation}</p>
              <div style="margin-bottom:4px">
                <div style="font-size:10px;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:0.07em">Confidence</div>
                <div class="mbar-wrap"><div class="mbar" style="width:{conf_pct}%;background:{conf_color}"></div></div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Signal rows
            if f.signals:
                st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);margin:10px 0 6px 4px">Triggered Signals</div>', unsafe_allow_html=True)
                for s in f.signals:
                    sev_cls = {"severe":"sev-severe","moderate":"sev-moderate","mild":"sev-mild"}.get(s.severity,"sev-mild")
                    st.markdown(f"""
                    <div class="signal-row">
                      <div class="signal-name">{s.name}</div>
                      <div class="signal-desc">{s.description}</div>
                      <div class="signal-sev {sev_cls}">{s.severity}</div>
                    </div>""", unsafe_allow_html=True)

            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Raw trigger data:**")
                st.json(f.triggered_data)
            with cb:
                st.markdown("**Suggested actions:**")
                for a in f.suggested_actions:
                    st.markdown(f"✦ {a}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    else:
        st.success("✅  No risk findings detected — this user appears mentally healthy across all monitored domains.")

    # Social & work context
    st.markdown('<div class="sec-hdr">Social & Work Context</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Social / week",    si["social_interactions_per_week"])
    s2.metric("Exercise / week",  si["exercise_sessions_per_week"])
    s3.metric("Work h / day",     f"{si['work_hours_daily']:.1f}")
    s4.metric("Screen h / day",   f"{si.get('screen_time_hours_daily', '—')}")

    # Journal entries
    st.markdown('<div class="sec-hdr">Recent Journal Entries</div>', unsafe_allow_html=True)
    for e in reversed(user["journal_entries"][-5:]):
        sentiment = e.get("sentiment","neutral")
        sc  = {"negative":"#f43f5e","positive":"#34d399"}.get(sentiment,"#484f58")
        lc  = {"negative":"#f43f5e","positive":"#34d399","neutral":"#232d42"}.get(sentiment,"#232d42")
        st.markdown(f"""
        <div class="journal-entry" style="border-left-color:{lc}">
          <div class="journal-date">{e['date']} &nbsp;·&nbsp; <span style="color:{sc};text-transform:uppercase;font-size:10px;letter-spacing:0.07em">{sentiment}</span></div>
          <div class="journal-text">{e['entry']}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — AI ASSISTANT  (two tabs: Conversation + Clinical Q&A)
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Assistant":
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-title">🤖 AI Clinical Assistant</div>
      <div class="page-hdr-sub">Conversational mental health check-in + clinical Q&A — powered by Ollama LLM + FAISS RAG</div>
      <div style="margin-top:8px">
        <span style="font-size:12px;background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.3);border-radius:6px;padding:4px 10px;color:#4f8ef7;font-weight:600;margin-right:8px">LLM: {asst.llm_status}</span>
        <span style="font-size:12px;background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);border-radius:6px;padding:4px 10px;color:#a78bfa;font-weight:600">{asst.memory_status}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── User selector (shared across both tabs) ───────────────
    user_opts = {"— Select a user —": None}
    user_opts.update({f"{u['user_id']} — {u['name']}": u["user_id"] for u in users})
    col_sel, col_clr = st.columns([5, 1])
    with col_sel:
        scope   = st.selectbox("User scope:", list(user_opts.keys()))
        user_id = user_opts[scope]
    with col_clr:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.checkin_msgs   = []
            st.session_state.checkin_stage  = "idle"
            st.rerun()

    # ── Two tabs ──────────────────────────────────────────────
    tab1, tab2 = st.tabs(["💬  Check-in Conversation", "🔬  Clinical Q&A"])

    # ════════════════════════════════
    #  TAB 1 — CONVERSATIONAL CHECK-IN
    # ════════════════════════════════
    with tab1:
        # Session state init
        if "checkin_msgs"  not in st.session_state: st.session_state.checkin_msgs  = []
        if "checkin_stage" not in st.session_state: st.session_state.checkin_stage = "idle"
        if "checkin_uid"   not in st.session_state: st.session_state.checkin_uid   = None

        # Reset conversation when user changes
        if user_id != st.session_state.checkin_uid:
            st.session_state.checkin_msgs  = []
            st.session_state.checkin_stage = "idle"
            st.session_state.checkin_uid   = user_id

        if not user_id:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">👤</div>
              <div class="es-title">Select a user to begin</div>
              <div class="es-hint">Choose a user from the dropdown above to start a personalised check-in</div>
            </div>""", unsafe_allow_html=True)

        else:
            # ── Start button ──
            if st.session_state.checkin_stage == "idle":
                user_obj = user_map[user_id]
                res_obj  = screen_user(user_obj)
                rc_color = {"critical":"#f43f5e","high":"#f97316",
                            "moderate":"#fbbf24","low":"#34d399"}.get(res_obj.risk_level,"#6b7a96")
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border);
                            border-left:4px solid {rc_color};border-radius:10px;
                            padding:18px 22px;margin-bottom:16px">
                  <div style="font-size:15px;font-weight:700;color:var(--text-pri);margin-bottom:6px">
                    Ready to check in with {user_obj['name']}
                  </div>
                  <div style="font-size:13px;color:var(--text-muted);margin-bottom:12px">
                    Risk level: <span style="color:{rc_color};font-weight:700">{res_obj.risk_level.upper()}</span>
                    &nbsp;·&nbsp; Score: {res_obj.overall_risk_score:.0f}/100
                    &nbsp;·&nbsp; Avg stress: {user_obj['aggregates']['avg_stress_14d']:.1f}/10
                    &nbsp;·&nbsp; Avg sleep: {user_obj['aggregates']['avg_sleep_14d']:.1f}h
                  </div>
                  <div style="font-size:12px;color:var(--text-faint)">
                    The assistant will greet the user, ask adaptive questions based on their data,
                    respond supportively, and close with an emotional pattern summary.
                  </div>
                </div>""", unsafe_allow_html=True)

                if st.button("▶  Start Check-in Conversation", use_container_width=True, key="start_checkin"):
                    with st.spinner("🧠 Preparing personalised check-in…"):
                        resp = asst.start_checkin(user_id)
                    st.session_state.checkin_msgs  = [{"role":"assistant","content":resp["message"],"meta":resp}]
                    st.session_state.checkin_stage = "checkin"
                    st.rerun()

            else:
                # ── Chat display ──
                for msg in st.session_state.checkin_msgs:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Stage badge
                stage = st.session_state.checkin_stage
                if stage == "summary":
                    st.success("✅  Check-in complete — emotional pattern summary provided above.")
                    if st.button("🔄  Start a new check-in", key="restart_checkin"):
                        st.session_state.checkin_msgs  = []
                        st.session_state.checkin_stage = "idle"
                        st.rerun()
                else:
                    # ── Adaptive quick-reply buttons ──────────────────
                    qs = asst.adaptive_questions(user_id)
                    if qs:
                        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin:12px 0 6px">Suggested replies:</div>', unsafe_allow_html=True)
                        qcols = st.columns(len(qs))
                        for qi, q in enumerate(qs):
                            if qcols[qi].button(q, key=f"aq_{qi}", use_container_width=True):
                                st.session_state.checkin_msgs.append({"role":"user","content":q})
                                with st.spinner("🧠 Responding…"):
                                    resp = asst.reply_to_checkin(
                                        q, user_id,
                                        st.session_state.checkin_msgs
                                    )
                                st.session_state.checkin_msgs.append({"role":"assistant","content":resp["message"],"meta":resp})
                                st.session_state.checkin_stage = resp["stage"]
                                st.rerun()

                    # ── Free-text input ───────────────────────────────
                    if user_input := st.chat_input("Type your response…", key="checkin_input"):
                        st.session_state.checkin_msgs.append({"role":"user","content":user_input})
                        with st.spinner("🧠 Responding…"):
                            resp = asst.reply_to_checkin(
                                user_input, user_id,
                                st.session_state.checkin_msgs
                            )
                        st.session_state.checkin_msgs.append({"role":"assistant","content":resp["message"],"meta":resp})
                        st.session_state.checkin_stage = resp["stage"]
                        st.rerun()

    # ════════════════════════════════
    #  TAB 2 — CLINICAL Q&A
    # ════════════════════════════════
    with tab2:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Quick questions
        st.markdown('<div class="sec-hdr">Quick Questions</div>', unsafe_allow_html=True)
        samples = [
            "What mental health risks are visible?",
            "Has stress increased recently?",
            "Summarize this user's emotional state",
            "How does stress affect their sleep?",
            "What actions do you recommend?",
            "Show me the journal entries",
        ]
        qc = st.columns(3)
        for i, q in enumerate(samples):
            if qc[i%3].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.messages.append({"role":"user","content":q})
                with st.spinner("🧠 Analyzing…"):
                    resp = asst.ask(q, user_id=user_id)
                st.session_state.messages.append({"role":"assistant","content":resp["answer"],"meta":resp})
                st.rerun()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">🔬</div>
              <div class="es-title">Ask a clinical question</div>
              <div class="es-hint">Select a user above, then click a quick question or type below</div>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta",{}).get("risk_level"):
                    m = msg["meta"]
                    mc = st.columns(3)
                    mc[0].metric("Risk Level", m["risk_level"].upper())
                    mc[1].metric("Risk Score", f"{m.get('risk_score',0):.0f}/100")
                    mc[2].metric("Intent",     m.get("intent","—"))

        if prompt := st.chat_input("Ask a clinical question…", key="qa_input"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("🧠 Analyzing…"):
                resp = asst.ask(prompt, user_id=user_id)
            with st.chat_message("assistant"):
                st.markdown(resp["answer"])
                if resp.get("risk_level"):
                    mc = st.columns(3)
                    mc[0].metric("Risk Level", resp["risk_level"].upper())
                    mc[1].metric("Risk Score", f"{resp.get('risk_score',0):.0f}/100")
                    mc[2].metric("Intent",     resp.get("intent","—"))
            st.session_state.messages.append({"role":"assistant","content":resp["answer"],"meta":resp})


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — RAG SEARCH
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Search":
    st.markdown("""
    <div class="page-hdr">
      <div class="page-hdr-title">🔍 Semantic Search</div>
      <div class="page-hdr-sub">Vector similarity search across all user profiles using FAISS + TF-IDF embeddings</div>
    </div>""", unsafe_allow_html=True)

    c_in, c_k = st.columns([5,1])
    with c_in:
        query = st.text_input("", placeholder="e.g. users with severe burnout and poor sleep patterns…",
                              label_visibility="collapsed")
    with c_k:
        top_k = st.number_input("Results", min_value=3, max_value=20, value=8, step=1)

    if query:
        with st.spinner("Searching FAISS index…"):
            results = retrieve_by_query(query, top_k=top_k)

        st.markdown(f"<p style='color:var(--text-muted);font-size:13px;margin-bottom:12px'>Found <b style='color:var(--accent)'>{len(results)}</b> matches for <em>\"{query}\"</em></p>",
                    unsafe_allow_html=True)

        for idx, r in enumerate(results):
            u      = r["user"]
            res    = screen_user(u)
            pcls   = rp(res.risk_level)
            rc_    = rc(res.risk_level)
            sim_pct = int(r["similarity"] * 100)
            sim_col = "#34d399" if sim_pct>70 else "#fbbf24" if sim_pct>45 else "#f43f5e"

            # Full card — no st.expander at all
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid {rc_};
                        border-radius:10px;padding:18px 20px;margin-bottom:10px;">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px">
                <div style="font-size:15px;font-weight:700;color:var(--text-pri)">
                  #{idx+1} &nbsp; {u['user_id']} — {u['name']}
                </div>
                <div style="display:flex;gap:6px;flex-wrap:wrap">
                  <span class="pill {pcls}">{res.risk_level.upper()}</span>
                  <span class="pill pill-grey">{sim_pct}% match</span>
                  <span class="pill pill-grey">Score {res.overall_risk_score:.0f}/100</span>
                </div>
              </div>
              <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
                <span class="pill pill-grey">{u['occupation']}</span>
                <span class="pill pill-grey">{u['condition_label'].replace('_',' ').title()}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            sc = st.columns(5)
            sc[0].metric("Mood",       f"{u['aggregates']['avg_mood_14d']:.1f}/10")
            sc[1].metric("Stress",     f"{u['aggregates']['avg_stress_14d']:.1f}/10")
            sc[2].metric("Sleep",      f"{u['aggregates']['avg_sleep_14d']:.1f}h")
            sc[3].metric("Energy",     f"{u['aggregates']['avg_energy_14d']:.1f}/10")
            sc[4].metric("Risk Score", f"{res.overall_risk_score:.0f}/100")

            sim_bar_w = min(100, sim_pct * 1.3)
            st.markdown(f"""
            <div style="margin:8px 0 4px">
              <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">Similarity: {r['similarity']:.4f}</div>
              <div class="mbar-wrap"><div class="mbar" style="width:{sim_bar_w:.0f}%;background:{sim_col}"></div></div>
            </div>
            <p style="color:var(--text-sec);font-size:13px;margin:10px 0 4px">
              <b>Latest journal:</b> <em>{u['journal_entries'][-1]['entry'][:140]}…</em>
            </p>""", unsafe_allow_html=True)

            if res.findings:
                findings_str = "  ·  ".join(f.label for f in res.findings)
                st.markdown(f"<p style='font-size:13px;color:var(--text-muted);margin:4px 0 16px'><b style='color:var(--text-sec)'>Findings:</b> {findings_str}</p>", unsafe_allow_html=True)

            st.markdown("<hr style='border-color:var(--border);margin:4px 0 12px'>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="es-icon">🔍</div>
          <div class="es-title">Enter a clinical query to begin</div>
          <div class="es-hint">Try: <em>"high stress burnout risk"</em> &nbsp;·&nbsp; <em>"declining mood and poor sleep"</em> &nbsp;·&nbsp; <em>"users with anxiety and social isolation"</em></div>
        </div>""", unsafe_allow_html=True)
