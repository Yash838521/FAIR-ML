"""
streamlit_app.py
----------------
Fairness-Aware ML Dashboard
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairML · Bias Audit",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --ink:         #0d0f0e;
    --ink-2:       #141614;
    --ink-3:       #1a1d1a;
    --ink-4:       #222522;
    --rule:        rgba(255,255,255,0.07);
    --rule-strong: rgba(255,255,255,0.13);
    --text-1:  #f0ede8;
    --text-2:  #a8a49e;
    --text-3:  #6b6762;
    --text-4:  #3d3b38;
    --accent:  #e8c870;
    --serif: 'Playfair Display', Georgia, serif;
    --sans:  'IBM Plex Sans', system-ui, sans-serif;
    --mono:  'IBM Plex Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--ink);
    color: var(--text-1);
    -webkit-font-smoothing: antialiased;
}

.stApp { background: var(--ink); }

[data-testid="stSidebar"] {
    background: var(--ink-2) !important;
    border-right: 1px solid var(--rule) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }

[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--rule-strong) !important;
    gap: 0;
}
button[data-baseweb="tab"] {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
    color: var(--text-3) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
    transition: color 0.2s, border-color 0.2s;
}
button[data-baseweb="tab"]:hover { color: var(--text-2) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

.kpi-card {
    background: var(--ink-3);
    border: 1px solid var(--rule);
    border-top: 2px solid var(--kpi-color, var(--accent));
    border-radius: 3px;
    padding: 20px 18px 16px;
    text-align: left;
    transition: border-color 0.2s;
}
.kpi-card:hover {
    border-color: var(--rule-strong);
    border-top-color: var(--kpi-color, var(--accent));
}
.kpi-num {
    font-family: var(--serif);
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text-1);
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-3);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 8px;
}
.kpi-sub {
    font-family: var(--sans);
    font-size: 0.73rem;
    color: var(--text-3);
    margin-top: 3px;
    font-style: italic;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
}
.pill-fair   { background: rgba(91,165,140,0.12); color: #5ba58c; border: 1px solid rgba(91,165,140,0.25); }
.pill-unfair { background: rgba(196,96,74,0.12);  color: #c4604a; border: 1px solid rgba(196,96,74,0.25); }
.pill::before {
    content: '';
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
}

.eyebrow {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    color: var(--text-3);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 16px;
}

.ed-head {
    font-family: var(--serif);
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-1);
    line-height: 1.3;
    letter-spacing: -0.01em;
    margin-bottom: 6px;
}
.ed-deck {
    font-family: var(--sans);
    font-size: 0.82rem;
    color: var(--text-2);
    line-height: 1.65;
    font-style: italic;
    margin-bottom: 18px;
}

.step-card {
    background: var(--ink-3);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 16px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: var(--step-clr, var(--accent));
    border-radius: 3px 0 0 3px;
}
.step-card:hover { background: var(--ink-4); border-color: var(--rule-strong); }
.step-n {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-4);
    letter-spacing: 0.14em;
}
.step-t {
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-1);
    margin-top: 3px;
}
.step-d {
    font-family: var(--sans);
    font-size: 0.77rem;
    color: var(--text-2);
    line-height: 1.65;
    margin-top: 5px;
}

.callout {
    border-left: 2px solid var(--callout-clr, var(--accent));
    padding: 12px 18px;
    background: var(--ink-3);
    border-radius: 0 3px 3px 0;
    margin: 12px 0;
}
.callout-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--callout-clr, var(--accent));
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.callout-body {
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--text-2);
    line-height: 1.7;
}

table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
th {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3);
    border-bottom: 1px solid var(--rule-strong);
    padding: 9px 14px;
    text-align: left;
    font-weight: 400;
}
td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--rule);
    color: var(--text-1);
    vertical-align: middle;
    font-family: var(--sans);
}
tr:hover td { background: var(--ink-3); }

hr { border-color: var(--rule) !important; margin: 20px 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--ink-4); border-radius: 2px; }

[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--serif) !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 4px !important;
}
[data-testid="stMarkdownContainer"] h4 {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-3) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}
.stRadio label span { font-family: var(--sans) !important; font-size: 0.82rem !important; }
.stSlider label { font-family: var(--mono) !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Plotly layout factory ───────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#6b6762", size=10.5),
    margin=dict(l=48, r=24, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color="#6b6762"),
        title_font=dict(family="IBM Plex Mono, monospace", size=10, color="#6b6762"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color="#6b6762"),
        title_font=dict(family="IBM Plex Mono, monospace", size=10, color="#6b6762"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.07)",
        borderwidth=1,
        font=dict(family="IBM Plex Mono, monospace", size=10, color="#a8a49e"),
    ),
    hoverlabel=dict(
        bgcolor="#1a1d1a",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#f0ede8"),
    ),
)

def CL(**kwargs):
    """Produce a clean chart layout without duplicate-key collisions."""
    layout = {k: dict(v) if isinstance(v, dict) else v for k, v in _BASE.items()}
    for k, v in kwargs.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            merged = dict(layout[k])
            merged.update(v)
            layout[k] = merged
        else:
            layout[k] = v
    return layout

# ─── Palette ─────────────────────────────────────────────────────────────────
GOLD  = "#e8c870"
TEAL  = "#5ba58c"
CORAL = "#c4604a"
SLATE = "#7a8fa6"
SAGE  = "#7fa67a"
OCHRE = "#c4934a"
DUST  = "#a89880"

# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

@st.cache_data
def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path): return None
    return pd.read_csv(path)

def ensure_feature_col(df):
    if df is None: return None
    df = df.copy()
    if "feature" not in df.columns:
        df = df.reset_index().rename(columns={"index": "feature"})
    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})
    return df

def pill(is_fair, fair_text="Fair", unfair_text="Biased"):
    cls  = "pill-fair" if is_fair else "pill-unfair"
    text = fair_text  if is_fair else unfair_text
    return f'<span class="pill {cls}">{text}</span>'

def kpi_card(number, label, color=None, sub=None):
    color = color or GOLD
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-card" style="--kpi-color:{color};">'
            f'<div class="kpi-num">{number}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')

def eyebrow(text):
    return f'<div class="eyebrow">{text}</div>'

def callout(label, body, color=None):
    color = color or GOLD
    return (f'<div class="callout" style="--callout-clr:{color};">'
            f'<div class="callout-label">{label}</div>'
            f'<div class="callout-body">{body}</div></div>')

def results_available():
    return os.path.exists(os.path.join(RESULTS_DIR, "model_metrics.json"))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 22px;">
        <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.4rem;
                    font-weight:700;color:#f0ede8;letter-spacing:-0.02em;line-height:1.1;">
            FairML
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3d3b38;
                    letter-spacing:0.16em;text-transform:uppercase;margin-top:6px;">
            Bias Audit · Adult Income
        </div>
        <div style="width:32px;height:1px;background:#e8c870;margin-top:12px;opacity:0.5;"></div>
    </div>
    """, unsafe_allow_html=True)

    if not results_available():
        st.error("Results not found. Run `python pipeline.py` first.")
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
                    background:rgba(91,165,140,0.08);border:1px solid rgba(91,165,140,0.18);
                    border-radius:3px;margin-bottom:4px;">
            <span style="width:6px;height:6px;border-radius:50%;background:{TEAL};flex-shrink:0;"></span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                         color:{TEAL};letter-spacing:0.1em;text-transform:uppercase;">Results loaded</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(eyebrow("Dataset"), unsafe_allow_html=True)
    for k, v in [("Target",">£50K annual income"),("Source","UCI Adult Income"),
                  ("Sensitive","Gender, Race"),("Models","Logistic Reg. + RF")]:
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3d3b38;
                        letter-spacing:0.14em;text-transform:uppercase;">{k}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.79rem;
                        color:#a8a49e;margin-top:2px;">{v}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(eyebrow("Fairness Thresholds"), unsafe_allow_html=True)
    for name, thresh in [("SPD","< 0.10"),("DI","> 0.80"),("TPR Gap","< 0.10"),("Cal. Gap","< 0.05")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:baseline;
                    padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.07);">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#a8a49e;">{name}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#e8c870;">{thresh}</span>
        </div>""", unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:18px 0 10px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:24px;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3d3b38;
                letter-spacing:0.18em;text-transform:uppercase;margin-bottom:10px;">
        · RESPONSIBLE AI
    </div>
    <div style="font-family:'Playfair Display',Georgia,serif;font-size:2.6rem;font-weight:700;
                color:#f0ede8;line-height:1.05;letter-spacing:-0.03em;">
        Fairness-Aware<br>Machine Learning
    </div>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.9rem;color:#a8a49e;
                line-height:1.7;margin-top:12px;max-width:640px;font-style:italic;">
        Auditing algorithmic bias in income classification — measuring statistical parity,
        disparate impact and equalized odds across protected demographic attributes.
    </div>
</div>
""", unsafe_allow_html=True)

if not results_available():
    st.warning("Pipeline not run yet. Execute `python pipeline.py` to generate results.")
    st.stop()

# ─── Load data ────────────────────────────────────────────────────────────────
metrics     = load_json("model_metrics.json")
bias_before = load_json("bias_before.json")
bias_rw     = load_json("bias_after_reweighing.json")
bias_thresh = load_json("bias_after_threshold.json")
mitigation  = load_json("mitigation_comparison.json")
shap_imp    = ensure_feature_col(load_csv("shap_importance.csv"))
shap_grp    = ensure_feature_col(load_csv("shap_by_group.csv"))
proxy_df    = load_csv("proxy_variables.csv")
intersect   = load_csv("intersectional_bias.csv")

tabs = st.tabs(["Overview","Model Performance","Bias Detection",
                "Intersectionality","Explainability","Mitigation"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    lr = metrics["LogisticRegression"]
    rf = metrics["RandomForest"]
    bg = bias_before["gender"]

    st.markdown(eyebrow("Key Findings"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    spd_val = abs(bg['statistical_parity']['spd'])
    di_val  = bg['disparate_impact']['worst_di']

    c1.markdown(kpi_card(f"{lr['accuracy']:.1%}", "LR Accuracy",      GOLD,  "Logistic Regression"), unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{rf['roc_auc']:.3f}",  "RF ROC-AUC",       SLATE, "Random Forest"),       unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{spd_val:.3f}",         "Gender SPD",
                         CORAL if not bg['statistical_parity']['is_fair'] else TEAL,
                         "Stat. parity diff."), unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{di_val:.3f}",           "Gender DI",
                         CORAL if not bg['disparate_impact']['is_fair'] else TEAL,
                         "Disparate impact"),  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Methodology"), unsafe_allow_html=True)

    STEP_COLORS = [GOLD, SLATE, CORAL, OCHRE, TEAL, SAGE]
    steps = [
        ("01", "Data Preprocessing",   "Cleaning the UCI Adult Income dataset. Encoding categoricals, scaling numerics, preserving sensitive attributes without introducing data leakage."),
        ("02", "Model Training",        "Training Logistic Regression for interpretability alongside Random Forest as a benchmark. Evaluating accuracy, F1, and ROC-AUC."),
        ("03", "Bias Detection",        "Quantifying Statistical Parity Difference, Disparate Impact, Equalized Odds, and Calibration across gender and race."),
        ("04", "Intersectionality",     "Analysing bias at the intersection of gender × race — uncovering compounded disadvantage invisible to single-attribute audits."),
        ("05", "SHAP Explainability",   "Computing global and per-group SHAP values. Detecting proxy variables that encode sensitive attributes indirectly."),
        ("06", "Bias Mitigation",       "Applying Reweighing and Threshold Adjustment. Validating change significance with McNemar's test."),
    ]
    c1, c2 = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div class="step-card" style="--step-clr:{STEP_COLORS[i]};">
            <div class="step-n">Step {num}</div>
            <div class="step-t">{title}</div>
            <div class="step-d">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Fairness Summary"), unsafe_allow_html=True)

    fg = bias_before["gender"]
    fr = bias_before["race"]
    rows = [
        ["Gender", "Statistical Parity",      f"{fg['statistical_parity']['spd']:.4f}",          "< 0.10", pill(fg['statistical_parity']['is_fair'])],
        ["Gender", "Disparate Impact",         f"{fg['disparate_impact']['worst_di']:.4f}",        "> 0.80", pill(fg['disparate_impact']['is_fair'])],
        ["Gender", "Equalized Odds (TPR gap)", f"{fg['equalized_odds']['tpr_gap']:.4f}",           "< 0.10", pill(fg['equalized_odds']['is_fair_tpr'])],
        ["Gender", "Calibration Gap",          f"{fg['calibration']['max_calibration_gap']:.4f}", "< 0.05", pill(fg['calibration']['is_calibrated'])],
        ["Race",   "Statistical Parity",       f"{fr['statistical_parity']['spd']:.4f}",          "< 0.10", pill(fr['statistical_parity']['is_fair'])],
        ["Race",   "Disparate Impact",         f"{fr['disparate_impact']['worst_di']:.4f}",        "> 0.80", pill(fr['disparate_impact']['is_fair'])],
        ["Race",   "Equalized Odds (TPR gap)", f"{fr['equalized_odds']['tpr_gap']:.4f}",           "< 0.10", pill(fr['equalized_odds']['is_fair_tpr'])],
    ]
    tbl = pd.DataFrame(rows, columns=["Attribute", "Metric", "Value", "Threshold", "Status"])
    st.write(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    lr = metrics["LogisticRegression"]
    rf = metrics["RandomForest"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_lbls = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    st.markdown(eyebrow("Logistic Regression"), unsafe_allow_html=True)
    cols = st.columns(5)
    for col, k, lbl in zip(cols, metric_keys, metric_lbls):
        col.markdown(kpi_card(f"{lr[k]:.3f}", lbl, GOLD), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Random Forest"), unsafe_allow_html=True)
    cols2 = st.columns(5)
    for col, k, lbl in zip(cols2, metric_keys, metric_lbls):
        col.markdown(kpi_card(f"{rf[k]:.3f}", lbl, SLATE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(eyebrow("Metrics Comparison"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Logistic Regression", x=metric_lbls,
            y=[lr[k] for k in metric_keys],
            marker_color=GOLD, marker_line_width=0, opacity=0.9,
        ))
        fig.add_trace(go.Bar(
            name="Random Forest", x=metric_lbls,
            y=[rf[k] for k in metric_keys],
            marker_color=SLATE, marker_line_width=0, opacity=0.9,
        ))
        fig.update_layout(**CL(barmode="group", height=320,
                               yaxis={"range": [0, 1]},
                               bargap=0.28, bargroupgap=0.1))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown(eyebrow("Confusion Matrices"), unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Logistic Regression", "Random Forest"])
        for i, m in enumerate([lr, rf], 1):
            cm = m["confusion_matrix"]
            fig.add_trace(go.Heatmap(
                z=cm, x=["≤50K", ">50K"], y=["≤50K", ">50K"],
                colorscale=[[0, "#141614"], [1, GOLD]],
                showscale=False,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=15, family="IBM Plex Sans", color="#f0ede8"),
            ), row=1, col=i)
        fig.update_layout(**CL(height=320))
        fig.update_annotations(font=dict(family="IBM Plex Mono, monospace", color="#6b6762", size=10.5))
        st.plotly_chart(fig, width='stretch')

    auc_lr = lr['roc_auc']
    auc_rf = rf['roc_auc']
    better = "Random Forest" if auc_rf > auc_lr else "Logistic Regression"
    diff   = abs(auc_rf - auc_lr)
    st.markdown(callout(
        "ROC-AUC Analysis",
        f"LR: <strong style='color:#f0ede8'>{auc_lr:.4f}</strong> &nbsp;·&nbsp; "
        f"RF: <strong style='color:#f0ede8'>{auc_rf:.4f}</strong> &nbsp;·&nbsp; "
        f"{better} leads by {diff:.4f}. "
        "Both exceed 0.85. Logistic Regression preferred for this audit — "
        "coefficients are directly interpretable by stakeholders and regulators.",
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — BIAS DETECTION
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    attr_sel = st.radio("Sensitive attribute", ["Gender", "Race"], horizontal=True)
    attr_key = "gender" if attr_sel == "Gender" else "race"
    data = bias_before[attr_key]

    # Statistical Parity
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Statistical Parity — Positive Prediction Rates"), unsafe_allow_html=True)
    sp    = data["statistical_parity"]
    rates = sp["group_rates"]
    max_v = max(rates.values())

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            x=list(rates.keys()),
            y=list(rates.values()),
            marker_color=[TEAL if v == max_v else CORAL for v in rates.values()],
            marker_line_width=0,
            text=[f"{v:.1%}" for v in rates.values()],
            textposition="outside",
            textfont=dict(color="#a8a49e", family="IBM Plex Mono, monospace", size=10),
        ))
        fig.add_hline(y=list(rates.values())[0], line_dash="dot",
                      line_color=GOLD, line_width=1,
                      annotation_text="reference",
                      annotation_font=dict(color=GOLD, family="IBM Plex Mono, monospace", size=9))
        fig.update_layout(**CL(height=280, yaxis={"range": [0, 1], "title": "P(ŷ=1)"}))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown(kpi_card(f"{sp['spd']:.4f}", "SPD",
                    TEAL if sp['is_fair'] else CORAL, "< 0.10 fair"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Fair" if sp['is_fair'] else "Biased", "Status",
                    TEAL if sp['is_fair'] else CORAL), unsafe_allow_html=True)
    st.caption("SPD = 0 implies equal positive prediction rates. SPD < 0.10 is the accepted threshold.")

    # Disparate Impact
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Disparate Impact"), unsafe_allow_html=True)
    di      = data["disparate_impact"]
    di_vals = di["di_per_group"]

    c1, c2 = st.columns([3, 1])
    with c1:
        fig = go.Figure(go.Bar(
            x=list(di_vals.keys()),
            y=list(di_vals.values()),
            marker_color=[TEAL if v >= 0.8 else CORAL for v in di_vals.values()],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in di_vals.values()],
            textposition="outside",
            textfont=dict(color="#a8a49e", family="IBM Plex Mono, monospace", size=10),
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color=GOLD, line_width=1,
                      annotation_text="80% rule",
                      annotation_font=dict(color=GOLD, family="IBM Plex Mono, monospace", size=9))
        fig.add_hline(y=1.0, line_dash="dot",
                      line_color="rgba(255,255,255,0.1)", line_width=1)
        fig.update_layout(**CL(height=270, yaxis={"range": [0, 1.3], "title": "DI ratio"}))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown(kpi_card(f"{di['worst_di']:.4f}", "Worst DI",
                    TEAL if di['is_fair'] else CORAL, "> 0.80 fair"), unsafe_allow_html=True)
    st.caption("DI = 1 is perfect parity. Values below 0.80 carry legal risk under UK Equality Act 2010.")

    # Equalized Odds
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Equalized Odds — TPR & FPR per Group"), unsafe_allow_html=True)
    eo    = data["equalized_odds"]["per_group"]
    eo_df = pd.DataFrame(eo).T.reset_index().rename(columns={"index": "group"})

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="TPR", x=eo_df["group"], y=eo_df["TPR"].astype(float),
            marker_color=SLATE, marker_line_width=0,
            text=eo_df["TPR"].apply(lambda v: f"{float(v):.3f}"),
            textposition="outside",
            textfont=dict(color="#a8a49e", family="IBM Plex Mono, monospace", size=10),
        ))
        fig.add_trace(go.Bar(
            name="FPR", x=eo_df["group"], y=eo_df["FPR"].astype(float),
            marker_color=CORAL, marker_line_width=0,
            text=eo_df["FPR"].apply(lambda v: f"{float(v):.3f}"),
            textposition="outside",
            textfont=dict(color="#a8a49e", family="IBM Plex Mono, monospace", size=10),
        ))
        fig.update_layout(**CL(barmode="group", height=280, yaxis={"range": [0, 1]}))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown(kpi_card(f"{data['equalized_odds']['tpr_gap']:.4f}", "TPR Gap",
                    TEAL if data['equalized_odds']['is_fair_tpr'] else CORAL,
                    "< 0.10 fair"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(f"{data['equalized_odds']['fpr_gap']:.4f}", "FPR Gap",
                    TEAL if data['equalized_odds']['is_fair_fpr'] else CORAL,
                    "< 0.10 fair"), unsafe_allow_html=True)

    # Calibration
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(eyebrow("Calibration — Predicted vs Actual Rate"), unsafe_allow_html=True)
    cal    = data["calibration"]["per_group"]
    cal_df = pd.DataFrame(cal).T.reset_index().rename(columns={"index": "group"})

    c1, c2 = st.columns([3, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Predicted", x=cal_df["group"],
            y=cal_df["mean_predicted"].astype(float),
            marker_color=GOLD, marker_line_width=0, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Actual", x=cal_df["group"],
            y=cal_df["actual_rate"].astype(float),
            marker_color=DUST, marker_line_width=0, opacity=0.85,
        ))
        fig.update_layout(**CL(barmode="group", height=270, yaxis={"range": [0, 0.8]}))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown(kpi_card(f"{data['calibration']['max_calibration_gap']:.4f}", "Max Cal. Gap",
                    TEAL if data['calibration']['is_calibrated'] else CORAL,
                    "< 0.05 fair"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — INTERSECTIONALITY
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class="ed-head">Intersectional Bias — Gender × Race</div>
    <div class="ed-deck">
        Single-attribute audits can mask compounded disadvantage. A model may appear fair
        on gender in isolation, yet systematically disadvantage specific gender–race
        intersections. This analysis surfaces those hidden disparities.
    </div>
    """, unsafe_allow_html=True)

    if intersect is None:
        st.warning("Run `python pipeline.py` to generate intersectional data.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(eyebrow("Positive Prediction Rate by Group"), unsafe_allow_html=True)
            fig = px.bar(
                intersect.sort_values("positive_rate"),
                x="positive_rate", y="group", orientation="h",
                color="positive_rate",
                color_continuous_scale=[[0, CORAL], [0.45, OCHRE], [1, TEAL]],
                text="positive_rate",
                labels={"positive_rate": "P(ŷ=1)", "group": ""},
            )
            fig.update_traces(
                texttemplate="%{text:.1%}", textposition="outside",
                textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#a8a49e"),
                marker_line_width=0,
            )
            fig.update_layout(**CL(
                coloraxis_showscale=False,
                height=max(360, len(intersect) * 44),
                xaxis={"range": [0, 1]},
            ))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.markdown(eyebrow("Range"), unsafe_allow_html=True)
            max_r = intersect["positive_rate"].max()
            min_r = intersect["positive_rate"].min()
            gap   = max_r - min_r
            st.markdown(kpi_card(f"{max_r:.1%}", "Highest Group Rate", TEAL, "Best outcome"),  unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(kpi_card(f"{min_r:.1%}", "Lowest Group Rate",  CORAL, "Worst outcome"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(kpi_card(f"{gap:.3f}", "Intersectional Gap",
                        CORAL if gap > 0.1 else TEAL, "SPD across groups"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("TPR & FPR by Intersectional Group"), unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="TPR", x=intersect["group"], y=intersect["TPR"],
                              marker_color=SLATE, marker_line_width=0))
        fig2.add_trace(go.Bar(name="FPR", x=intersect["group"], y=intersect["FPR"],
                              marker_color=CORAL, marker_line_width=0))
        fig2.update_layout(**CL(barmode="group", xaxis={"tickangle": -30}))
        st.plotly_chart(fig2, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("Raw Data"), unsafe_allow_html=True)
        st.dataframe(
            intersect.style.background_gradient(subset=["positive_rate"], cmap="RdYlGn"),
            width='stretch',
        )


# ═══════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class="ed-head">SHAP-Based Explainability</div>
    <div class="ed-deck">
        SHapley Additive exPlanations decompose each prediction into per-feature contributions —
        enabling transparent, auditable model behaviour across demographic groups.
    </div>
    """, unsafe_allow_html=True)

    if shap_imp is None:
        st.warning("Run `python pipeline.py` to generate SHAP data.")
    else:
        top_n  = st.slider("Top N features", 5, 20, 15)
        df_top = shap_imp.head(top_n).copy()
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(eyebrow("Global Feature Importance"), unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=df_top["mean_abs_shap"].iloc[::-1].values,
                y=df_top["feature"].iloc[::-1].values,
                orientation="h",
                marker=dict(
                    color=df_top["mean_abs_shap"].iloc[::-1].values,
                    colorscale=[[0, "#1a1d1a"], [0.5, SLATE], [1, GOLD]],
                    line_width=0,
                ),
                text=[f"{v:.4f}" for v in df_top["mean_abs_shap"].iloc[::-1].values],
                textposition="outside",
                textfont=dict(color="#6b6762", family="IBM Plex Mono, monospace", size=9),
            ))
            fig.update_layout(**CL(height=500, xaxis={"title": "Mean |SHAP|"}))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.markdown(eyebrow("Per-Group SHAP"), unsafe_allow_html=True)
            if shap_grp is not None:
                grp_top    = shap_grp.head(top_n).copy()
                group_cols = [c for c in grp_top.columns if c not in ("feature", "abs_diff")]
                if len(group_cols) > 0:
                    palette = [GOLD, SLATE, TEAL, CORAL, OCHRE, SAGE, DUST]
                    fig2 = go.Figure()
                    for i, gcol in enumerate(group_cols):
                        fig2.add_trace(go.Bar(
                            name=gcol, x=grp_top["feature"],
                            y=grp_top[gcol].astype(float),
                            marker_color=palette[i % len(palette)],
                            marker_line_width=0,
                        ))
                    fig2.update_layout(**CL(
                        barmode="group", height=500,
                        xaxis={"tickangle": -35},
                        yaxis={"title": "Mean |SHAP|"},
                    ))
                    st.plotly_chart(fig2, width='stretch')
                else:
                    st.info("No group columns found.")
            else:
                st.info("shap_by_group.csv not found.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("Proxy Variable Detection"), unsafe_allow_html=True)
        st.markdown(callout(
            "What are proxy variables?",
            "Features with high Spearman correlation to protected attributes act as proxy "
            "variables — enabling indirect discrimination even when gender and race are "
            "excluded from model inputs. Identifying them is a regulatory best practice.",
            OCHRE
        ), unsafe_allow_html=True)

        if proxy_df is not None and len(proxy_df) > 0:
            fig3 = go.Figure(go.Bar(
                x=proxy_df["abs_corr"],
                y=proxy_df["feature"],
                orientation="h",
                marker=dict(
                    color=proxy_df["abs_corr"],
                    colorscale=[[0, OCHRE], [0.6, "#b07040"], [1, CORAL]],
                    line_width=0,
                ),
                text=[f"{v:.3f}  ({r})" for v, r in
                      zip(proxy_df["abs_corr"], proxy_df["sensitive_attr"])],
                textposition="outside",
                textfont=dict(color="#6b6762", family="IBM Plex Mono, monospace", size=9),
            ))
            fig3.add_vline(x=0.3, line_dash="dash", line_color=GOLD, line_width=1,
                           annotation_text="high risk (0.30)",
                           annotation_font=dict(color=GOLD, family="IBM Plex Mono, monospace", size=9))
            fig3.update_layout(**CL(
                xaxis={"title": "|Spearman Correlation|", "range": [0, 1]},
                height=max(260, len(proxy_df) * 44),
            ))
            st.plotly_chart(fig3, width='stretch')
            st.dataframe(proxy_df, width='stretch')
        else:
            st.success("No strong proxy variables detected (|correlation| < 0.20).")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("Interpretation Guide"), unsafe_allow_html=True)
        for clr, insight, meaning in [
            (GOLD,  "High mean |SHAP|",           "Feature has strong global influence on predictions."),
            (CORAL, "Large per-group difference",  "Feature affects groups differently — potential proxy."),
            (OCHRE, "High proxy correlation",      "Feature encodes sensitive group membership indirectly."),
            (TEAL,  "Positive SHAP value",         "Pushes prediction toward > £50K income class."),
            (SLATE, "Negative SHAP value",         "Pushes prediction toward ≤ £50K income class."),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:baseline;gap:14px;padding:9px 0;
                        border-bottom:1px solid rgba(255,255,255,0.07);">
                <div style="width:4px;height:4px;border-radius:50%;background:{clr};
                            flex-shrink:0;margin-top:8px;"></div>
                <div><span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;
                                  font-weight:600;color:#f0ede8;">{insight}</span>
                     <span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;
                                  color:#a8a49e;"> — {meaning}</span></div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 6 — MITIGATION
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("""
    <div class="ed-head">Bias Mitigation — Before vs After</div>
    <div class="ed-deck">
        Two complementary strategies were applied: Reweighing (pre-processing) adjusts
        training sample weights; Threshold Adjustment (post-processing) shifts the
        classification boundary per group. McNemar's test confirms whether changes
        are statistically significant.
    </div>
    """, unsafe_allow_html=True)

    if mitigation is None:
        st.warning("Run `python pipeline.py` to generate mitigation results.")
    else:
        orig = mitigation.get("original", {})
        rw   = mitigation.get("reweighing", mitigation.get("reweighing_", {}))
        thr  = mitigation.get(
            "threshold_adjust",
            mitigation.get("threshold_adjustment",
            mitigation.get("threshold",
            mitigation.get("threshold_adj", {}))))

        if not orig:
            st.error("Could not find 'original' key in mitigation_comparison.json.")
            st.json(mitigation); st.stop()
        if not thr:
            st.warning(f"Threshold adjustment not found. Keys: {list(mitigation.keys())}")

        st.markdown(eyebrow("Performance & Fairness Comparison"), unsafe_allow_html=True)
        row_defs = [
            ("Accuracy",     "accuracy",       False),
            ("F1 Score",     "f1",             False),
            ("ROC-AUC",      "roc_auc",        False),
            ("Gender SPD ↓", "gender_spd",     True),
            ("Gender DI ↑",  "gender_di",      False),
            ("TPR Gap ↓",    "gender_tpr_gap", True),
        ]
        def sfmt(d, k):
            v = d.get(k); return f"{v:.4f}" if v is not None else "—"

        rows_data = [[lbl, sfmt(orig,k), sfmt(rw,k), sfmt(thr,k)] for lbl,k,_ in row_defs]
        cmp_df = pd.DataFrame(rows_data, columns=["Metric","Original","Reweighing","Threshold Adj."])
        st.dataframe(cmp_df, width='stretch', hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        scenarios = [(orig, CORAL, "Original"), (rw, OCHRE, "Reweighing")]
        if thr: scenarios.append((thr, TEAL, "Threshold Adj."))

        with c1:
            st.markdown(eyebrow("Fairness Improvement"), unsafe_allow_html=True)
            fig = go.Figure()
            for scenario, color, name in scenarios:
                fig.add_trace(go.Bar(
                    name=name, x=["SPD", "TPR Gap"],
                    y=[abs(scenario.get("gender_spd", 0)),
                       abs(scenario.get("gender_tpr_gap", 0))],
                    marker_color=color, marker_line_width=0, opacity=0.88,
                ))
            fig.add_hline(y=0.1, line_dash="dash",
                          line_color="rgba(255,255,255,0.12)", line_width=1,
                          annotation_text="threshold (0.10)",
                          annotation_font=dict(color="#6b6762", family="IBM Plex Mono, monospace", size=9))
            fig.update_layout(**CL(barmode="group", height=320,
                                   yaxis={"range":[0,0.5], "title":"Value (lower = fairer)"}))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.markdown(eyebrow("Accuracy Trade-off"), unsafe_allow_html=True)
            fig2 = go.Figure()
            for scenario, color, name in scenarios:
                fig2.add_trace(go.Bar(
                    name=name, x=["Accuracy", "F1", "ROC-AUC"],
                    y=[scenario.get("accuracy",0), scenario.get("f1",0), scenario.get("roc_auc",0)],
                    marker_color=color, marker_line_width=0, opacity=0.88,
                ))
            fig2.update_layout(**CL(barmode="group", height=320, yaxis={"range":[0.75,1.0]}))
            st.plotly_chart(fig2, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("Statistical Significance — McNemar's Test"), unsafe_allow_html=True)
        st.markdown(callout(
            "Why McNemar's test?",
            "McNemar's test evaluates whether the pattern of prediction disagreements between "
            "two models is statistically random. A significant result (p < 0.05) confirms the "
            "mitigation genuinely altered model behaviour — not by chance.",
            SLATE
        ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        mc_rw  = rw.get("mcnemar", {})
        mc_thr = thr.get("mcnemar", {}) if thr else {}
        pairs  = [(mc_rw, "Reweighing", "")]
        if thr:
            pairs.append((mc_thr, "Threshold Adjustment",
                          f"Threshold used: {thr.get('threshold_used','—')}"))

        cols_mc = st.columns(len(pairs))
        for col, (mc, label, note) in zip(cols_mc, pairs):
            sig    = mc.get("significant", False)
            accent = TEAL if sig else CORAL
            with col:
                st.markdown(f"""
                <div style="background:#1a1d1a;border:1px solid rgba(255,255,255,0.07);
                            border-top:2px solid {accent};border-radius:3px;padding:20px;">
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-weight:600;
                                font-size:0.85rem;color:#f0ede8;margin-bottom:12px;">{label}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                                color:#a8a49e;line-height:1.9;">
                        χ² = {mc.get('chi2','—')}<br>p = {mc.get('p_value','—')}
                    </div>
                    <div style="margin-top:14px;">{pill(sig, 'Significant', 'Not Significant')}</div>
                    {'<div style="font-family:var(--mono);font-size:0.65rem;color:#3d3b38;margin-top:8px;">' + note + '</div>' if note else ''}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(eyebrow("Fairness–Accuracy Trade-off"), unsafe_allow_html=True)

        orig_spd = orig.get("gender_spd", 0) or 1e-9
        rw_spd   = rw.get("gender_spd", 0)
        spd_rr   = (orig_spd - rw_spd) / abs(orig_spd) * 100
        acc_rr   = (orig.get("accuracy", 0) - rw.get("accuracy", 0)) * 100

        rows_summary = [("Reweighing", f"{spd_rr:.1f}%", f"{acc_rr:+.2f}%",
                         "Pre-processing — minimal accuracy cost", OCHRE)]
        if thr:
            thr_spd = thr.get("gender_spd", 0)
            spd_tr  = (orig_spd - thr_spd) / abs(orig_spd) * 100
            acc_tr  = (orig.get("accuracy", 0) - thr.get("accuracy", 0)) * 100
            rows_summary.append(("Threshold Adj.", f"{spd_tr:.1f}%", f"{acc_tr:+.2f}%",
                                 "Post-processing — largest fairness gain", TEAL))

        for tech, spd_r, acc_c, note, clr in rows_summary:
            st.markdown(f"""
            <div style="display:flex;align-items:center;flex-wrap:wrap;
                        background:#1a1d1a;border:1px solid rgba(255,255,255,0.07);
                        border-left:2px solid {clr};border-radius:3px;margin-bottom:8px;overflow:hidden;">
                <div style="padding:14px 20px;min-width:150px;border-right:1px solid rgba(255,255,255,0.07);">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3d3b38;
                                letter-spacing:0.14em;text-transform:uppercase;">Technique</div>
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.85rem;
                                font-weight:600;color:{clr};margin-top:3px;">{tech}</div>
                </div>
                <div style="padding:14px 20px;min-width:120px;border-right:1px solid rgba(255,255,255,0.07);">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3d3b38;
                                letter-spacing:0.14em;text-transform:uppercase;">SPD Reduction</div>
                    <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.15rem;
                                font-weight:700;color:#f0ede8;margin-top:3px;">{spd_r}</div>
                </div>
                <div style="padding:14px 20px;min-width:130px;border-right:1px solid rgba(255,255,255,0.07);">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3d3b38;
                                letter-spacing:0.14em;text-transform:uppercase;">Accuracy Cost</div>
                    <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.15rem;
                                font-weight:700;color:#f0ede8;margin-top:3px;">{acc_c}</div>
                </div>
                <div style="padding:14px 20px;flex:1;min-width:200px;">
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.78rem;
                                color:#a8a49e;font-style:italic;">{note}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(callout(
            "Key Insight",
            "Both techniques substantially reduce statistical parity difference. "
            "Threshold adjustment typically achieves the greatest fairness gain at a modest "
            "accuracy cost; reweighing is more conservative. The right choice depends on the "
            "deployment context and the organisation's acceptable fairness–accuracy trade-off — "
            "a decision that should involve both technical and ethics stakeholders.",
            GOLD
        ), unsafe_allow_html=True)