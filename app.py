import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US-Iran War | GDP Impact on Asia",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.main { background-color: #0a0e1a; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1f3c 50%, #0a0e1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(220,53,69,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(220,53,69,0.15);
    border: 1px solid rgba(220,53,69,0.4);
    color: #e05a5a;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 900;
    color: #f0f4f8;
    line-height: 1.2;
    margin: 0 0 12px 0;
}
.hero-subtitle {
    font-size: 16px;
    color: #8fa3b8;
    font-weight: 300;
    max-width: 600px;
    line-height: 1.6;
}

/* Metric cards */
.metric-card {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2e6da4; }
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8fa3b8;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #f0f4f8;
}
.metric-sub {
    font-size: 12px;
    color: #8fa3b8;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: #f0f4f8;
    border-left: 4px solid #dc3545;
    padding-left: 16px;
    margin: 32px 0 20px 0;
}

/* Vulnerability badge */
.vuln-high   { color: #dc3545; font-weight: 700; }
.vuln-medium { color: #fd7e14; font-weight: 700; }
.vuln-low    { color: #28a745; font-weight: 700; }

/* Info box */
.info-box {
    background: rgba(30,58,95,0.4);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #2e6da4;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 14px;
    color: #c8d8e8;
    line-height: 1.6;
    margin: 12px 0;
}
.warn-box {
    background: rgba(220,53,69,0.08);
    border: 1px solid rgba(220,53,69,0.25);
    border-left: 4px solid #dc3545;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 14px;
    color: #e8c8c8;
    line-height: 1.6;
    margin: 12px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1524 !important;
    border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
COLORS = {
    'Bangladesh':  '#e05a5a',
    'India':       '#fd9a4a',
    'Japan':       '#5b8fd4',
    'Pakistan':    '#a775d4',
    'Singapore':   '#5bbfd4',
    'South Korea': '#4ab87a',
}

COUNTRIES = list(COLORS.keys())

BASELINE = {
    'Bangladesh': 5.70, 'India': 7.76, 'Japan': 0.84,
    'Pakistan': 2.47,   'Singapore': 3.44, 'South Korea': 2.10,
}

# Hybrid forecast deviations (pp from baseline)
FORECASTS = {
    'Bangladesh':  {'Moderate': {'2025':-0.40,'2026':-0.37,'2027':-0.45},
                    'Prolonged':{'2025':-0.40,'2026':-0.69,'2027':-1.10},
                    'Extreme':  {'2025':-0.40,'2026':-0.69,'2027':-1.10}},
    'India':       {'Moderate': {'2025':-2.50,'2026':-2.08,'2027':-2.50},
                    'Prolonged':{'2025':-2.50,'2026':-2.40,'2027':-2.93},
                    'Extreme':  {'2025':-2.50,'2026':-2.44,'2027':-3.19}},
    'Japan':       {'Moderate': {'2025':-1.00,'2026':-0.83,'2027':-1.00},
                    'Prolonged':{'2025':-1.00,'2026':-0.86,'2027':-1.03},
                    'Extreme':  {'2025':-1.00,'2026':-0.86,'2027':-1.03}},
    'Pakistan':    {'Moderate': {'2025':-0.28,'2026':-0.99,'2027':-0.68},
                    'Prolonged':{'2025':-0.44,'2026':-2.22,'2027':-1.84},
                    'Extreme':  {'2025':-0.56,'2026':-3.33,'2027':-3.03}},
    'Singapore':   {'Moderate': {'2025':-1.15,'2026':-0.96,'2027':-1.15},
                    'Prolonged':{'2025':-1.15,'2026':-0.99,'2027':-1.18},
                    'Extreme':  {'2025':-1.15,'2026':-0.99,'2027':-1.18}},
    'South Korea': {'Moderate': {'2025':-0.17,'2026':-0.38,'2027':-0.27},
                    'Prolonged':{'2025':-0.26,'2026':-0.72,'2027':-0.61},
                    'Extreme':  {'2025':-0.33,'2026':-1.03,'2027':-0.96}},
}

MODEL_RESULTS = {
    'Country':     COUNTRIES,
    'VAR':         [1.34, 2.01, 0.60, 2.62, 2.22, 1.33],
    'Random Forest':[0.95, 2.12, 0.64, 3.57, 2.24, 0.87],
    'XGBoost':     [0.68, 2.76, 0.62, 2.74, 2.26, 1.25],
    'Best Model':  ['XGBoost','VAR','VAR','VAR','VAR','Random Forest'],
}

SHAP = {
    'Feature': ['reserve_days','debt_gdp','oil_shock_signal_lag1',
                'energy_imports_pct','gpr_normalised_lag1',
                'remittance_pct_gdp','unemployment','inflation',
                'hormuz_disruption','fertilizer_price_index'],
    'SHAP':    [1.113,0.895,0.335,0.314,0.245,0.198,0.162,0.141,0.118,0.095],
}

VULNERABILITY = {
    'Country':    COUNTRIES,
    'Ranking':    [6,2,4,1,3,5],
    'Label':      ['Low','High','Medium','Critical','Medium','Low'],
    'Reserve Days':[18,30,150,18,90,100],
    'Energy Import %':[95,35,90,85,60,70],
}

# ── Try load real panel data ───────────────────────────────────────────────────
@st.cache_data
def load_panel():
    paths = [
        'dissertation_data/processed/panel_final.csv',
        '../dissertation_data/processed/panel_final.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

panel = load_panel()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:Playfair Display,serif;font-size:18px;
                    font-weight:700;color:#f0f4f8;'>🌏 GDP Impact</div>
        <div style='font-size:11px;color:#8fa3b8;margin-top:4px;
                    letter-spacing:1px;text-transform:uppercase;'>
            US-Iran War Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Scenario Forecasts",
        "🤖 Model Comparison",
        "🔍 SHAP Analysis",
        "🗺️ Vulnerability Map",
        "📋 About",
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#8fa3b8;line-height:1.6;'>
        <b style='color:#c8d8e8;'>MSc Data Science</b><br>
        Dissertation Project<br>
        VAR · Random Forest · XGBoost<br>
        6 Countries · 1990–2027
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("""
    <div class='hero'>
        <div class='hero-tag'>🔴 Live Geopolitical Crisis · February 2026</div>
        <div class='hero-title'>Economic Impact of the<br>US-Iran War on Asia</div>
        <div class='hero-subtitle'>
            A machine learning study predicting GDP growth deviations across
            six Asian economies under three conflict duration scenarios,
            using VAR, Random Forest, and XGBoost with SHAP interpretability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Countries Studied</div>
            <div class='metric-value'>6</div>
            <div class='metric-sub'>Asian economies</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Data Coverage</div>
            <div class='metric-value'>34yrs</div>
            <div class='metric-sub'>1990 – 2024</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Variables Used</div>
            <div class='metric-value'>20</div>
            <div class='metric-sub'>Across 6 categories</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Worst Case Impact</div>
            <div class='metric-value'>−3.33pp</div>
            <div class='metric-sub'>Pakistan, Extreme 2026</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Vulnerability at a Glance</div>",
                unsafe_allow_html=True)

    st.markdown("""<div class='warn-box'>
        ⚠️ <b>Over 80% of oil and LNG</b> bound for Asian markets transits
        the Strait of Hormuz annually. A prolonged US-Iran conflict risks
        severe energy supply disruption across all six economies studied.
    </div>""", unsafe_allow_html=True)

    # Country cards
    cols = st.columns(3)
    vuln_map = {'Critical':'🔴','High':'🟠','Medium':'🟡','Low':'🟢'}
    details = {
        'Bangladesh': ('XGBoost','−0.69 pp','~95% energy import dependency'),
        'India':      ('XGBoost','−2.44 pp','Largest absolute GDP exposure'),
        'Japan':      ('XGBoost','−0.86 pp','~150 days reserve buffer'),
        'Pakistan':   ('Calibrated','−3.33 pp','Only ~18 days reserve cover'),
        'Singapore':  ('XGBoost','−0.99 pp','Major oil trading hub'),
        'South Korea':('Calibrated','−1.03 pp','~70% crude from Middle East'),
    }
    vuln_labels = {'Bangladesh':'Low','India':'High','Japan':'Medium',
                   'Pakistan':'Critical','Singapore':'Medium','South Korea':'Low'}

    for i, country in enumerate(COUNTRIES):
        with cols[i % 3]:
            vlabel = vuln_labels[country]
            icon   = vuln_map[vlabel]
            model, worst, note = details[country]
            color  = COLORS[country]
            st.markdown(f"""
            <div style='background:#0d1f3c;border:1px solid #1e3a5f;
                        border-top:3px solid {color};border-radius:12px;
                        padding:16px;margin-bottom:12px;'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:8px;'>
                    <span style='font-weight:700;color:{color};
                                 font-size:15px;'>{country}</span>
                    <span style='font-size:12px;'>{icon} {vlabel}</span>
                </div>
                <div style='font-size:22px;font-family:Playfair Display,serif;
                            font-weight:700;color:#f0f4f8;
                            margin-bottom:4px;'>{worst}</div>
                <div style='font-size:11px;color:#8fa3b8;
                            margin-bottom:6px;'>Extreme scenario, 2026</div>
                <div style='font-size:11px;color:#c8d8e8;
                            background:rgba(255,255,255,0.04);
                            padding:6px 8px;border-radius:6px;'>{note}</div>
                <div style='font-size:10px;color:#8fa3b8;
                            margin-top:6px;'>Model: {model}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Scenario Forecasts":

    st.markdown("<div class='section-header'>GDP Deviation Forecasts</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Forecasts show percentage point (pp) deviation from each country's
        2022–2024 baseline GDP growth rate. All values are negative —
        confirming the war suppresses growth across all countries and scenarios.
    </div>""", unsafe_allow_html=True)

    scenario = st.selectbox("Select Conflict Scenario",
                            ["Moderate","Prolonged","Extreme"],
                            index=2)

    SCENARIO_INFO = {
        "Moderate":  "Analogue: 2011 Iran sanctions · Brent ~$105–115/bbl · Partial Hormuz disruption",
        "Prolonged": "Analogue: 1990 Gulf War · Brent ~$120–130/bbl · Sustained 8–12 month disruption",
        "Extreme":   "Composite worst-case · Brent ~$145–155/bbl · Full Hormuz closure 12+ months",
    }
    st.markdown(f"<div class='info-box'>ℹ️ {SCENARIO_INFO[scenario]}</div>",
                unsafe_allow_html=True)

    # Build forecast table
    rows = []
    for country in COUNTRIES:
        d = FORECASTS[country][scenario]
        base = BASELINE[country]
        rows.append({
            'Country':    country,
            'Baseline %': f"{base:.2f}%",
            '2025 Dev':   f"{d['2025']:.2f} pp",
            '2026 Dev':   f"{d['2026']:.2f} pp",
            '2027 Dev':   f"{d['2027']:.2f} pp",
            '2026 GDP':   f"~{base + d['2026']:.2f}%",
        })
    df = pd.DataFrame(rows)

    st.markdown("**GDP Deviation Table (pp from baseline)**")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart — 2026 deviations
    st.markdown("**War Year 2026 — GDP Deviation by Country**")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0d1f3c')
    ax.set_facecolor('#0d1f3c')

    devs    = [FORECASTS[c][scenario]['2026'] for c in COUNTRIES]
    colors  = [COLORS[c] for c in COUNTRIES]
    bars    = ax.bar(COUNTRIES, devs, color=colors, width=0.55,
                     edgecolor='none', zorder=3)

    for bar, val in zip(bars, devs):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.08,
                f'{val:.2f} pp', ha='center', va='top',
                fontsize=9, color='white', fontweight='bold')

    ax.axhline(0, color='#8fa3b8', linewidth=0.8, linestyle='--')
    ax.set_ylabel('GDP Deviation (pp)', color='#c8d8e8', fontsize=10)
    ax.set_title(f'{scenario} Scenario — War Year 2026',
                 color='#f0f4f8', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#c8d8e8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(axis='y', alpha=0.15, color='#8fa3b8')
    ax.set_ylim(min(devs) - 0.5, 0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Line chart — 2025-2027 all countries
    st.markdown("**GDP Deviation Trend 2025–2027**")
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    fig2.patch.set_facecolor('#0d1f3c')
    ax2.set_facecolor('#0d1f3c')

    years = ['2025','2026','2027']
    for country in COUNTRIES:
        vals = [FORECASTS[country][scenario][y] for y in years]
        ax2.plot(years, vals, color=COLORS[country],
                 linewidth=2.2, marker='o', markersize=6,
                 label=country, zorder=3)
        ax2.annotate(f'{vals[-1]:.2f}',
                     xy=('2027', vals[-1]),
                     xytext=(8, 0), textcoords='offset points',
                     color=COLORS[country], fontsize=8, va='center')

    ax2.axhline(0, color='#8fa3b8', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('GDP Deviation (pp)', color='#c8d8e8', fontsize=10)
    ax2.set_title(f'{scenario} Scenario — GDP Deviation 2025–2027',
                  color='#f0f4f8', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(fontsize=8, facecolor='#0d1f3c',
               edgecolor='#1e3a5f', labelcolor='#c8d8e8', ncol=2)
    ax2.tick_params(colors='#c8d8e8', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax2.grid(alpha=0.15, color='#8fa3b8')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":

    st.markdown("<div class='section-header'>Model Performance Comparison</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Three models were trained and evaluated on the same held-out test set
        (2022–2024). Lower RMSE = better prediction accuracy.
        VAR achieved the best average test RMSE overall.
    </div>""", unsafe_allow_html=True)

    # Overall summary
    c1, c2, c3 = st.columns(3)
    for col, model, rmse, note in [
        (c1, "VAR",           "1.69", "✅ Best overall"),
        (c2, "Random Forest", "1.73", "Best for S.Korea"),
        (c3, "XGBoost",       "1.75", "Best for Bangladesh & Japan"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{model}</div>
                <div class='metric-value'>{rmse}</div>
                <div class='metric-sub'>Avg Test RMSE · {note}</div>
            </div>""", unsafe_allow_html=True)

    # Per country table
    df_model = pd.DataFrame(MODEL_RESULTS)
    st.markdown("**Per-Country Test RMSE**")
    st.dataframe(df_model, use_container_width=True, hide_index=True)

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0d1f3c')
    ax.set_facecolor('#0d1f3c')

    x      = np.arange(len(COUNTRIES))
    width  = 0.25
    models = ['VAR','Random Forest','XGBoost']
    mcols  = ['#5b8fd4','#4ab87a','#e05a5a']

    for i, (m, c) in enumerate(zip(models, mcols)):
        vals = df_model[m].values
        bars = ax.bar(x + i*width, vals, width, label=m,
                      color=c, alpha=0.85, edgecolor='none', zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.03, f'{val:.2f}',
                    ha='center', va='bottom',
                    fontsize=7.5, color='#c8d8e8')

    ax.set_xticks(x + width)
    ax.set_xticklabels(COUNTRIES, color='#c8d8e8', fontsize=9)
    ax.set_ylabel('Test RMSE', color='#c8d8e8', fontsize=10)
    ax.set_title('Per-Country Test RMSE — VAR vs RF vs XGBoost',
                 color='#f0f4f8', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9, facecolor='#0d1f3c',
              edgecolor='#1e3a5f', labelcolor='#c8d8e8')
    ax.tick_params(colors='#c8d8e8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(axis='y', alpha=0.15, color='#8fa3b8')
    ax.set_ylim(0, 4.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""<div class='info-box'>
        <b>Why VAR outperformed ML models overall?</b><br>
        The test period (2022–2024) captured the Ukraine war energy shock —
        which differs from the Hormuz scenario. VAR's direct inclusion of
        inflation and GPR as inputs captured these dynamics well.
        ML models showed moderate overfitting (RF: 1.06→2.01 train-to-test gap).
        No single model dominated all countries, motivating the hybrid approach.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Analysis":

    st.markdown("<div class='section-header'>SHAP Feature Importance</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        SHAP (SHapley Additive exPlanations) shows which variables drove
        the XGBoost model's GDP predictions. Higher SHAP value = stronger
        influence on the prediction. Results were consistent with Random Forest
        importance rankings — validating the theoretical framework.
    </div>""", unsafe_allow_html=True)

    df_shap = pd.DataFrame(SHAP).sort_values('SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1f3c')
    ax.set_facecolor('#0d1f3c')

    bar_colors = ['#e05a5a' if i >= 7 else '#5b8fd4'
                  for i in range(len(df_shap))]
    bars = ax.barh(df_shap['Feature'], df_shap['SHAP'],
                   color=bar_colors, edgecolor='none',
                   height=0.6, zorder=3)

    for bar, val in zip(bars, df_shap['SHAP']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9,
                color='#c8d8e8', fontweight='bold')

    ax.set_xlabel('Mean |SHAP Value|', color='#c8d8e8', fontsize=10)
    ax.set_title('XGBoost SHAP Feature Importance\n'
                 'Top features driving GDP vulnerability predictions',
                 color='#f0f4f8', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#c8d8e8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(axis='x', alpha=0.15, color='#8fa3b8')

    red_patch   = mpatches.Patch(color='#e05a5a', label='Top 3 features')
    blue_patch  = mpatches.Patch(color='#5b8fd4', label='Other features')
    ax.legend(handles=[red_patch, blue_patch], fontsize=8,
              facecolor='#0d1f3c', edgecolor='#1e3a5f',
              labelcolor='#c8d8e8')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature explanations
    st.markdown("**What each top feature means:**")
    features_info = [
        ("🛢️ reserve_days",         "1.113",
         "Strategic oil reserve coverage in days of import supply. "
         "Countries with fewer days (Pakistan: ~18) face immediate GDP impact "
         "when Hormuz is disrupted. Japan (~150 days) can absorb short-term shocks."),
        ("💰 debt_gdp",             "0.895",
         "Government debt as % of GDP. High-debt countries (Pakistan >75%) "
         "have less fiscal space to absorb energy shocks through subsidies "
         "or counter-cyclical spending."),
        ("📈 oil_shock_signal_lag1","0.335",
         "Lagged normalised oil price shock. The lag reflects that GDP "
         "statistics capture energy shock effects with a one-year delay, "
         "as firms adjust through inventory depletion and supply chain restructuring."),
        ("⚡ energy_imports_pct",   "0.314",
         "Energy imports as % of GDP — the structural amplification coefficient. "
         "A one-unit oil price shock has larger GDP impact where energy "
         "imports represent 8% of GDP vs 2%."),
        ("🌍 gpr_normalised_lag1",  "0.245",
         "Lagged Geopolitical Risk Index. Elevated geopolitical uncertainty "
         "suppresses GDP through investment withdrawal and trade disruption "
         "with a one-year transmission lag."),
    ]

    for feat, shap_val, desc in features_info:
        st.markdown(f"""
        <div style='background:#0d1f3c;border:1px solid #1e3a5f;
                    border-radius:10px;padding:14px 18px;margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:6px;'>
                <span style='font-weight:700;color:#f0f4f8;
                             font-size:14px;'>{feat}</span>
                <span style='font-family:Playfair Display,serif;
                             font-size:18px;font-weight:700;
                             color:#e05a5a;'>SHAP: {shap_val}</span>
            </div>
            <div style='font-size:13px;color:#c8d8e8;
                        line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VULNERABILITY MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Vulnerability Map":

    st.markdown("<div class='section-header'>Country Vulnerability Analysis</div>",
                unsafe_allow_html=True)

    # Heatmap of all scenarios
    st.markdown("**GDP Deviation Heatmap — All Countries × All Scenarios × All Years**")

    scenarios = ['Moderate','Prolonged','Extreme']
    years     = ['2025','2026','2027']
    labels    = [f"{s}\n{y}" for s in scenarios for y in years]
    matrix    = []

    for country in COUNTRIES:
        row = []
        for s in scenarios:
            for y in years:
                row.append(FORECASTS[country][s][y])
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1f3c')
    ax.set_facecolor('#0d1f3c')

    mat = np.array(matrix)
    im  = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                    vmin=-3.5, vmax=0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color='#c8d8e8', fontsize=8)
    ax.set_yticks(range(len(COUNTRIES)))
    ax.set_yticklabels(COUNTRIES, color='#c8d8e8', fontsize=9)

    for i in range(len(COUNTRIES)):
        for j in range(len(labels)):
            val = mat[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if val < -1.5 else '#0a0e1a',
                    fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors='#c8d8e8', labelsize=8)
    cbar.set_label('GDP Deviation (pp)', color='#c8d8e8', fontsize=9)

    ax.set_title('GDP Deviation Heatmap — Darker Red = Greater Economic Impact',
                 color='#f0f4f8', fontsize=12, fontweight='bold', pad=12)

    # Vertical separators between scenarios
    for x in [2.5, 5.5]:
        ax.axvline(x, color='#8fa3b8', linewidth=1.5, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Vulnerability ranking
    st.markdown("**Vulnerability Ranking — Extreme Scenario 2026**")
    df_vuln = pd.DataFrame(VULNERABILITY).sort_values('Ranking')

    rank_colors = {'Critical':'🔴','High':'🟠','Medium':'🟡','Low':'🟢'}

    for _, row in df_vuln.iterrows():
        icon  = rank_colors[row['Label']]
        color = COLORS[row['Country']]
        dev   = FORECASTS[row['Country']]['Extreme']['2026']
        st.markdown(f"""
        <div style='background:#0d1f3c;border:1px solid #1e3a5f;
                    border-left:4px solid {color};border-radius:10px;
                    padding:12px 18px;margin-bottom:8px;
                    display:flex;align-items:center;gap:16px;'>
            <div style='font-size:22px;font-weight:700;
                        color:#8fa3b8;width:28px;'>#{row['Ranking']}</div>
            <div style='flex:1;'>
                <div style='font-weight:700;color:{color};
                            font-size:15px;'>{icon} {row['Country']}</div>
                <div style='font-size:12px;color:#8fa3b8;margin-top:2px;'>
                    Reserve: {row['Reserve Days']} days ·
                    Energy imports: {row['Energy Import %']}% of GDP
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-family:Playfair Display,serif;
                            font-size:22px;font-weight:700;
                            color:#e05a5a;'>{dev:.2f} pp</div>
                <div style='font-size:11px;color:#8fa3b8;'>Extreme 2026</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 About":

    st.markdown("<div class='section-header'>About This Project</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1f3c;border:1px solid #1e3a5f;border-radius:12px;
                padding:28px 32px;margin-bottom:20px;'>
        <div style='font-family:Playfair Display,serif;font-size:20px;
                    font-weight:700;color:#f0f4f8;margin-bottom:12px;'>
            Predicting Macroeconomic Shock Propagation in Asian Economies
        </div>
        <div style='font-size:14px;color:#c8d8e8;line-height:1.8;'>
            This dissertation develops and evaluates a machine learning framework
            to predict the GDP growth impact of the US-Iran War on six Asian
            economies under three conflict duration scenarios spanning 2025 to 2027.<br><br>
            A panel dataset covering six countries over 1990–2024 was constructed
            from 20 variables spanning energy market indicators, macroeconomic
            controls, geopolitical risk indices, and structural vulnerability measures.
            Three models were trained, compared, and combined into a hybrid forecasting
            architecture that produces economically consistent GDP deviation forecasts
            for all six countries across all three scenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Models Used**")
        for m, desc in [
            ("VAR","Econometric baseline · Impulse Response Functions"),
            ("Random Forest","Ensemble ML · Feature importance rankings"),
            ("XGBoost + SHAP","Gradient boosting · Full interpretability"),
        ]:
            st.markdown(f"""<div class='metric-card'>
                <div style='font-weight:700;color:#f0f4f8;
                            font-size:14px;'>{m}</div>
                <div style='font-size:12px;color:#8fa3b8;
                            margin-top:4px;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**Data Sources**")
        for src, vars_ in [
            ("World Bank / IMF","GDP growth, debt, remittances"),
            ("IEA","Hormuz flow, strategic reserves"),
            ("EIA / IMF","Brent crude, energy prices"),
            ("Caldara & Iacoviello","Geopolitical Risk Index (GPR)"),
            ("FAO / World Bank","Fertilizer price index"),
            ("World Bank via FRED","World GDP growth control"),
        ]:
            st.markdown(f"""<div class='metric-card'>
                <div style='font-weight:700;color:#f0f4f8;
                            font-size:13px;'>{src}</div>
                <div style='font-size:11px;color:#8fa3b8;
                            margin-top:3px;'>{vars_}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
        <b>GitHub Repository:</b>
        <a href='https://github.com/msmamata/dissertation_project'
           style='color:#5b8fd4;'>
            github.com/msmamata/dissertation_project
        </a><br>
        <b>Programme:</b> MSc Data Science<br>
        <b>Dissertation Deadline:</b> May 22, 2026
    </div>""", unsafe_allow_html=True)
