# ============================================================
# DAY 3 — AI Credit Risk Assessment System
# Streamlit Web Application
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Credit Risk Assessor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f8f9fb; }

    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #f8f9fb 100%);
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        color: white;
    }
    .app-header p {
        font-size: 0.95rem;
        opacity: 0.8;
        margin: 0.4rem 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #eef0f4;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3a5f;
    }

    /* Risk verdict boxes */
    .verdict-approve {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-review {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-deny {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-sub {
        font-size: 0.9rem;
        margin-top: 0.4rem;
        opacity: 0.75;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 3px solid #2563eb;
        padding-left: 0.6rem;
        margin: 1.5rem 0 1rem;
    }

    /* Risk factor pills */
    .risk-pill-high {
        background: #fef2f2; color: #b91c1c;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.82rem; font-weight: 500;
        display: inline-block; margin: 3px;
    }
    .risk-pill-low {
        background: #ecfdf5; color: #065f46;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.82rem; font-weight: 500;
        display: inline-block; margin: 3px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #eef0f4;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #1e40af;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_artifacts():
    rf       = joblib.load('rf_model.pkl')
    features = joblib.load('features.pkl')
    explainer = shap.TreeExplainer(rf)
    return rf, features, explainer

try:
    model, FEATURES, explainer = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model files. Make sure rf_model.pkl and features.pkl are in the same folder.\nError: {e}")
    model_loaded = False
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>🏦 AI Credit Risk Assessment System</h1>
    <p>Powered by Random Forest + SHAP Explainability &nbsp;|&nbsp; Give Me Some Credit Dataset &nbsp;|&nbsp; AUC-ROC: 0.8482</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — INPUT FORM
# ============================================================
st.sidebar.markdown("## 📋 Applicant Information")
st.sidebar.markdown("Fill in the applicant's financial details below.")
st.sidebar.markdown("---")

st.sidebar.markdown("**Credit Behaviour**")
revolving_util = st.sidebar.slider(
    "Revolving Utilization (%)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="Credit card balance / credit limit. Higher = more risk."
)
late_30_59 = st.sidebar.number_input(
    "Times 30–59 Days Late", min_value=0, max_value=20, value=0,
    help="Number of times 30-59 days past due in last 2 years."
)
late_60_89 = st.sidebar.number_input(
    "Times 60–89 Days Late", min_value=0, max_value=20, value=0
)
late_90 = st.sidebar.number_input(
    "Times 90+ Days Late", min_value=0, max_value=20, value=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Personal & Financial**")
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=40)
monthly_income = st.sidebar.number_input(
    "Monthly Income (USD)", min_value=0, max_value=100000, value=5000, step=100
)
debt_ratio = st.sidebar.number_input(
    "Debt Ratio", min_value=0.0, max_value=10.0, value=0.35, step=0.01,
    help="Monthly debt payments / monthly income."
)
dependents = st.sidebar.number_input(
    "Number of Dependents", min_value=0, max_value=20, value=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Credit Portfolio**")
open_credit = st.sidebar.number_input(
    "Open Credit Lines & Loans", min_value=0, max_value=50, value=8
)
real_estate = st.sidebar.number_input(
    "Real Estate Loans or Lines", min_value=0, max_value=20, value=1
)

st.sidebar.markdown("---")
assess_btn = st.sidebar.button("🔍 Assess Credit Risk", use_container_width=True)

st.sidebar.markdown("""
<div class="info-box">
    <b>About this system</b><br>
    This AI model was trained on 150,000 historical borrower records.
    SHAP values explain <i>why</i> the model made its decision.
</div>
""", unsafe_allow_html=True)


# ============================================================
# COMPUTE DERIVED FEATURES
# ============================================================
def build_input(revolving_util, age, late_30_59, debt_ratio,
                monthly_income, open_credit, late_90,
                real_estate, late_60_89, dependents):
    total_late       = late_30_59 + late_60_89 + late_90
    income_per_dep   = monthly_income / (dependents + 1)
    high_util        = int(revolving_util > 0.75)

    row = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age':                                  age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio':                            debt_ratio,
        'MonthlyIncome':                        monthly_income,
        'NumberOfOpenCreditLinesAndLoans':      open_credit,
        'NumberOfTimes90DaysLate':              late_90,
        'NumberRealEstateLoansOrLines':         real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents':                   dependents,
        'TotalLatePayments':                    total_late,
        'IncomePerDependent':                   income_per_dep,
        'HighUtilization':                      high_util,
    }
    return pd.DataFrame([row])[FEATURES]


# ============================================================
# DEFAULT DASHBOARD (before assessment)
# ============================================================
if not assess_btn:
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Training Records", "150,000"),
        ("Model Type",       "Random Forest"),
        ("AUC-ROC Score",   "0.8482"),
        ("Features Used",    "13"),
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👈 Fill in applicant details in the sidebar and click **Assess Credit Risk**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Enter** applicant financial details in the sidebar
        2. **Click** Assess Credit Risk
        3. **AI model** predicts probability of default
        4. **SHAP** explains the top risk factors
        5. **Verdict**: Approve / Manual Review / Deny
        """)
    with col_b:
        st.markdown('<div class="section-header">Key Risk Factors (from training)</div>', unsafe_allow_html=True)
        factors = {
            "Revolving Utilization": 0.384,
            "Total Late Payments":   0.243,
            "90+ Days Late":         0.067,
            "30–59 Days Late":       0.066,
            "Age":                   0.058,
            "Debt Ratio":            0.043,
        }
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#dc2626', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']
        bars = ax.barh(list(factors.keys())[::-1],
                       list(factors.values())[::-1],
                       color=colors[::-1], edgecolor='white')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance', fontweight='bold', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ============================================================
# ASSESSMENT RESULTS
# ============================================================
else:
    X_input = build_input(
        revolving_util, age, late_30_59, debt_ratio,
        monthly_income, open_credit, late_90,
        real_estate, late_60_89, dependents
    )

    # Predict
    prob_default = model.predict_proba(X_input)[0][1]
    risk_score   = int(prob_default * 100)

    # Verdict thresholds
    if prob_default < 0.30:
        verdict = "APPROVED"
        verdict_class = "verdict-approve"
        verdict_icon  = "✅"
        verdict_color = "#10b981"
        verdict_sub   = "Low default risk — application can proceed."
    elif prob_default < 0.55:
        verdict = "MANUAL REVIEW"
        verdict_class = "verdict-review"
        verdict_icon  = "⚠️"
        verdict_color = "#f59e0b"
        verdict_sub   = "Moderate risk — requires human officer review."
    else:
        verdict = "DENIED"
        verdict_class = "verdict-deny"
        verdict_icon  = "❌"
        verdict_color = "#ef4444"
        verdict_sub   = "High default risk — application does not meet criteria."

    # ── Top row: verdict + score ─────────────────────────────
    col_v, col_s, col_p = st.columns([2, 1, 1])

    with col_v:
        st.markdown(f"""
        <div class="{verdict_class}">
            <div class="verdict-title" style="color:{verdict_color}">
                {verdict_icon} {verdict}
            </div>
            <div class="verdict-sub">{verdict_sub}</div>
        </div>""", unsafe_allow_html=True)

    with col_s:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value" style="color:{verdict_color}">{risk_score}/100</div>
        </div>""", unsafe_allow_html=True)

    with col_p:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Probability</div>
            <div class="metric-value" style="color:{verdict_color}">{prob_default:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Risk gauge ───────────────────────────────────────────
    col_g, col_shap = st.columns([1, 2])

    with col_g:
        st.markdown('<div class="section-header">Risk Gauge</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 2.5))
        # Background bar
        ax.barh(['Risk'], [100], color='#e5e7eb', height=0.5, edgecolor='none')
        # Filled portion
        bar_color = '#10b981' if risk_score < 30 else ('#f59e0b' if risk_score < 55 else '#ef4444')
        ax.barh(['Risk'], [risk_score], color=bar_color, height=0.5, edgecolor='none')
        # Score label
        ax.text(risk_score + 1, 0, f'{risk_score}', va='center', fontsize=16, fontweight='bold', color=bar_color)
        ax.set_xlim(0, 110)
        ax.axis('off')
        ax.set_title('Default Risk Score (0 = Safe, 100 = High Risk)',
                     fontsize=9, color='#6b7280')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Derived features display
        total_late = late_30_59 + late_60_89 + late_90
        income_per_dep = monthly_income / (dependents + 1)
        st.markdown('<div class="section-header">Computed Features</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | Total Late Payments | {total_late} |
        | Income per Dependent | ${income_per_dep:,.0f} |
        | High Utilization Flag | {"Yes ⚠" if revolving_util > 0.75 else "No ✓"} |
        """)

    # ── SHAP explanation ─────────────────────────────────────
    with col_shap:
        st.markdown('<div class="section-header">AI Explanation — Why this decision?</div>',
                    unsafe_allow_html=True)

        shap_vals = explainer.shap_values(X_input)

        # Handle SHAP output format
        if isinstance(shap_vals, list):
            sv_row = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            sv_row = shap_vals[0, :, 1]
        else:
            sv_row = shap_vals[0]

        shap_df = pd.DataFrame({
            'Feature': FEATURES,
            'SHAP':    sv_row,
            'Value':   X_input.values[0]
        }).reindex(columns=['Feature', 'SHAP', 'Value'])
        shap_df['Abs'] = shap_df['SHAP'].abs()
        shap_df = shap_df.sort_values('Abs', ascending=False).head(8)

        # SHAP bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_shap = ['#dc2626' if v > 0 else '#10b981' for v in shap_df['SHAP']]
        bars = ax.barh(shap_df['Feature'][::-1],
                       shap_df['SHAP'][::-1],
                       color=colors_shap[::-1], edgecolor='white', height=0.6)
        ax.axvline(0, color='#374151', linewidth=0.8, linestyle='--')
        ax.set_xlabel('SHAP Value (red = increases risk, green = reduces risk)')
        ax.set_title('Top Features Driving This Decision', fontweight='bold', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        red_patch   = mpatches.Patch(color='#dc2626', label='Increases default risk')
        green_patch = mpatches.Patch(color='#10b981', label='Decreases default risk')
        ax.legend(handles=[red_patch, green_patch], fontsize=8, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top 3 risk factors in plain English
        st.markdown('<div class="section-header">Top Risk Factors (Plain English)</div>',
                    unsafe_allow_html=True)

        risk_increasing = shap_df[shap_df['SHAP'] > 0].head(3)
        risk_reducing   = shap_df[shap_df['SHAP'] < 0].head(2)

        PLAIN_NAMES = {
            'RevolvingUtilizationOfUnsecuredLines': 'High credit utilization',
            'TotalLatePayments':                    'History of late payments',
            'NumberOfTimes90DaysLate':              'Severely overdue payments (90+ days)',
            'NumberOfTime30-59DaysPastDueNotWorse': 'Minor late payments (30–59 days)',
            'NumberOfTime60-89DaysPastDueNotWorse': 'Moderate late payments (60–89 days)',
            'age':                                  'Applicant age',
            'DebtRatio':                            'High debt-to-income ratio',
            'MonthlyIncome':                        'Monthly income level',
            'NumberOfOpenCreditLinesAndLoans':      'Number of open credit lines',
            'NumberRealEstateLoansOrLines':         'Real estate loans',
            'NumberOfDependents':                   'Number of dependents',
            'IncomePerDependent':                   'Income per dependent',
            'HighUtilization':                      'Credit utilization above 75%',
        }

        if not risk_increasing.empty:
            st.markdown("**Factors increasing risk:**")
            for _, row in risk_increasing.iterrows():
                name = PLAIN_NAMES.get(row['Feature'], row['Feature'])
                st.markdown(f'<span class="risk-pill-high">↑ {name}</span>',
                            unsafe_allow_html=True)

        if not risk_reducing.empty:
            st.markdown("**Factors reducing risk:**")
            for _, row in risk_reducing.iterrows():
                name = PLAIN_NAMES.get(row['Feature'], row['Feature'])
                st.markdown(f'<span class="risk-pill-low">↓ {name}</span>',
                            unsafe_allow_html=True)

    st.markdown("---")

    # ── Full input summary ───────────────────────────────────
    st.markdown('<div class="section-header">Applicant Input Summary</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        | Field | Value |
        |-------|-------|
        | Age | {age} |
        | Monthly Income | ${monthly_income:,} |
        | Debt Ratio | {debt_ratio:.2f} |
        | Dependents | {dependents} |
        """)
    with col2:
        st.markdown(f"""
        | Field | Value |
        |-------|-------|
        | Revolving Utilization | {revolving_util:.0%} |
        | 30–59 Days Late | {late_30_59} |
        | 60–89 Days Late | {late_60_89} |
        | 90+ Days Late | {late_90} |
        """)
    with col3:
        st.markdown(f"""
        | Field | Value |
        |-------|-------|
        | Open Credit Lines | {open_credit} |
        | Real Estate Loans | {real_estate} |
        | Total Late Payments | {late_30_59 + late_60_89 + late_90} |
        | High Utilization | {"Yes" if revolving_util > 0.75 else "No"} |
        """)

    st.markdown(f"""
    <div class="info-box">
        <b>Disclaimer:</b> This AI assessment is a decision-support tool only.
        Final credit decisions must be reviewed by a qualified human officer in accordance
        with applicable lending regulations. Model AUC-ROC: 0.8482 on 30,000 held-out test records.
    </div>
    """, unsafe_allow_html=True)