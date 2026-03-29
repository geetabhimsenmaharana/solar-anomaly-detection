"""
dashboard.py
------------
Solar Anomaly Detection Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json

st.set_page_config(
    page_title="Solar Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)
# ── APP INTRODUCTION ─────────────────────────────────────────────────────────
st.markdown("""
# 🔍 Solar Panel Anomaly Detection Dashboard

Welcome! This dashboard helps monitor **solar panel performance** across multiple sites and automatically detects anomalies that could impact energy production.

**Key Features:**
- ✅ Tracks actual vs expected energy output (kWh) per site
- ✅ Highlights anomalies with severity labels: CRITICAL, HIGH, MEDIUM
- ✅ Displays top-level metrics: number of sites, alerts, F1 score, false positive rate
- ✅ Ensemble of 3 models: Isolation Forest, PyTorch Autoencoder, Prophet
- ✅ Visualizes site-specific anomaly timeline and model consensus
- ✅ Provides detailed severity and performance ratio for operational decisions

**Why it matters:**  
Solar panels can experience faults, degradation, or abnormal behavior that reduces energy output. Early detection of anomalies allows for proactive maintenance, higher efficiency, and cost savings.

*Scroll down to explore active alerts, site timelines, severity scores, and model consensus.*
""")
st.divider()



st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.metric-value { font-size: 28px; font-weight: 600; color: #1F4E79; }
.metric-label { font-size: 12px; color: #666; margin-top: 4px; }
.critical { color: #C00000 !important; }
.high     { color: #ED7D31 !important; }
.medium   { color: #FFC000 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    scores   = None
    severity = None
    metrics  = None

    if os.path.exists("outputs/anomaly_scores.csv"):
        scores = pd.read_csv("outputs/anomaly_scores.csv")
        scores['month_year'] = pd.to_datetime(scores['month_year'])

    if os.path.exists("outputs/severity_report.csv"):
        severity = pd.read_csv("outputs/severity_report.csv")
        severity['month_year'] = pd.to_datetime(severity['month_year'])

    if os.path.exists("outputs/evaluation_metrics.json"):
        with open("outputs/evaluation_metrics.json") as f:
            metrics = json.load(f)

    return scores, severity, metrics


scores, severity, metrics = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔍 Solar Site Anomaly Detection")
st.markdown("**3-model ensemble** — Isolation Forest · PyTorch Autoencoder · Prophet")
st.divider()

if scores is None:
    st.warning("Run `anomaly_detection_pipeline.py` first to generate results.")
    st.stop()

# ── TOP METRICS ───────────────────────────────────────────────────────────────
label_counts = scores['severity_label'].value_counts()
critical = label_counts.get('CRITICAL', 0)
high     = label_counts.get('HIGH', 0)
medium   = label_counts.get('MEDIUM', 0)
normal   = label_counts.get('NORMAL', 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{scores['site_id'].nunique()}</div>
        <div class="metric-label">Sites Monitored</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value critical">{critical}</div>
        <div class="metric-label">Critical Alerts</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value high">{high}</div>
        <div class="metric-label">High Alerts</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value medium">{medium}</div>
        <div class="metric-label">Medium Alerts</div>
    </div>""", unsafe_allow_html=True)
with col5:
    if metrics:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics['f1']:.2f}</div>
            <div class="metric-label">F1 Score</div>
        </div>""", unsafe_allow_html=True)
with col6:
    if metrics:
        fpr = metrics['false_positive_rate']
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{fpr:.2f}</div>
            <div class="metric-label">False Positive Rate</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── SITE SELECTOR ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🚨 Active Alerts — Priority Queue")

    if severity is not None and len(severity) > 0:
        display = severity[['site_id', 'month_year', 'severity_label',
                            'severity_score', 'model_consensus',
                            'performance_ratio']].copy()
        display['month_year']       = display['month_year'].dt.strftime('%b %Y')
        display['severity_score']   = display['severity_score'].round(3)
        display['performance_ratio'] = (display['performance_ratio'] * 100).round(1).astype(str) + '%'
        display = display.rename(columns={
            'site_id': 'Site', 'month_year': 'Month',
            'severity_label': 'Severity', 'severity_score': 'Score',
            'model_consensus': 'Models', 'performance_ratio': 'Performance'
        })

        def highlight_severity(row):
            if row['Severity'] == 'CRITICAL':
                return ['background-color: #FFEBEB'] * len(row)
            elif row['Severity'] == 'HIGH':
                return ['background-color: #FFF3EB'] * len(row)
            elif row['Severity'] == 'MEDIUM':
                return ['background-color: #FFFAEB'] * len(row)
            return [''] * len(row)

        st.dataframe(
            display.style.apply(highlight_severity, axis=1),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.success("No active alerts — all sites normal")

with col_right:
    st.subheader("📈 Site Anomaly Timeline")

    sites = sorted(scores['site_id'].unique())
    selected_site = st.selectbox("Select site", sites)
    site_data = scores[scores['site_id'] == selected_site].sort_values('month_year')

    fig = go.Figure()

    # Actual vs Expected
    fig.add_trace(go.Scatter(
        x=site_data['month_year'], y=site_data['actual_kwh'],
        mode='lines+markers', name='Actual kWh',
        line=dict(color='#2E75B6', width=2), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=site_data['month_year'], y=site_data['expected_kwh'],
        mode='lines', name='Expected kWh',
        line=dict(color='#A0A0A0', width=1, dash='dash')
    ))

    # Highlight anomalies
    severity_colors = {
        'CRITICAL': '#C00000', 'HIGH': '#ED7D31', 'MEDIUM': '#FFC000'
    }
    for sev, color in severity_colors.items():
        flagged = site_data[site_data['severity_label'] == sev]
        if len(flagged) > 0:
            fig.add_trace(go.Scatter(
                x=flagged['month_year'], y=flagged['actual_kwh'],
                mode='markers', name=sev,
                marker=dict(color=color, size=10, symbol='triangle-down')
            ))

    fig.update_layout(
        height=380,
        xaxis_title='Month', yaxis_title='Energy (kWh)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white',
        yaxis=dict(gridcolor='#f0f0f0'),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── SEVERITY SCORE CHART ──────────────────────────────────────────────────────
st.subheader("🎯 Ensemble Severity Score — " + selected_site)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=site_data['month_year'], y=site_data['severity_score'],
    fill='tozeroy', fillcolor='rgba(237,125,49,0.2)',
    line=dict(color='#ED7D31', width=2), name='Severity Score'
))
fig2.add_hline(y=0.5, line_dash='dash', line_color='red',
               annotation_text='High threshold', annotation_position='right')
fig2.add_hline(y=0.3, line_dash='dash', line_color='orange',
               annotation_text='Medium threshold', annotation_position='right')
fig2.update_layout(
    height=220, yaxis=dict(range=[0, 1], gridcolor='#f0f0f0'),
    plot_bgcolor='white', margin=dict(t=10, b=30),
    xaxis_title='Month', yaxis_title='Score (0-1)'
)
st.plotly_chart(fig2, use_container_width=True)

# ── MODEL CONSENSUS ───────────────────────────────────────────────────────────
st.subheader("🤝 Model Consensus — How Many Models Agree")

col_a, col_b, col_c = st.columns(3)

models_info = [
    ('Isolation Forest', 'iso_flagged', '#1F4E79',
     'Detects sudden production drops using random partitioning'),
    ('PyTorch Autoencoder', 'ae_flagged', '#ED7D31',
     'Detects pattern anomalies via reconstruction error'),
    ('Prophet', 'prophet_flagged', '#70AD47',
     'Detects months outside seasonal confidence intervals'),
]

for col, (name, col_name, color, desc) in zip([col_a, col_b, col_c], models_info):
    with col:
        if col_name in site_data.columns:
            flagged_count = site_data[col_name].sum()
            total         = len(site_data)
            st.metric(name, f"{int(flagged_count)} flagged",
                      f"{flagged_count/total*100:.0f}% of months")
            st.caption(desc)

st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:12px'>
Self-initiated ML project | Real solar operational data (anonymized) |
Stack: Python · Isolation Forest · PyTorch · Prophet · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
