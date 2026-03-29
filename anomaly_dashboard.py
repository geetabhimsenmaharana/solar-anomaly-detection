"""
dashboard.py
------------
Solar Anomaly Detection Dashboard
Run with: streamlit run dashboard.py
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st           # For creating the interactive dashboard
import pandas as pd             # For working with tables of data
import numpy as np              # For math operations
import plotly.graph_objects as go   # For interactive charts
import plotly.express as px         # For simpler interactive charts
import os                        # To check if files exist
import json                      # To read JSON files

# ── PAGE CONFIGURATION ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Anomaly Detection",  # Title of the dashboard
    page_icon="🔍",                        # Small icon in the browser tab
    layout="wide"                           # Use full width of the browser
)

# ── CUSTOM STYLES ─────────────────────────────────────────────────────────
# This changes how the metrics and alerts look (colors, sizes, etc.)
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

# ── FUNCTION TO LOAD DATA ─────────────────────────────────────────────────
@st.cache_data  # Cache so data is not reloaded every time we change something
def load_data():
    """
    Loads anomaly scores, severity report, and evaluation metrics
    from output files if they exist.
    """
    scores   = None    # Table of monthly actual vs expected kWh per site
    severity = None    # Table of severity of anomalies
    metrics  = None    # JSON with model performance metrics (F1, false positive rate)

    # Load anomaly scores CSV if exists
    if os.path.exists("outputs/anomaly_scores.csv"):
        scores = pd.read_csv("outputs/anomaly_scores.csv")
        scores['month_year'] = pd.to_datetime(scores['month_year'])  # Convert string to date

    # Load severity report CSV if exists
    if os.path.exists("outputs/severity_report.csv"):
        severity = pd.read_csv("outputs/severity_report.csv")
        severity['month_year'] = pd.to_datetime(severity['month_year'])

    # Load evaluation metrics JSON if exists
    if os.path.exists("outputs/evaluation_metrics.json"):
        with open("outputs/evaluation_metrics.json") as f:
            metrics = json.load(f)

    return scores, severity, metrics

# Load data into variables
scores, severity, metrics = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────
st.title("🔍 Solar Site Anomaly Detection")
st.markdown("**3-model ensemble** — Isolation Forest · PyTorch Autoencoder · Prophet")
st.divider()

# If there is no data yet, stop the dashboard and ask the user to run pipeline
if scores is None:
    st.warning("Run `anomaly_detection_pipeline.py` first to generate results.")
    st.stop()

# ── TOP METRICS CARDS ─────────────────────────────────────────────────────
# Count alerts by severity
label_counts = scores['severity_label'].value_counts()
critical = label_counts.get('CRITICAL', 0)
high     = label_counts.get('HIGH', 0)
medium   = label_counts.get('MEDIUM', 0)
normal   = label_counts.get('NORMAL', 0)

# Create 6 columns to display metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Column 1: Number of sites
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{scores['site_id'].nunique()}</div>
        <div class="metric-label">Sites Monitored</div>
    </div>""", unsafe_allow_html=True)

# Column 2-4: Critical, High, Medium alerts
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

# Column 5-6: F1 score and False Positive Rate from models
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

# ── SITE SELECTOR & ALERTS ────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

# Left column: Table of active alerts
with col_left:
    st.subheader("🚨 Active Alerts — Priority Queue")

    if severity is not None and len(severity) > 0:
        # Prepare table to display
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

        # Highlight rows with colors based on severity
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

# Right column: Site timeline chart
with col_right:
    st.subheader("📈 Site Anomaly Timeline")

    sites = sorted(scores['site_id'].unique())
    selected_site = st.selectbox("Select site", sites)
    site_data = scores[scores['site_id'] == selected_site].sort_values('month_year')

    fig = go.Figure()

    # Plot actual vs expected energy output
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

    # Highlight anomalies with colored markers
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

# ── SEVERITY SCORE CHART ──────────────────────────────────────────────────
st.subheader("🎯 Ensemble Severity Score — " + selected_site)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=site_data['month_year'], y=site_data['severity_score'],
    fill='tozeroy', fillcolor='rgba(237,125,49,0.2)',
    line=dict(color='#ED7D31', width=2), name='Severity Score'
))
# Threshold lines for high/medium severity
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

# ── MODEL CONSENSUS ───────────────────────────────────────────────────────
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

# ── FOOTER ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:12px'>
Self-initiated ML project | Real solar operational data (anonymized) |
Stack: Python · Isolation Forest · PyTorch · Prophet · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
