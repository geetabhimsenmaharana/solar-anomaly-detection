"""
dashboard.py
------------
Solar Anomaly Detection Dashboard
Run with: streamlit run dashboard.py

This dashboard shows solar energy production anomalies across different sites.
It helps operators understand which sites are underperforming and why.
All explanations are simplified for beginners.
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st           # For creating interactive web dashboards
import pandas as pd             # For working with tables (like Excel)
import numpy as np              # For math calculations
import plotly.graph_objects as go   # For interactive charts
import plotly.express as px         # For simpler charts
import os                        # To check if files exist
import json                      # To read model metrics stored in JSON

# ── PAGE CONFIGURATION ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Anomaly Detection",  # Browser tab title
    page_icon="🔍",                        # Small icon in the browser tab
    layout="wide"                           # Full width dashboard
)

# ── CUSTOM STYLES ─────────────────────────────────────────────────────────
# This changes the colors, fonts, and styles of the metrics and alerts
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
.critical { color: #C00000 !important; }  /* Red = Critical */
.high     { color: #ED7D31 !important; }  /* Orange = High */
.medium   { color: #FFC000 !important; }  /* Yellow = Medium */
</style>
""", unsafe_allow_html=True)

# ── FUNCTION TO LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Loads 3 types of data generated from the anomaly detection pipeline:
    
    1. anomaly_scores.csv -> Contains energy production per site per month
    2. severity_report.csv -> Shows severity of anomalies for each site/month
    3. evaluation_metrics.json -> Model performance metrics (F1 score, false positives)
    
    Returns None if files do not exist.
    """
    scores   = None
    severity = None
    metrics  = None

    if os.path.exists("outputs/anomaly_scores.csv"):
        scores = pd.read_csv("outputs/anomaly_scores.csv")
        scores['month_year'] = pd.to_datetime(scores['month_year'])  # Convert text to date

    if os.path.exists("outputs/severity_report.csv"):
        severity = pd.read_csv("outputs/severity_report.csv")
        severity['month_year'] = pd.to_datetime(severity['month_year'])

    if os.path.exists("outputs/evaluation_metrics.json"):
        with open("outputs/evaluation_metrics.json") as f:
            metrics = json.load(f)

    return scores, severity, metrics

# Load data
scores, severity, metrics = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────
st.title("🔍 Solar Site Anomaly Detection")
st.markdown("**3-model ensemble** — Isolation Forest · PyTorch Autoencoder · Prophet")
st.divider()

# Stop dashboard if no data exists yet
if scores is None:
    st.warning("Run `anomaly_detection_pipeline.py` first to generate results.")
    st.stop()

# ── TOP METRICS CARDS ─────────────────────────────────────────────────────
# These are summary metrics across all sites

label_counts = scores['severity_label'].value_counts()
critical = label_counts.get('CRITICAL', 0)   # Red alerts = serious problems
high     = label_counts.get('HIGH', 0)       # Orange alerts = moderate problems
medium   = label_counts.get('MEDIUM', 0)    # Yellow alerts = minor problems
normal   = label_counts.get('NORMAL', 0)    # Normal months

# Create 6 columns to display summary metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Column 1: Number of sites monitored
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{scores['site_id'].nunique()}</div>
        <div class="metric-label">Sites Monitored</div>
        <div style='font-size:10px; color:#999'>Number of solar sites in the dataset</div>
    </div>""", unsafe_allow_html=True)

# Column 2-4: Critical, High, Medium alerts
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value critical">{critical}</div>
        <div class="metric-label">Critical Alerts</div>
        <div style='font-size:10px; color:#999'>Serious issues detected</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value high">{high}</div>
        <div class="metric-label">High Alerts</div>
        <div style='font-size:10px; color:#999'>Moderate issues detected</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value medium">{medium}</div>
        <div class="metric-label">Medium Alerts</div>
        <div style='font-size:10px; color:#999'>Minor issues detected</div>
    </div>""", unsafe_allow_html=True)

# Column 5-6: Model performance metrics
with col5:
    if metrics:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics['f1']:.2f}</div>
            <div class="metric-label">F1 Score</div>
            <div style='font-size:10px; color:#999'>How accurate our models are (1 = perfect)</div>
        </div>""", unsafe_allow_html=True)
with col6:
    if metrics:
        fpr = metrics['false_positive_rate']
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{fpr:.2f}</div>
            <div class="metric-label">False Positive Rate</div>
            <div style='font-size:10px; color:#999'>How often the model raises false alarms</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── LEFT PANEL: ACTIVE ALERTS ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🚨 Active Alerts — Priority Queue")
    if severity is not None and len(severity) > 0:
        # Prepare display table with simplified names
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

        # Color rows by severity
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

# ── RIGHT PANEL: SITE ANOMALY TIMELINE ───────────────────────────────────
with col_right:
    st.subheader("📈 Site Anomaly Timeline")

    sites = sorted(scores['site_id'].unique())
    selected_site = st.selectbox("Select site", sites)
    site_data = scores[scores['site_id'] == selected_site].sort_values('month_year')

    fig = go.Figure()

    # Plot actual vs expected energy
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

# ── SEVERITY SCORE CHART ──────────────────────────────────────────────────
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

# ── MODEL CONSENSUS ───────────────────────────────────────────────────────
st.subheader("🤝 Model Consensus — How Many Models Agree")

# Beginner-friendly explanation for all models
st.info("""
**Beginner-Friendly Explanation:**

Each model checks your solar site's monthly energy to find unusual behavior:

1. **Isolation Forest**:  
   - Looks for months where energy drops abnormally compared to usual monthly values.  
   - **Example:** If a site usually produces 1000 kWh/month:  
     - March = 700 kWh ✅ (flagged)  
     - June  = 600 kWh ✅ (flagged)  
     - Total months = 8  
     - Result: **2 flagged → 25% of months**
     -2 flagged: This means 2 months were flagged by the Isolation Forest model as unusual or problematic. In other words, out of all the months recorded for a particular site, 2 of them looked abnormal in energy production.
     - Suppose we have 8 months of energy production: Say, March and June are the 2 flagged months.
     -25% of months: This is the percentage of months that were flagged.
     -Calculation: Percentage = (Flagged Months​/Total Months) x 100. = 2/8*100= 25%
     -So the dashboard shows 2 flagged → 25% of months, meaning one-quarter of the months had abnormal energy detected by this model.
     -Isolation Forest looks for months where energy drops abnormally.
     -2 flagged → 2 months had unusual energy.
     -25% of months → These 2 months make up 25% of all months for this site.
     -Colors: Red = critical, orange = high, yellow = medium.

2. **PyTorch Autoencoder**:  
   - Detects unusual patterns in energy production over time (not just sudden drops, but weird trends).  
   - **Example:** Normally, energy slightly fluctuates: 980, 1000, 995, 1005…  
     - February = 850 kWh ✅ (flagged)  
     - May      = 1100 kWh ✅ (flagged)  
     - August   = 1200 kWh ✅ (flagged)  
     - Total months = 8  
     - Result: **3 flagged → 38% of months**

3. **Prophet**:  
   - Prophet is a model that predicts what energy production should be each month based on past data and seasonal patterns.  - **Example:** All months are within expected seasonal range:  
   - It looks at patterns across months or seasons (like summer, winter) and says: It looks at patterns across months or seasons (like summer, winter) and says
   - In other words, it’s checking if your site is producing energy like it normally would at that time of year.
   
   - Suppose we have a solar site with energy production for 8 months; Let say The actual energy is very close to expected energy each month. Prophet compares each month’s actual value to the expected value ± some tolerance.
   -Since all months are within normal seasonal ranges, no month is flagged as unusual.
   -What “0 flagged → 0% of months” means:
   -0 flagged: Prophet did not find any month that was unusual compared to seasonal expectations.
   -0% of months: Out of all months checked, 0% were abnormal according to Prophet.
   - Result: **0 flagged → 0% of months**
""")

# Create columns for each model metric
col_a, col_b, col_c = st.columns(3)

models_info = [
    ('Isolation Forest', 'iso_flagged', '#1F4E79',
     'Flags sudden drops in energy compared to usual monthly values'),
    ('PyTorch Autoencoder', 'ae_flagged', '#ED7D31',
     'Flags unusual patterns in energy production over time'),
    ('Prophet', 'prophet_flagged', '#70AD47',
     'Flags months outside normal seasonal trends')
]

# Display metrics for each model
for col, (name, col_name, color, desc) in zip([col_a, col_b, col_c], models_info):
    with col:
        if col_name in site_data.columns:
            flagged_count = site_data[col_name].sum()  # Number of months flagged
            total = len(site_data)                     # Total months
            st.metric(
                name,
                f"{int(flagged_count)} flagged",        # Number of flagged months
                f"{flagged_count/total*100:.0f}% of months"  # % of months flagged
            )
            # Short simple explanation for each model
            st.caption(desc + f" (Example: see info box above for how months are flagged)")

# ── FOOTER ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:12px'>
Self-initiated ML project | Real solar operational data (anonymized) |
Stack: Python · Isolation Forest · PyTorch · Prophet · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
