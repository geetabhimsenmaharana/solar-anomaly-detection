import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------------------
# Sidebar - About the App
# ---------------------------------------
st.sidebar.title("About This App")
st.sidebar.info("""
**Solar Panel Anomaly Detection Dashboard**  
This app monitors solar panel performance and detects anomalies in real-time.  

- Detect issues affecting energy output  
- Visualize panel metrics over time  
- Highlight unusual patterns automatically  

Built with Python, Streamlit, and Machine Learning.  
[View Source Code on GitHub](https://github.com/geetabhimsenmaharana/solar-anomaly-detection)
""")

# ---------------------------------------
# Main Title
# ---------------------------------------
st.title("☀️ Solar Panel Anomaly Detection Dashboard")

# ---------------------------------------
# Introduction
# ---------------------------------------
st.markdown("""
This dashboard helps you **monitor solar panel performance** and detect any anomalies that could reduce energy efficiency.

**Why it matters:**  
Solar panels may experience faults, degradation, or abnormal behavior that reduces output. Early detection helps save energy, reduce maintenance costs, and optimize performance.
""")

# ---------------------------------------
# Upload Data
# ---------------------------------------
st.header("📥 Upload Your Data")
st.markdown("Upload a CSV file containing solar panel readings (voltage, current, temperature, power).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ---------------------------------------
    # Visualizations
    # ---------------------------------------
    st.header("📊 Time Series Metrics")
    metrics = df.columns.tolist()[1:]  # skip timestamp column if first

    selected_metric = st.selectbox("Select Metric to Visualize", metrics)

    fig = px.line(df, x=df.columns[0], y=selected_metric, title=f"{selected_metric} over Time")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Anomaly Detection Section
    # ---------------------------------------
    st.header("⚠️ Detected Anomalies")
    st.markdown("""
    Anomalies are automatically highlighted based on deviations from normal behavior.
    """)
    
    # Example: simple threshold anomaly detection
    threshold = st.slider("Set anomaly threshold", min_value=0.0, max_value=float(df[selected_metric].max()), value=float(df[selected_metric].mean() + 2*df[selected_metric].std()))
    anomalies = df[df[selected_metric] > threshold]

    st.write(f"Number of anomalies detected: {anomalies.shape[0]}")
    st.dataframe(anomalies)

    fig2 = px.scatter(df, x=df.columns[0], y=selected_metric, title=f"{selected_metric} with Anomalies Highlighted")
    fig2.add_scatter(x=anomalies[df.columns[0]], y=anomalies[selected_metric], mode='markers', marker=dict(color='red', size=10), name='Anomalies')
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Waiting for CSV file to be uploaded.")
