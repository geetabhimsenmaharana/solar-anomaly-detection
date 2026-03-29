# 🔍 Solar Site Anomaly Detection

> **Self-initiated ML system** — 3-model ensemble detecting anomalies in solar energy production across 25 sites using real operational data.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://share.streamlit.io)

## 🎯 Problem

Manual monitoring of 25+ solar sites is slow and reactive. By the time an engineer notices underperformance, energy has already been lost. This system detects anomalies automatically — flagging sites that need attention before problems escalate.

## 🔧 Why 3 Models Instead of 1

Each model catches a different type of anomaly:

| Model | What it catches | How |
|---|---|---|
| **Isolation Forest** | Sudden production drops | Random partitioning — anomalies isolate faster |
| **PyTorch Autoencoder** | Pattern degradation | High reconstruction error = unusual pattern |
| **Prophet** | Seasonal deviations | Actual outside 95% confidence interval |

A single model would miss anomaly types outside its detection method. The ensemble maximizes coverage and reduces false positives through consensus scoring.

## 🏗️ Severity Scoring

The 3 model scores are combined into a single severity score:

```
severity = (0.40 × Isolation Forest) + (0.35 × Autoencoder) + (0.25 × Prophet)
```

Isolation Forest is weighted highest because it's most sensitive to sudden production drops — the highest operational priority.

**Severity levels:**
- 🔴 **CRITICAL** — All 3 models agree, score > 0.7
- 🟠 **HIGH** — 2+ models agree, score > 0.5
- 🟡 **MEDIUM** — 1+ model flags, score > 0.3
- 🟢 **NORMAL** — Score ≤ 0.15

## 📁 Project Structure

```
solar-anomaly-detection/
├── data/
│   ├── anomaly_all_sites.csv       # 25 sites, 912 rows, 46 features
│   ├── master_final.csv            # Full dataset
│   └── site_alert_features.csv    # SolarEdge alert history per site
├── models/
│   ├── isolation_forest.pkl        # Trained IF model
│   ├── autoencoder.pt              # Trained PyTorch model
│   ├── autoencoder_config.json     # Model config + threshold
│   └── scaler.pkl                  # Feature scaler
├── outputs/
│   ├── anomaly_scores.csv          # All sites, all months with scores
│   ├── severity_report.csv         # Flagged anomalies only
│   ├── evaluation_metrics.json     # Precision, recall, F1
│   └── anomaly_chart.png           # Visualization
├── anomaly_detection_pipeline.py   # Main pipeline
├── dashboard.py                    # Streamlit dashboard
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/geetabhimsenmaharana/solar-anomaly-detection
cd solar-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python anomaly_detection_pipeline.py

# 4. Launch dashboard
streamlit run dashboard.py
```

## 📊 Key Results

- **25 sites** monitored across Massachusetts
- **3 model types** — unsupervised, neural network, time-series
- **Multi-model consensus** reduces false positives
- **Priority queue** — operations team sees highest severity first
- **SolarEdge alert integration** — model validated against real fault records

## 🔍 PyTorch Autoencoder Architecture

```
Input (13 features)
    → Linear(13→32) + ReLU + Dropout(0.1)
    → Linear(32→16) + ReLU
    → Linear(16→4)          ← latent space
    → Linear(4→16) + ReLU
    → Linear(16→32) + ReLU + Dropout(0.1)
    → Linear(32→13)         ← reconstruction
```

Trained exclusively on **normal production months**. High reconstruction error on a new month signals it looks nothing like normal — flagged as anomaly.

## 📈 Live Dashboard

🔗 [View Live Demo](https://share.streamlit.io) ← *Add Streamlit Cloud URL after deploying*

## ⚠️ Data Note

All site addresses anonymized (Site_001 through Site_035). Real addresses not included.

## 🛠️ Tech Stack

`Python` · `Isolation Forest` · `PyTorch` · `Prophet` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit` · `Plotly` · `Matplotlib`

---

*Part of a 3-project self-initiated ML portfolio built on real solar operational data.*
*→ [Solar Forecasting](https://github.com/geetabhimsenmaharana/solar-energy-forecasting)*
*→ [Automation Platform](https://github.com/geetabhimsenmaharana/solar-automation-platform)*
