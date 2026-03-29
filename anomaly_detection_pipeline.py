"""
anomaly_detection_pipeline.py
------------------------------
Solar Site Anomaly Detection System

3-model ensemble:
  1. Isolation Forest    — unsupervised outlier detection
  2. PyTorch Autoencoder — reconstruction error detection
  3. Prophet             — time-series forecast deviation

Outputs a severity score per site-month combining all 3 signals.

Usage:
    python anomaly_detection_pipeline.py

Outputs:
    models/isolation_forest.pkl
    models/autoencoder.pt
    outputs/anomaly_scores.csv
    outputs/severity_report.csv
    outputs/anomaly_chart.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import joblib
import os
import json
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/anomaly_all_sites.csv"
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Features for anomaly detection
ANOMALY_FEATURES = [
    'performance_ratio', 'deviation_pct',
    'month_sin', 'month_cos',
    'is_summer', 'is_winter',
    'lag_1m', 'lag_3m', 'lag_12m',
    'roll_avg_3m', 'roll_avg_6m',
    'expected_kwh', 'actual_kwh',
]


# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("="*55)
    print("SOLAR ANOMALY DETECTION PIPELINE")
    print("="*55)
    print("\nStep 1: Loading data...")

    df = pd.read_csv(DATA_PATH)
    df['month_year'] = pd.to_datetime(df['month_year'])
    df = df.sort_values(['site_id', 'month_year']).reset_index(drop=True)

    # Only rows with actual data
    known = df[df['actual_kwh'] > 0].copy()

    print(f"  Sites: {known['site_id'].nunique()}")
    print(f"  Total rows: {len(known)}")
    print(f"  Anomalies labeled: {int(known['is_anomaly'].sum())} ({known['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Date range: {known['month_year'].min().strftime('%b %Y')} → {known['month_year'].max().strftime('%b %Y')}")

    return df, known


# ── STEP 2: PREPARE FEATURES ──────────────────────────────────────────────────
def prepare_features(known):
    print("\nStep 2: Preparing features...")

    ready = known.copy()
    for col in ANOMALY_FEATURES:
        if col in ready.columns:
            ready[col] = ready[col].fillna(0)
        else:
            ready[col] = 0

    # Replace inf values
    ready = ready.replace([np.inf, -np.inf], 0)

    X = ready[ANOMALY_FEATURES].values
    y = ready['is_anomaly'].fillna(0).values.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f"  Features: {len(ANOMALY_FEATURES)}")
    print(f"  Samples: {len(X_scaled)}")

    return ready, X_scaled, y, scaler


# ── STEP 3: ISOLATION FOREST ──────────────────────────────────────────────────
def run_isolation_forest(X_scaled, y, ready):
    print("\nStep 3: Training Isolation Forest...")

    contamination = max(0.05, min(0.4, y.mean()))

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    # Scores: lower = more anomalous
    raw_scores   = iso.decision_function(X_scaled)
    predictions  = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal

    # Normalize to 0-1 (1 = most anomalous)
    iso_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    iso_binary = (predictions == -1).astype(int)

    # Evaluate against rule-based labels
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() > 0:
        from sklearn.metrics import f1_score
        f1 = f1_score(y[valid_mask], iso_binary[valid_mask], zero_division=0)
        print(f"  Contamination rate: {contamination:.1%}")
        print(f"  Anomalies detected: {iso_binary.sum()} ({iso_binary.mean()*100:.1f}%)")
        print(f"  F1 vs rule-based labels: {f1:.3f}")

    joblib.dump(iso, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    print(f"  Model saved → models/isolation_forest.pkl")

    return iso_scores, iso_binary


# ── STEP 4: PYTORCH AUTOENCODER ───────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(X_scaled, y):
    print("\nStep 4: Training PyTorch Autoencoder...")

    # Train only on normal data
    normal_mask = y == 0
    X_normal    = X_scaled[normal_mask]

    # Convert to tensors
    X_tensor        = torch.FloatTensor(X_scaled)
    X_normal_tensor = torch.FloatTensor(X_normal)

    dataset    = TensorDataset(X_normal_tensor, X_normal_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim  = X_scaled.shape[1]
    model      = Autoencoder(input_dim, latent_dim=4)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion  = nn.MSELoss()

    # Train
    model.train()
    losses = []
    for epoch in range(100):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100 — Loss: {losses[-1]:.4f}")

    # Reconstruction errors on all data
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

    # Normalize to 0-1
    ae_scores  = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

    # Threshold at 90th percentile of normal reconstruction errors
    normal_errors = errors[normal_mask]
    threshold     = np.percentile(normal_errors, 90)
    ae_binary     = (errors > threshold).astype(int)

    print(f"  Reconstruction threshold: {threshold:.4f}")
    print(f"  Anomalies detected: {ae_binary.sum()} ({ae_binary.mean()*100:.1f}%)")

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'autoencoder.pt'))
    # Save config
    with open(os.path.join(MODEL_DIR, 'autoencoder_config.json'), 'w') as f:
        json.dump({'input_dim': input_dim, 'latent_dim': 4, 'threshold': float(threshold)}, f)
    print(f"  Model saved → models/autoencoder.pt")

    return ae_scores, ae_binary, model


# ── STEP 5: PROPHET ───────────────────────────────────────────────────────────
def run_prophet(known):
    print("\nStep 5: Running Prophet time-series anomaly detection...")

    try:
        from prophet import Prophet
    except ImportError:
        print("  Prophet not available — skipping")
        n = len(known)
        return np.zeros(n), np.zeros(n, dtype=int)

    prophet_scores  = np.zeros(len(known))
    prophet_binary  = np.zeros(len(known), dtype=int)
    sites_processed = 0

    for site_id in known['site_id'].unique():
        site_df = known[known['site_id'] == site_id].copy()

        if len(site_df) < 12:
            continue

        # Prophet requires ds and y columns
        prophet_df = site_df[['month_year', 'actual_kwh']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df[prophet_df['y'] > 0]

        if len(prophet_df) < 12:
            continue

        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05
            )
            m.fit(prophet_df)

            forecast = m.predict(prophet_df[['ds']])

            # Anomaly = actual outside confidence interval
            actual   = prophet_df['y'].values
            lower    = forecast['yhat_lower'].values
            upper    = forecast['yhat_upper'].values
            yhat     = forecast['yhat'].values

            # Score = normalized distance outside confidence interval
            below = np.maximum(0, lower - actual)
            above = np.maximum(0, actual - upper)
            distance = below + above

            # Normalized score
            max_dist = distance.max() if distance.max() > 0 else 1
            scores   = distance / max_dist
            binary   = ((actual < lower) | (actual > upper * 1.3)).astype(int)

            # Map back to full dataframe index
            site_idx = known[known['site_id'] == site_id].index
            valid    = known.loc[site_idx, 'actual_kwh'] > 0
            valid_idx = site_idx[valid[:len(scores)]]

            for i, idx in enumerate(valid_idx[:len(scores)]):
                if idx in known.index:
                    pos = known.index.get_loc(idx)
                    prophet_scores[pos] = scores[i]
                    prophet_binary[pos] = binary[i]

            sites_processed += 1

        except Exception as e:
            continue

    print(f"  Sites processed: {sites_processed}")
    print(f"  Anomalies detected: {prophet_binary.sum()} ({prophet_binary.mean()*100:.1f}%)")

    return prophet_scores, prophet_binary


# ── STEP 6: ENSEMBLE SEVERITY SCORING ────────────────────────────────────────
def compute_severity(known, iso_scores, ae_scores, prophet_scores,
                     iso_binary, ae_binary, prophet_binary):
    print("\nStep 6: Computing ensemble severity scores...")

    # Weighted consensus — Isolation Forest gets highest weight
    # as it's most sensitive to sudden production drops
    W_ISO     = 0.40
    W_AE      = 0.35
    W_PROPHET = 0.25

    severity = (
        W_ISO     * iso_scores +
        W_AE      * ae_scores +
        W_PROPHET * prophet_scores
    )

    # Model consensus — how many models agree it's an anomaly
    consensus = iso_binary + ae_binary + prophet_binary

    # Severity buckets
    def severity_label(score, consensus_count):
        if consensus_count == 3 and score > 0.7:
            return 'CRITICAL'
        elif consensus_count >= 2 and score > 0.5:
            return 'HIGH'
        elif consensus_count >= 1 and score > 0.3:
            return 'MEDIUM'
        elif score > 0.15:
            return 'LOW'
        else:
            return 'NORMAL'

    results = known[['site_id', 'month_year', 'actual_kwh',
                      'expected_kwh', 'performance_ratio',
                      'deviation_pct', 'is_anomaly']].copy()

    results['iso_score']      = iso_scores
    results['ae_score']       = ae_scores
    results['prophet_score']  = prophet_scores
    results['severity_score'] = severity
    results['model_consensus'] = consensus
    results['iso_flagged']    = iso_binary
    results['ae_flagged']     = ae_binary
    results['prophet_flagged'] = prophet_binary
    results['severity_label'] = [
        severity_label(s, c) for s, c in zip(severity, consensus)
    ]

    # Print summary
    label_counts = results['severity_label'].value_counts()
    print("\n  Severity distribution:")
    for label in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']:
        count = label_counts.get(label, 0)
        pct   = count / len(results) * 100
        bar   = '█' * int(pct / 2)
        print(f"    {label:<10} {bar:<25} {count:>4} ({pct:.1f}%)")

    # Save
    results.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_scores.csv'), index=False)

    # Severity report — only flagged items
    flagged = results[results['severity_label'] != 'NORMAL'].sort_values(
        'severity_score', ascending=False
    )
    flagged.to_csv(os.path.join(OUTPUT_DIR, 'severity_report.csv'), index=False)

    print(f"\n  Full scores saved → outputs/anomaly_scores.csv")
    print(f"  Severity report  → outputs/severity_report.csv ({len(flagged)} flagged)")

    return results


# ── STEP 7: EVALUATE VS LABELS ────────────────────────────────────────────────
def evaluate(results):
    print("\nStep 7: Evaluating against rule-based labels...")

    labeled = results[results['is_anomaly'].notna()].copy()
    y_true  = labeled['is_anomaly'].astype(int)
    y_pred  = (labeled['severity_label'].isin(['CRITICAL', 'HIGH', 'MEDIUM'])).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    false_positive_rate = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)

    print(f"  Precision:          {precision:.3f}")
    print(f"  Recall:             {recall:.3f}")
    print(f"  F1 Score:           {f1:.3f}")
    print(f"  False positive rate: {false_positive_rate:.3f}")

    metrics = {
        'precision': round(precision, 3),
        'recall':    round(recall, 3),
        'f1':        round(f1, 3),
        'false_positive_rate': round(false_positive_rate, 3)
    }
    with open(os.path.join(OUTPUT_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── STEP 8: VISUALIZE ─────────────────────────────────────────────────────────
def visualize(results):
    print("\nStep 8: Generating anomaly charts...")

    # Get site with most anomalies
    site_anomalies = results[results['severity_label'] != 'NORMAL'].groupby('site_id').size()
    if len(site_anomalies) == 0:
        print("  No anomalies to plot")
        return

    top_site    = site_anomalies.idxmax()
    site_data   = results[results['site_id'] == top_site].copy()
    site_data   = site_data.sort_values('month_year')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Anomaly Detection — {top_site}', fontsize=13, y=0.98)

    # Plot 1: Production vs Expected with anomalies highlighted
    ax1 = axes[0]
    ax1.plot(site_data['month_year'], site_data['actual_kwh'],
             color='#2E75B6', linewidth=1.5, label='Actual kWh', zorder=3)
    ax1.plot(site_data['month_year'], site_data['expected_kwh'],
             color='#A0A0A0', linewidth=1, linestyle='--', label='Expected kWh')

    colors = {'CRITICAL': '#C00000', 'HIGH': '#ED7D31', 'MEDIUM': '#FFC000'}
    for label, color in colors.items():
        flagged = site_data[site_data['severity_label'] == label]
        if len(flagged) > 0:
            ax1.scatter(flagged['month_year'], flagged['actual_kwh'],
                       color=color, zorder=5, s=60, label=f'{label}', marker='v')

    ax1.set_ylabel('Energy (kWh)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Plot 2: Severity score over time
    ax2 = axes[1]
    ax2.fill_between(site_data['month_year'], site_data['severity_score'],
                     alpha=0.4, color='#ED7D31')
    ax2.plot(site_data['month_year'], site_data['severity_score'],
             color='#ED7D31', linewidth=1.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='High threshold')
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Medium threshold')
    ax2.set_ylabel('Severity Score')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    # Plot 3: Model consensus
    ax3 = axes[2]
    bars = ax3.bar(site_data['month_year'], site_data['model_consensus'],
                   color=['#C00000' if c == 3 else '#ED7D31' if c == 2
                          else '#FFC000' if c == 1 else '#D9D9D9'
                          for c in site_data['model_consensus']],
                   width=20, alpha=0.8)
    ax3.set_ylabel('Models Agreeing')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['0', '1', '2', '3 (all)'])
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.set_xlabel('Month')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'anomaly_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved → {chart_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, known                           = load_data()
    ready, X_scaled, y, scaler         = prepare_features(known)
    iso_scores, iso_binary              = run_isolation_forest(X_scaled, y, ready)
    ae_scores, ae_binary, ae_model      = run_autoencoder(X_scaled, y)
    prophet_scores, prophet_binary      = run_prophet(known)
    results                             = compute_severity(
                                            known,
                                            iso_scores, ae_scores, prophet_scores,
                                            iso_binary, ae_binary, prophet_binary
                                          )
    metrics                             = evaluate(results)
    visualize(results)

    print("\n" + "="*55)
    print("ANOMALY DETECTION COMPLETE")
    print("="*55)
    print(f"  Models:          Isolation Forest + Autoencoder + Prophet")
    print(f"  Sites analyzed:  {results['site_id'].nunique()}")
    print(f"  F1 Score:        {metrics['f1']:.3f}")
    print(f"  False pos. rate: {metrics['false_positive_rate']:.3f}")
    print(f"  Full report:     outputs/anomaly_scores.csv")
    print(f"  Severity report: outputs/severity_report.csv")
    print("="*55)
