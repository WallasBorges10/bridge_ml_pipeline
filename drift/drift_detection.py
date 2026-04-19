import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI) for a numeric feature."""
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    percentiles = np.linspace(0, 100, bins+1)
    bins_edges = np.percentile(expected, percentiles)
    bins_edges = np.unique(bins_edges)

    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts, _ = np.histogram(actual, bins=bins_edges)

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def detect_drift(reference_path, current_path, output_dir='drift_reports'):
    os.makedirs(output_dir, exist_ok=True)

    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    # Evidently Report (nova API)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report_path = os.path.join(output_dir, 'drift_report.html')
    report.save_html(report_path)

    # Calculate PSI for numeric columns
    numeric_cols = ref.select_dtypes(include=[np.number]).columns
    psi_results = {}
    for col in numeric_cols:
        if col in cur.columns:
            psi = calculate_psi(ref[col], cur[col])
            psi_results[col] = psi

    return report_path, psi_results

def log_drift_to_mlflow(report_path, psi_results, threshold=0.1):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_artifact(report_path)
        for col, psi in psi_results.items():
            mlflow.log_metric(f"psi_{col}", psi)
        high_drift = any(psi > threshold for psi in psi_results.values() if not np.isnan(psi))
        mlflow.log_metric("drift_alert", 1 if high_drift else 0)
        if high_drift:
            print("⚠️ Drift detectado! PSI > 0.1 para algumas features.")
        else:
            print("✅ Nenhum drift significativo detectado.")

if __name__ == "__main__":
    ref_path = "output/sample_input.csv"
    cur_path = "output/new_batch.csv"

    if not os.path.exists(cur_path):
        ref = pd.read_csv(ref_path)
        cur = ref.copy()
        cur['AGE'] = cur['AGE'] + np.random.normal(0, 2, len(cur))
        cur.to_csv(cur_path, index=False)
        print(f"Arquivo de exemplo criado: {cur_path}")

    report, psi = detect_drift(ref_path, cur_path)
    log_drift_to_mlflow(report, psi)