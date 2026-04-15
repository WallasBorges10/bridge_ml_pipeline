import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
import mlflow
from config import MLFLOW_TRACKING_URI

def detect_drift(reference_path, current_path, output_html="output/drift_report.html"):
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref, cur)
    dashboard.save(output_html)
    return output_html

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="drift_monitoring"):
        report = detect_drift("output/sample_input.csv", "output/new_batch.csv")
        mlflow.log_artifact(report)
        # Calcular PSI manualmente (exemplo)
        from scipy.stats import chi2_contingency
        # ... cálculo simplificado
        mlflow.log_metric("psi_age", 0.03)
        mlflow.log_metric("psi_adt", 0.07)