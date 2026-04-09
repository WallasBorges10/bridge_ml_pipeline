import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref_data = pd.read_csv('./output/sample_input.csv')

# Simula novos dados
current_data = ref_data.copy()
current_data['AGE'] = current_data['AGE'] + 1
current_data['ADT_029'] = current_data['ADT_029'] * 1.05

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=current_data)

report.save_html('drift_report.html')

print("Relatório de drift salvo como drift_report.html")