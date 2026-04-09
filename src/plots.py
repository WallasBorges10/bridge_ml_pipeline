"""Plotting utilities for model comparison and insights."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import mlflow
import os

logger = logging.getLogger(__name__)

def plot_comparison(metrics_dict, save_path, mlflow_run=None):
    """Plot bar chart comparing models."""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    data = {m: [metrics_dict[m][k] for k in metric_names] for m in models}
    df_plot = pd.DataFrame(data, index=metric_names).T
    ax = df_plot.plot(kind='bar', width=0.8, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    plt.ylabel('Score (0 to 1)')
    plt.ylim(0.6, 1.0)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"Comparison plot saved to {save_path}")
    plt.savefig(save_path, dpi=300)
    if mlflow_run:
        mlflow.log_artifact(save_path)
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, save_path: str = 'matriz_confusao.png'):
    """Plot normalized confusion matrix."""
    y_pred = model.predict(X_test)
    cm = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            cm[i,j] = np.sum((y_test == i) & (y_pred == j))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Bom (0)', 'Crítico (1)'],
                yticklabels=['Bom (0)', 'Crítico (1)'])
    plt.xlabel('Predição do Modelo')
    plt.ylabel('Condição Real (NBI)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_feature_importance(model, save_path: str = 'importancia_features.png'):
    """Plot feature importance (simplified, using predefined important features)."""
    # This is a demonstration plot from the notebook.
    features = ['Idade da Estrutura (AGE)', 'Tráfego Diário (ADT)', 'Largura do Deck',
                'Material da Superestrutura', 'Nº de Vãos', 'Classe da Rodovia',
                'Comprimento Máximo', 'Tipo de Projeto']
    importances_demo = [0.35, 0.22, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances_demo, y=features, palette='viridis')
    plt.xlabel('Importância Relativa (Gini Importance)')
    plt.ylabel('Variáveis (Features)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"Feature importance plot saved to {save_path}")
    plt.close()

def log_artifact_if_needed(file_path):
    if mlflow.active_run():
        mlflow.log_artifact(file_path)