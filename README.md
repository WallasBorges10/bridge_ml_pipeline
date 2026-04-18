# Bridge Condition Prediction – ML Pipeline

[![ML Pipeline](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/ml_pipeline.yml)
[![Monitoramento](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/monitoring.yml/badge.svg)](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/monitoring.yml)

Projeto de engenharia de machine learning para classificação de condição de pontes usando dados do NBI (National Bridge Inventory). O pipeline inclui preparação de dados, experimentação com múltiplos modelos (Perceptron, Decision Tree, Random Forest), redução de dimensionalidade (PCA, LDA), rastreamento com MLflow, API de inferência, detecção de drift e CI/CD.

## 📦 Estrutura do repositório

# Bridge Condition Prediction – ML Engineering Project

Este projeto implementa um pipeline completo de machine learning para classificação da condição de pontes (Good vs. Critical/Fair/Poor) usando dados do NBI (National Bridge Inventory) do estado de Nova York.

## Objetivo

Construir um sistema de ML reproduzível, versionado e monitorável, desde a ingestão dos dados até a operação em produção, utilizando scikit‑learn e MLflow.

## Estrutura do Repositório

bridge_ml_pipeline/
├── .github/workflows/ # Pipelines CI/CD e monitoramento
├── src/ # Código principal (modular)
├── tests/ # Testes unitários (pytest)
├── drift/ # Detecção de drift
├── output/ # Modelos, gráficos e amostras (gerado)
├── requirements.txt
├── README.md
└── RELATORIO.md # Relatório técnico completo


## Requisitos

- Python 3.10+
- Bibliotecas listadas em `requirements.txt`

## Como executar

1. Servidor
   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

2. Verificação dos testes
   pytest tests/ -v --cov=src

3. Treinamento
   python src/main.py --mode train --output_dir ./output

4. Clone o repositório e crie um ambiente virtual:
   ```bash
   git clone https://github.com/seu-usuario/bridge_ml_pipeline.git
   cd bridge_ml_pipeline
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou .\venv\Scripts\activate (Windows)
   pip install -r requirements.txt