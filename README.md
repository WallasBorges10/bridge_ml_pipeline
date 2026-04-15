# Bridge Condition Prediction – ML Engineering Project

Este projeto implementa um pipeline completo de machine learning para classificação da condição de pontes (Good vs. Critical/Fair/Poor) usando dados do NBI (National Bridge Inventory) do estado de Nova York.

## Objetivo

Construir um sistema de ML reproduzível, versionado e monitorável, desde a ingestão dos dados até a operação em produção, utilizando scikit‑learn e MLflow.

## Estrutura do Repositório

bridge_ml_pipeline/
├── .github/workflows/ # CI/CD simulado
├── src/ # Código principal
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── pipeline_builder.py
│ ├── train.py
│ ├── evaluate.py
│ ├── plots.py
│ ├── predict.py
│ ├── main.py
│ └── app.py # API Flask
├── notebooks/ # Exploração inicial
├── tests/ # Testes unitários
├── drift/ # Detecção de drift
├── output/ # Modelos e artefatos gerados
├── requirements.txt
└── README.md


## Requisitos

- Python 3.10+
- Bibliotecas listadas em `requirements.txt`

## Como executar

1. Clone o repositório e crie um ambiente virtual:
   ```bash
   git clone https://github.com/seu-usuario/bridge_ml_pipeline.git
   cd bridge_ml_pipeline
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou .\venv\Scripts\activate (Windows)
   pip install -r requirements.txt