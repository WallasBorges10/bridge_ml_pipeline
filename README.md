# 🏗️ Projeto de Engenharia de Machine Learning: Classificação de Pontes

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](http://localhost:5000)
[![CI/CD](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/WallasBorges10/bridge_ml_pipeline/actions)

## 📌 Visão Geral
Este repositório consolida a transição de um projeto de modelagem exploratória para um **sistema de Machine Learning profissional**. O objetivo técnico é estruturar um pipeline reprodutível para classificação de condições de pontes, utilizando `scikit-learn` e `MLflow`, com foco em **rastreabilidade, controle de complexidade e simulação de produção**.

## 🎯 Objetivos de Negócio e Técnicos
- **Negócio:** Prever a condição estrutural de pontes para priorizar inspeções e alocar recursos de manutenção de forma eficiente.
- **Técnico:** Desenvolver um sistema que balanceie **desempenho preditivo (F1-Score)** com **custo computacional** e **interpretabilidade**, garantindo reprodutibilidade via pipelines e monitoramento contínuo.

## 🚀 Como Executar

### 1. Clone e Prepare o Ambiente
```bash
git clone https://github.com/WallasBorges10/bridge_ml_pipeline.git
cd bridge_ml_pipeline
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 2. Execute o Pipeline de Treinamento (Rastreamento MLflow)
- Inicie o servidor de UI do MLflow em um terminal separado
   mlflow ui
- Execute o script principal de experimentação
   python src/main.py

Acesse http://localhost:5000 para comparar os experimentos.

### 3. Simule a API de Produção
- python src/app.py
Envie uma requisição POST para http://localhost:5001/predict com um JSON de features.

# 4. Executar Testes e CI/CD (Simulado)
- pytest tests/
O pipeline de CI/CD está configurado em .github/workflows/ci.yml e valida testes e linting automaticamente.

# 🧱 Estrutura do Projeto (Engenharia)
.
├── .github/workflows/      # CI/CD (Testes Automatizados)
├── data/                   # Dados Brutos e Processados
├── notebooks/              # Exploração (Apenas Visualização)
├── src/                    # Código Fonte Modular (Reutilizável)
│   ├── data/               # Ingestão e Pré-processamento
│   ├── models/             # Treinamento e Persistência
│   └── app.py              # API Flask para Inferência
├── tests/                  # Testes Unitários
├── RELATORIO.md            # Documento Técnico Completo (Análise Crítica)
├── requirements.txt        # Dependências Exatas
└── README.md               # Este Arquivo  