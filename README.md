# 🌉 Bridge Condition Prediction – ML Pipeline

[](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/ml_pipeline.yml)
[](https://github.com/WallasBorges10/bridge_ml_pipeline/actions/workflows/monitoring.yml)

## 📌 Visão Geral

Este é um projeto de **Engenharia de Machine Learning** focado na classificação das condições de pontes utilizando dados reais do **National Bridge Inventory (NBI)**.

O objetivo é prever se uma ponte está em estado **Crítico** ou **Bom**, permitindo a priorização inteligente de inspeções e manutenção. O projeto vai além do modelo: ele abrange todo o ciclo de vida (ML Lifecycle), desde a ingestão modular até o monitoramento de drift em produção.

-----

## 🛠️ Tecnologias e Ferramentas

  * **Modelagem:** Scikit-learn (Random Forest, Decision Tree, Perceptron)
  * **Rastreamento:** MLflow (Métricas, Parâmetros e Model Registry)
  * **Interface:** Streamlit (Dashboard Interativo)
  * **Serviço:** Flask (API REST de Inferência)
  * **MLOps:** Evidently AI (Drift), PSI, GitHub Actions (CI/CD)
  * **Qualidade:** Pytest (Testes Unitários)

-----

## 📂 Estrutura do Projeto

```text
bridge_ml_pipeline/
├── .github/workflows/    # Automação CI/CD (Treino e Monitoramento)
├── src/                  # Core do Pipeline (Modularizado)
│   ├── config.py         # Hiperparâmetros e configurações
│   ├── data_loader.py    # Ingestão e limpeza (Remoção de Leakage)
│   ├── preprocessing.py  # Feature Engineering (AGE, TRAFFIC_DENSITY)
│   ├── pipeline_builder.py # Construção de Pipelines (Sklearn)
│   ├── train.py          # Treinamento com GridSearchCV
│   ├── evaluate.py       # Validação e Métricas
│   ├── plots.py          # Visualizações técnicas
│   ├── predict.py        # Wrapper de inferência
│   ├── app.py            # API de Produção (Flask)
│   └── main.py           # Orquestrador principal
├── drift/                # Monitoramento (Evidently + PSI)
├── tests/                # Garantia de qualidade (Pytest)
├── output/               # Artefatos gerados (Modelos e Gráficos)
├── app_streamlit.py      # Dashboard do usuário final
├── requirements.txt      # Dependências do projeto
└── RELATORIO.md          # Documentação técnica profunda
```

-----

## 🚀 Como Executar

### 1\. Preparação do Ambiente

```bash
# Clone o repositório
git clone https://github.com/WallasBorges10/bridge_ml_pipeline.git
cd bridge_ml_pipeline

# Configure o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 2\. Ciclo de Treinamento e Rastreamento

Para rodar o pipeline completo e registrar no MLflow:

```bash
python src/main.py --mode train --output_dir ./output
```

Para visualizar a UI do MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 3\. Servindo o Modelo (API e UI)

  * **API Flask:** `python src/app.py` (Porta 5001)
  * **Dashboard Streamlit:** `streamlit run app_streamlit.py` (Porta 8501)

-----

## 📈 Resultados e Performance

Os modelos foram avaliados com foco em **Recall**, garantindo que pontes críticas não sejam ignoradas.

| Modelo | Recall | F1-Score | Precisão | Tempo Treino |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **0.9045** | **0.8703** | 0.8386 | 31.5s |
| Decision Tree | 0.8232 | 0.8405 | 0.8586 | 2.1s |
| Perceptron (Baseline) | 0.8026 | 0.8194 | 0.8371 | 0.5s |

-----

## 🛡️ Monitoramento e CI/CD

O projeto implementa práticas modernas de **MLOps**:

1.  **CI/CD Pipeline:** Testes automatizados e retreinamento são disparados a cada `push` na branch `main`.
2.  **Detecção de Drift:** Um processo semanal analisa o **PSI (Population Stability Index)**. Se os dados de entrada mudarem significativamente (PSI \> 0.1), alertas são registrados no MLflow para análise de retreinamento.

-----

## 📖 Documentação Adicional

  * Para uma análise detalhada sobre a redução de dimensionalidade (PCA/LDA) e decisões arquiteturais, acesse o [**Relatório Técnico (RELATORIO.md)**](https://www.google.com/search?q=RELATORIO.md).

-----

## 🎥 Demonstração

*(Vídeo em breve - Demonstrando o uso do Dashboard Streamlit e predições em lote)*

-----

**Desenvolvido por Wallas Borges** – *Projeto Acadêmico para Pós-Graduação em MLOps.*