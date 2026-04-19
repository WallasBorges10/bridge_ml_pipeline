# Relatório Técnico – Projeto de Engenharia de Machine Learning

## 1. Estruturação do Projeto de Machine Learning

### 1.1 Mapeamento dos Experimentos Realizados
A transição de notebooks exploratórios para um projeto estruturado de engenharia permitiu o rastreamento sistemático de múltiplos modelos via **MLflow**.

| Modelo | Principais Hiperparâmetros | Métricas (Validação Cruzada) | Limitações / Observações |
| :--- | :--- | :--- | :--- |
| **Perceptron** | `max_iter=1000`, `class_weight='balanced'` | Acurácia ≈ 0,760 | Modelo linear simples; incapaz de capturar padrões complexos. |
| **Decision Tree** | `max_depth=5`, `class_weight='balanced'` | F1 ≈ 0,840 | Boa interpretabilidade, mas propenso a *overfitting* sem poda. |
| **Random Forest** | `n_estimators=200`, `class_weight='balanced'` | **Recall ≈ 0,904**, F1 ≈ 0,870 | **Melhor performance geral**; maior custo computacional. |
| **RF + PCA** | `n_components=0.95` (95% variância) | Recall ≈ 0,891 | Leve perda de recall; ganho marginal em tempo de treino. |
| **RF + LDA** | `n_components=1` | Recall ≈ 0,790 | Perda significativa de recall; inviável para o negócio. |

> **Nota:** Dados extraídos do MLflow (experimento `bridge_condition_prediction`, runs de 19/04/2026).

### 1.2 Objetivo Técnico e Métricas de Sucesso
* **Objetivo:** Construir um classificador binário (*Good* vs *Critical*) para priorizar a manutenção de infraestruturas.
* **Métrica Primária:** **Recall da Classe Crítica** – O foco é minimizar Falsos Negativos (pontes críticas não detectadas).
* **Métrica Secundária:** **F1-Score** – Para garantir que o modelo não classifique todas as pontes como críticas (equilíbrio com Precisão).
* **Impacto de Negócio:** Otimização de recursos públicos e aumento da segurança viária.

### 1.3 Arquitetura Modular do Código (`src/`)
A lógica foi fragmentada em módulos independentes para facilitar a manutenção e o deploy:
* `config.py`: Gestão de constantes e parâmetros.
* `data_loader.py`: Ingestão de dados brutos e remoção de *leakage*.
* `preprocessing.py`: Engenharia de novas variáveis (`AGE`, `TRAFFIC_DENSITY`).
* `pipeline_builder.py`: Automação do `ColumnTransformer` e transformações dimensionais.
* `train.py` & `evaluate.py`: Treinamento com *GridSearchCV* e cálculo de performance.
* `app.py`: API Flask para servir predições em tempo real.
* `drift_detection.py`: Monitoramento de desvio de dados com **Evidently AI**.

---

## 2. Fundação de Dados e Diagnóstico Inicial

### 2.1 Estrutura de Ingestão e Amostragem
* **Dataset:** FHWA NBI - Nova York (17.666 registros).
* **Divisão:** Amostragem estratificada (70% treino / 30% teste) para lidar com o desbalanceamento.

### 2.2 Diagnóstico de Qualidade e Ações Corretivas

| Problema Identificado | Ação Tomada |
| :--- | :--- |
| **Missing Values (>50%)** | Remoção de 11 colunas altamente incompletas. |
| **Data Leakage** | Exclusão de 14 variáveis registradas apenas *após* a inspeção física. |
| **Desbalanceamento** | Aplicação de `class_weight='balanced'` no treinamento. |
| **Inconsistências** | Padronização de strings (`lowercase`) e aplicação de *One-Hot Encoding*. |

### 2.3 Limitações Estruturais
O modelo é altamente dependente da qualidade do campo `ADT_029` (Tráfego Diário), que pode estar defasado. Além disso, a ausência de dados geoespaciais impede a análise de fatores climáticos locais (ex: proximidade do mar e corrosão).

---

## 3. Experimentação Sistemática (Resultados Reais)

A tabela abaixo consolida as métricas registradas na run `full_pipeline` do MLflow (ID: `de0ad91c944c485ebd076dd7738ffded`).

| Modelo | Recall | F1-Score | Precisão | Acurácia | Treino (s) | Inferência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Perceptron (Baseline) | 0,8026 | 0,8194 | 0,8371 | 0,7604 | 0,5 | 0,02 |
| Decision Tree (Tuned) | 0,8232 | 0,8405 | 0,8586 | 0,7883 | 2,1 | 0,03 |
| **Random Forest (Tuned)** | **0,9045** | **0,8703** | 0,8386 | 0,8174 | 31,5 | 0,12 |
| RF + PCA (95% Var) | 0,8910 | 0,8590 | 0,8097 | 0,8009 | 28,1 | 0,10 |
| RF + LDA (1 Comp.) | 0,7900 | 0,8241 | 0,8613 | 0,7715 | 24,7 | 0,09 |

---

## 4. Controle de Complexidade e Redução de Dimensionalidade

### 4.1 Avaliação de PCA e LDA
* **PCA:** Reduziu levemente a performance, mas não trouxe ganho computacional que justificasse a perda de interpretabilidade das *features*.
* **LDA:** Degradou severamente o Recall (-11,4%), tornando o modelo perigoso para a aplicação de segurança pública (muitos Falsos Negativos).

**Conclusão:** O uso de redução de dimensionalidade foi **descartado** para o modelo de produção, optando-se pelo Random Forest sobre o conjunto completo de features.

---

## 5. Seleção e Justificativa do Modelo Final

O **Random Forest (Tuned)** foi o escolhido para o deploy.

* **Hiperparâmetros:** `n_estimators=200`, `max_depth=None`, `class_weight='balanced'`.
* **Poder de Decisão:** O Recall de **90,45%** garante que 9 em cada 10 pontes críticas serão corretamente sinalizadas pelo sistema.
* **Interpretabilidade:** Permite extrair a importância das variáveis para auditoria técnica de engenheiros civis.

---

## 6. Operacionalização e Produção (MLOps)

### 6.1 Versionamento (Model Registry)
O modelo está registrado no MLflow Model Registry como `BridgeConditionRF` (Versão 1), sob o estágio **Production**.

### 6.2 API de Inferência
Exemplo de requisição para o serviço Flask:
```bash
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"YEAR_BUILT_027": 1950, "TRAFFIC_DENSITY": 150.5, "STRUCTURE_KIND_043A": 1}'
```

### 6.3 Monitoramento de Drift
Utilizamos o **PSI (Population Stability Index)** para monitorar mudanças na distribuição dos dados de entrada.
* **Alerta:** Caso o PSI seja superior a **0.1**, um alerta é disparado no MLflow.
* **Ação:** Retreinamento automático é agendado caso a performance de Recall caia abaixo de 85% em novos dados rotulados.

---

## 7. Conclusão

Este projeto demonstra a maturidade de um pipeline de engenharia completo. A escolha do **Random Forest** validada por experimentos rastreáveis garante uma solução robusta, enquanto a automação via CI/CD e o monitoramento de *drift* asseguram que o modelo permaneça confiável ao longo do tempo.

---
*Relatório gerado automaticamente a partir de logs do MLflow.*
*Repositório:* [bridge_ml_pipeline](https://github.com/WallasBorges10/bridge_ml_pipeline.git)