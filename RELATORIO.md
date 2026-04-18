# Relatório Técnico – Projeto de Engenharia de Machine Learning

## 1. Contexto e Evolução do Projeto

### 1.1 Mapeamento de Experimentos Anteriores
O ponto de partida deste projeto foram os experimentos exploratórios conduzidos em notebooks, onde modelos foram testados de forma isolada. A transição para um projeto de engenharia visa resolver a **falta de rastreabilidade** e a **dificuldade de operacionalização** do código original.

| Modelo | Acurácia (Validação) | Limitações Observadas |
| :--- | :--- | :--- |
| **Perceptron Simples** | ~74% | Alta variância; incapaz de capturar padrões não-lineares. |
| **Árvore de Decisão** | ~82% | Sobreajuste (overfitting) severo sem controle de profundidade. |
| **KNN** | ~78% | Custo computacional elevado durante a inferência. |

### 1.2 Objetivo Técnico e Métricas de Sucesso
Construir um classificador binário para prever se uma ponte está em condição **crítica (Fair/Poor)** ou **boa (Good)**, utilizando apenas variáveis disponíveis *antes* da inspeção para evitar o vazamento de dados (*data leakage*).

* **Métrica Primária:** Recall da classe crítica (maximizar identificação de riscos).
* **Métrica Secundária:** F1-score (equilíbrio entre precisão e recall).
* **Objetivo de Negócio:** Redução de custos operacionais e aumento da segurança estrutural.

### 1.3 Engenharia de Features e Pipeline
Foram selecionadas **62 features** de domínio, com a exclusão de 14 variáveis que causavam *leakage*.
* **Novas Features:** `AGE`, `TRAFFIC_DENSITY` e `AGE_NORMALIZED`.
* **Tratamento de Dados:** * Remoção de colunas com >50% de dados faltantes.
    * Imputação de mediana (numéricas) e moda (categóricas).
    * Codificação via *One-Hot Encoding*.
    * Padronização via `StandardScaler`.

---

## 2. Fundação de Dados e Diagnóstico de Qualidade

### 2.1 Estratégia de Ingestão e Amostragem
* **Fonte:** Dataset público de Pontes (National Bridge Inventory).
* **Amostragem:** Divisão estratificada (`train_test_split` com `stratify=y`) para preservar a distribuição das classes de condição da ponte.

### 2.2 Diagnóstico de Riscos e Mitigação
1.  **Dados Ausentes (Missing Values):** Colunas com alta vacância foram removidas para evitar ruído. A imputação por mediana foi escolhida por sua robustez contra outliers.
2.  **Viés de Representação:** Identificou-se predominância de pontes federais. O risco de degradação de performance em pontes municipais (Drift) será monitorado via **PSI (Population Stability Index)**.
3.  **Inconsistências Categóricas:** Padronização de strings (`lowercase` e `strip`) aplicada no pré-processamento.

---

## 3. Análise Comparativa de Experimentos

Os resultados abaixo foram registrados e versionados utilizando o **MLflow** no experimento `bridge_condition_prediction`.

| Modelo | Recall | F1 | Precisão | Acurácia | Treino (s) | Inferência (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Perceptron (baseline) | 0,8026 | 0,8194 | 0,8371 | 0,7604 | 0,5 | 0,02 |
| Decision Tree (tuned) | 0,8232 | 0,8405 | 0,8586 | 0,7883 | 2,1 | 0,03 |
| **Random Forest (tuned)** | **0,9045** | **0,8703** | 0,8386 | 0,8174 | 31,5 | 0,12 |
| RF + PCA (95% var) | 0,8910 | 0,8590 | 0,8097 | 0,8009 | 28,1 | 0,10 |
| RF + LDA (1 comp.) | 0,7900 | 0,8241 | 0,8613 | 0,7715 | 24,7 | 0,09 |

---

## 4. Redução de Dimensionalidade

### 4.1 Técnicas Avaliadas
* **PCA (Principal Component Analysis):** Utilizada para redução de ruído, mantendo 95% da variância explicada.
* **LDA (Linear Discriminant Analysis):** Focada na maximização da separabilidade das classes (reduzida a 1 componente).

### 4.2 Impacto e Conclusão
Embora o PCA tenha reduzido o tempo de treino em 10%, houve uma queda no Recall e F1. O LDA mostrou-se inviável, com perda significativa de performance (Recall caiu para 0,79). **Conclusão:** A redução de dimensionalidade não é recomendada para o deploy final, visto que o Random Forest lida bem com a dimensionalidade original e a perda de sensibilidade clínica/técnica é inaceitável para o negócio.

---

## 5. Modelo Final Escolhido

O modelo selecionado para produção foi o **Random Forest (Tuned)**.

* **Configuração de Hiperparâmetros:**
    * `n_estimators`: 200
    * `max_depth`: None (expansão total)
    * `class_weight`: 'balanced' (correção de desbalanceamento)

* **Justificativa:** Apresentou o maior **Recall (0,9045)**, garantindo que a maioria das pontes em estado crítico seja identificada. Além disso, oferece interpretabilidade através da importância das variáveis (*feature importances*).

---

## 6. Operacionalização e Monitoramento

### 6.1 Versionamento e Registro
O artefato final foi salvo como `modelo_pontes_ny_rf_deploy.joblib` e registrado no **MLflow Model Registry**:
* **Nome:** `BridgeConditionRF`
* **Versão:** 1
* **Estágio:** Production

### 6.2 API de Inferência
Uma API Flask foi desenvolvida para servir o modelo em tempo real através do endpoint `/predict`.

**Exemplo de Requisição (cURL):**
```bash
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d @input.json
```

### 6.3 Estratégia de Monitoramento
O sistema conta com um detector de drift (`drift_detector.py`) que avalia periodicamente a estabilidade da população de entrada para identificar quando o modelo precisa ser retreinado devido a mudanças nas características das pontes inspecionadas.