# Relatório Técnico – Projeto de Engenharia de Machine Learning

## 1. Decisões de Projeto

### 1.1 Objetivo técnico
Construir um classificador binário para prever se uma ponte está em condição **crítica (Fair/Poor)** ou **boa (Good)**, utilizando apenas variáveis disponíveis *antes* da inspeção (evitando data leakage). O modelo será usado para priorizar inspeções e manutenção.

### 1.2 Métricas de sucesso
- **Primária:** Recall da classe crítica – maximizar a identificação de pontes que precisam de intervenção.
- **Secundária:** F1-score – equilíbrio entre recall e precisão.
- **Negócio:** Redução de custos com inspeções desnecessárias e aumento da segurança.

### 1.3 Escolha de features
Foram selecionadas 62 features de domínio (excluídas 14 leakage features). Engenharia de features adicionou `AGE`, `TRAFFIC_DENSITY` e `AGE_NORMALIZED`.

### 1.4 Pipeline de dados
- Remoção de colunas com >50% de missing.
- Imputação: mediana para numéricas, moda para categóricas.
- One‑hot encoding para categóricas.
- Padronização (`StandardScaler`) para numéricas.

## 2. Análise Comparativa de Experimentos

| Modelo                     | Recall | F1    | Precisão | Acurácia | Tempo treino (s) | Inferência (ms) |
|----------------------------|--------|-------|----------|----------|------------------|-----------------|
| Perceptron (baseline)      | 0,8026 | 0,8194| 0,8371   | 0,7604   | 0,5              | 0,02            |
| Decision Tree (tuned)      | 0,8232 | 0,8405| 0,8586   | 0,7883   | 2,1              | 0,03            |
| Random Forest (tuned)      | 0,9045 | 0,8703| 0,8386   | 0,8174   | 31,5             | 0,12            |
| RF + PCA (95% var)         | 0,8910 | 0,8590| 0,8097   | 0,8009   | 28,1             | 0,10            |
| RF + LDA (1 componente)    | 0,7900 | 0,8241| 0,8613   | 0,7715   | 24,7             | 0,09            |

*Resultados registrados no MLflow – experimento `bridge_condition_prediction`.*

## 3. Redução de Dimensionalidade

### 3.1 Técnicas escolhidas
- **PCA** (não supervisionada): reduz ruído e correlações lineares. Manteve 95% da variância.
- **LDA** (supervisionada): maximiza separabilidade entre classes. Reduziu para 1 componente (por ser binário).

### 3.2 Impacto observado
- **PCA:** pequena queda no recall (−1,5%) e F1 (−1,1%), mas redução de 10% no tempo de treino.
- **LDA:** perda significativa de recall (de 0,9045 para 0,7900), inviável para o negócio.

### 3.3 Conclusão sobre dimensionalidade
A redução não é recomendada para este problema. O ganho computacional é marginal e a perda de recall compromete o objetivo de negócio. O Random Forest original já lida bem com a dimensionalidade e fornece importância de features (interpretável).

## 4. Modelo Final Escolhido

**Random Forest** com hiperparâmetros:
- `n_estimators=200`
- `max_depth=None`
- `class_weight='balanced'`
- Demais parâmetros _default_ do scikit‑learn.

**Justificativa:**
- Melhor recall (0,9045) – essencial para não deixar pontes críticas sem inspeção.
- Melhor F1 (0,8703) – bom equilíbrio.
- Interpretabilidade via `feature_importances_`.
- Custo computacional aceitável (31 s treino, 0,12 ms inferência).

## 5. Operacionalização e Monitoramento

### 5.1 Versionamento
Modelo salvo como `modelo_pontes_ny_rf_deploy.joblib` e registrado no MLflow Model Registry como `BridgeConditionRF` (versão 1, estágio Production).

### 5.2 API de inferência
Serviço Flask (`src/app.py`) expõe endpoint `/predict`. Exemplo de requisição:
```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d @input.json