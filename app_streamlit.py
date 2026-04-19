import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Bridge Condition Predictor - ML Engineering",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# 1. Carregamento do modelo e artefatos
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "output/modelo_pontes_ny_rf_deploy.joblib"
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado em {model_path}. Execute o treinamento primeiro.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_sample():
    sample_path = "output/sample_input.csv"
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return None

@st.cache_data
def load_confusion_matrix():
    img_path = "output/matriz_confusao.png"
    if os.path.exists(img_path):
        return img_path
    return None

@st.cache_data
def load_comparison_plot():
    img_path = "output/comparacao_modelos.png"
    if os.path.exists(img_path):
        return img_path
    return None

model = load_model()
sample_df = load_sample()

# Extrair feature importances do modelo (RandomForest dentro do pipeline)
def get_feature_importances():
    """Extrai feature importances do Random Forest e nomes das features após pré-processamento."""
    try:
        # Acessa o classificador dentro do pipeline
        clf = model.named_steps['classifier']
        if not hasattr(clf, 'feature_importances_'):
            st.warning("O modelo não possui feature_importances_ (não é uma árvore/ensemble).")
            return None
        
        importances = clf.feature_importances_
        
        # Acessa o pré-processador
        preprocessor = model.named_steps['preprocessor']
        
        # Obtém nomes das features após transformação (inclui one-hot)
        # Para versões mais antigas do sklearn, pode ser necessário usar get_feature_names()
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            # Fallback para versões antigas
            feature_names = preprocessor.get_feature_names()
        
        # Verifica se o tamanho coincide
        if len(importances) != len(feature_names):
            st.warning(f"Tamanho diferente: importâncias ({len(importances)}) vs nomes ({len(feature_names)}). Usando fallback.")
            # Fallback: usar nomes das colunas originais (sem expansão)
            num_cols = preprocessor.transformers_[0][2]
            cat_cols = preprocessor.transformers_[1][2]
            # Como one-hot expande, pegamos só as primeiras n importâncias
            fallback_names = list(num_cols) + list(cat_cols)
            feature_names = fallback_names[:len(importances)]
        
        # Cria DataFrame
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feat_imp
        
    except Exception as e:
        st.warning(f"Não foi possível extrair feature importances: {e}")
        # Retorna dados de exemplo (apenas para demonstração)
        return pd.DataFrame({
            'feature': ['AGE', 'ADT_029', 'TRAFFIC_DENSITY', 'STRUCTURE_LEN_MT_049', 
                        'MAX_SPAN_LEN_MT_048', 'DECK_WIDTH_MT_052', 'YEAR_BUILT_027',
                        'STRUCTURE_KIND_043A', 'FUNCTIONAL_CLASS_026', 'TOLL_020'],
            'importance': [0.35, 0.22, 0.12, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01]
        })
# -------------------------------------------------------------------
# 2. Função de predição (compatível com pipeline)
# -------------------------------------------------------------------
def predict_from_dataframe(df):
    """Faz predição usando o pipeline carregado."""
    try:
        pred = model.predict(df)
        proba = model.predict_proba(df)
        return pred, proba
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None

# -------------------------------------------------------------------
# 3. Sidebar com informações do projeto
# -------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("🌉 Projeto")
    st.markdown("""
    **Classificação de Condição de Pontes**  
    *National Bridge Inventory (NBI)*  
    """)
    st.markdown("---")
    st.markdown("**Modelo Final:** Random Forest")
    st.markdown("**Recall (Crítico):** 90,45%")
    st.markdown("**F1-score:** 87,03%")
    st.markdown("**Acurácia:** 81,74%")
    st.markdown("---")
    st.markdown("📁 **Repositório:** [GitHub](https://github.com/WallasBorges10/bridge_ml_pipeline)")
    st.markdown("📊 **MLflow:** UI em http://localhost:5000")

# -------------------------------------------------------------------
# 4. Abas principais
# -------------------------------------------------------------------
tab_overview, tab_importance, tab_performance, tab_batch, tab_manual = st.tabs(
    ["📋 Visão Geral", "📊 Importância das Features", "📈 Performance", "📂 Predição em Lote", "✍️ Predição Manual"]
)

# ------------------- TAB 1: VISÃO GERAL -------------------
with tab_overview:
    st.header("Sistema de Apoio à Decisão - Manutenção de Pontes")
    st.markdown("""
    Este aplicativo utiliza um modelo de **Machine Learning** treinado com dados históricos do 
    *National Bridge Inventory (NBI)* para classificar pontes em:
    - ✅ **Boa (Good)** – condição estrutural satisfatória, inspeção rotineira.
    - ⚠️ **Crítica (Fair/Poor)** – necessita de atenção ou reparos em curto prazo.
    
    **Objetivo de Negócio:** Priorizar inspeções e alocar recursos de manutenção de forma eficiente, 
    reduzindo custos e aumentando a segurança.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recall (Classe Crítica)", "90,45%", delta="+8% vs Baseline")
    with col2:
        st.metric("F1-score", "87,03%", delta="+3%")
    with col3:
        st.metric("Tempo Médio de Inferência", "0.12 ms", delta="< 1 ms")
    
    st.subheader("Fluxo do Pipeline de ML")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*0W7CrN9VZ8k5JqMqE3vQqA.png", caption="Pipeline de Engenharia de ML")
    st.caption("Fonte: ilustração conceitual - Dados → Pré-processamento → Modelo → Predição → Monitoramento")

# ------------------- TAB 2: IMPORTÂNCIA DAS FEATURES -------------------
with tab_importance:
    st.header("Top Features mais Relevantes para a Decisão")
    st.markdown("""
    A importância das features (Gini Importance) indica quais variáveis mais influenciam a classificação.
    As cinco principais são:
    1. **Idade da estrutura (AGE)** – quanto mais antiga, maior risco.
    2. **Tráfego diário (ADT)** – maior volume, maior desgaste.
    3. **Densidade de tráfego** – ADT / faixas.
    4. **Comprimento da ponte** – estruturas mais longas tendem a mais problemas.
    5. **Vão máximo** – impacto na distribuição de cargas.
    """)
    feat_imp = get_feature_importances()
    if feat_imp is not None and not feat_imp.empty:
        top10 = feat_imp.head(10)
        fig = px.bar(top10, x='importance', y='feature', orientation='h',
                     title="Importância das Features (Random Forest)",
                     labels={'importance': 'Importância (Gini)', 'feature': 'Variável'},
                     color='importance', color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados de importância não disponíveis. Execute o treinamento para gerar.")
    
    # Explicação adicional
    with st.expander("🔍 Entenda como a importância é calculada"):
        st.markdown("""
        A importância de cada feature é calculada pela **redução média de impureza (Gini)** ao longo de todas as árvores do Random Forest.
        Quanto maior o valor, mais a feature contribui para a separação das classes (Boa vs. Crítica).
        """)

# ------------------- TAB 3: PERFORMANCE -------------------
with tab_performance:
    st.header("Análise de Performance do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Confusão (Normalizada)")
        cm_img = load_confusion_matrix()
        if cm_img:
            st.image(cm_img, use_container_width=True)
        else:
            st.info("Matriz de confusão não encontrada. Execute o treinamento para gerar.")
    with col2:
        st.subheader("Comparação entre Modelos")
        comp_img = load_comparison_plot()
        if comp_img:
            st.image(comp_img, use_container_width=True)
        else:
            st.info("Gráfico de comparação não encontrado. Execute o treinamento para gerar.")
    
    st.subheader("Métricas Detalhadas")
    metrics_data = {
        "Modelo": ["Perceptron", "Decision Tree", "Random Forest", "RF + PCA", "RF + LDA"],
        "Recall": [0.8026, 0.8232, 0.9045, 0.8910, 0.7900],
        "Precisão": [0.8371, 0.8586, 0.8386, 0.8097, 0.8613],
        "F1-score": [0.8194, 0.8405, 0.8703, 0.8590, 0.8241],
        "Acurácia": [0.7604, 0.7883, 0.8174, 0.8009, 0.7715]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Gráfico de barras comparativo
    fig_comp = px.bar(df_metrics, x="Modelo", y=["Recall", "Precisão", "F1-score", "Acurácia"],
                      barmode="group", title="Comparação de Métricas por Modelo")
    st.plotly_chart(fig_comp, use_container_width=True)

# ------------------- TAB 4: PREDIÇÃO EM LOTE (UPLOAD CSV) -------------------
with tab_batch:
    st.header("Predição em Lote - Upload de Arquivo CSV")
    st.markdown("""
    Envie um arquivo CSV contendo os dados das pontes. O arquivo deve conter as mesmas colunas 
    utilizadas no treinamento (exemplo disponível em `output/sample_input.csv`).
    """)
    uploaded_file = st.file_uploader("Selecione um arquivo CSV", type="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("**Prévia dos dados carregados:**")
        st.dataframe(df_input.head())
        if st.button("🔍 Executar Predição", key="batch_btn"):
            with st.spinner("Processando..."):
                pred, proba = predict_from_dataframe(df_input)
                if pred is not None:
                    results = pd.DataFrame({
                        "Registro": range(1, len(pred)+1),
                        "Condição Prevista": ["Crítica" if p == 1 else "Boa" for p in pred],
                        "Probabilidade (Crítica)": [f"{p:.2%}" for p in proba[:, 1]]
                    })
                    st.success("Predição concluída!")
                    st.dataframe(results, use_container_width=True)
                    # Gráfico de distribuição das probabilidades
                    fig_hist = px.histogram(x=proba[:, 1], nbins=20, 
                                            labels={'x': 'Probabilidade de ser Crítica'},
                                            title="Distribuição das Probabilidades")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Opção de download dos resultados
                    csv_results = results.to_csv(index=False)
                    st.download_button("📥 Baixar resultados (CSV)", csv_results, "predicoes.csv", "text/csv")

# ------------------- TAB 5: PREDIÇÃO MANUAL (FORMULÁRIO) -------------------
with tab_manual:
    st.header("Predição Individual - Preencha os dados da ponte")
    st.markdown("Utilize os campos abaixo para simular a condição de uma ponte específica.")
    
    # Carregar uma amostra base para manter as colunas esperadas
    if sample_df is None:
        st.error("Arquivo sample_input.csv não encontrado. Execute o treinamento primeiro.")
        st.stop()
    base_sample = sample_df.iloc[0:1].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        year_built = st.number_input("Ano de Construção", min_value=1800, max_value=2025, value=1970)
        adt = st.number_input("Tráfego Diário (ADT)", min_value=0, value=5000)
        lanes = st.number_input("Número de Faixas", min_value=1, value=2)
        length = st.number_input("Comprimento da Ponte (m)", min_value=0.0, value=50.0)
        max_span = st.number_input("Vão Máximo (m)", min_value=0.0, value=15.0)
    with col2:
        deck_width = st.number_input("Largura do Tabuleiro (m)", min_value=0.0, value=8.0)
        struct_kind = st.selectbox("Tipo de Estrutura", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
        service_on = st.selectbox("Serviço na Ponte", [1, 2, 3, 4, 5], index=1)
        functional_class = st.selectbox("Classe Funcional", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], index=0)
        toll = st.selectbox("Pedágio", [3, 1, 2], index=0, format_func=lambda x: {3: "Sem pedágio", 1: "Pedágio", 2: "Em túnel"}.get(x, "Desconhecido"))
    
    # Botão de predição
    if st.button("🔍 Predizer Condição", key="manual_btn"):
        # Criar DataFrame a partir da amostra, substituindo valores
        input_df = base_sample.copy()
        # Atualizar campos
        input_df['YEAR_BUILT_027'] = year_built
        input_df['ADT_029'] = adt
        input_df['TRAFFIC_LANES_ON_028A'] = lanes
        input_df['STRUCTURE_LEN_MT_049'] = length
        input_df['MAX_SPAN_LEN_MT_048'] = max_span
        input_df['DECK_WIDTH_MT_052'] = deck_width
        input_df['STRUCTURE_KIND_043A'] = struct_kind
        input_df['SERVICE_ON_042A'] = service_on
        input_df['FUNCTIONAL_CLASS_026'] = functional_class
        input_df['TOLL_020'] = toll
        # Recalcular features derivadas
        current_year = datetime.now().year
        input_df['AGE'] = current_year - year_built
        input_df['TRAFFIC_DENSITY'] = adt / (lanes + 1)
        # Usar média e std da amostra original para normalizar AGE (aprox.)
        age_mean = sample_df['AGE'].mean()
        age_std = sample_df['AGE'].std()
        input_df['AGE_NORMALIZED'] = (input_df['AGE'] - age_mean) / age_std
        
        with st.spinner("Processando..."):
            pred, proba = predict_from_dataframe(input_df)
            if pred is not None:
                resultado = "Crítica" if pred[0] == 1 else "Boa"
                prob_critica = proba[0][1]
                st.success(f"### Resultado: **{resultado}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilidade de ser Crítica", f"{prob_critica:.2%}")
                with col2:
                    st.metric("Nível de Confiança", f"{max(proba[0]):.2%}")
                # Gráfico gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_critica * 100,
                    title = {'text': "Risco Crítico (%)"},
                    domain = {'x': [0,1], 'y': [0,1]},
                    gauge = {'axis': {'range': [0,100]},
                             'bar': {'color': "darkred"},
                             'steps': [
                                 {'range': [0, 30], 'color': "lightgreen"},
                                 {'range': [30, 70], 'color': "orange"},
                                 {'range': [70, 100], 'color': "red"}],
                             'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Explicação
                if prob_critica > 0.7:
                    st.warning("⚠️ Alta probabilidade de condição crítica. Recomenda-se inspeção prioritária.")
                elif prob_critica > 0.3:
                    st.info("📌 Probabilidade moderada. Monitoramento regular.")
                else:
                    st.success("✅ Baixa probabilidade de condição crítica. Manutenção rotineira.")

# Rodapé
st.markdown("---")
st.caption("Projeto de Engenharia de Machine Learning - Pós-Graduação em MLOps | Modelo Random Forest | Monitoramento de Drift integrado com Evidently e MLflow")