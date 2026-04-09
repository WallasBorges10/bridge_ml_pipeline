"""Main entry point for the bridge ML pipeline."""

import os
import sys
import argparse
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from data_loader import load_data, clean_data, remove_leakage, select_domain_features
from preprocessing import engineer_features, create_target
from pipeline_builder import build_preprocessor
from train import train_perceptron, train_decision_tree, train_random_forest
from evaluate import evaluate_model
from plots import plot_comparison, plot_confusion_matrix, plot_feature_importance, log_artifact_if_needed

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_training(output_dir='.'):
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Configura MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Run principal
    with mlflow.start_run(run_name="full_pipeline") as parent_run:
        # Log de parâmetros gerais
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # 1. Carregar e preparar dados
        df_raw = load_data()
        df_clean = clean_data(df_raw)
        df_no_leak = remove_leakage(df_clean)
        df_selected = select_domain_features(df_no_leak)
        df_feat = engineer_features(df_selected)
        df_target = create_target(df_feat)

        # 2. Split
        X = df_target.drop(columns=['TARGET'])
        y = df_target['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # 3. Pré-processador
        preprocessor = build_preprocessor(X_train)

        # 4. Treinar modelos (cada um em sua própria run aninhada)
        perc_model = train_perceptron(X_train, y_train, preprocessor)
        dt_grid = train_decision_tree(X_train, y_train, preprocessor)
        rf_grid = train_random_forest(X_train, y_train, preprocessor)

        # 5. Avaliar (métricas logadas na run principal)
        perc_metrics = evaluate_model(perc_model, X_test, y_test, "Perceptron")
        dt_metrics = evaluate_model(dt_grid.best_estimator_, X_test, y_test, "DecisionTree")
        rf_metrics = evaluate_model(rf_grid.best_estimator_, X_test, y_test, "RandomForest")

        # Log das métricas na run principal
        for name, metrics in [("Perceptron", perc_metrics), ("DecisionTree", dt_metrics), ("RandomForest", rf_metrics)]:
            for k, v in metrics.items():
                mlflow.log_metric(f"{name}_{k}", v)

        # 6. Gráficos e artefatos
        metrics_dict = {
            'Perceptron': perc_metrics,
            'Decision Tree': dt_metrics,
            'Random Forest': rf_metrics
        }
        plot_comparison(metrics_dict, os.path.join(output_dir, 'output/comparacao_modelos.png'))
        plot_confusion_matrix(rf_grid.best_estimator_, X_test, y_test, os.path.join(output_dir, 'output/matrizes_confusao.png'))
        plot_feature_importance(rf_grid.best_estimator_, os.path.join(output_dir, 'output/importancia_features.png'))

        # Log dos gráficos como artefatos
        for fname in ['output/comparacao_modelos.png', 'output/matrizes_confusao.png', 'output/importancia_features.png']:
            log_artifact_if_needed(os.path.join(output_dir, fname))

        # 7. Salvar sample_input.csv e logar
        sample_X_test = X_test.head(5)
        sample_path = os.path.join(output_dir, 'sample_input.csv')
        sample_X_test.to_csv(sample_path, index=False)
        log_artifact_if_needed(sample_path)

        # 8. Salvar modelo final e logar como artefato e no Model Registry
        final_model_path = os.path.join(output_dir, 'output/modelo_pontes_ny_rf_deploy.joblib')
        joblib.dump(rf_grid.best_estimator_, final_model_path)
        log_artifact_if_needed(final_model_path)

        # Log do modelo no MLflow (para registro)
        mlflow.sklearn.log_model(rf_grid.best_estimator_, "random_forest_best_model")
        mlflow.register_model(f"runs:/{parent_run.info.run_id}/random_forest_best_model", "BridgeConditionRF")

        logger.info(f"Models saved to {output_dir}")

    return rf_grid.best_estimator_

def run_prediction(model_path: str, sample_path: str):
    """Load model and run prediction on sample."""
    from predict import load_model, load_sample_input, predict
    model = load_model(model_path)
    sample = load_sample_input(sample_path)
    predict(model, sample)

def main():
    parser = argparse.ArgumentParser(description="Bridge Condition Prediction Pipeline")
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                        help="Run training pipeline or load model for prediction")
    parser.add_argument('--model_path', type=str, default='modelo_pontes_ny_rf_deploy.joblib',
                        help="Path to saved model (for predict mode)")
    parser.add_argument('--sample_path', type=str, default='./output/sample_input.csv',
                        help="Path to sample input CSV (for predict mode)")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="Directory to save models and plots")
    args = parser.parse_args()

    if args.mode == 'train':
        run_training(output_dir=args.output_dir)
    elif args.mode == 'predict':
        run_prediction(args.model_path, args.sample_path)
    else:
        logger.error("Invalid mode. Use 'train' or 'predict'.")

if __name__ == '__main__':
    main()