import pandas as pd
from unittest.mock import patch
from src.pipeline_builder import build_preprocessor
from src.train import train_perceptron, train_random_forest

@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_params")
@patch("mlflow.start_run")
def test_train_perceptron(mock_start_run, mock_log_params, mock_log_model):
    X = pd.DataFrame({'feat': list(range(30))})
    y = [0, 1] * 15
    preprocessor = build_preprocessor(X)
    model = train_perceptron(X, y, preprocessor)
    assert hasattr(model, 'predict')

@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_params")
@patch("mlflow.start_run")
def test_train_random_forest(mock_start_run, mock_log_params, mock_log_model):
    X = pd.DataFrame({'feat': list(range(30))})
    y = [0, 1] * 15
    preprocessor = build_preprocessor(X)
    grid = train_random_forest(X, y, preprocessor)
    assert hasattr(grid, 'best_estimator_')