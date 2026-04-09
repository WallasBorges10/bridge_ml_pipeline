import mlflow
import mlflow.sklearn
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from config import RANDOM_STATE, DT_PARAM_GRID, RF_PARAM_GRID, CV_FOLDS, SCORING
from pipeline_builder import create_pipeline

def train_perceptron(X_train, y_train, preprocessor):
    with mlflow.start_run(run_name="Perceptron", nested=True):
        model = create_pipeline(
            preprocessor,
            Perceptron(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE, class_weight='balanced')
        )
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "perceptron_model")
        return model

def train_decision_tree(X_train, y_train, preprocessor):
    with mlflow.start_run(run_name="DecisionTree_tuned", nested=True):
        base_pipeline = create_pipeline(preprocessor, DecisionTreeClassifier(random_state=RANDOM_STATE))
        grid_search = GridSearchCV(base_pipeline, DT_PARAM_GRID, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "decision_tree_model")
        return grid_search

def train_random_forest(X_train, y_train, preprocessor):
    with mlflow.start_run(run_name="RandomForest_tuned", nested=True):
        base_pipeline = create_pipeline(
            preprocessor,
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        )
        grid_search = GridSearchCV(base_pipeline, RF_PARAM_GRID, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")
        return grid_search