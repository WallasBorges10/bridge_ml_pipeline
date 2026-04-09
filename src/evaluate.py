from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics