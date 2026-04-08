"""Prediction utilities for loaded models."""

import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """Load saved model from disk."""
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def load_sample_input(sample_path: str) -> pd.DataFrame:
    """Load a sample input from CSV file (saved during training)."""
    logger.info(f"Loading sample input from {sample_path}")
    return pd.read_csv(sample_path)

def predict(model, input_data: pd.DataFrame):
    """Run prediction and return class and probabilities."""
    pred = model.predict(input_data)
    proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
    
    for i, (p, prob) in enumerate(zip(pred, proba)):
        logger.info(f"Sample {i+1} - Prediction: {p} (0=Good, 1=Critical/Fair/Poor), Probabilities: {prob}")
    
    return pred, proba