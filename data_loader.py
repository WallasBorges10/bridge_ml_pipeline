"""Data loading and initial cleaning."""

import pandas as pd
import logging
from config import DATA_URL, TARGET_COL, LEAKY_FEATURES

logger = logging.getLogger(__name__)

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load raw NBI data from URL."""
    logger.info(f"Loading data from {url}")
    df = pd.read_csv(url, sep=",", quotechar="'", low_memory=False)
    logger.info(f"Initial shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with >50% missing, duplicates, rows with null target."""
    logger.info("Cleaning data...")
    missing_perc = df.isnull().sum() / len(df)
    cols_to_drop = missing_perc[missing_perc > 0.5].index
    df_clean = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns (>50% missing)")

    df_clean = df_clean.drop_duplicates()
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[TARGET_COL])
    logger.info(f"Dropped {initial_len - len(df_clean)} rows with null target")
    logger.info(f"Shape after cleaning: {df_clean.shape}")
    return df_clean

def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known leakage features."""
    present = [c for c in LEAKY_FEATURES if c in df.columns]
    df_no_leak = df.drop(columns=present, errors='ignore')
    logger.info(f"Removed {len(present)} leakage features")
    return df_no_leak

def select_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only domain-defined non-conditional features."""
    from config import DOMAIN_FEATURES
    existing = [c for c in DOMAIN_FEATURES if c in df.columns]
    df_selected = df[existing].copy()
    logger.info(f"Selected {len(existing)} domain features")
    return df_selected