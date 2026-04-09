"""Feature engineering and target creation."""

import numpy as np
import pandas as pd
import logging
from config import TARGET_MAP, TARGET_COL, CURRENT_YEAR

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: AGE, TRAFFIC_DENSITY, AGE_NORMALIZED."""
    df = df.copy()
    df['AGE'] = CURRENT_YEAR - df['YEAR_BUILT_027']
    df.loc[df['AGE'] < 0, 'AGE'] = 0

    if 'ADT_029' in df.columns and 'TRAFFIC_LANES_ON_028A' in df.columns:
        df['TRAFFIC_DENSITY'] = df['ADT_029'] / (df['TRAFFIC_LANES_ON_028A'] + 1)
        df['TRAFFIC_DENSITY'] = df['TRAFFIC_DENSITY'].replace([np.inf, -np.inf], np.nan)

    df['AGE_NORMALIZED'] = (df['AGE'] - df['AGE'].mean()) / df['AGE'].std()
    logger.info("Engineered features: AGE, TRAFFIC_DENSITY, AGE_NORMALIZED")
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map BRIDGE_CONDITION to binary target."""
    df = df.copy()
    df['TARGET'] = df[TARGET_COL].map(TARGET_MAP)
    df = df.dropna(subset=['TARGET'])
    df['TARGET'] = df['TARGET'].astype(int)
    df = df.drop(columns=[TARGET_COL])
    logger.info(f"Target distribution:\n{df['TARGET'].value_counts(normalize=True)}")
    return df