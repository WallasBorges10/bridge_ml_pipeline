import pandas as pd
import numpy as np
from src.preprocessing import engineer_features, create_target

def test_engineer_features():
    df = pd.DataFrame({
        'YEAR_BUILT_027': [2000, 1990],
        'ADT_029': [5000, 3000],
        'TRAFFIC_LANES_ON_028A': [2, 1],
        'BRIDGE_CONDITION': ['G', 'P']
    })
    df = engineer_features(df)
    assert 'AGE' in df.columns
    assert 'TRAFFIC_DENSITY' in df.columns
    assert 'AGE_NORMALIZED' in df.columns
    assert not df['AGE'].isnull().any()
    assert not df['TRAFFIC_DENSITY'].isnull().any()

def test_create_target():
    df = pd.DataFrame({'BRIDGE_CONDITION': ['G', 'F', 'P', 'N']})
    df = create_target(df)
    assert 'TARGET' in df.columns
    # A linha 'N' vira NaN e é removida, restam 3 linhas
    assert df['TARGET'].tolist() == [0, 1, 1]
    assert len(df) == 3