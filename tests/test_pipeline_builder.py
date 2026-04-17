import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.pipeline_builder import build_preprocessor, build_pipeline_with_pca, build_pipeline_with_lda

def test_build_preprocessor():
    X = pd.DataFrame({
        'num1': [1, 2, None],
        'num2': [10, 20, 30],
        'cat1': ['a', 'b', 'a']
    })
    preprocessor = build_preprocessor(X)
    assert preprocessor is not None
    # apenas testa se não lança erro
    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == 3

def test_build_pipeline_with_pca():
    preprocessor = build_preprocessor(pd.DataFrame({'a': [1,2,3]}))
    clf = RandomForestClassifier()
    pipeline = build_pipeline_with_pca(preprocessor, clf, n_components=0.95)
    assert 'pca' in pipeline.named_steps

def test_build_pipeline_with_lda():
    preprocessor = build_preprocessor(pd.DataFrame({'a': [1,2,3]}))
    clf = RandomForestClassifier()
    pipeline = build_pipeline_with_lda(preprocessor, clf, n_components=1)
    assert 'lda' in pipeline.named_steps