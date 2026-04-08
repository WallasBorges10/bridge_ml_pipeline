"""Build preprocessing pipeline and model pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import logging

logger = logging.getLogger(__name__)

def build_preprocessor(X):
    """Build ColumnTransformer for numeric and categorical features."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    logger.info(f"Preprocessor built: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    return preprocessor

def create_pipeline(preprocessor, classifier):
    """Create a full pipeline with preprocessor and classifier."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def build_pipeline_with_pca(preprocessor, n_components=0.95):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components)),
        ('classifier', RandomForestClassifier(...))
    ])