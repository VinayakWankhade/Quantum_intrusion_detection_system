import pandas as pd
from sklearn.decomposition import PCA
from config.config import PCA_EXPLAINED_VARIANCE
from utils.helpers import get_logger

logger = get_logger(__name__)

def apply_pca(X, n_components=None):
    """
    Applies PCA reduction. If n_components is not provided, it falls back
    to the configured PCA_EXPLAINED_VARIANCE target.
    Returns (X_pca_df, pca)
    """
    target_components = n_components if n_components is not None else PCA_EXPLAINED_VARIANCE
    logger.info(f"Applying PCA with target components/variance: {target_components}")

    pca = PCA(n_components=target_components)
    X_pca = pca.fit_transform(X)

    # Convert back to DataFrame
    pca_columns = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

    logger.info(f"PCA reduced dimensions from {X.shape[1]} to {X_pca_df.shape[1]}")
    logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    return X_pca_df, pca