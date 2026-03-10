import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.helpers import get_logger

logger = get_logger(__name__)

def encode_features(X):
    """
    Applies OneHotEncoding to categorical features, maintaining a DataFrame structure.
    Handles unseen categories in future test sets by ignoring them.
    """
    logger.info("Encoding categorical features...")

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns

    if len(categorical_cols) == 0:
        logger.info("No categorical features found to encode.")
        return X

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform categorical columns
    encoded_data = encoder.fit_transform(X[categorical_cols])
    
    # Create descriptive column names for the encoded features
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=X.index)

    # Drop original categorical columns and concatenate encoded ones
    X_numerical = X.drop(columns=categorical_cols)
    X_encoded = pd.concat([X_numerical, encoded_df], axis=1)

    logger.info(f"Encoding complete. Features expanded from {X.shape[1]} to {X_encoded.shape[1]}")
    return X_encoded