import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from config.config import SCALER_TYPE
from utils.helpers import get_logger

logger = get_logger(__name__)

def scale_features(X):
    """
    Scales features based on configuration (RobustScaler or StandardScaler).
    Returns a pandas DataFrame to maintain feature names. 
    """
    logger.info(f"Scaling features using {SCALER_TYPE} scaler...")

    if SCALER_TYPE.lower() == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # We convert back to DataFrame to preserve column names
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled