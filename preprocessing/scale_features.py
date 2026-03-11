import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from config.config import SCALER_TYPE
from utils.helpers import get_logger

logger = get_logger(__name__)

def scale_features(X):
    """
    Scales features based on configuration (RobustScaler or StandardScaler).
    Returns (X_scaled, scaler)
    """
    logger.info(f"Scaling features using {SCALER_TYPE} scaler...")

    if SCALER_TYPE.lower() == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Fit and transform
    X_scaled_raw = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_raw, columns=X.columns, index=X.index)

    return X_scaled, scaler