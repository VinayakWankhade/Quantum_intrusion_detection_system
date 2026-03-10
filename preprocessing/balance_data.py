from imblearn.over_sampling import SMOTE
from utils.helpers import get_logger

logger = get_logger(__name__)

def balance_classes(X, y, random_state=42):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance 
    class distribution in the training data.
    """
    logger.info("Balancing dataset classes using SMOTE...")
    logger.info(f"Original class distribution:\n{y.value_counts()}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logger.info(f"Resampled class distribution:\n{y_resampled.value_counts()}")
    
    return X_resampled, y_resampled
