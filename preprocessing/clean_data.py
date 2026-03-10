from sklearn.impute import SimpleImputer
import pandas as pd
from utils.helpers import get_logger

logger = get_logger(__name__)

def clean_data(df, strategy="median"):
    """
    Cleans the datasets by dropping exact duplicates and handling missing values
    using a robust imputation strategy.
    """
    logger.info("Cleaning dataset...")
    
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Separate categorical and numerical columns for safe imputation
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Impute numericals
    if len(numerical_cols) > 0:
        logger.info(f"Imputing missing numerical values using '{strategy}' strategy.")
        num_imputer = SimpleImputer(strategy=strategy)
        df.loc[:, numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categoricals
    if len(categorical_cols) > 0:
        logger.info("Imputing missing categorical values using 'most_frequent' strategy.")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df.loc[:, categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df