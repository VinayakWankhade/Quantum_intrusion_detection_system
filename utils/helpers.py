import os
import logging
import pandas as pd
import joblib

def get_logger(name="quantum_ids", log_file="quantum_ids.log"):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler if log_file is provided
        if log_file:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(f"logs/{log_file}")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

logger = get_logger()

def load_csv(path):
    logger.info(f"Loading dataset from {path}")
    return pd.read_csv(path)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved dataset to {path}")

def log_shape(name, df):
    logger.info(f"{name} shape: {df.shape}")

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath):
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)