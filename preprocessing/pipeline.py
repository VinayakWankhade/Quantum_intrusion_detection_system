import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import TEST_SIZE, RANDOM_STATE, PCA_EXPLAINED_VARIANCE
from utils.helpers import load_csv, save_csv, log_shape, get_logger

from preprocessing.clean_data import clean_data
from preprocessing.encode_features import encode_features
from preprocessing.scale_features import scale_features
from preprocessing.pca_reduction import apply_pca
from preprocessing.balance_data import balance_classes

logger = get_logger(__name__)

class PreprocessingPipeline:

    def __init__(self, input_path):
        self.input_path = input_path

    def run(self):
        logger.info(f"Starting Preprocessing pipeline for {self.input_path}")

        # 1. Load dataset
        df = load_csv(self.input_path)
        log_shape("Raw dataset", df)

        # 2. Clean data (Handle duplicates & imputation)
        df = clean_data(df)
        log_shape("Clean dataset", df)

        # 3. Separate features and label
        label_col = "label" if "label" in df.columns else "Label"
        if label_col not in df.columns:
            logger.error("Label column not found in dataset")
            raise Exception("Label column not found")

        y = df[label_col]
        X = df.drop(columns=[label_col])

        # Convert textual labels to binary (if applicable, e.g., NSL-KDD anomaly mapping)
        # Note: Depending on your exact format, y might already be numeric.
        if y.dtype == 'object':
            logger.info("Converting string labels to binary (anomaly vs normal)")
            y = y.apply(lambda val: 0 if val.lower() == 'normal' else 1)
        elif y.nunique() > 2:
            logger.info(f"Found {y.nunique()} classes. Assuming label 21 or 'normal' means 0 (benign), else 1.")
            # Based on common NSL-KDD mapping if encoded early
            y = y.apply(lambda val: 0 if val == 21 else 1)
            
        # 4. Encode categorical features
        X_encoded = encode_features(X)
        log_shape("Encoded dataset", X_encoded)

        # 5. Scale features
        X_scaled = scale_features(X_encoded)

        # 6. PCA reduction
        X_pca = apply_pca(X_scaled, n_components=PCA_EXPLAINED_VARIANCE)
        log_shape("PCA reduced dataset", X_pca)

        # 7. Balance Data (SMOTE)
        # It's better to balance the training split rather than full data to avoid data leakage
        # However, for simplicity across pipelines, let's balance after PCA.
        X_balanced, y_balanced = balance_classes(X_pca, y, random_state=RANDOM_STATE)
        
        # 8. Convert to final dataframe
        processed = X_balanced.copy()
        processed["label"] = y_balanced.values
        log_shape("Final balanced dataset", processed)

        # 9. Train test split
        logger.info(f"Splitting dataset (TEST_SIZE={TEST_SIZE}, RANDOM_STATE={RANDOM_STATE})")
        train, test = train_test_split(
            processed,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=processed["label"]  # Ensure identical class distribution in train/test
        )

        # 10. Save datasets
        save_csv(train, "data/processed/train_dataset.csv")
        save_csv(test, "data/processed/test_dataset.csv")

        logger.info("Preprocessing pipeline completed successfully.")