import sys
import os

# Add the project root directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, RANDOM_STATE
from utils.helpers import get_logger, save_model
from models.evaluate import evaluate_model

logger = get_logger(__name__)

def train_random_forest(tune_hyperparameters=False):
    logger.info("Loading training and testing datasets for Random Forest...")
    
    try:
        train = pd.read_csv(TRAIN_DATA_PATH)
        test = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        logger.error("Datasets not found. Please run the preprocessing pipeline first.")
        return None

    X_train, y_train = train.drop("label", axis=1), train["label"]
    X_test, y_test = test.drop("label", axis=1), test["label"]

    logger.info("Initializing Random Forest Classifier...")
    model = RandomForestClassifier(random_state=RANDOM_STATE)

    if tune_hyperparameters:
        logger.info("Starting GridSearchCV for Random Forest hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        logger.info(f"GridSearchCV completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        logger.info("Training Random Forest model with default parameters...")
        start_time = time.time()
        model.fit(X_train, y_train)
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Evaluating Random Forest model on test set...")
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1] if len(set(y_test)) == 2 else None

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")

    evaluate_model(model, X_test, y_test, model_name="RandomForest")

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    save_model(model, model_path)
    
    return model

if __name__ == "__main__":
    train_random_forest(tune_hyperparameters=False)