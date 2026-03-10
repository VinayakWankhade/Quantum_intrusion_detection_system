import sys
import os

# Add the project root directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from sklearn.metrics import accuracy_score, classification_report
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, QSVM_FEATURE_MAP_REPS
from utils.helpers import get_logger, save_model
from utils.quantum_utils import subset_for_quantum
from models.evaluate import evaluate_model

logger = get_logger(__name__)

def train_qsvm():
    logger.info("Loading datasets for QSVM...")
    
    try:
        train = pd.read_csv(TRAIN_DATA_PATH)
        test = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        logger.error("Datasets not found. Please run the preprocessing pipeline first.")
        return None

    X_train_full, y_train_full = train.drop("label", axis=1), train["label"]
    X_test_full, y_test_full = test.drop("label", axis=1), test["label"]

    # Downsample for Quantum Compatibility
    X_train, y_train = subset_for_quantum(X_train_full, y_train_full)
    
    # We also subset the testing data so evaluation doesn't take hours
    logger.info("Subsetting test data for QSVM evaluation...")
    X_test, y_test = subset_for_quantum(X_test_full, y_test_full, subset_size=200)

    num_features = X_train.shape[1]
    
    logger.info(f"Initializing ZZFeatureMap with {num_features} qubits and {QSVM_FEATURE_MAP_REPS} reps...")
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=QSVM_FEATURE_MAP_REPS, entanglement="linear")

    logger.info("Setting up FidelityQuantumKernel...")
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    logger.info("Initializing QSVC...")
    qsvc = QSVC(quantum_kernel=qkernel)

    logger.info("Training QSVM model (this will take time on a classical simulator)...")
    start_time = time.time()
    qsvc.fit(X_train, y_train)
    logger.info(f"QSVM Training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Evaluating QSVM model on quantum test subset...")
    eval_start_time = time.time()
    predictions = qsvc.predict(X_test)
    logger.info(f"QSVM Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"QSVM Accuracy: {accuracy:.4f}")

    evaluate_model(qsvc, X_test, y_test, model_name="QSVM")

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, "qsvm_model.pkl")
    save_model(qsvc, model_path)
    
    return qsvc

if __name__ == "__main__":
    train_qsvm()
