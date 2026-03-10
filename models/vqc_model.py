import sys
import os

# Add the project root directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from sklearn.metrics import accuracy_score, classification_report
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

from config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, VQC_LAYERS, VQC_OPTIMIZER
from utils.helpers import get_logger, save_model
from utils.quantum_utils import subset_for_quantum
from models.evaluate import evaluate_model

logger = get_logger(__name__)

def train_vqc():
    logger.info("Loading datasets for VQC...")
    
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
    X_test, y_test = subset_for_quantum(X_test_full, y_test_full, subset_size=200)

    num_features = X_train.shape[1]
    
    logger.info(f"Setting up ZZFeatureMap with {num_features} qubits...")
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1, entanglement="linear")

    logger.info(f"Setting up RealAmplitudes ansatz with {VQC_LAYERS} layers...")
    ansatz = RealAmplitudes(num_qubits=num_features, reps=VQC_LAYERS)

    logger.info(f"Setting up {VQC_OPTIMIZER} optimizer...")
    if VQC_OPTIMIZER.upper() == "COBYLA":
        optimizer = COBYLA(maxiter=100)
    else:
        # Fallback to default COBYLA if not supported here
        optimizer = COBYLA(maxiter=100)

    logger.info("Initializing VQC...")
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    logger.info("Training VQC model (this will take time on a classical simulator)...")
    start_time = time.time()
    vqc.fit(X_train, y_train)
    logger.info(f"VQC Training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Evaluating VQC model on quantum test subset...")
    eval_start_time = time.time()
    predictions = vqc.predict(X_test)
    logger.info(f"VQC Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"VQC Accuracy: {accuracy:.4f}")

    evaluate_model(vqc, X_test, y_test, model_name="VQC")

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, "vqc_model.pkl")
    save_model(vqc, model_path)
    
    return vqc

if __name__ == "__main__":
    train_vqc()
