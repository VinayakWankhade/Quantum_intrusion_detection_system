import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from utils.helpers import get_logger

logger = get_logger(__name__)

def train_quantum_specialists():
    logger.info("Initializing Quantum Rare-Attack Specialist Training...")
    
    cic_path = "data/processed/cicids_merged.csv"
    if not os.path.exists(cic_path):
        logger.error("CIC-IDS merged file not found!")
        return

    # 1. Mine Rare Attacks
    df = pd.read_csv(cic_path)
    # Rarest labels - use fuzzy matching
    rare_df = df[df['label'].str.contains('Infiltration|Sql Injection|Heartbleed', case=False, na=False)].copy()
    rare_df['label_bin'] = 1 
    
    num_rare = len(rare_df)
    logger.info(f"Extracted {num_rare} rare attack samples.")

    # 2. Extract Benign (1:1 Balance for extreme sensitivity)
    benign_df = df[df['label'].str.lower() == 'benign'].sample(num_rare, random_state=42).copy()
    benign_df['label_bin'] = 0
    
    # Combined Quantum Dataset
    q_df = pd.concat([rare_df, benign_df], ignore_index=True)
    
    # 3. Feature Mapping
    X_raw = pd.DataFrame()
    X_raw['duration'] = q_df['Flow Duration'] / 1000000.0
    X_raw['src_bytes'] = q_df['Fwd Packets Length Total']
    X_raw['dst_bytes'] = q_df['Bwd Packets Length Total']
    X_raw['count'] = q_df['Total Fwd Packets']
    X_raw['srv_count'] = q_df['Total Backward Packets']
    
    y = q_df['label_bin']

    # 4. Preprocessing (4 Qubits)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=4) # UPGRADED TO 4 QUBITS
    X_pca = pca.fit_transform(X_scaled)
    
    # Save Preprocessing
    pre_dir = "models/saved/preprocessing/quantum"
    os.makedirs(pre_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(pre_dir, "scaler_specialist.pkl"))
    joblib.dump(pca, os.path.join(pre_dir, "pca_specialist.pkl"))

    # 5. Quantum Training (4-Qubit QSVC)
    logger.info("Training 4-Qubit QSVC Rare-Attack Specialist...")
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2) # Higher reps for entanglement
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_pca, y)
    
    joblib.dump(qsvc, "models/saved/qsvc_specialist.pkl")
    logger.info("QSVC Specialist Saved.")
    
    logger.info("Quantum Rare-Attack Specialization Complete!")

if __name__ == "__main__":
    train_quantum_specialists()
