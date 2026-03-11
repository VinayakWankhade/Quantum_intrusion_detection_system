import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

def retrain_hybrid():
    print("🚀 Starting Unified Hybrid Model Refresh & Tuning...")
    
    # 1. Load Unified Dataset
    data_path = "data/processed/cicids_merged.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df_raw = pd.read_csv(data_path)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Feature Engineering (6 Core Vitals)
    X_raw = pd.DataFrame()
    X_raw['duration'] = df_raw['Flow Duration'] / 1000000.0
    X_raw['src_bytes'] = df_raw['Fwd Packets Length Total']
    X_raw['dst_bytes'] = df_raw['Bwd Packets Length Total']
    X_raw['count'] = df_raw['Total Fwd Packets']
    X_raw['srv_count'] = df_raw['Total Backward Packets']
    X_raw['protocol_tcp'] = (df_raw['Protocol'] == 6).astype(int)
    X_raw['protocol_udp'] = (df_raw['Protocol'] == 17).astype(int)
    X_raw['protocol_icmp'] = (df_raw['Protocol'] == 1).astype(int)
    
    y = (df_raw['label'] != 'Benign').astype(int)
    
    print(f"Dataset Loaded: {len(X_raw)} samples. Starting Classical Tuning...")

    # BRAIN 1: Classical Random Forest with Hyperparameter Tuning
    # Use a 100k subset for fast tuning
    X_tune, _, y_tune, _ = train_test_split(X_raw, y, train_size=100000, stratify=y, random_state=42)
    
    scaler_unified = RobustScaler()
    X_tune_scaled = scaler_unified.fit_transform(X_tune)
    
    pca_unified = PCA(n_components=6) # 6 components for classical (max features=8)
    X_tune_pca = pca_unified.fit_transform(X_tune_scaled)
    
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    print("Executing GridSearchCV for Random Forest...")
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
    rf_grid.fit(X_tune_pca, y_tune)
    
    best_rf = rf_grid.best_estimator_
    print(f"Best Classical RF Params: {rf_grid.best_params_}")
    print(f"Best Tune Accuracy: {rf_grid.best_score_:.4f}")

    # BRAIN 2: Quantum Specialist (QSVC) Tuning
    print("\nStarting Quantum Specialist Tuning...")
    # Mine Rare Attacks
    rare_df = df_raw[df_raw['label'].str.contains('Infiltration|Sql Injection|Heartbleed', case=False, na=False)]
    benign_df = df_raw[df_raw['label'] == 'Benign'].sample(len(rare_df), random_state=42)
    q_df = pd.concat([rare_df, benign_df], ignore_index=True)
    
    X_q = pd.DataFrame()
    X_q['duration'] = q_df['Flow Duration'] / 1000000.0
    X_q['src_bytes'] = q_df['Fwd Packets Length Total']
    X_q['dst_bytes'] = q_df['Bwd Packets Length Total']
    X_q['count'] = q_df['Total Fwd Packets']
    X_q['srv_count'] = q_df['Total Backward Packets']
    # Protocol omitted for 4-qubit PCA focus
    
    y_q = (q_df['label'] != 'Benign').astype(int)
    
    scaler_specialist = RobustScaler()
    X_q_scaled = scaler_specialist.fit_transform(X_q)
    
    pca_specialist = PCA(n_components=4)
    X_q_pca = pca_specialist.fit_transform(X_q_scaled)
    
    print("Training Tuned QSVC Specialist...")
    # Using 4 qubits, 2 repetitions for high entanglement
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_q_pca, y_q)
    print("Quantum Specialist Trained Successfully.")

    # SAVE EVERYTHING
    print("\nSaving Optimized Model Pipeline...")
    models_dir = "models/saved"
    pre_dir = "models/saved/preprocessing"
    q_pre_dir = "models/saved/preprocessing/quantum"
    
    os.makedirs(q_pre_dir, exist_ok=True)
    
    joblib.dump(best_rf, os.path.join(models_dir, "unified_rf_model.pkl"))
    joblib.dump(qsvc, os.path.join(models_dir, "qsvc_specialist.pkl"))
    
    joblib.dump(scaler_unified, os.path.join(pre_dir, "scaler_unified.pkl"))
    joblib.dump(pca_unified, os.path.join(pre_dir, "pca_unified.pkl"))
    joblib.dump(X_tune.columns.tolist(), os.path.join(pre_dir, "feature_names_unified.pkl"))
    
    joblib.dump(scaler_specialist, os.path.join(q_pre_dir, "scaler_specialist.pkl"))
    joblib.dump(pca_specialist, os.path.join(q_pre_dir, "pca_specialist.pkl"))
    
    print("✅ Model Refresh & Tuning Complete. Systems Restored.")

if __name__ == "__main__":
    retrain_hybrid()
