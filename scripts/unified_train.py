import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from utils.helpers import get_logger

logger = get_logger(__name__)

def advanced_unified_pipeline():
    logger.info("Initializing Advanced Unified Pipeline...")
    nsl_path = "data/processed/nslkdd_merged.csv"
    cic_path = "data/processed/cicids_merged.csv"
    
    # 1. Load and Standardize Labels
    logger.info("Standardizing labels across datasets...")
    nsl_df = pd.read_csv(nsl_path)
    nsl_df['label'] = nsl_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    cic_df = pd.read_csv(cic_path)
    cic_df['label'] = cic_df['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
    
    # 2. Unified Feature Selection (Common intersection or mapped features)
    # We will use the NSL-KDD baseline as the common feature set and map CIC-IDS features to it
    # This ensures consistency for the real-time sniffer too.
    
    # Mapping logic for CIC-IDS to NSL-KDD features
    cic_mapped = pd.DataFrame()
    cic_mapped['duration'] = cic_df['Flow Duration'] / 1000000.0 # to seconds
    cic_mapped['protocol_type'] = cic_df['Protocol'].map({6: 'tcp', 17: 'udp', 1: 'icmp'}).fillna('tcp')
    cic_mapped['src_bytes'] = cic_df['Fwd Packets Length Total']
    cic_mapped['dst_bytes'] = cic_df['Bwd Packets Length Total']
    cic_mapped['count'] = cic_df['Total Fwd Packets']
    cic_mapped['srv_count'] = cic_df['Total Backward Packets']
    cic_mapped['label'] = cic_df['label']
    
    # For NSL-KDD, we keep the core features
    nsl_subset = nsl_df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'label']].copy()
    
    # 3. Merging into Super-Dataset
    logger.info("Combining datasets into Super-Dataset...")
    combined_df = pd.concat([nsl_subset, cic_mapped], ignore_index=True)
    logger.info(f"Super-Dataset Shape: {combined_df.shape}")
    
    # 4. Final Preprocessing
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']
    
    # Encode
    logger.info("Final Encoding...")
    X_encoded = pd.get_dummies(X, columns=['protocol_type'])
    
    # Scale
    logger.info("Final Scaling (RobustScaler)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # PCA (Reduce to 5 core features for speed since we downsampled features)
    logger.info("Final PCA Reduction...")
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    # 5. Split and Save
    logger.info("Final Data Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    os.makedirs("data/final", exist_ok=True)
    np.save("data/final/X_train.npy", X_train)
    np.save("data/final/X_test.npy", X_test)
    np.save("data/final/y_train.npy", y_train)
    np.save("data/final/y_test.npy", y_test)
    
    # Save Preprocessing Objects for Sniffer
    pre_dir = "models/saved/preprocessing"
    os.makedirs(pre_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(pre_dir, "scaler_unified.pkl"))
    joblib.dump(pca, os.path.join(pre_dir, "pca_unified.pkl"))
    joblib.dump(list(X_encoded.columns), os.path.join(pre_dir, "feature_names_unified.pkl"))
    
    logger.info("Unified Pipeline complete. Starting Training...")
    
    # 6. Final Model Training (Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    joblib.dump(rf, "models/saved/unified_rf_model.pkl")
    logger.info("Unified Random Forest Model trained and saved!")
    
    # 7. Evaluation
    from sklearn.metrics import classification_report
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"\nFinal Model Performance:\n{report}")

if __name__ == "__main__":
    advanced_unified_pipeline()
