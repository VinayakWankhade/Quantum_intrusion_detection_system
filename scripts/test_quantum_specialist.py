import pandas as pd
import joblib
import os
import sys

# Mocking the backend logic for a clean test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_on_rare_attacks():
    print("Starting Quantum Hybrid Stress Test...")
    
    # Load Models
    models_dir = "models/saved"
    rf = joblib.load(os.path.join(models_dir, "unified_rf_model.pkl"))
    q_specialist = joblib.load(os.path.join(models_dir, "qsvc_specialist.pkl"))
    
    pre_dir = "models/saved/preprocessing"
    scaler_unified = joblib.load(os.path.join(pre_dir, "scaler_unified.pkl"))
    pca_unified = joblib.load(os.path.join(pre_dir, "pca_unified.pkl"))
    feature_names = joblib.load(os.path.join(pre_dir, "feature_names_unified.pkl"))
    
    q_pre_dir = "models/saved/preprocessing/quantum"
    q_scaler = joblib.load(os.path.join(q_pre_dir, "scaler_specialist.pkl"))
    q_pca = joblib.load(os.path.join(q_pre_dir, "pca_specialist.pkl"))
    
    # Load Rare Attacks from CIC-IDS
    cic_path = "data/processed/cicids_merged.csv"
    df = pd.read_csv(cic_path)
    # Target SQL Injection specifically as it's the hardest
    rare_df = df[df['label'].str.contains('Sql Injection|Heartbleed|Infiltration', case=False, na=False)].sample(10, random_state=42)
    
    print(f"Testing on {len(rare_df)} rare samples")
    
    catches = 0
    for i, row in rare_df.iterrows():
        clean_label = "".join([c if ord(c) < 128 else " " for f in [row['label']] for c in str(f)])
        print(f"\n--- Sample {i} [Actual: {clean_label}] ---")
        
        # Prepare Features
        feat = pd.DataFrame([{
            'duration': row['Flow Duration'] / 1000000.0,
            'protocol_type': 'tcp' if row['Protocol'] == 6 else 'udp' if row['Protocol'] == 17 else 'icmp',
            'src_bytes': row['Fwd Packets Length Total'],
            'dst_bytes': row['Bwd Packets Length Total'],
            'count': row['Total Fwd Packets'],
            'srv_count': row['Total Backward Packets']
        }])
        
        # 1. Classical
        feat_enc = pd.get_dummies(feat)
        for col in feature_names:
            if col not in feat_enc.columns: feat_enc[col] = 0
        feat_enc = feat_enc[feature_names]
        pca_feat = pca_unified.transform(scaler_unified.transform(feat_enc))
        rf_pred = rf.predict(pca_feat)[0]
        
        # 2. Quantum
        q_core = feat[['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']]
        q_pca_feat = q_pca.transform(q_scaler.transform(q_core))
        q_pred = q_specialist.predict(q_pca_feat)[0]
        
        status = "PASSED"
        if q_pred == 1 and rf_pred == 0:
            status = "QUANTUM CATCH! (Quantum found what RF missed)"
            catches += 1
        elif q_pred == 1 and rf_pred == 1:
            status = "DUAL DETECTION (Both Correct)"
        elif q_pred == 0 and rf_pred == 1:
            status = "CLASSICAL ONLY (Quantum missed)"
        else:
            status = "TOTAL MISS"
            
        print(f"Verdict: {status}")

    print(f"\nFinal Test Summary: Quantum Specialist caught {catches} elusive threats missed by Classical.")

if __name__ == "__main__":
    test_on_rare_attacks()
