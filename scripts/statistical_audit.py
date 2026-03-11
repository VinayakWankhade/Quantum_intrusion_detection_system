import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway, zscore
import os

def run_statistical_audit():
    cic_path = "c:/Users/wankh/Downloads/quantum_ml_model/quantum_ids_project/data/processed/cicids_merged.csv"
    if not os.path.exists(cic_path):
        print("Data not found.")
        return

    # Load representative sample for stats
    df_raw = pd.read_csv(cic_path, nrows=1000000)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Target
    df_raw['is_attack'] = (df_raw['label'] != 'Benign').astype(int)
    
    # Balance for plotting
    benign_pool = df_raw[df_raw['label'] == 'Benign']
    attack_pool = df_raw[df_raw['label'] != 'Benign']
    
    n_samples = min(5000, len(benign_pool), len(attack_pool))
    benign = benign_pool.sample(n_samples, random_state=42)
    attacks = attack_pool.sample(n_samples, random_state=42)
    df_plot = pd.concat([benign, attacks])

    # 1. Visualization (KDE Plots)
    features_to_plot = {
        'Flow Duration': 'Duration (ms)',
        'Fwd Packets Length Total': 'Source Bytes',
        'Bwd Packets Length Total': 'Dest Bytes',
        'Total Fwd Packets': 'Fwd Packet Count'
    }
    
    plt.figure(figsize=(15, 10))
    for i, (col, title) in enumerate(features_to_plot.items(), 1):
        plt.subplot(2, 2, i)
        # Use log scale for network data visualization as it's highly skewed
        sns.kdeplot(data=df_plot, x=np.log1p(df_plot[col]), hue='label', fill=True, common_norm=False)
        plt.title(f"Log-Distribution: {title}")
        plt.xlabel(f"Log({title} + 1)")
    
    plot_path = "C:/Users/wankh/.gemini/antigravity/brain/9bc1769d-bf3f-4a63-bca2-4a559b2882a3/feature_distributions.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    # print(f"Distribution plots saved to {plot_path}")

    # 2. Chi-Square Test (Categorical Independence)
    contingency = pd.crosstab(df_plot['Protocol'], df_plot['is_attack'])
    chi2, p_chi2, dof, ex = chi2_contingency(contingency)
    
    # 3. F-Test / ANOVA
    f_results = {}
    for col in features_to_plot.keys():
        group_b = df_plot[df_plot['is_attack'] == 0][col]
        group_a = df_plot[df_plot['is_attack'] == 1][col]
        f_stat, p_f = f_oneway(group_b, group_a)
        f_results[col] = (f_stat, p_f)

    # 4. Z-Score Deviation (How weird are rare attacks?)
    rare_attacks = df_raw[df_raw['label'].str.contains('Infiltration|Sql Injection|Heartbleed', case=False, na=False)]
    benign_full = df_raw[df_raw['label'] == 'Benign']
    
    print("\n--- Statistical Significance Audit ---")
    print(f"Chi-Square (Protocol Significance): {chi2:.2f} (p-value: {p_chi2:.4e})")
    
    print("\n--- ANOVA (F-Test) Feature Importance ---")
    for feat, (f_stat, p_val) in f_results.items():
        sig = "HIGHLY SIGNIFICANT" if p_val < 0.001 else "SIGNIFICANT" if p_val < 0.05 else "INSIGNIFICANT"
        print(f"Feature: {feat:<25} | p-value: {p_val:>8.4e} | Conclusion: {sig}")

    print("\n--- Rare Attack Signature Deviation (Z-Scores) ---")
    for feat in features_to_plot.keys():
        b_mean = np.nanmean(benign_full[feat])
        b_std = np.nanstd(benign_full[feat])
        r_mean = np.nanmean(rare_attacks[feat])
        z_dev = (r_mean - b_mean) / b_std if (b_std and not np.isnan(b_std)) else 0
        status = "MAJOR DEVIATION" if abs(z_dev) > 1 else "SUBTLE"
        print(f"Feature: {feat:<25} | Deviation: {z_dev:>8.2f} sigma | Type: {status}")

if __name__ == "__main__":
    run_statistical_audit()
