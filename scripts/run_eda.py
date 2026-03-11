import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.helpers import get_logger

logger = get_logger(__name__)

def run_eda(file_path, output_dir):
    logger.info(f"Running EDA on {file_path}")
    df = pd.read_csv(file_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Basic Stats
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Missing Values:\n{df.isnull().sum().to_string()}\n")
        f.write(f"Class Distribution:\n{df['label'].value_counts().to_string()}\n")

    # 2. Class Distribution Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='label')
    plt.title(f"Class Distribution - {os.path.basename(file_path)}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    # 3. Correlation Heatmap (Numeric features only)
    plt.figure(figsize=(15, 10))
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] > 1:
        # Limit to top correlated features for visibility if too many
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    logger.info(f"EDA completed for {file_path}. Results in {output_dir}")

if __name__ == "__main__":
    nsl_path = "data/processed/nslkdd_merged.csv"
    if os.path.exists(nsl_path):
        run_eda(nsl_path, "results/eda/nslkdd")
    
    cic_path = "data/processed/cicids_merged.csv"
    if os.path.exists(cic_path):
        run_eda(cic_path, "results/eda/cicids")
