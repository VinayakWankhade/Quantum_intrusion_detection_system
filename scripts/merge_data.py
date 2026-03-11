import pandas as pd
import os
import glob
from utils.helpers import get_logger

logger = get_logger(__name__)

def merge_datasets():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. NSL-KDD Merging
    logger.info("Merging NSL-KDD files...")
    nsl_files = ["KDDTrain+.txt", "KDDTest+.txt"]
    nsl_dfs = []
    
    # NSL-KDD column names (standard 41 features + label + difficulty)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    for f in nsl_files:
        path = os.path.join(raw_dir, f)
        if os.path.exists(path):
            nsl_dfs.append(pd.read_csv(path, header=None, names=columns))
    
    if nsl_dfs:
        nsl_merged = pd.concat(nsl_dfs, ignore_index=True)
        nsl_merged.to_csv(os.path.join(processed_dir, "nslkdd_merged.csv"), index=False)
        logger.info(f"NSL-KDD merged: {nsl_merged.shape}")

    # 2. CIC-IDS 2017 Merging (Parquet files)
    logger.info("Merging CIC-IDS 2017 Parquet files...")
    cic_files = glob.glob(os.path.join(raw_dir, "*-no-metadata.parquet"))
    cic_dfs = []
    
    for f in cic_files:
        logger.info(f"Reading {os.path.basename(f)}...")
        cic_dfs.append(pd.read_parquet(f))
    
    if cic_dfs:
        cic_merged = pd.concat(cic_dfs, ignore_index=True)
        # Standardizing label column name if it varies
        if 'Label' in cic_merged.columns:
            cic_merged.rename(columns={'Label': 'label'}, inplace=True)
        
        cic_merged.to_csv(os.path.join(processed_dir, "cicids_merged.csv"), index=False)
        logger.info(f"CIC-IDS merged: {cic_merged.shape}")

if __name__ == "__main__":
    merge_datasets()
