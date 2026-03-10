# # config.py
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'nsl_kdd.csv')
# PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'processed_ids.csv')

# # Model parameters
# RANDOM_STATE = 42
# TEST_SIZE = 0.2


"""
Project Configuration File
Central place for all project parameters
"""


# -----------------------------
# DATA PATHS
# -----------------------------

RAW_DATA_DIR = "data/raw"

PROCESSED_DATA_DIR = "data/processed"

MODELS_DIR = "models/saved"

RESULTS_DIR = "results"

NSLKDD_DATA_PATH = "data/processed/nslkdd_merged.csv"

CICIDS_DATA_PATH = "data/processed/cicids_merged.csv"

TRAIN_DATA_PATH = "data/processed/train_dataset.csv"

TEST_DATA_PATH = "data/processed/test_dataset.csv"


# -----------------------------
# PREPROCESSING PARAMETERS
# -----------------------------

TEST_SIZE = 0.2

RANDOM_STATE = 42

PCA_EXPLAINED_VARIANCE = 0.95  # Target 95% variance

IMPUTATION_STRATEGY = "median"  # robust handling of missing numericals

SCALER_TYPE = "robust"  # "robust" or "standard"


# -----------------------------
# MODEL PARAMETERS
# -----------------------------

SVM_KERNEL = "rbf"

SVM_C = 1.0

RANDOM_FOREST_TREES = 100

RANDOM_FOREST_MAX_DEPTH = None


# -----------------------------
# QUANTUM MODEL PARAMETERS
# -----------------------------

QSVM_FEATURE_MAP_REPS = 2

VQC_LAYERS = 3

VQC_OPTIMIZER = "COBYLA"

QUANTUM_SUBSET_SIZE = 1000  # Number of samples to train quantum models on


# -----------------------------
# LOGGING SETTINGS
# -----------------------------

VERBOSE = True