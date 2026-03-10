import pandas as pd
from config.config import QUANTUM_SUBSET_SIZE
from utils.helpers import get_logger

logger = get_logger(__name__)

def subset_for_quantum(X, y, subset_size=QUANTUM_SUBSET_SIZE, random_state=42):
    """
    Downsamples the dataset explicitly for Quantum models. Quantum simulators 
    are extremely slow on large datasets, so a representative fraction is needed.
    """
    logger.warning(
        f"Downsampling dataset to {subset_size} samples for Quantum compatibility. "
        "Training on full data with Qiskit simulators is generally infeasible."
    )

    if len(X) <= subset_size:
        return X, y

    # We use pandas sample to randomly subset while maintaining indices
    # We stratify by joining temporarily if y is a Series
    temp_df = X.copy()
    temp_df['__target__'] = y

    # Stratified sampling
    subset_df = temp_df.groupby('__target__', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, subset_size // 2)), random_state=random_state)
    )

    y_subset = subset_df['__target__']
    X_subset = subset_df.drop(columns=['__target__'])

    logger.info(f"Quantum subset created. New shape: {X_subset.shape}. Class dist: {y_subset.value_counts().to_dict()}")

    # Ensure returning NumPy arrays since Qiskit algorithms expect them
    return X_subset.values, y_subset.values
