import argparse
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import NSLKDD_DATA_PATH
from utils.helpers import get_logger

logger = get_logger("main")

def load_environment():
    """Import heavy models/pipelines here to prevent slow execution if a flag isn't called."""
    from preprocessing.pipeline import PreprocessingPipeline
    from models.svm_model import train_svm
    from models.random_forest import train_random_forest
    from models.qsvm_model import train_qsvm
    from models.vqc_model import train_vqc
    return PreprocessingPipeline, train_svm, train_random_forest, train_qsvm, train_vqc

def main():
    parser = argparse.ArgumentParser(description="Advanced Quantum IDS Project Runner")
    parser.add_argument('--preprocess', action='store_true', help="Run full robust ML data preprocessing pipeline")
    parser.add_argument('--train-classical', action='store_true', help="Train all Classical ML models (SVM, Random Forest)")
    parser.add_argument('--train-quantum', action='store_true', help="Train all Quantum ML models on downsampled subsets (QSVM, VQC)")
    parser.add_argument('--tune', action='store_true', help="When combined with --train-classical, rigorously tunes hyperparameters")
    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    logger.info("="*50)
    logger.info("Initializing Advanced Quantum IDS Processing Center")
    logger.info("="*50)
    
    # Lazy Load to avoid initialization delay for simple flags
    PreprocessingPipeline, train_svm, train_random_forest, train_qsvm, train_vqc = load_environment()
    
    if args.preprocess:
        logger.info(f"Triggering Advanced Preprocessing from source: {NSLKDD_DATA_PATH}")
        pipeline = PreprocessingPipeline(input_path=NSLKDD_DATA_PATH)
        pipeline.run()
        
    if args.train_classical:
        logger.info(f"Triggering Classical Model Training (Hyperparameter Tuning: {args.tune})")
        logger.info("Training SVM...")
        train_svm(tune_hyperparameters=args.tune)
        
        logger.info("Training Random Forest...")
        train_random_forest(tune_hyperparameters=args.tune)
        
    if args.train_quantum:
        logger.info("Triggering Quantum Model Training (Using Data Subsets)")
        logger.info("Training Quantum SVM with ZZFeatureMap...")
        train_qsvm()
        
        logger.info("Training Variational Quantum Classifier (VQC)...")
        train_vqc()

if __name__ == "__main__":
    main()