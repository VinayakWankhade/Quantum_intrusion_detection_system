import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from config.config import RESULTS_DIR
from utils.helpers import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Centralized evaluation function that calculates metrics and generates/saves
    Confusion Matrices, ROC Curves, and PR curves.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info(f"Evaluating {model_name}...")
    predictions = model.predict(X_test)
    
    # Textual Reports
    logger.info(f"\n{model_name} Classification Report\n" + "-"*50)
    logger.info("\n" + classification_report(y_test, predictions))

    # Confusion Matrix Plotting
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix plot to {cm_path}")
    plt.close()

    # Probability based curves (ROC / PR)
    # Check if the model has predict_proba
    if hasattr(model, "predict_proba"):
        prob_predictions = model.predict_proba(X_test)
        if prob_predictions.shape[1] == 2:  # Binary classification
            prob_predictions = prob_predictions[:, 1]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, prob_predictions)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{model_name} - ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, prob_predictions)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f"{model_name} - Precision-Recall Curve")
            plt.tight_layout()
            pr_path = os.path.join(RESULTS_DIR, f"{model_name}_pr_curve.png")
            plt.savefig(pr_path)
            plt.close()

    logger.info(f"{model_name} evaluation complete. Assets saved to {RESULTS_DIR}")
