"""
Evaluation module.
Handles model evaluation, metrics calculation, and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)

    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    return plt

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """Plot feature importance."""
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[sorted_idx], color='skyblue')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    logger.info(f"Cross-validation: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    return scores

def analyze_errors(y_true, rf_pred, xgb_pred, combined_pred):
    """Analyze disagreements between models."""
    rf_correct = (rf_pred == y_true)
    xgb_correct = (xgb_pred == y_true)
    combined_correct = (combined_pred == y_true)

    disagreements = np.where(rf_pred != xgb_pred)[0]

    analysis = {
        'rf_accuracy': np.mean(rf_correct),
        'xgb_accuracy': np.mean(xgb_correct),
        'combined_accuracy': np.mean(combined_correct),
        'disagreements_count': len(disagreements),
        'disagreements_rate': len(disagreements) / len(y_true)
    }

    return analysis
