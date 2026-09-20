"""
Model training module.
Handles training of Random Forest and SVM models, with functions to avoid duplication.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train, rf_params):
    """Train Random Forest model."""
    logger.info("Training Random Forest model")
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    logger.info("Random Forest training completed")
    return rf

def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    logger.info("Training XGBoost model")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    logger.info("XGBoost training completed")
    return xgb_model

def get_model_predictions(rf_model, xgb_model, X_test):
    """Get predictions from RF and XGBoost models."""
    # RF predictions
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    # XGBoost predictions
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Combined predictions (average probabilities)
    combined_proba = (rf_proba + xgb_proba) / 2
    combined_pred = (combined_proba > 0.5).astype(int)

    return {
        'rf': {'pred': rf_pred, 'proba': rf_proba},
        'xgb': {'pred': xgb_pred, 'proba': xgb_proba},
        'combined': {'pred': combined_pred, 'proba': combined_proba}
    }

def evaluate_training_performance(rf_model, xgb_model, X_train, y_train, X_test, y_test):
    """Evaluate training performance and check for overfitting for both models."""
    # RF evaluation
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_test_acc = rf_model.score(X_test, y_test)
    rf_overfitting_gap = rf_train_acc - rf_test_acc

    # XGBoost evaluation
    xgb_train_acc = xgb_model.score(X_train, y_train)
    xgb_test_acc = xgb_model.score(X_test, y_test)
    xgb_overfitting_gap = xgb_train_acc - xgb_test_acc

    logger.info(f"RF Training accuracy: {rf_train_acc:.3f}, Test: {rf_test_acc:.3f}, Gap: {rf_overfitting_gap:.3f}")
    logger.info(f"XGBoost Training accuracy: {xgb_train_acc:.3f}, Test: {xgb_test_acc:.3f}, Gap: {xgb_overfitting_gap:.3f}")

    return {
        'rf': {
            'train_accuracy': rf_train_acc,
            'test_accuracy': rf_test_acc,
            'overfitting_gap': rf_overfitting_gap
        },
        'xgb': {
            'train_accuracy': xgb_train_acc,
            'test_accuracy': xgb_test_acc,
            'overfitting_gap': xgb_overfitting_gap
        }
    }
