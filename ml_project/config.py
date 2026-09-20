"""
Configuration file for the ML project.
Centralizes all parameters, paths, and hyperparameters.
"""

import os

CONFIG = {
    # Data paths
    'data_path': None,  # Will be set after download
    'ms_path': None,
    'normal_path': None,

    # Dataset parameters
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,

    # Feature extraction thresholds
    'feature_thresholds': {
        'blanc': 200,
        'noir': 30
    },

    # Model hyperparameters
    'rf_params': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    },

    'svm_params': {
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    },

    # Logging
    'log_level': 'INFO',
    'log_file': 'ml_project.log'
}
