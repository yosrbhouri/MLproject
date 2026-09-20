"""
Main script for the ML project.
Orchestrates data preparation, model training, and evaluation.
"""

import logging
import numpy as np
from config import CONFIG
from data_prep import download_dataset, load_image_paths, balance_dataset, extract_features_batch, split_data
from model_training import train_random_forest, train_xgboost, get_model_predictions, evaluate_training_performance
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, analyze_errors

# Setup logging
logging.basicConfig(level=CONFIG['log_level'], filename=CONFIG['log_file'], format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    logger.info("Starting ML project execution")

    try:
        # 1. Download and load data
        logger.info("Step 1: Downloading dataset")
        data_path = download_dataset()
        CONFIG['data_path'] = data_path

        logger.info("Step 2: Loading image paths")
        ms_images, normal_images = load_image_paths(data_path)

        logger.info("Step 3: Balancing dataset")
        all_images, labels = balance_dataset(ms_images, normal_images, CONFIG['random_state'])

        # 2. Extract features
        logger.info("Step 4: Extracting features")
        X = extract_features_batch(all_images, CONFIG['feature_thresholds'])
        y = np.array(labels)

        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)

        # 3. Split data
        logger.info("Step 5: Splitting data")
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=CONFIG['stratify']
        )

        # 4. Train models
        logger.info("Step 6: Training models")
        rf_model = train_random_forest(X_train, y_train, CONFIG['rf_params'])
        xgb_model = train_xgboost(X_train, y_train)

        # Evaluate training
        training_eval = evaluate_training_performance(rf_model, xgb_model, X_train, y_train, X_test, y_test)

        # 5. Get predictions
        logger.info("Step 7: Getting predictions")
        predictions = get_model_predictions(rf_model, xgb_model, X_test)

        # 6. Evaluate models
        logger.info("Step 8: Evaluating models")
        rf_metrics = evaluate_model(y_test, predictions['rf']['pred'], predictions['rf']['proba'])
        xgb_metrics = evaluate_model(y_test, predictions['xgb']['pred'], predictions['xgb']['proba'])
        combined_metrics = evaluate_model(y_test, predictions['combined']['pred'], predictions['combined']['proba'])

        # 7. Analyze errors
        error_analysis = analyze_errors(
            y_test,
            predictions['rf']['pred'],
            predictions['xgb']['pred'],
            predictions['combined']['pred']
        )

        # 8. Print results
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        print(f"Dataset: {len(X)} images ({sum(y)} MS, {len(y)-sum(y)} Normal)")
        print(f"Train/Test split: {len(X_train)}/{len(X_test)}")

        print(f"\nRandom Forest:")
        print(f"  Accuracy: {rf_metrics['accuracy']:.3f}")
        print(f"  AUC: {rf_metrics.get('auc', 'N/A'):.3f}")

        print(f"\nXGBoost:")
        print(f"  Accuracy: {xgb_metrics['accuracy']:.3f}")
        print(f"  AUC: {xgb_metrics.get('auc', 'N/A'):.3f}")

        print(f"\nCombined Model:")
        print(f"  Accuracy: {combined_metrics['accuracy']:.3f}")
        print(f"  AUC: {combined_metrics.get('auc', 'N/A'):.3f}")

        print(f"\nError Analysis:")
        print(f"  Disagreements: {error_analysis['disagreements_count']} ({error_analysis['disagreements_rate']:.1%})")

        # 9. Feature importance
        feature_names = ["Luminosity", "Contrast", "% White", "% Black", "% Gray", "Edges"]
        importances = rf_model.feature_importances_
        print(f"\nTop Features:")
        sorted_idx = np.argsort(importances)[::-1]
        for i, idx in enumerate(sorted_idx[:3]):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

        logger.info("Execution completed successfully")

        # Return models and data for web interface
        return {
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'feature_names': feature_names,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': predictions,
            'metrics': {
                'rf': rf_metrics,
                'xgb': xgb_metrics,
                'combined': combined_metrics
            }
        }

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    results = main()
