# TODO List for Modifying ML Project

## 1. Update Requirements and Dependencies
- [x] Add XGBoost to requirements.txt

## 2. Modify Model Training (model_training.py)
- [x] Replace SVM with XGBoost training function
- [x] Update get_model_predictions to use XGBoost instead of SVM
- [x] Update evaluate_training_performance if needed

## 3. Modify Main Script (main.py)
- [x] Update imports to include XGBoost
- [x] Replace SVM training with XGBoost training
- [x] Update predictions and metrics to use XGBoost

## 4. Modify Evaluation Module (evaluation.py)
- [x] Update analyze_errors to use XGBoost instead of SVM
- [x] Ensure all evaluation functions work with RF and XGBoost only

## 5. Modify Web Interface (app.py)
- [x] Update to load and use RF and XGBoost models
- [x] Modify prediction logic to focus on luminosity and white spots features
- [x] Update UI to reflect the new model combination (RF + XGBoost)
- [x] Add explanation text about classification based on luminosity and white spots

## 6. Adjust Feature Extraction if Needed (data_prep.py)
- [x] Review extract_6_features_simple to ensure it properly captures white spots (% white pixels)
- [x] Potentially adjust thresholds for better detection of white spots

## 7. Testing and Validation
- [x] Test the modified models
- [x] Validate web interface functionality
- [x] Ensure classification logic works as expected
