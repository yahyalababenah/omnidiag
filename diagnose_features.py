import pandas as pd
import joblib
import xgboost as xgb
import os

model_path = 'models/heart_disease/omni_diag_xgb_optimized.pkl'
data_path = 'data/heart_disease/processed/final_ready_data.csv'

print(f"--- Diagnosing Feature Mismatch ---")

# 1. Load the model and inspect its features
if os.path.exists(model_path):
    print(f"\nLoading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        if isinstance(model, xgb.Booster):
            model_feature_names = model.feature_names
        elif isinstance(model, xgb.XGBClassifier):
            # For newer XGBoost versions, feature_names might be in _Booster
            if hasattr(model, 'get_booster'):
                model_feature_names = model.get_booster().feature_names
            else: # Fallback for older versions or other cases
                print("Warning: Could not extract feature names directly from XGBClassifier. "
                      "This might be an older XGBoost version or configuration.")
                model_feature_names = None
        else:
            print(f"Warning: Model is of type {type(model)}, expected XGBoost Booster or XGBClassifier.")
            model_feature_names = None

        if model_feature_names:
            print(f"Model expects {len(model_feature_names)} features.")
            print(f"Model feature names: {model_feature_names}")
        else:
            print("Could not determine model feature names.")

    except Exception as e:
        print(f"Error loading or inspecting model: {e}")
        model_feature_names = None
else:
    print(f"Error: Model file not found at {model_path}")
    model_feature_names = None

# 2. Load the training data and inspect its columns
if os.path.exists(data_path):
    print(f"\nLoading training data from: {data_path}")
    try:
        df_train = pd.read_csv(data_path)
        data_columns = list(df_train.drop(columns=['HeartDisease'], errors='ignore').columns) # Assuming HeartDisease is target
        print(f"Training data has {len(data_columns)} features (excluding target).")
        print(f"Training data columns: {data_columns}")
    except Exception as e:
        print(f"Error loading or inspecting training data: {e}")
        data_columns = None
else:
    print(f"Error: Training data file not found at {data_path}")
    data_columns = None

# 3. Compare
if model_feature_names and data_columns:
    print("\n--- Comparison ---")
    if set(model_feature_names) == set(data_columns) and len(model_feature_names) == len(data_columns):
        print("✅ Feature sets match in name and count.")
    else:
        print("❌ Feature sets DO NOT match.")
        missing_in_model = set(data_columns) - set(model_feature_names)
        extra_in_model = set(model_feature_names) - set(data_columns)
        if missing_in_model:
            print(f"Features in data but MISSING in model: {sorted(list(missing_in_model))}")
        if extra_in_model:
            print(f"Features in model but EXTRA in data: {sorted(list(extra_in_model))}")
        if len(model_feature_names) != len(data_columns):
            print(f"Model expects {len(model_feature_names)} features, but data has {len(data_columns)} features.")
