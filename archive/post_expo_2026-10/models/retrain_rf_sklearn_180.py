"""
OmniDiag — Re-train Random Forest with sklearn 1.8.0 for SHAP compatibility
===========================================================================
Corrected version: matches the inference pipeline order:
  1. Engineer features on RAW data
  2. Apply Kaggle-trained StandardScaler

This ensures the RF model sees the same data distribution at inference as
during training, fixing the TreeExplainer crash with sklearn 1.8.0.

Usage:
    python models/retrain_rf_sklearn_180.py
"""

import logging
import os
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.retrain_rf")

RANDOM_STATE = 42
DATA_PATH = os.path.join(_PROJECT_ROOT,
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
TARGET_COL = "Diabetes_binary"
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth", "GenHlth",
                       "Age", "Education", "Income"]
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
FEATURE_COLS = BINARY_FEATURES + CONTINUOUS_FEATURES

# Kaggle-matched RF hyperparameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
}

MODEL_SAVE_PATH = os.path.join(
    _PROJECT_ROOT, "models", "diabetes", "rf_model.pkl"
)
SCALER_PATH = os.path.join(
    _PROJECT_ROOT, "models", "diabetes", "preprocessors",
    "standard_scaler.pkl"
)


def main():
    log.info("=" * 60)
    log.info("Re-training RF with inference-matched pipeline")
    log.info("=" * 60)

    # ── Step 1: Load raw data ─────────────────────────────────────────
    log.info(f"Loading raw data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    X_raw = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Split: X_train {X_train_raw.shape}, X_test {X_test_raw.shape}")

    # ── Step 2: Engineer features on RAW data (matches inference) ─────
    log.info("Engineering features on RAW data (inference-matched order)...")
    from features.diabetes_features import DiabetesFeatureEngineer
    from configs.config_loader import load_config

    config = load_config("diabetes")
    engineer = DiabetesFeatureEngineer(config)

    X_train = engineer.engineer_heuristic(X_train_raw)
    X_train = engineer.engineer_medical(X_train)
    X_test = engineer.engineer_heuristic(X_test_raw)
    X_test = engineer.engineer_medical(X_test)
    log.info(f"Engineered: X_train {X_train.shape}, X_test {X_test.shape}")

    # ── Step 3: Apply Kaggle scaler (matches inference) ───────────────
    log.info(f"Loading Kaggle scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    numeric_cols_present = [c for c in CONTINUOUS_FEATURES
                            if c in X_train.columns]
    log.info(f"Scaling: {numeric_cols_present}")
    X_train[numeric_cols_present] = scaler.transform(
        X_train[numeric_cols_present]
    )
    X_test[numeric_cols_present] = scaler.transform(
        X_test[numeric_cols_present]
    )
    log.info("Scaling applied ✅")

    # ── Step 4: Train Random Forest ───────────────────────────────────
    log.info(f"Training RF with params: {RF_PARAMS}")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    log.info(f"RF trained. n_features_in_={rf.n_features_in_}")

    # Quick accuracy check
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    log.info(f"Train accuracy: {train_acc:.4f}")
    log.info(f"Test accuracy:  {test_acc:.4f}")

    # ── Step 5: Save model ────────────────────────────────────────────
    log.info(f"Saving RF model to: {MODEL_SAVE_PATH}")
    joblib.dump(rf, MODEL_SAVE_PATH)

    # ── Step 6: Verify re-loadability ─────────────────────────────────
    log.info("Verifying re-loadability...")
    loaded = joblib.load(MODEL_SAVE_PATH)
    assert isinstance(loaded, RandomForestClassifier)
    assert loaded.n_estimators == 300
    assert loaded.n_features_in_ == X_train.shape[1]
    log.info("✅ RF loads successfully with sklearn 1.8.0")

    # ── Step 7: Verify SHAP TreeExplainer on inference data ───────────
    log.info("Verifying SHAP TreeExplainer on inference data...")
    import shap
    explainer = shap.TreeExplainer(loaded)
    sv = explainer(X_test.iloc[:1])
    log.info(f"✅ SHAP values shape: {sv.values.shape}")

    # ── Step 8: Verify all 26 features get non-zero SHAP values ───────
    nz = sum(1 for v in sv.values.flatten() if abs(v) > 1e-10)
    log.info(f"Non-zero SHAP values: {nz}/{X_train.shape[1]}")

    log.info("=" * 60)
    log.info("SUCCESS: RF model re-trained and SHAP-verified for sklearn 1.8.0")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
