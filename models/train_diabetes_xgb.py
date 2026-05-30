"""
OmniDiag — Diabetes XGBoost Training Script
=============================================
Trains an XGBoost classifier on the preprocessed CDC BRFSS 2015 Diabetes
Health Indicators dataset. Loads preprocessed data from the pipeline,
applies feature engineering, trains the model, and saves artifacts.

Usage:
    python models/train_diabetes_xgb.py
"""

import logging
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path for config import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from configs.config_loader import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.train_diabetes")


def load_preprocessed_data(
    config: dict,
) -> tuple:
    """
    Load preprocessed data from the preprocessing pipeline.
    
    For now, runs the preprocessing pipeline inline since we want
    to ensure the exact same transformation is applied to training data.
    
    In production, this would load from cached CSV files.
    """
    from experiment_files.data_pipeline.preprocess_diabetes import (
        run_diabetes_preprocessing,
    )

    preprocessors_path = resolve_path(config, "model", "preprocessors_path")
    scaler_path = os.path.join(preprocessors_path, "standard_scaler.pkl")

    # If scaler already exists (from separate preprocessing run), load it
    if os.path.exists(scaler_path):
        log.info(f"Found existing scaler at: {scaler_path}")
        scaler = joblib.load(scaler_path)
        log.info(f"Loaded scaler: mean={scaler.mean_.tolist()}, scale={scaler.scale_.tolist()}")

    X_train, X_test, y_train, y_test = run_diabetes_preprocessing()
    log.info(f"Loaded preprocessed data: {X_train.shape}, {X_test.shape}")
    return X_train, X_test, y_train, y_test


def engineer_features(
    df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Apply feature engineering to a DataFrame using the diabetes feature engineer.
    
    Args:
        df: Input DataFrame with base features.
        config: Disease configuration dict.
    
    Returns:
        DataFrame with engineered features appended.
    """
    import importlib

    module_path = config.get("features", {}).get("module", "")
    class_name = config.get("features", {}).get("class", "")
    if not module_path or not class_name:
        log.warning("Feature engineer not configured — returning raw DataFrame.")
        return df

    module = importlib.import_module(module_path)
    engineer_class = getattr(module, class_name)
    engineer = engineer_class(config)

    df = engineer.engineer_heuristic(df)
    df = engineer.engineer_medical(df)
    log.info(f"Feature engineering complete. Shape: {df.shape}")
    log.info(f"New columns: {set(df.columns)}")
    return df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> xgb.XGBClassifier:
    """
    Train the XGBoost classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        params: XGBoost hyperparameters.
    
    Returns:
        Trained XGBClassifier.
    """
    log.info("Training XGBoost classifier...")
    log.info(f"Parameters: {json.dumps(params, indent=2)}")

    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    log.info("Training complete.")
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained XGBClassifier.
        X_test: Test features.
        y_test: Test labels.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    log.info("Evaluating model on test set...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    log.info("=" * 50)
    log.info("Test Set Evaluation Metrics:")
    for metric, value in metrics.items():
        log.info(f"  {metric}: {value:.4f}")
    log.info("=" * 50)

    log.info("\nClassification Report:")
    log.info("\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"Confusion Matrix:\n{cm}")

    return metrics


def save_artifacts(
    model: xgb.XGBClassifier,
    metrics: dict,
    config: dict,
) -> None:
    """
    Save the trained model and metrics to disk.
    
    Args:
        model: Trained XGBClassifier.
        metrics: Evaluation metrics dict.
        config: Disease configuration dict.
    """
    weights_path = resolve_path(config, "model", "weights_path")
    preprocessors_path = resolve_path(config, "model", "preprocessors_path")

    # Save model
    weights_dir = os.path.dirname(weights_path)
    os.makedirs(weights_dir, exist_ok=True)
    joblib.dump(model, weights_path)
    log.info(f"Model saved to: {weights_path}")

    # Save metrics
    metrics_path = os.path.join(os.path.dirname(weights_path), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to: {metrics_path}")

    # Save feature names for inference
    feature_names_path = os.path.join(preprocessors_path, "feature_names.json")
    os.makedirs(preprocessors_path, exist_ok=True)
    with open(feature_names_path, "w") as f:
        json.dump(list(model.feature_names_in_), f, indent=2)
    log.info(f"Feature names saved to: {feature_names_path}")


def main():
    """Main training pipeline."""
    log.info("=" * 60)
    log.info("OmniDiag Diabetes Training Pipeline — Starting")
    log.info("=" * 60)

    # Load config
    config = load_config("diabetes")
    log.info(f"Loaded config for disease: {config.get('disease', {}).get('name')}")

    # Default XGBoost parameters (can be tuned later)
    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    # Use best_params from config if available, otherwise defaults
    best_params = config.get("model", {}).get("best_params", {})
    if best_params:
        params = {**default_params, **best_params}
        log.info("Using best_params from config (with defaults for missing keys).")
    else:
        params = default_params
        log.info("Using default parameters (no best_params in config yet).")

    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data(config)

    # Apply feature engineering
    log.info("Applying feature engineering to training set...")
    X_train_fe = engineer_features(X_train, config)
    log.info("Applying feature engineering to test set...")
    X_test_fe = engineer_features(X_test, config)

    # Verify feature consistency
    assert set(X_train_fe.columns) == set(X_test_fe.columns), (
        f"Feature mismatch after engineering:\n"
        f"  Train only: {set(X_train_fe.columns) - set(X_test_fe.columns)}\n"
        f"  Test only:  {set(X_test_fe.columns) - set(X_train_fe.columns)}"
    )
    log.info(f"Feature engineering verified. Total features: {X_train_fe.shape[1]}")

    # Train model
    model = train_model(X_train_fe, y_train, params)

    # Evaluate
    metrics = evaluate_model(model, X_test_fe, y_test)

    # Save artifacts
    save_artifacts(model, metrics, config)

    log.info("=" * 60)
    log.info("OmniDiag Diabetes Training Pipeline — Complete")
    log.info("=" * 60)

    return model, metrics


if __name__ == "__main__":
    main()
