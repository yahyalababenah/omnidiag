"""
OmniDiag — Diabetes Model Hyperparameter Optimization
======================================================
Runs Optuna hyperparameter optimization for XGBoost and LightGBM
on the CDC BRFSS 2015 Diabetes Health Indicators dataset.

Key characteristics of this dataset:
    - Balanced classes (50/50) → no scale_pos_weight needed
    - 70,692 samples, 26 features (21 base + 5 engineered)
    - Binary classification: diabetes vs no diabetes

Search spaces designed based on:
    - XGBoost: Medium-depth trees (max_depth 4-10), moderate regularization
    - LightGBM: Leaf-wise trees with leaf count and subsampling controls

Usage:
    python experiment_files/models/optimize_diabetes_models.py
"""

import os
import sys
import json
import datetime
import logging
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("omnidiag.optimize_diabetes")

# ── Constants ────────────────────────────────────────────────────────
N_TRIALS = 100
N_FOLDS = 5
RANDOM_STATE = 42
TARGET_COL = "Diabetes_binary"

# Model output paths
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "diabetes")
GRID_BEST_PATH = os.path.join(MODELS_DIR, "grid_best.json")
PREPROCESSORS_DIR = os.path.join(MODELS_DIR, "preprocessors")

# ── Data Loading ─────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Run the preprocessing pipeline and feature engineering,
    returning a single DataFrame ready for model training.

    Returns:
        DataFrame with all 26 features + target column.
    """
    from experiment_files.data_pipeline.preprocess_diabetes import (
        run_diabetes_preprocessing,
    )
    from features.diabetes_features import DiabetesFeatureEngineer
    from configs.config_loader import load_config

    config = load_config("diabetes")
    X_train, X_test, y_train, y_test = run_diabetes_preprocessing()

    # Combine train + test for cross-validation
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    # Apply feature engineering
    engineer = DiabetesFeatureEngineer(config)
    X = engineer.engineer_heuristic(X)
    X = engineer.engineer_medical(X)

    # Add target back
    df = X.copy()
    df[TARGET_COL] = y.values

    log.info(f"Optimization dataset: {df.shape[0]} samples, {df.shape[1] - 1} features")
    log.info(f"Target distribution: {y.value_counts().to_dict()}")
    log.info(f"Features: {list(X.columns)}")

    return df


def prepare_xy(df: pd.DataFrame):
    """Split DataFrame into features and target."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


# ── XGBoost Optimization ─────────────────────────────────────────────

def xgb_objective(trial, X, y):
    """Optuna objective for XGBoost — maximize CV accuracy."""

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    import xgboost as xgb
    model = xgb.XGBClassifier(**param, use_label_encoder=False)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def optimize_xgboost(X, y) -> Dict[str, Any]:
    """Run Optuna optimization for XGBoost."""
    log.info("\n" + "=" * 60)
    log.info("XGBoost Hyperparameter Optimization — Starting")
    log.info(f"Trials: {N_TRIALS} | CV folds: {N_FOLDS} | Dataset: {X.shape[0]} × {X.shape[1]}")
    log.info("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        study_name="diabetes_xgb",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: xgb_objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info(f"\n🏆 Best XGBoost Trial (#{best.number})")
    log.info(f"   CV Accuracy: {best.value:.4f}")
    log.info(f"   Params: {best.params}")

    return {
        "model": "xgboost",
        "best_score": round(float(best.value), 6),
        "best_params": best.params,
    }


# ── LightGBM Optimization ────────────────────────────────────────────

def lgb_objective(trial, X, y):
    """Optuna objective for LightGBM — maximize CV accuracy."""
    import lightgbm as lgb

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**param)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def optimize_lightgbm(X, y) -> Dict[str, Any]:
    """Run Optuna optimization for LightGBM."""
    log.info("\n" + "=" * 60)
    log.info("LightGBM Hyperparameter Optimization — Starting")
    log.info(f"Trials: {N_TRIALS} | CV folds: {N_FOLDS} | Dataset: {X.shape[0]} × {X.shape[1]}")
    log.info("=" * 60)

    import lightgbm as lgb  # noqa: ensure importable
    study = optuna.create_study(
        direction="maximize",
        study_name="diabetes_lgb",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: lgb_objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info(f"\n🏆 Best LightGBM Trial (#{best.number})")
    log.info(f"   CV Accuracy: {best.value:.4f}")
    log.info(f"   Params: {best.params}")

    return {
        "model": "lightgbm",
        "best_score": round(float(best.value), 6),
        "best_params": best.params,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("OmniDiag Diabetes — Hyperparameter Optimization")
    log.info(f"LightGBM version: {__import__('lightgbm').__version__}")
    log.info(f"Optuna version: {optuna.__version__}")
    log.info("=" * 60)

    # Load and prepare data
    df = load_data()
    X, y = prepare_xy(df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Run optimizations ────────────────────────────────────────
    results = {
        "optimization_date": datetime.datetime.now().isoformat(),
        "dataset": {
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "class_balance": y.value_counts(normalize=True).to_dict(),
            "feature_names": list(X.columns),
        },
        "optimization": {
            "n_trials": N_TRIALS,
            "n_folds": N_FOLDS,
            "metric": "accuracy",
            "random_state": RANDOM_STATE,
        },
        "results": {},
    }

    # XGBoost
    xgb_result = optimize_xgboost(X, y)
    results["results"]["xgboost"] = xgb_result

    # LightGBM
    lgb_result = optimize_lightgbm(X, y)
    results["results"]["lightgbm"] = lgb_result

    # ── Compare and select best ──────────────────────────────────
    best_model = max(results["results"], key=lambda m: results["results"][m]["best_score"])
    results["best_overall"] = {
        "model": best_model,
        "score": results["results"][best_model]["best_score"],
        "params": results["results"][best_model]["best_params"],
    }

    log.info("\n" + "=" * 60)
    log.info("Optimization Complete — Results Summary")
    log.info("=" * 60)
    log.info(f"  XGBoost  CV Accuracy: {xgb_result['best_score']:.4f}")
    log.info(f"  LightGBM CV Accuracy: {lgb_result['best_score']:.4f}")
    log.info(f"  🏆 Best model: {best_model} ({results['best_overall']['score']:.4f})")

    # Save results
    with open(GRID_BEST_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to: {GRID_BEST_PATH}")

    return results


if __name__ == "__main__":
    main()
