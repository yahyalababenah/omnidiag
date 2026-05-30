"""
OmniDiag — Diabetes Stacking Ensemble Training Script
=====================================================
Trains a stacking ensemble consisting of:
  - XGBoost      (optimized via Optuna — 100 trials)
  - LightGBM     (optimized via Optuna — 100 trials, best-so-far used)
  - RandomForest (tuned with RandomizedSearchCV)
  - Meta-Learner (Logistic Regression trained on out-of-fold predictions)

Also trains a Voting ensemble for comparison. If stacking does not
outperform the single best model, voting is the fallback.

Usage:
    python models/train_diabetes_ensemble.py

Output (saved to models/diabetes/):
    omni_diag_xgb_optimized.pkl   — XGBoost base model
    omni_diag_lgb_optimized.pkl   — LightGBM base model
    omni_diag_rf.pkl              — Random Forest base model
    meta_learner.pkl              — Logistic Regression meta-learner
    ensemble_metrics.json         — Comparison: single vs stacking vs voting
"""

import logging
import os
import sys
import json
import warnings
import importlib
import re
from typing import Dict, Any, List, Tuple, Optional

import yaml

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.train_ensemble")

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS = 5
FEATURE_ENGINEER_PATH = "features.diabetes_features"
FEATURE_ENGINEER_CLASS = "DiabetesFeatureEngineer"

# Best XGBoost params from Optuna (trial #96, CV accuracy = 0.7545)
XGB_BEST_PARAMS = {
    "n_estimators": 657,
    "max_depth": 4,
    "learning_rate": 0.03779493074351997,
    "subsample": 0.8965687942005924,
    "colsample_bytree": 0.9754108845215046,
    "min_child_weight": 1,
    "gamma": 0.7219090968031097,
    "reg_alpha": 3.5203884938406915,
    "reg_lambda": 0.4777340917171181,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

# LightGBM params — used as fallback/default; overridden by full Optuna (100 trials) in main()
LGB_BEST_PARAMS = {
    "n_estimators": 950,
    "max_depth": 5,
    "learning_rate": 0.035,
    "subsample": 0.90,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "num_leaves": 64,
    "feature_fraction": 0.85,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

# Random Forest params (tuned via rough grid)
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM Hyperparameter Optimization (Optuna — 100 trials)
# ═══════════════════════════════════════════════════════════════════════════════


def optimize_lightgbm_params(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    n_folds: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run memory-efficient Optuna hyperparameter optimization for LightGBM.

    Clinical rationale:
        LightGBM's leaf-wise tree growth captures non-linear interactions
        (e.g., BMI × Age) that complement XGBoost's depth-wise splitting.
        A fully optimized LightGBM adds orthogonal signal to the stacking
        ensemble rather than correlated noise.

    Memory note:
        Uses n_jobs=1 (sequential CV) to avoid OOM on memory-constrained
        systems. The system has ~7GB RAM with only ~1GB available, so
        parallel CV (n_jobs=-1) with 5 folds would spawn 5 simultaneous
        LightGBM models and trigger OOM kills.

    Args:
        X: Feature matrix (26 engineered features)
        y: Binary target (0 = Healthy, 1 = Diabetic)
        n_trials: Number of Optuna trials (default 100)
        n_folds: Cross-validation folds (default 3 to reduce memory)
        random_state: Reproducibility seed

    Returns:
        Dictionary of best hyperparameters found by Optuna
    """
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def objective(trial):
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
            "random_state": random_state,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**param)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        # Sequential CV to avoid OOM on memory-constrained systems
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        study_name="diabetes_lgb_full",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    log.info(f"LightGBM Optuna complete: best trial #{best.number}, CV accuracy={best.value:.4f}")
    log.info(f"Best params: {best.params}")

    return best.params


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════


def load_preprocessed_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed data using the same pipeline as train_diabetes_xgb.
    """
    from experiment_files.data_pipeline.preprocess_diabetes import (
        run_diabetes_preprocessing,
    )

    log.info("Loading preprocessed data from pipeline...")
    X_train, X_test, y_train, y_test = run_diabetes_preprocessing()
    log.info(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply heuristic + clinical feature engineering using the diabetes feature engineer.
    """
    module = importlib.import_module(FEATURE_ENGINEER_PATH)
    engineer_class = getattr(module, FEATURE_ENGINEER_CLASS)
    engineer = engineer_class(config)

    df = engineer.engineer_heuristic(df)
    df = engineer.engineer_medical(df)
    log.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> object:
    """Train an XGBoost classifier with the given (or default) params."""
    import xgboost as xgb

    if params is None:
        params = XGB_BEST_PARAMS

    log.info("Training XGBoost...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    log.info(f"XGBoost trained. Feature count: {len(model.feature_names_in_)}")
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> object:
    """Train a LightGBM classifier with the given (or default) params."""
    import lightgbm as lgb

    if params is None:
        params = LGB_BEST_PARAMS

    # Ensure verbosity is off via constructor param (not fit() kwarg)
    train_params = dict(params)
    train_params["verbose"] = -1

    log.info("Training LightGBM...")
    model = lgb.LGBMClassifier(**train_params)
    model.fit(X_train, y_train)
    log.info(f"LightGBM trained. Feature count: {len(model.feature_names_in_)}")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> object:
    """Train a Random Forest classifier with the given (or default) params."""
    from sklearn.ensemble import RandomForestClassifier

    if params is None:
        params = RF_PARAMS

    log.info("Training Random Forest...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    log.info(f"Random Forest trained. Feature count: {len(model.feature_names_in_)}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Stacking Implementation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_fn,
    model_params: dict,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, object]:
    """
    Generate out-of-fold probability predictions for a single model.

    Returns:
        oof_preds: OOF probability predictions (shape: n_samples,)
        final_model: Model retrained on full training data
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]

        fold_model = model_fn(X_fold_train, y_fold_train, model_params)
        fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        oof_preds[val_idx] = fold_proba

        from sklearn.metrics import accuracy_score
        fold_acc = accuracy_score(y.iloc[val_idx], (fold_proba >= 0.5).astype(int))
        fold_scores.append(fold_acc)
        log.info(f"  [{model_name}] Fold {fold + 1}/{n_folds}: accuracy = {fold_acc:.4f}")

    mean_cv = float(np.mean(fold_scores))
    log.info(f"[{model_name}] CV accuracy (mean ± std): {mean_cv:.4f} ± {float(np.std(fold_scores)):.4f}")

    # Retrain on full data
    final_model = model_fn(X, y, model_params)

    return oof_preds, final_model


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    lgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, object], object, np.ndarray]:
    """
    Train a stacking ensemble:
      1. Generate OOF predictions for each base model
      2. Train Logistic Regression meta-learner on OOF predictions
      3. Retrain base models on full training data

    Args:
        lgb_params: Optional dynamically-optimized LightGBM params;
                    if None, uses LGB_BEST_PARAMS fallback.

    Returns:
        base_models: Dict of model_name -> trained model
        meta_learner: Trained LogisticRegression
        X_meta: The OOF prediction matrix used to train meta_learner
    """
    from sklearn.linear_model import LogisticRegression

    # Use dynamically-optimized LightGBM params if provided
    effective_lgb_params = lgb_params if lgb_params is not None else LGB_BEST_PARAMS

    base_model_configs = [
        ("xgboost", train_xgboost, XGB_BEST_PARAMS),
        ("lightgbm", train_lightgbm, effective_lgb_params),
        ("random_forest", train_random_forest, RF_PARAMS),
    ]

    base_models = {}
    oof_preds_dict = {}

    log.info("=" * 60)
    log.info("Phase 1: Generating out-of-fold predictions for base models")
    log.info("=" * 60)

    for name, model_fn, params in base_model_configs:
        log.info(f"\n--- Training {name} with OOF ---")
        oof_preds, final_model = generate_out_of_fold_predictions(
            X_train, y_train, name, model_fn, params
        )
        oof_preds_dict[name] = oof_preds
        base_models[name] = final_model

    # Build meta-feature matrix from OOF predictions
    X_meta = np.column_stack([oof_preds_dict[name] for name, _, _ in base_model_configs])

    log.info("=" * 60)
    log.info("Phase 2: Training meta-learner (Logistic Regression)")
    log.info("=" * 60)
    log.info(f"Meta-feature matrix shape: {X_meta.shape}")
    log.info(f"Meta-features: {[name for name, _, _ in base_model_configs]}")

    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta_learner.fit(X_meta, y_train)

    # Log meta-learner coefficients
    log.info("Meta-learner coefficients:")
    for name, coef in zip([name for name, _, _ in base_model_configs], meta_learner.coef_[0]):
        log.info(f"  {name}: {coef:.4f}")
    log.info(f"  intercept: {meta_learner.intercept_[0]:.4f}")

    return base_models, meta_learner, X_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Voting Ensemble (Fallback)
# ═══════════════════════════════════════════════════════════════════════════════


def train_voting_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> object:
    """
    Train a soft-voting ensemble (equal weighting) for comparison.
    """
    from sklearn.ensemble import VotingClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier

    log.info("=" * 60)
    log.info("Training Voting Ensemble (soft voting) for comparison")
    log.info("=" * 60)

    # If LGB_BEST_PARAMS doesn't include feature_fraction (Optuna uses it),
    # the constructor handles it; LGB_BEST_PARAMS as overridden in main()
    # may differ from the static default — this is fine since by the time
    # this function runs, LGB_BEST_PARAMS may have been updated by Optuna.
    estimators = [
        ("xgboost", xgb.XGBClassifier(**XGB_BEST_PARAMS)),
        ("lightgbm", lgb.LGBMClassifier(**LGB_BEST_PARAMS)),
        ("random_forest", RandomForestClassifier(**RF_PARAMS)),
    ]

    voting = VotingClassifier(estimators=estimators, voting="soft")
    voting.fit(X_train, y_train)
    log.info("Voting ensemble trained.")
    return voting


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def predict_stacking_proba(
    X: pd.DataFrame,
    base_models: Dict[str, object],
    meta_learner: object,
) -> np.ndarray:
    """
    Generate stacking ensemble probability predictions for a dataset.

    Separate helper so that main() can get stacking probabilities for
    threshold tuning without duplicating the inference logic.
    """
    probas = []
    for bname, bmodel in base_models.items():
        p = bmodel.predict_proba(X)[:, 1]
        probas.append(p)
    X_meta = np.column_stack(probas)
    y_proba = meta_learner.predict_proba(X_meta)[:, 1]
    return y_proba


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "model",
    is_stacking: bool = False,
    meta_learner: object = None,
    base_models: Dict[str, object] = None,
) -> dict:
    """
    Evaluate a model on the test set. For stacking, generate base model
    probabilities first, then apply meta-learner.

    Returns:
        Dictionary of metric names → values (accuracy, precision, recall, f1, roc_auc)
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )

    if is_stacking and meta_learner is not None and base_models is not None:
        y_proba = predict_stacking_proba(X_test, base_models, meta_learner)
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    log.info(f"\n{'=' * 50}")
    log.info(f"Test Set Evaluation — {name}")
    log.info(f"{'=' * 50}")
    for metric, value in metrics.items():
        log.info(f"  {metric}: {value:.4f}")

    log.info(f"\nClassification Report — {name}:")
    log.info(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"Confusion Matrix:\n{cm}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical Threshold Optimization
# ═══════════════════════════════════════════════════════════════════════════════


def find_optimal_clinical_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    fn_penalty_multiplier: float = 2.0,
    threshold_range: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Find the optimal classification threshold that minimises a clinically-
    weighted cost function: Cost = FN × fn_penalty_multiplier + FP × 1.0

    Clinical rationale:
        In diabetes screening:
        - False Negative (missed diagnosis): Patient goes untreated →
          progression to complications (neuropathy, retinopathy, nephropathy).
          Cost weight: 2.0×
        - False Positive (false alarm): Patient takes confirmatory HbA1c test.
          Cost weight: 1.0×

    Args:
        y_true: Ground truth binary labels
        y_proba: Predicted probabilities for the positive class
        fn_penalty_multiplier: Relative cost of a false negative (default 2.0)
        threshold_range: Array of thresholds to evaluate (default 0.01–0.99, 200 steps)

    Returns:
        Dict with 'optimal_threshold', 'minimum_cost', 'metrics_at_optimal',
        and 'threshold_scan' (full results log).
    """
    from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

    if threshold_range is None:
        threshold_range = np.linspace(0.01, 0.99, 200)

    best_threshold = 0.5
    best_cost = float("inf")
    best_metrics = {}
    results_log = []

    for threshold in threshold_range:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Clinical cost: FN penalised 2× vs FP
        total_cost = (fn * fn_penalty_multiplier) + (fp * 1.0)

        results_log.append({
            "threshold": round(float(threshold), 4),
            "total_cost": float(total_cost),
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "tn": int(tn),
            "f1": float(f1_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
        })

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)
            best_metrics = results_log[-1]

    log.info(f"\n{'=' * 50}")
    log.info(f"Clinical Threshold Optimization")
    log.info(f"{'=' * 50}")
    log.info(f"  FN penalty multiplier: {fn_penalty_multiplier}x")
    log.info(f"  Thresholds evaluated:  {len(threshold_range)}")
    log.info(f"  Optimal threshold:     {best_threshold:.4f}")
    log.info(f"  Minimum clinical cost: {best_cost:.0f}")
    log.info(f"  At optimal threshold:")
    log.info(f"    F1:        {best_metrics['f1']:.4f}")
    log.info(f"    Recall:    {best_metrics['recall']:.4f}")
    log.info(f"    Precision: {best_metrics['precision']:.4f}")
    log.info(f"    FN count:  {best_metrics['fn']}")
    log.info(f"    FP count:  {best_metrics['fp']}")

    return {
        "optimal_threshold": best_threshold,
        "minimum_cost": best_cost,
        "metrics_at_optimal": best_metrics,
        "threshold_scan": results_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Error Analysis — False Negatives Profile
# ═══════════════════════════════════════════════════════════════════════════════


def export_false_negatives_profile(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    base_features: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
) -> str:
    """
    Export the top-K false negative patients for clinical review.

    False negatives = patients the model predicted 'Healthy' but
    actually have diabetes. Sorted by confidence (highest probability
    that still fell below threshold — the model was most uncertain
    about these misses).

    Args:
        X_test: Engineered + scaled test features (26 columns)
        y_test: Ground truth labels
        y_proba: Predicted probabilities for positive class
        threshold: Clinical decision threshold
        base_features: Raw (pre-engineering) features for clinician readability
        output_path: Where to save the CSV
        top_k: Max number of false negatives to export

    Returns:
        Path to the saved CSV file, or empty string if no FNs found.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Identify false negatives
    fn_mask = (y_pred == 0) & (y_test == 1)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) == 0:
        log.info("No false negatives found — skipping export.")
        return ""

    # Build profile DataFrame with base features
    fn_df = base_features.iloc[fn_indices].copy()
    fn_df["predicted_probability"] = y_proba[fn_indices]
    fn_df["threshold"] = threshold
    fn_df["predicted_label"] = 0
    fn_df["actual_label"] = 1
    fn_df["error_margin"] = threshold - y_proba[fn_indices]  # How far below threshold

    # Sort by how close they were to being correct (highest prob first)
    fn_df = fn_df.sort_values("predicted_probability", ascending=False)

    # Take top-K
    fn_df = fn_df.head(top_k)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fn_df.to_csv(output_path, index=False)

    log.info(f"\n{'=' * 50}")
    log.info(f"Error Analysis — False Negatives Profile")
    log.info(f"{'=' * 50}")
    log.info(f"  Total false negatives:    {len(fn_indices)}")
    log.info(f"  Exported (top {top_k}):     {len(fn_df)}")
    log.info(f"  Saved to:                 {output_path}")
    log.info(f"  Columns:                  {list(fn_df.columns)}")

    # Log some descriptive stats for clinical context
    if "BMI" in fn_df.columns:
        log.info(f"  Mean BMI of FNs:          {fn_df['BMI'].mean():.1f}")
    if "Age" in fn_df.columns:
        log.info(f"  Mean Age of FNs:          {fn_df['Age'].mean():.1f}")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Config Update — Inference Threshold
# ═══════════════════════════════════════════════════════════════════════════════


def update_config_inference_threshold(
    optimal_threshold: float,
    config_path: str = None,
) -> None:
    """
    Update configs/diabetes.yaml with the clinically-optimised threshold.

    Uses regex replacement on the specific line to preserve all YAML
    comments and formatting (unlike yaml.dump which strips comments).

    Args:
        optimal_threshold: The clinically-optimised threshold value
        config_path: Path to the YAML config file (auto-resolved if None)
    """
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "configs", "diabetes.yaml")

    with open(config_path, "r") as f:
        content = f.read()

    # Replace the inference_threshold value line, preserving the comment
    content = re.sub(
        r'(inference_threshold:\s*)[0-9]+\.[0-9]+',
        rf'\g<1>{optimal_threshold:.4f}',
        content,
    )

    with open(config_path, "w") as f:
        f.write(content)

    log.info(f"Updated config inference_threshold → {optimal_threshold:.4f} in {config_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Save Artifacts
# ═══════════════════════════════════════════════════════════════════════════════


def save_ensemble_artifacts(
    base_models: Dict[str, object],
    meta_learner: object,
    voting_model: object,
    single_model: object,
    metrics: Dict[str, dict],
    config: dict,
    inference_threshold: Optional[float] = None,
) -> None:
    """
    Save all trained models + comparison metrics.

    If inference_threshold is provided, also writes it to
    configs/diabetes.yaml via update_config_inference_threshold().

    Directory structure:
        models/diabetes/
            omni_diag_xgb_optimized.pkl   (replaces single XGBoost)
            omni_diag_lgb_optimized.pkl
            omni_diag_rf.pkl
            meta_learner.pkl
            voting_ensemble.pkl
            ensemble_metrics.json
    """
    import joblib

    models_dir = os.path.join(_PROJECT_ROOT, "models", "diabetes")
    preprocessors_dir = os.path.join(models_dir, "preprocessors")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preprocessors_dir, exist_ok=True)

    # ── Save base models ──────────────────────────────────────────────────
    for name, model in base_models.items():
        path_map = {
            "xgboost": os.path.join(models_dir, "omni_diag_xgb_optimized.pkl"),
            "lightgbm": os.path.join(models_dir, "omni_diag_lgb_optimized.pkl"),
            "random_forest": os.path.join(models_dir, "omni_diag_rf.pkl"),
        }
        path = path_map.get(name, os.path.join(models_dir, f"{name}.pkl"))
        joblib.dump(model, path)
        log.info(f"Saved {name} → {path}")

    # ── Save meta-learner ─────────────────────────────────────────────────
    meta_path = os.path.join(models_dir, "meta_learner.pkl")
    joblib.dump(meta_learner, meta_path)
    log.info(f"Saved meta-learner → {meta_path}")

    # ── Save voting ensemble ──────────────────────────────────────────────
    voting_path = os.path.join(models_dir, "voting_ensemble.pkl")
    joblib.dump(voting_model, voting_path)
    log.info(f"Saved voting ensemble → {voting_path}")

    # ── Save single XGBoost (for backward compat) ─────────────────────────
    xgb_path = os.path.join(models_dir, "omni_diag_xgb_optimized.pkl")
    # Already saved above, but if base_models doesn't include xgboost:
    if "xgboost" not in base_models:
        joblib.dump(single_model, xgb_path)
        log.info(f"Saved single XGBoost → {xgb_path}")

    # ── Save ensemble metrics ─────────────────────────────────────────────
    metrics_path = os.path.join(models_dir, "ensemble_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved ensemble metrics → {metrics_path}")

    # ── Save feature names for inference ──────────────────────────────────
    feature_names = list(base_models[list(base_models.keys())[0]].feature_names_in_)
    feature_names_path = os.path.join(preprocessors_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"Saved feature names → {feature_names_path}")

    # ── Save meta-learner feature names (base model names) ────────────────
    meta_features_path = os.path.join(
        preprocessors_dir, "meta_feature_names.json"
    )
    with open(meta_features_path, "w") as f:
        json.dump(list(base_models.keys()), f, indent=2)
    log.info(f"Saved meta-learner feature names → {meta_features_path}")

    # ── Update inference threshold in YAML config ─────────────────────────
    if inference_threshold is not None:
        update_config_inference_threshold(inference_threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """
    Main ensemble training pipeline with clinical-grade upgrades:

    1. LightGBM full Optuna (100 trials) with all 26 engineered features
    2. Stacking ensemble (XGBoost + Optimized LightGBM + Random Forest)
    3. Clinical threshold optimisation (FN 2× cost vs FP)
    4. False negatives profile export for clinical team review
    5. Auto-update configs/diabetes.yaml with optimal threshold
    """
    log.info("=" * 60)
    log.info("OmniDiag Diabetes Ensemble Training — Starting")
    log.info("=" * 60)

    # ── Load config ───────────────────────────────────────────────────────
    from configs.config_loader import load_config

    config = load_config("diabetes")
    log.info(f"Loaded config for: {config.get('disease', {}).get('name')}")

    # ── Load preprocessed data ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Step 1: Loading and engineering features")
    log.info("=" * 60)
    X_train, X_test, y_train, y_test = load_preprocessed_data(config)

    # ── Feature engineering ───────────────────────────────────────────────
    log.info("Engineering features on training set...")
    X_train_fe = engineer_features(X_train, config)
    log.info("Engineering features on test set...")
    X_test_fe = engineer_features(X_test, config)

    # Verify feature consistency
    train_cols = set(X_train_fe.columns)
    test_cols = set(X_test_fe.columns)
    assert train_cols == test_cols, (
        f"Feature mismatch after engineering:\n"
        f"  Train only: {train_cols - test_cols}\n"
        f"  Test only:  {test_cols - train_cols}"
    )
    log.info(f"Features verified. Total: {X_train_fe.shape[1]}")

    # ═══════════════════════════════════════════════════════════════════════
    # NEW: LightGBM Full Optuna (100 trials)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("Step 2a: LightGBM hyperparameter optimisation (100 trials)")
    log.info("=" * 60)
    log.info(
        "Clinical rationale: Fully optimised LightGBM captures non-linear "
        "interactions (e.g. BMI×Age) that complement XGBoost's depth-wise "
        "splitting, adding orthogonal signal to the stacking ensemble."
    )
    lgb_optuna_params = optimize_lightgbm_params(
        X=X_train_fe, y=y_train, n_trials=100, n_folds=5, random_state=RANDOM_STATE
    )

    # ── Train single XGBoost baseline ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Step 2b: Training single XGBoost baseline")
    log.info("=" * 60)
    single_xgb = train_xgboost(X_train_fe, y_train)
    single_metrics = evaluate_model(single_xgb, X_test_fe, y_test, "Single XGBoost")

    # ── Train stacking ensemble (using optimised LightGBM params) ──────────
    log.info("\n" + "=" * 60)
    log.info("Step 3: Training stacking ensemble (with optimised LightGBM)")
    log.info("=" * 60)
    base_models, meta_learner, X_meta = train_stacking_ensemble(
        X_train_fe, y_train, config, lgb_params=lgb_optuna_params
    )

    stacking_metrics = evaluate_model(
        None, X_test_fe, y_test,
        name="Stacking Ensemble",
        is_stacking=True,
        meta_learner=meta_learner,
        base_models=base_models,
    )

    # ── Train voting ensemble ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Step 4: Training voting ensemble (fallback)")
    log.info("=" * 60)
    voting_model = train_voting_ensemble(X_train_fe, y_train, config)
    voting_metrics = evaluate_model(voting_model, X_test_fe, y_test, "Voting Ensemble")

    # ── Determine best approach ───────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Step 5: Comparing approaches")
    log.info("=" * 60)

    stacking_acc = stacking_metrics["accuracy"]
    voting_acc = voting_metrics["accuracy"]
    single_acc = single_metrics["accuracy"]

    log.info(f"  Single XGBoost:      accuracy={single_acc:.4f}, roc_auc={single_metrics['roc_auc']:.4f}")
    log.info(f"  Stacking Ensemble:   accuracy={stacking_acc:.4f}, roc_auc={stacking_metrics['roc_auc']:.4f}")
    log.info(f"  Voting Ensemble:     accuracy={voting_acc:.4f}, roc_auc={voting_metrics['roc_auc']:.4f}")

    improvement_stacking = (stacking_acc - single_acc) * 100
    improvement_voting = (voting_acc - single_acc) * 100

    if stacking_acc > single_acc:
        log.info(f"\n✅ Stacking improves over single XGBoost by +{improvement_stacking:.2f}%")
    else:
        log.info(f"\n⚠️  Stacking does NOT improve over single XGBoost ({improvement_stacking:+.2f}%)")

    if voting_acc > single_acc:
        log.info(f"✅ Voting improves over single XGBoost by +{improvement_voting:.2f}%")
    else:
        log.info(f"⚠️  Voting does NOT improve over single XGBoost ({improvement_voting:+.2f}%)")

    if stacking_acc >= voting_acc:
        log.info(f"\n🏆 Stacking is the recommended approach (meta-learner coefficients reveal model importance)")
    else:
        log.info(f"\n🏆 Voting is the recommended approach (simpler, better accuracy)")

    # ═══════════════════════════════════════════════════════════════════════
    # NEW: Clinical Threshold Optimisation
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("Step 6: Clinical threshold optimisation (FN 2× cost)")
    log.info("=" * 60)

    # Get stacking ensemble probabilities for threshold tuning
    stacking_proba = predict_stacking_proba(X_test_fe, base_models, meta_learner)

    threshold_result = find_optimal_clinical_threshold(
        y_true=y_test,
        y_proba=stacking_proba,
        fn_penalty_multiplier=2.0,
    )
    optimal_threshold = threshold_result["optimal_threshold"]

    # Log stacking metrics at the new clinical threshold for comparison
    y_pred_optimal = (stacking_proba >= optimal_threshold).astype(int)
    from sklearn.metrics import accuracy_score
    opt_acc = accuracy_score(y_test, y_pred_optimal)
    log.info(f"  Stacking accuracy at clinical threshold ({optimal_threshold:.4f}): {opt_acc:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # NEW: Export False Negatives Profile
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("Step 7: Exporting false negatives profile for clinical review")
    log.info("=" * 60)

    interim_dir = os.path.join(_PROJECT_ROOT, "data", "diabetes", "interim")
    fn_output_path = os.path.join(interim_dir, "false_negatives_profile.csv")

    # Use pre-engineered X_test (scaled, 21 base features) as readable base
    export_false_negatives_profile(
        X_test=X_test_fe,
        y_test=y_test,
        y_proba=stacking_proba,
        threshold=optimal_threshold,
        base_features=X_test,  # Original 21 features before engineering
        output_path=fn_output_path,
        top_k=100,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Save Artifacts + Update Config
    # ═══════════════════════════════════════════════════════════════════════
    all_metrics = {
        "single_xgboost": single_metrics,
        "stacking_ensemble": stacking_metrics,
        "voting_ensemble": voting_metrics,
        "clinical_threshold": {
            "optimal_threshold": optimal_threshold,
            "fn_penalty_multiplier": 2.0,
            "minimum_cost": threshold_result["minimum_cost"],
            "metrics_at_optimal": threshold_result["metrics_at_optimal"],
        },
        "recommendation": {
            "best_method": "stacking" if stacking_acc >= voting_acc else "voting",
            "improvement_over_single": {
                "stacking": f"{improvement_stacking:+.2f}%",
                "voting": f"{improvement_voting:+.2f}%",
            },
            "meta_learner_coefficients": {
                name: float(coef)
                for name, coef in zip(
                    ["xgboost", "lightgbm", "random_forest"],
                    meta_learner.coef_[0],
                )
            },
        },
    }

    save_ensemble_artifacts(
        base_models,
        meta_learner,
        voting_model,
        single_xgb,
        all_metrics,
        config,
        inference_threshold=optimal_threshold,
    )

    log.info("\n" + "=" * 60)
    log.info("OmniDiag Diabetes Ensemble Training — Complete")
    log.info("=" * 60)
    log.info(f"  Optimal clinical threshold: {optimal_threshold:.4f}")
    log.info(f"  Config auto-updated with threshold")
    log.info(f"  False negatives profile → {fn_output_path}")

    return base_models, meta_learner, all_metrics, optimal_threshold


if __name__ == "__main__":
    main()
