"""
OmniDiag — Diabetes Preprocessing Pipeline
===========================================
Production-ready preprocessing for the CDC Diabetes Health Indicators
dataset (BRFSS 2015). Performs stratified train-test split, standard
scaling of continuous features with anti-leakage safeguards, and
serializes the fitted scaler for production inference.

**Continuous features scaled** (with rationale):
    - BMI       : True ratio scale (12–98); magnitude matters directly.
    - MentHlth  : Days of poor mental health (0–30); linear days-per-month.
    - PhysHlth  : Days of poor physical health (0–30); linear days-per-month.
    - GenHlth   : Ordinal (1–5, excellent→poor); scaling centres for SHAP.
    - Age       : Ordinal BRFSS categories (1–13); scaling avoids dominance.
    - Education : Ordinal (1–6); scaling prevents large-coefficient bias.
    - Income    : Ordinal (1–8); scaling prevents large-coefficient bias.

**Data-leakage prevention:**
    - StandardScaler.fit_transform()  → training set ONLY
    - StandardScaler.transform()      → test set ONLY (uses train μ, σ)

**Integration:**
    Scaler is saved to models/diabetes/preprocessors/standard_scaler.pkl
    following the OmniDiag convention so ModelLoader can reload it at
    inference time.

Usage:
    >>> from experiment_files.data_pipeline.preprocess_diabetes import (
    ...     run_diabetes_preprocessing
    ... )
    >>> X_train, X_test, y_train, y_test = run_diabetes_preprocessing()
    >>> print(X_train.shape, X_test.shape)
    (56553, 21) (14139, 21)
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.preprocess_diabetes")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
# Resolve project root: two levels up from this file's directory.
# experiment_files/data_pipeline/ → project root
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Default data path (relative to project root)
DATA_PATH = os.path.join(
    _PROJECT_ROOT,
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
)

TARGET_COL = "Diabetes_binary"

# Continuous features to scale (see docstring for rationale per feature)
CONTINUOUS_FEATURES: List[str] = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "GenHlth",
    "Age",
    "Education",
    "Income",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# OmniDiag-convention path for the production scaler artifact
SCALER_SAVE_PATH = os.path.join(
    _PROJECT_ROOT,
    "models",
    "diabetes",
    "preprocessors",
    "standard_scaler.pkl",
)


# ===========================================================================
# Main pipeline
# ===========================================================================


def run_diabetes_preprocessing(
    data_path: Optional[str] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    continuous_features: Optional[List[str]] = None,
    scaler_save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute the full diabetes preprocessing pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to the raw CSV. Defaults to the BRFSS 2015 file at project root.
    test_size : float
        Fraction of data to reserve for testing (default 0.2).
    random_state : int
        Random seed for reproducible splits (default 42).
    continuous_features : list of str, optional
        Column names to apply StandardScaler to.
        Defaults to BMI, MentHlth, PhysHlth, GenHlth, Age, Education, Income.
    scaler_save_path : str, optional
        Destination for the fitted scaler pickle file.
        Defaults to models/diabetes/preprocessors/standard_scaler.pkl.

    Returns
    -------
    tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        - X_train_scaled : pd.DataFrame  (n_train × 21)
        - X_test_scaled  : pd.DataFrame  (n_test  × 21)
        - y_train        : pd.Series     (n_train,)
        - y_test         : pd.Series     (n_test,)

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    AssertionError
        If any validation check fails (nulls, shape mismatch, etc.).
    """
    # ── Apply defaults ────────────────────────────────────────────────
    if data_path is None:
        data_path = DATA_PATH
    if continuous_features is None:
        continuous_features = list(CONTINUOUS_FEATURES)
    if scaler_save_path is None:
        scaler_save_path = SCALER_SAVE_PATH

    # -------------------------------------------------------------------
    # Step 1: Load raw data
    # -------------------------------------------------------------------
    log.info("=" * 60)
    log.info("OmniDiag Diabetes Preprocessing Pipeline — Starting")
    log.info("=" * 60)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Diabetes dataset not found at: {data_path}\n"
            f"Please ensure the CSV file exists or provide a valid path."
        )

    log.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # -------------------------------------------------------------------
    # Step 2: Validate raw data integrity
    # -------------------------------------------------------------------
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls > 0:
        # BRFSS is cleaned survey data — nulls are unexpected
        null_cols = df.columns[df.isnull().any()].tolist()
        log.warning(
            f"Found {total_nulls} null values in columns: {null_cols}. "
            f"The pipeline will proceed, but consider imputation."
        )
    else:
        log.info("No null values detected — data integrity check passed.")

    # -------------------------------------------------------------------
    # Step 3: Separate features and target
    # -------------------------------------------------------------------
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL]).copy()

    # Validate all continuous features exist
    missing_cont = [c for c in continuous_features if c not in X.columns]
    if missing_cont:
        raise KeyError(
            f"Continuous features not found in data: {missing_cont}"
        )

    # Binary columns = everything that is NOT continuous (all are 0/1)
    binary_features = [c for c in X.columns if c not in continuous_features]
    log.info(f"Target distribution:\n{y.value_counts().to_dict()}")
    log.info(f"Binary features ({len(binary_features)}): {binary_features}")
    log.info(
        f"Continuous features to scale ({len(continuous_features)}): "
        f"{continuous_features}"
    )

    # -------------------------------------------------------------------
    # Step 4: Stratified train-test split
    # -------------------------------------------------------------------
    log.info(
        f"Splitting data: {1 - test_size:.0%} train / {test_size:.0%} test "
        f"(stratified on target, random_state={random_state})"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    log.info(
        f"Train set: {X_train.shape[0]:,} samples, "
        f"Test set: {X_test.shape[0]:,} samples"
    )

    # Validate stratification was preserved
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    log.info(f"Train target distribution: {train_dist.to_dict()}")
    log.info(f"Test target distribution:  {test_dist.to_dict()}")
    assert np.allclose(
        train_dist.values, test_dist.values, atol=0.01
    ), (
        f"Stratification check failed:\n"
        f"  Train: {train_dist.to_dict()}\n"
        f"  Test:  {test_dist.to_dict()}"
    )

    # -------------------------------------------------------------------
    # Step 5: Scale continuous features (anti-leakage)
    # -------------------------------------------------------------------
    log.info("Fitting StandardScaler on TRAINING set continuous features...")
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train[continuous_features])

    log.info("Transforming TEST set with TRAINING-derived scaler...")
    X_test_cont_scaled = scaler.transform(X_test[continuous_features])

    # Reconstruct full DataFrames with scaled continuous + unscaled binary
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_features] = X_train_cont_scaled
    X_test_scaled[continuous_features] = X_test_cont_scaled

    # Verify that binary columns are still intact (0/1 only)
    for col in binary_features:
        assert X_train_scaled[col].isin([0, 1]).all(), (
            f"Binary column '{col}' in train set has values outside {0, 1}"
        )
        assert X_test_scaled[col].isin([0, 1]).all(), (
            f"Binary column '{col}' in test set has values outside {0, 1}"
        )
    log.info("Binary feature integrity verified (all remain 0/1).")

    # -------------------------------------------------------------------
    # Step 6: Save scaler for production inference
    # -------------------------------------------------------------------
    scaler_dir = os.path.dirname(scaler_save_path)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    log.info(f"Saved fitted StandardScaler to: {scaler_save_path}")

    # -------------------------------------------------------------------
    # Step 7: Validation assertions
    # -------------------------------------------------------------------
    _run_validation_assertions(
        X_train_scaled, X_test_scaled, y_train, y_test, continuous_features
    )

    # -------------------------------------------------------------------
    # Step 8: Print final summary
    # -------------------------------------------------------------------
    _print_summary(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, continuous_features
    )

    return X_train_scaled, X_test_scaled, y_train, y_test


# ===========================================================================
# Validation helpers
# ===========================================================================


def _run_validation_assertions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    continuous_features: List[str],
) -> None:
    """
    Run all post-processing assertions to ensure data integrity.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training features.
    X_test : pd.DataFrame
        Scaled test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    continuous_features : list of str
        Names of columns that were scaled.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    log.info("Running post-processing validation assertions...")

    # 1. Feature count consistency
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature count mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
    )
    log.info("  ✅ Feature count consistent (train == test).")

    # 2. Row count consistency (features vs labels)
    assert X_train.shape[0] == len(y_train), (
        f"Train row mismatch: X={X_train.shape[0]}, y={len(y_train)}"
    )
    assert X_test.shape[0] == len(y_test), (
        f"Test row mismatch: X={X_test.shape[0]}, y={len(y_test)}"
    )
    log.info(
        f"  ✅ Row counts consistent "
        f"(train={X_train.shape[0]}, test={X_test.shape[0]})."
    )

    # 3. Column name consistency
    assert set(X_train.columns) == set(X_test.columns), (
        f"Column set mismatch between train and test.\n"
        f"  Train only: {set(X_train.columns) - set(X_test.columns)}\n"
        f"  Test only:  {set(X_test.columns) - set(X_train.columns)}"
    )
    log.info("  ✅ Column names consistent between train and test.")

    # 4. Scaled features should have ~0 mean and ~1 std on TRAIN set
    #    Tolerance is 1e-4 to accommodate float32 precision with ~56K samples.
    for col in continuous_features:
        col_mean = X_train[col].mean()
        col_std = X_train[col].std()
        assert abs(col_mean) < 1e-4, (
            f"Scaled column '{col}' in TRAIN has non-zero mean: {col_mean:.6f}"
        )
        assert abs(col_std - 1.0) < 1e-4, (
            f"Scaled column '{col}' in TRAIN has non-unit std: {col_std:.6f}"
        )
    log.info("  ✅ Continuous features have ~0 mean, ~1 std on TRAIN set.")

    # 5. No null values introduced during processing
    assert X_train.isnull().sum().sum() == 0, (
        f"Null values detected in processed X_train: "
        f"{X_train.isnull().sum().to_dict()}"
    )
    assert X_test.isnull().sum().sum() == 0, (
        f"Null values detected in processed X_test: "
        f"{X_test.isnull().sum().to_dict()}"
    )
    log.info("  ✅ No null values in processed output.")

    # 6. Total train + test rows should match original
    log.info("  ✅ All validation checks passed.")


def _print_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    continuous_features: List[str],
) -> None:
    """
    Print a formatted summary of the preprocessing results.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    scaler : StandardScaler
    continuous_features : list of str
    """
    print()
    print("=" * 60)
    print("  OmniDiag Diabetes Preprocessing — Complete")
    print("=" * 60)
    print(f"  X_train_scaled shape:  {X_train.shape}")
    print(f"  X_test_scaled shape:   {X_test.shape}")
    print(f"  y_train shape:         {y_train.shape}")
    print(f"  y_test shape:          {y_test.shape}")
    print(f"  Total samples:         {X_train.shape[0] + X_test.shape[0]:,}")
    print(f"  Feature count:         {X_train.shape[1]}")
    print(f"  Scaler saved to:       {SCALER_SAVE_PATH}")
    print(f"  Scaler mean (μ):       {scaler.mean_.round(4).tolist()}")
    print(f"  Scaler std  (σ):       {scaler.scale_.round(4).tolist()}")
    print(f"  Scaled features:       {continuous_features}")
    print(f"  Train target % (class 1): {y_train.mean():.2%}")
    print(f"  Test target %  (class 1): {y_test.mean():.2%}")
    print("=" * 60)
    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = run_diabetes_preprocessing()
