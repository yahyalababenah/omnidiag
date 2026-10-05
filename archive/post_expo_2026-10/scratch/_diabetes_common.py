"""
Shared, read-only helpers for the diabetes scratch scripts.

Reproduces the training split of experiment_files/data_pipeline/preprocess_diabetes.py
WITHOUT calling run_diabetes_preprocessing() — that function re-fits and
overwrites models/diabetes/preprocessors/standard_scaler.pkl. Here the shipped
scaler is loaded and only ever used for .transform().

Nothing in this module writes to models/ or configs/.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from features.diabetes_features import DiabetesFeatureEngineer

CONFIG_PATH = os.path.join(ROOT, "configs", "diabetes.yaml")
CSV_PATH = os.path.join(
    ROOT, "data", "diabetes", "raw",
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
)
EVIDENCE_DIR = os.path.join(ROOT, "evaluation_evidence", "diabetes")
TARGET = "Diabetes_binary"
SEED = 42
FN_COST, FP_COST = 2.0, 1.0
OLD_THRESHOLD = 0.275          # value shipped before this fix (selected on y_test)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_split(cfg):
    """Same split as preprocess_diabetes: test_size=0.2, random_state=42, stratify=y."""
    df = pd.read_csv(CSV_PATH)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    tcfg = cfg.get("training", {})
    return train_test_split(
        X, y,
        test_size=tcfg.get("test_size", 0.2),
        random_state=tcfg.get("random_state", SEED),
        stratify=y,
    )


def make_preparer(cfg):
    """scale (shipped scaler, transform only) -> engineer. Same order as training + loader."""
    scaler = joblib.load(os.path.join(ROOT, cfg["model"]["preprocessors_path"], "standard_scaler.pkl"))
    cont = list(scaler.feature_names_in_)
    fe = DiabetesFeatureEngineer(cfg)

    def prep(X_raw):
        d = X_raw.copy()
        d[cont] = scaler.transform(d[cont])
        d = fe.engineer_heuristic(d)
        d = fe.engineer_medical(d)
        return d

    return prep


def load_models(cfg):
    ens = cfg["model"]["ensemble"]
    base = {
        b["name"]: joblib.load(os.path.join(ROOT, b["weights_path"]))
        for b in ens["base_models"]
    }
    meta = joblib.load(os.path.join(ROOT, ens["meta_learner"]["weights_path"]))
    return base, meta


def align(df, model):
    return df[[str(c) for c in model.feature_names_in_]]


def stacking_proba(X_prepped, base, meta):
    cols = [base[n].predict_proba(align(X_prepped, base[n]))[:, 1] for n in base]
    return meta.predict_proba(np.column_stack(cols))[:, 1]


def confusion(y, p, t):
    """Returns (tn, fp, fn, tp) for p >= t."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tn, fp, fn, tp


def clinical_cost(y, p, t):
    tn, fp, fn, tp = confusion(y, p, t)
    return FN_COST * fn + FP_COST * fp
