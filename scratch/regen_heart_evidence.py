"""
Regenerate evaluation_evidence/heart/ pooled-CV metrics, per-site LOSO
confusion matrices, and threshold_log at the SHIPPED operating point
(threshold=0.3695), using the SAME pipeline/hyperparameters as
models/heart_disease/heart_full_tuned.pkl.

Source of truth for hyperparameters/pipeline shape:
  notebooks/omnidiag_hpo_thresholds_kaggle.ipynb (cells 1, 5, 6, 11, 15)
  models/heart_disease/heart_full_tuned.pkl (params field)
  evaluation_evidence/heart/hpo.json (best_params)

Data: data/heart_disease/processed/uci_heart_by_site.csv (920 rows, 4 sites)
Run with: backend/.venv/bin/python3 (pinned validated stack, matches
requirements.txt: pandas==3.0.3 numpy==2.4.6 scikit-learn==1.9.0 xgboost==3.3.0)
"""
import json, os
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = "/home/yahia/Desktop/Projects/Heart_Disease_Project"
OUT = os.path.join(ROOT, "evaluation_evidence/heart")
SEED = 42
THRESHOLD = 0.3695  # shipped operating point, from heart_full_tuned.pkl["threshold"]

NUM = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]
CAT = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FULL, TGT = NUM + CAT, "HeartDisease"

# Best params, verbatim from evaluation_evidence/heart/hpo.json:best_params
# (cross-checked against heart_full_tuned.pkl["params"] -- identical)
PROD = dict(
    n_estimators=500, max_depth=4, learning_rate=0.028928994165742683,
    subsample=0.8077672166682721, colsample_bytree=0.44881626783908657,
    min_child_weight=4, gamma=3.2247939623734005,
    reg_alpha=4.970097824015917e-07, reg_lambda=0.05334501049618681,
    scale_pos_weight=1.565386715651742,
    random_state=SEED, eval_metric="logloss",
)

shipped = joblib.load(os.path.join(ROOT, "models/heart_disease/heart_full_tuned.pkl"))
assert shipped["threshold"] == THRESHOLD, shipped["threshold"]
assert shipped["params"] == PROD, "PROD params mismatch vs shipped pkl -- STOP"
print("Confirmed: PROD params == heart_full_tuned.pkl['params']; threshold == 0.3695")


def make_pipe(params, num=NUM, cat=CAT):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", IterativeImputer(random_state=SEED)),
                               ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                       unknown_value=-1))]), cat),
        ])),
        ("clf", xgb.XGBClassifier(**params)),
    ])


df = pd.read_csv(os.path.join(ROOT, "data/heart_disease/processed/uci_heart_by_site.csv"))
X, y, site = df[FULL], df[TGT], df["site"]
print(f"Dataset: uci_heart_by_site.csv shape={df.shape}, sites={sorted(site.unique())}")


def score_at(y_true, y_score, thr):
    yp = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = accuracy_score(y_true, yp)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sensitivity=round(sens * 100, 2), specificity=round(spec * 100, 2),
                accuracy=round(acc * 100, 2), roc_auc=round(float(auc), 4) if auc == auc else None,
                n=int(len(y_true)))


# ---- 1. Pooled CV at threshold 0.3695 (5-fold StratifiedKFold, same cv object as notebook) ----
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = cross_val_predict(make_pipe(PROD), X, y, cv=cv, method="predict_proba")[:, 1]
pooled = score_at(y.values, oof, THRESHOLD)
pooled_auc = round(roc_auc_score(y, oof), 4)
print(f"pooled CV roc_auc={pooled_auc} (hpo.json best_inner_cv_auc / comparison 'Optuna best' pooled_cv = 0.8886)")
print("pooled @0.3695:", pooled)

# ---- 2. Per-site LOSO at threshold 0.3695, fixed PROD hyperparameters (non-nested: same    ----
#         hyperparameters actually shipped; NOT re-searched per fold like hpo.json's
#         nested_loso, which is the headline honest-generalisation number, 0.813 mean)
loso = {}
for s in sorted(site.unique()):
    tr, te = (site != s).values, (site == s).values
    m = make_pipe(PROD).fit(X[tr], y[tr])
    pr = m.predict_proba(X[te])[:, 1]
    yt = y[te].values
    res = score_at(yt, pr, THRESHOLD)
    res["prevalence"] = round(float(yt.mean()) * 100, 1)
    res["controls"] = int((yt == 0).sum())
    res["cases"] = int((yt == 1).sum())
    loso[s] = res
    print(s, res)

loso_auc_mean = round(float(np.mean([v["roc_auc"] for v in loso.values()])), 4)
print(f"non-nested LOSO auc mean (fixed prod hyperparams) = {loso_auc_mean}")
print("compare to hpo.json nested_loso_mean (re-searched per fold, headline) = 0.813")

out = dict(
    generated_by="scratch/regen_heart_evidence.py, 2026-09-22",
    model="models/heart_disease/heart_full_tuned.pkl (shipped)",
    threshold=THRESHOLD,
    threshold_source="heart_full_tuned.pkl['threshold'] / hpo_threshold_summary.json shipped",
    hyperparameters_source="evaluation_evidence/heart/hpo.json:best_params (verified == heart_full_tuned.pkl['params'])",
    data_source="data/heart_disease/processed/uci_heart_by_site.csv (920 rows, 4 UCI sites)",
    pooled_cv=dict(cv="StratifiedKFold(5, shuffle=True, random_state=42)", **pooled),
    leave_one_site_out=loso,
    loso_roc_auc_mean_non_nested=loso_auc_mean,
    nested_loso_roc_auc_mean_reference="0.813 (evaluation_evidence/heart/hpo.json:nested_loso_mean; "
                                        "hyperparameter search re-run inside each fold -- this script's "
                                        "LOSO instead reuses the shipped/fixed hyperparameters, which is "
                                        "what actually happens in production per held-out site)",
)
with open(os.path.join(OUT, "loso_tuned.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote evaluation_evidence/heart/loso_tuned.json")

final_metrics = dict(
    generated_by="scratch/regen_heart_evidence.py, 2026-09-22 (see loso_tuned.json for full detail)",
    model="models/heart_disease/heart_full_tuned.pkl",
    threshold=THRESHOLD,
    training_data=dict(source="UCI Heart Disease, 4 original site files",
                        n=920,
                        sites={s: int((site == s).sum()) for s in sorted(site.unique())},
                        note="ca and thal excluded: invasive, absent from production"),
    external_data=dict(source="Z-Alizadeh Sani (Tehran)", n=303,
                        used_for="validation only, never training",
                        result_file="evaluation_evidence/heart/external_validation_tuned.json"),
    headline=dict(
        internal_pooled_cv_roc_auc=pooled_auc,
        internal_pooled_cv_at_threshold=pooled,
        nested_loso_mean_roc_auc=0.813,
        nested_loso_source="evaluation_evidence/heart/hpo.json:nested_loso_mean",
        non_nested_loso_mean_roc_auc_at_fixed_hparams=loso_auc_mean,
        external_tehran_roc_auc=76.34,
        external_tehran_source="evaluation_evidence/heart/external_validation_tuned.json",
        external_tehran_note="measured on the 7-feature reduced model; full 11-feature model "
                              "cannot be evaluated on Tehran data (4 features unavailable)",
    ),
    operating_point=dict(threshold=THRESHOLD,
                          selected_on="out-of-fold predictions of training data",
                          cost_function="2.0*FN + 1.0*FP",
                          pooled_cv=pooled),
    evidence_files=[
        "evaluation_evidence/heart/hpo.json",
        "evaluation_evidence/heart/loso_tuned.json",
        "evaluation_evidence/heart/threshold_log_tuned.json",
        "evaluation_evidence/heart/external_validation_tuned.json",
        "evaluation_evidence/heart/leakage_comparison.json",
    ],
)
with open(os.path.join(OUT, "final_metrics_tuned.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)
print("wrote evaluation_evidence/heart/final_metrics_tuned.json")

# ---- 3. threshold_log at 0.3695 -- same format as the stale threshold_log.json:  ----
#         per-example oof probability + prediction, pooled (all folds concatenated)
threshold_log = []
for i, (idx, row) in enumerate(df.iterrows()):
    p = float(oof[i])
    threshold_log.append(dict(
        index=int(idx), site=row["site"], y_true=int(row[TGT]),
        proba=round(p, 6), threshold=THRESHOLD,
        y_pred=int(p >= THRESHOLD),
    ))
with open(os.path.join(OUT, "threshold_log_tuned.json"), "w") as f:
    json.dump(dict(threshold=THRESHOLD,
                    note="Pooled 5-fold StratifiedKFold(random_state=42) out-of-fold probabilities, "
                         "shipped hyperparameters (hpo.json:best_params), evaluated at the shipped "
                         "threshold 0.3695. Same format/derivation method as the archived "
                         "threshold_log.json, but for the tuned model at the tuned threshold.",
                    records=threshold_log), f, indent=2)
print(f"wrote evaluation_evidence/heart/threshold_log_tuned.json ({len(threshold_log)} records)")
