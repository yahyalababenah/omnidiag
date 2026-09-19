"""
Re-derive the diabetes clinical threshold on OUT-OF-FOLD training predictions.

Why fold clones are fitted
--------------------------
The shipped base models were fitted on the whole training split, so their
predictions on training rows are in-sample and useless for threshold
selection. Genuine OOF predictions need, for each fold, a model that never saw
that fold. So each shipped model is sklearn.clone()-d (identical
hyper-parameters) and fitted on the other 4 folds, in memory only.
Nothing is saved to models/, and the shipped .pkl files are only read.

Procedure (mirrors models/train_diabetes_ensemble.py after the fix)
  1. StratifiedKFold(5, shuffle=True, random_state=42) on the training split
  2. base-model OOF = cross_val_predict(clone(shipped_base), ...)
  3. stacking OOF   = cross_val_predict(clone(shipped_meta), base_OOF, ...)
  4. threshold      = find_optimal_clinical_threshold(y_train, stacking_OOF)
     — the unchanged function imported from the training script.

Outputs (evaluation_evidence/diabetes/)
  oof_threshold.json          old vs new threshold, costs, both prior scales
  threshold_scan_oof.csv      full 200-point scan on the OOF predictions
  oof_predictions_train.csv   per-row OOF probabilities (for re-use / audit)

Run:  backend/.venv/bin/python scratch/diabetes_oof_threshold.py
"""
import json
import os
import time

from _diabetes_common import (EVIDENCE_DIR, OLD_THRESHOLD, ROOT, SEED,
                              align, clinical_cost, confusion, load_config,
                              load_models, load_split, make_preparer,
                              stacking_proba)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from backend.prevalence_correction import apply_prevalence_correction
from models.train_diabetes_ensemble import find_optimal_clinical_threshold

N_FOLDS = 5
N_JOBS_PER_MODEL = 2           # RAM on this host is tight; does not change results


def _limit_threads(est):
    if "n_jobs" in est.get_params():
        est.set_params(n_jobs=N_JOBS_PER_MODEL)
    return est


def main():
    t0 = time.time()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    cfg = load_config()
    X_tr, X_te, y_tr, y_te = load_split(cfg)
    prep = make_preparer(cfg)
    Xtr = prep(X_tr)
    base, meta = load_models(cfg)
    print(f"train rows {len(Xtr)}  test rows {len(X_te)}  train prevalence {y_tr.mean():.4f}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_base = {}
    for name, model in base.items():
        t = time.time()
        est = _limit_threads(clone(model))
        oof_base[name] = cross_val_predict(
            est, align(Xtr, model), y_tr, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        print(f"  OOF {name:<14} AUC {roc_auc_score(y_tr, oof_base[name]):.4f}   ({time.time()-t:.0f}s)")

    X_meta = np.column_stack([oof_base[n] for n in base])
    oof_stack = cross_val_predict(clone(meta), X_meta, y_tr, cv=cv, method="predict_proba")[:, 1]
    # Secondary view: shipped meta-learner applied to the OOF base matrix.
    # Its 4 parameters were fitted on these same labels, so this is reported
    # only as a sensitivity check, not used for the decision.
    oof_stack_shipped_meta = meta.predict_proba(X_meta)[:, 1]
    print(f"  OOF stacking   AUC {roc_auc_score(y_tr, oof_stack):.4f}")

    res = find_optimal_clinical_threshold(y_true=y_tr, y_proba=oof_stack, fn_penalty_multiplier=2.0)
    t_new = float(res["optimal_threshold"])
    res_alt = find_optimal_clinical_threshold(y_true=y_tr, y_proba=oof_stack_shipped_meta,
                                              fn_penalty_multiplier=2.0)

    # Old procedure, reproduced: threshold selected on the test labels.
    p_te = stacking_proba(prep(X_te), base, meta)
    res_test = find_optimal_clinical_threshold(y_true=y_te, y_proba=p_te, fn_penalty_multiplier=2.0)

    pi_tr = float(cfg["model"]["prevalence_train"])
    pi_dep = float(cfg["model"]["prevalence_deploy"])

    def on_test(t):
        tn, fp, fn, tp = confusion(y_te, p_te, t)
        return dict(threshold_raw=t, tn=tn, fp=fp, fn=fn, tp=tp,
                    sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                    cost=clinical_cost(y_te, p_te, t))

    old, new = on_test(OLD_THRESHOLD), on_test(t_new)
    out = {
        "procedure": "cross_val_predict, StratifiedKFold(5, shuffle=True, random_state=42), "
                     "clones of shipped base models + clone of shipped meta-learner; "
                     "find_optimal_clinical_threshold unchanged (Cost = 2*FN + 1*FP, "
                     "np.linspace(0.01, 0.99, 200))",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "oof_auc_stacking": float(roc_auc_score(y_tr, oof_stack)),
        "oof_auc_base": {n: float(roc_auc_score(y_tr, v)) for n, v in oof_base.items()},
        "threshold_old_raw": OLD_THRESHOLD,
        "threshold_old_reproduced_on_y_test": float(res_test["optimal_threshold"]),
        "threshold_new_raw": t_new,
        "threshold_new_raw_sensitivity_check_shipped_meta": float(res_alt["optimal_threshold"]),
        "oof_min_cost": float(res["minimum_cost"]),
        "prevalence_train": pi_tr,
        "prevalence_deploy": pi_dep,
        "threshold_old_corrected": float(apply_prevalence_correction(OLD_THRESHOLD, pi_tr, pi_dep)),
        "threshold_new_corrected": float(apply_prevalence_correction(t_new, pi_tr, pi_dep)),
        "test_at_old": old,
        "test_at_new": new,
        "test_cost_change_pct": 100.0 * (new["cost"] - old["cost"]) / old["cost"],
        "test_sensitivity_change_pp": 100.0 * (new["sensitivity"] - old["sensitivity"]),
        "test_specificity_change_pp": 100.0 * (new["specificity"] - old["specificity"]),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(EVIDENCE_DIR, "oof_threshold.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(res["threshold_scan"]).to_csv(
        os.path.join(EVIDENCE_DIR, "threshold_scan_oof.csv"), index=False)
    pd.DataFrame({"y_true": y_tr.values, "p_oof_stacking": oof_stack,
                  **{f"p_oof_{n}": v for n, v in oof_base.items()}}).to_csv(
        os.path.join(EVIDENCE_DIR, "oof_predictions_train.csv"), index=False)

    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
    print("test @ old:", old)
    print("test @ new:", new)


if __name__ == "__main__":
    main()
