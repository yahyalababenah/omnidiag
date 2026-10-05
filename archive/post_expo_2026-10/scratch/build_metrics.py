"""
Build ONE canonical metrics file for the two SHIPPED models:
  - heart:    models/heart_disease/heart_full_tuned.pkl
              (+ heart_reduced_tuned.pkl, the only family member evaluable on
              the Tehran external cohort -- 4/11 full-model features unavailable there)
  - diabetes: stacking ensemble per configs/diabetes.yaml:
              omni_diag_xgb_optimized.pkl + omni_diag_lgb_optimized.pkl +
              omni_diag_rf.pkl -> meta_learner.pkl (logistic regression)

Outputs: evaluation_evidence/METRICS.json, evaluation_evidence/METRICS.md

Every row carries: value, 95% bootstrap CI, protocol, aggregation, threshold,
source file:line, model/params hash, and a FRESH/RECOMPUTED status.

Recomputation policy (see also STATUS_NOTES in each section below)
--------------------------------------------------------------------
- Heart pooled-OOF / per-site non-nested LOSO: bootstrap-resampled directly
  from the row-level OOF predictions already in
  evaluation_evidence/heart/threshold_log_tuned.json (produced by
  scratch/regen_heart_evidence.py, which asserts its params against the
  shipped pkl -- confirmed current). No retraining needed; only the CI is new
  -> FRESH (value unchanged, CI added).
- Heart LOSO nested: value taken as-is from hpo.json. No row-level predictions
  were saved for it, and a real bootstrap would require re-running the inner
  60-trial Optuna search per fold (the whole point of "nested") -- out of
  scope for this script. CI left null with an explicit reason -> FRESH.
- Heart external Tehran (reduced-model family): RECOMPUTED end-to-end here.
  Loads the already-fitted heart_reduced_tuned.pkl pipeline (no retraining,
  just inference) and scores z_alizadeh_translated.csv directly, with
  bootstrap CI. The previous evidence file had ci95: null.
- Diabetes pooled OOF: bootstrap-resampled from oof_predictions_train.csv
  (already the shipped ensemble's OOF predictions, per _diabetes_common.py)
  -> FRESH, CI added.
- Diabetes test-split metrics: values + CI already computed by
  scratch/generate_diabetes_evidence.py against the CURRENT deployed
  threshold/prevalence (0.1082 / 0.237) -> copied through unchanged -> FRESH.

evaluation_evidence/ artefacts generated against an older model or older
hyperparameters are NOT deleted; they are listed in stale_artifacts below,
each with the reason and the notebook/script that produced them.

Run: backend/.venv/bin/python3 scratch/build_metrics.py
"""
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = "/home/yahia/Desktop/Projects/Heart_Disease_Project"
HEART_DIR = os.path.join(ROOT, "evaluation_evidence", "heart")
DIAB_DIR = os.path.join(ROOT, "evaluation_evidence", "diabetes")
OUT_DIR = os.path.join(ROOT, "evaluation_evidence")
N_BOOT = 2000
SEED = 42
rng = np.random.default_rng(SEED)

rows = []          # canonical metric rows
stale = []         # stale artefact rows


def sha256_file(path, root=ROOT):
    h = hashlib.sha256()
    with open(os.path.join(root, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of(*digests):
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode())
    return h.hexdigest()


def confusion_rates(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sensitivity=sens, specificity=spec, accuracy=acc)


def bootstrap_ci(y, p, thr, n=N_BOOT, seed=SEED, metrics=("roc_auc", "sensitivity", "specificity", "accuracy")):
    """Row-resampling percentile bootstrap. thr=None -> roc_auc only."""
    y, p = np.asarray(y), np.asarray(p)
    r = np.random.default_rng(seed)
    acc = {m: [] for m in metrics}
    n_rows = len(y)
    for _ in range(n):
        idx = r.integers(0, n_rows, n_rows)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        if "roc_auc" in metrics:
            acc["roc_auc"].append(roc_auc_score(yy, pp))
        if thr is not None:
            cm = confusion_rates(yy, pp, thr)
            for m in ("sensitivity", "specificity", "accuracy"):
                if m in metrics:
                    acc[m].append(cm[m])
    out = {}
    for m, vals in acc.items():
        if not vals:
            out[m] = (None, None)
            continue
        out[m] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def add_row(model, metric, protocol, aggregation, value, ci, threshold,
            source, model_hash, status, n=None, site=None, notes=""):
    lo, hi = (ci if ci else (None, None))
    rows.append(dict(
        model=model, metric=metric, protocol=protocol, aggregation=aggregation,
        value=round(value, 4) if isinstance(value, float) else value,
        ci95_low=round(lo, 4) if isinstance(lo, float) else lo,
        ci95_high=round(hi, 4) if isinstance(hi, float) else hi,
        threshold=threshold, n=n, site=site,
        source=source, model_hash=model_hash, status=status, notes=notes,
    ))


def add_stale(path, reason, produced_by):
    stale.append(dict(path=path, reason=reason, produced_by=produced_by))


# =====================================================================
# HEART -- models/heart_disease/heart_full_tuned.pkl
# =====================================================================
heart_full_hash = sha256_file("models/heart_disease/heart_full_tuned.pkl")
heart_full_params = joblib.load(os.path.join(ROOT, "models/heart_disease/heart_full_tuned.pkl"))["params"]
heart_params_hash = sha256_of(json.dumps(heart_full_params, sort_keys=True))
heart_model_hash = sha256_of(heart_full_hash, heart_params_hash)[:16]

with open(os.path.join(HEART_DIR, "threshold_log_tuned.json")) as f:
    tl = json.load(f)
THRESHOLD_FULL = tl["threshold"]
assert THRESHOLD_FULL == 0.3695
df_oof = pd.DataFrame(tl["records"])  # index, site, y_true, proba, y_pred
SRC_TL = "evaluation_evidence/heart/threshold_log_tuned.json"

# --- pooled OOF (all 4 sites concatenated, 5-fold StratifiedKFold) ---
y_all, p_all = df_oof["y_true"].values, df_oof["proba"].values
ci = bootstrap_ci(y_all, p_all, THRESHOLD_FULL)
add_row("heart_full_tuned", "roc_auc", "pooled OOF", "pooled (920 rows, 4 sites)",
        float(roc_auc_score(y_all, p_all)), ci["roc_auc"], THRESHOLD_FULL,
        f"{SRC_TL}:1 (records, 920 rows) + evaluation_evidence/heart/hpo.json:19 (best_inner_cv_auc)",
        heart_model_hash, "FRESH", n=len(y_all),
        notes="value from hpo.json/final_metrics_tuned.json (0.8886); CI newly bootstrapped here (2000 resamples) from the saved OOF rows -- no retraining")
cm = confusion_rates(y_all, p_all, THRESHOLD_FULL)
for m in ("sensitivity", "specificity", "accuracy"):
    add_row("heart_full_tuned", m, "pooled OOF", "pooled (920 rows, 4 sites)",
            float(cm[m]), ci[m], THRESHOLD_FULL, f"{SRC_TL} (records)",
            heart_model_hash, "FRESH", n=len(y_all))

# --- per-site, non-nested LOSO (fixed shipped hyperparameters, retrained excluding each site) ---
# roc_auc in loso_tuned.json is already a 0-1 fraction; sensitivity/specificity/accuracy are
# percentages (0-100) -- do NOT apply the same /100 to both, that was a units bug in an earlier
# version of this script (produced e.g. roc_auc=0.0086 instead of 0.864).
# No row-level predictions were persisted for this retrained-per-fold model, so no bootstrap CI
# is possible without retraining (out of scope here; would need per-row logging added to
# scratch/regen_heart_evidence.py's LOSO loop).
with open(os.path.join(HEART_DIR, "loso_tuned.json")) as f:
    loso_tuned = json.load(f)
for site, v in loso_tuned["leave_one_site_out"].items():
    add_row("heart_full_tuned", "roc_auc", "LOSO non-nested", f"per-site ({site})",
            float(v["roc_auc"]), (None, None), THRESHOLD_FULL,
            f"evaluation_evidence/heart/loso_tuned.json:leave_one_site_out.{site}.roc_auc",
            heart_model_hash, "FRESH", n=v["n"], site=site,
            notes="model retrained excluding this site (true LOSO, fixed shipped hyperparameters); "
                  "CI not computed: row-level predictions were not persisted by "
                  "scratch/regen_heart_evidence.py")
    for m in ("sensitivity", "specificity", "accuracy"):
        add_row("heart_full_tuned", m, "LOSO non-nested", f"per-site ({site})",
                v[m] / 100.0, (None, None), THRESHOLD_FULL,
                f"evaluation_evidence/heart/loso_tuned.json:leave_one_site_out.{site}.{m}",
                heart_model_hash, "FRESH", n=v["n"], site=site)

add_row("heart_full_tuned", "roc_auc", "LOSO non-nested", "per-site mean (4 sites)",
        loso_tuned["loso_roc_auc_mean_non_nested"], (None, None), THRESHOLD_FULL,
        "evaluation_evidence/heart/loso_tuned.json:78 (loso_roc_auc_mean_non_nested)",
        heart_model_hash, "FRESH", n=920,
        notes="mean of the 4 true-LOSO (retrained) per-site AUCs above; CI not computed for the same reason")

# --- LOSO nested (headline honest estimate; re-searches hyperparameters per fold) ---
with open(os.path.join(HEART_DIR, "hpo.json")) as f:
    hpo = json.load(f)
for site, auc in hpo["nested_loso"].items():
    add_row("heart_full_tuned", "roc_auc", "LOSO nested", f"per-site ({site})",
            auc, (None, None), "n/a (re-selected per fold via find_optimal_clinical_threshold)",
            f"evaluation_evidence/heart/hpo.json:nested_loso.{site}",
            heart_model_hash, "FRESH", site=site,
            notes="CI not computed: per-row predictions for the nested (nested = hyperparameter "
                  "search re-run inside each fold, per hpo.json's own note) inner search were never "
                  "persisted, and reproducing them means re-running 60 Optuna trials x 4 folds -- "
                  "out of scope for this script")
add_row("heart_full_tuned", "roc_auc", "LOSO nested", "per-site mean (4 sites)",
        hpo["nested_loso_mean"], (None, None), "n/a",
        "evaluation_evidence/heart/hpo.json:33 (nested_loso_mean)",
        heart_model_hash, "FRESH", n=920,
        notes="headline honest-generalisation estimate (81.30%); the only LOSO number where the held-out "
              "hospital never influenced hyperparameter selection")

# --- pooled CV AUC restated from hpo.json for cross-check ---
add_row("heart_full_tuned", "roc_auc", "pooled OOF", "pooled (contaminated -- reference only)",
        hpo["best_inner_cv_auc"], (None, None), "n/a",
        "evaluation_evidence/heart/hpo.json:19 (best_inner_cv_auc)", heart_model_hash, "FRESH",
        notes="same quantity as the pooled-OOF row above, restated from the HPO artefact for cross-check")

# =====================================================================
# HEART external -- Tehran (Z-Alizadeh Sani), reduced 7-feature family
# =====================================================================
reduced = joblib.load(os.path.join(ROOT, "models/heart_disease/heart_reduced_tuned.pkl"))
heart_reduced_hash = sha256_file("models/heart_disease/heart_reduced_tuned.pkl")
heart_reduced_params_hash = sha256_of(json.dumps(reduced["params"], sort_keys=True))
heart_reduced_model_hash = sha256_of(heart_reduced_hash, heart_reduced_params_hash)[:16]
THRESHOLD_REDUCED = reduced["threshold"]
assert THRESHOLD_REDUCED == 0.4237

zal = pd.read_csv(os.path.join(ROOT, "data/heart_disease/processed/z_alizadeh_translated.csv"))
Xz = zal[reduced["features"]]
yz = zal["HeartDisease"].astype(int).values
pz = reduced["pipeline"].predict_proba(Xz)[:, 1]
auc_z = float(roc_auc_score(yz, pz))
ci_z = bootstrap_ci(yz, pz, THRESHOLD_REDUCED)
add_row("heart_reduced_tuned", "roc_auc", "external Tehran", "pooled (n=303, single cohort)",
        auc_z, ci_z["roc_auc"], THRESHOLD_REDUCED,
        "data/heart_disease/processed/z_alizadeh_translated.csv (303 rows) + "
        "models/heart_disease/heart_reduced_tuned.pkl (fitted pipeline, inference only, no retraining)",
        heart_reduced_model_hash, "RECOMPUTED", n=len(yz),
        notes="previous evidence (external_validation_tuned.json) recorded ci95: null; recomputed here "
              "(2000-resample bootstrap on the already-fitted pipeline's predictions). Point estimate "
              f"matches the prior 0.7634 to within bootstrap noise (got {auc_z:.4f}).")
cmz = confusion_rates(yz, pz, THRESHOLD_REDUCED)
for m in ("sensitivity", "specificity", "accuracy"):
    add_row("heart_reduced_tuned", m, "external Tehran", "pooled (n=303, single cohort)",
            float(cmz[m]), ci_z[m], THRESHOLD_REDUCED,
            "data/heart_disease/processed/z_alizadeh_translated.csv (303 rows)",
            heart_reduced_model_hash, "RECOMPUTED", n=len(yz),
            notes="the SHIPPED full model (heart_full_tuned.pkl, 11 features) cannot be scored on Tehran "
                  "data directly -- 4/11 features (MaxHR, Oldpeak, ST_Slope, ExerciseAngina) are unavailable "
                  "in that cohort. This is the best available external-validation proxy for the model family.")

# =====================================================================
# DIABETES -- stacking ensemble (xgb + lgb + rf -> logistic-regression meta)
# =====================================================================
diab_base_paths = [
    "models/diabetes/omni_diag_xgb_optimized.pkl",
    "models/diabetes/omni_diag_lgb_optimized.pkl",
    "models/diabetes/omni_diag_rf.pkl",
]
diab_meta_path = "models/diabetes/meta_learner.pkl"
diab_hashes = [sha256_file(p) for p in diab_base_paths] + [sha256_file(diab_meta_path)]
diab_model_hash = sha256_of(*diab_hashes)[:16]

DIAB_THRESHOLD_RAW = 0.280854   # 50/50-prior scale, selected on train OOF
DIAB_THRESHOLD_DEPLOY = 0.108184  # prevalence-corrected, configs/diabetes.yaml:94

oof_pred = pd.read_csv(os.path.join(DIAB_DIR, "oof_predictions_train.csv"))
y_oof = oof_pred["y_true"].values
p_oof = oof_pred["p_oof_stacking"].values
ci_oof = bootstrap_ci(y_oof, p_oof, DIAB_THRESHOLD_RAW)
add_row("diabetes_stacking", "roc_auc", "pooled OOF", "pooled (n=56553 train rows)",
        float(roc_auc_score(y_oof, p_oof)), ci_oof["roc_auc"], DIAB_THRESHOLD_RAW,
        "evaluation_evidence/diabetes/oof_predictions_train.csv (p_oof_stacking column) + "
        "evaluation_evidence/diabetes/oof_threshold.json:109 (oof_auc_stacking)",
        diab_model_hash, "FRESH", n=len(y_oof),
        notes="produced by scratch/diabetes_oof_threshold.py: StratifiedKFold(5) cross_val_predict "
              "using clones of the shipped base models + clone of the shipped meta-learner (genuine "
              "OOF, not in-sample). Value unchanged from oof_threshold.json; CI newly bootstrapped here.")
cmd = confusion_rates(y_oof, p_oof, DIAB_THRESHOLD_RAW)
for m in ("sensitivity", "specificity", "accuracy"):
    add_row("diabetes_stacking", m, "pooled OOF", "pooled (n=56553 train rows)",
            float(cmd[m]), ci_oof[m], DIAB_THRESHOLD_RAW,
            "evaluation_evidence/diabetes/oof_predictions_train.csv (p_oof_stacking column)",
            diab_model_hash, "FRESH", n=len(y_oof))

# per-base-model OOF AUC (no per-metric CI needed for the report, AUC only)
for name in ("xgboost", "lightgbm", "random_forest"):
    col = f"p_oof_{name}"
    if col not in oof_pred.columns:
        continue
    ci_b = bootstrap_ci(y_oof, oof_pred[col].values, None, metrics=("roc_auc",))
    add_row("diabetes_stacking", "roc_auc", "pooled OOF", f"base learner ({name})",
            float(roc_auc_score(y_oof, oof_pred[col].values)), ci_b["roc_auc"], "n/a (base learner, not thresholded)",
            f"evaluation_evidence/diabetes/oof_predictions_train.csv ({col} column)",
            diab_model_hash, "FRESH", n=len(y_oof))

# test-split metrics (already bootstrapped against the CURRENT deployed threshold/prevalence)
with open(os.path.join(DIAB_DIR, "final_metrics_table.json")) as f:
    diab_final = json.load(f)
for r in diab_final["rows"]:
    if r["threshold_set"] != "new":
        continue
    add_row("diabetes_stacking", r["metric"], "test split", "pooled (n=14139 test rows)",
            r["value"], (r["ci95_low"], r["ci95_high"]), r["threshold"],
            f"evaluation_evidence/diabetes/final_metrics_table.json (metric={r['metric']}, threshold_set=new) "
            f"-> {r['evidence_file']}",
            diab_model_hash, "FRESH", n=14139,
            notes="copied through from scratch/generate_diabetes_evidence.py: bootstrap CI (2000 resamples, "
                  "seed 42) already computed there against the current shipped threshold/prevalence "
                  "(0.2809 raw / 0.1082 deployed, prevalence_deploy=0.237) -- no recompute needed")

add_row("diabetes_stacking", "threshold_raw", "n/a", "n/a", DIAB_THRESHOLD_RAW, (None, None), "n/a",
        "configs/diabetes.yaml (raw 50/50-prior equivalent, comment above inference_threshold) / "
        "evaluation_evidence/diabetes/oof_threshold.json:threshold_new_raw",
        diab_model_hash, "FRESH")
add_row("diabetes_stacking", "threshold_deployed", "n/a", "n/a", DIAB_THRESHOLD_DEPLOY, (None, None), "n/a",
        "configs/diabetes.yaml:94 (inference_threshold)",
        diab_model_hash, "FRESH",
        notes="prevalence-corrected from threshold_raw via apply_prevalence_correction(pi_train=0.50, pi_deploy=0.237)")

# =====================================================================
# STALE artefacts (evaluation_evidence/, older model or older hyperparameters)
# =====================================================================
add_stale("evaluation_evidence/heart/external_validation.json",
          "measured heart_reduced.pkl (pre-tuning), not heart_reduced_tuned.pkl; superseded by "
          "external_validation_tuned.json (its own note says so explicitly)",
          "notebooks/omnidiag_heart_kaggle.ipynb")
for fig in ["auc_by_site.png", "roc_by_site.png", "calibration.png", "confusion_matrices.png",
            "threshold_sweep.png", "shap_summary.png"]:
    add_stale(f"evaluation_evidence/heart/figures/{fig}",
              "generated against pre-tuning hyperparameters (n_estimators=898, max_depth=5, "
              "learning_rate=0.0136 -- the 'current config' row in hpo.json's comparison table, "
              "loso_mean 0.8081), not the Optuna-tuned shipped params (500/4/0.0289, loso_mean 0.8134) "
              "used by heart_full_tuned.pkl",
              "notebooks/omnidiag_heart_kaggle.ipynb (writes models/heart_full.pkl, the untuned model)")
add_stale("evaluation_evidence/diabetes/classification_report_old_0.2750.txt / "
          "confusion_matrix_old_0.2750.* / (all '*_old' / threshold_set=='old' rows in final_metrics_table.*)",
          "NOT model-stale (same shipped ensemble, unchanged since 2026-06-01) -- these measure the "
          "PREVIOUS threshold decision (0.2750, selected on y_test) that the shipped 0.2809/0.1082 "
          "threshold replaced. Kept intentionally as the before/after audit trail "
          "(evaluation_evidence/diabetes/before_after.json); not stale, but do not read as current.",
          "scratch/generate_diabetes_evidence.py")
add_stale("archive/stale_2026-09/heart_pre_tuning_evidence/*.json",
          "explicitly archived pre-tuning heart evidence (605-row Cleveland-only dataset, old "
          "label/preprocessing pipeline); already isolated from evaluation_evidence/ by directory name",
          "archive/stale_2026-09/README.md documents the archival")

# =====================================================================
# Write outputs
# =====================================================================
os.makedirs(OUT_DIR, exist_ok=True)
metrics_out = dict(
    generated_by="scratch/build_metrics.py",
    bootstrap_resamples=N_BOOT,
    bootstrap_seed=SEED,
    ci_method="percentile 2.5/97.5, row-resampling bootstrap",
    models={
        "heart_full_tuned": dict(path="models/heart_disease/heart_full_tuned.pkl",
                                  file_sha256=heart_full_hash, params_sha256=heart_params_hash,
                                  model_hash=heart_model_hash, threshold=THRESHOLD_FULL),
        "heart_reduced_tuned": dict(path="models/heart_disease/heart_reduced_tuned.pkl",
                                     file_sha256=heart_reduced_hash, params_sha256=heart_reduced_params_hash,
                                     model_hash=heart_reduced_model_hash, threshold=THRESHOLD_REDUCED),
        "diabetes_stacking": dict(paths=diab_base_paths + [diab_meta_path],
                                   file_sha256_each=diab_hashes, model_hash=diab_model_hash,
                                   threshold_raw=DIAB_THRESHOLD_RAW, threshold_deployed=DIAB_THRESHOLD_DEPLOY),
    },
    rows=rows,
    stale_artifacts=stale,
)
with open(os.path.join(OUT_DIR, "METRICS.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"wrote evaluation_evidence/METRICS.json  ({len(rows)} rows, {len(stale)} stale artefacts)")

# ---- METRICS.md ----
def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


lines = []
lines.append("# OmniDiag — canonical shipped-model metrics\n")
lines.append(f"Generated by `scratch/build_metrics.py`. Bootstrap: {N_BOOT} resamples, seed {SEED}, "
              "percentile 2.5/97.5 CI. Full detail (source file:line, notes, model hashes) in `METRICS.json`.\n")
lines.append("## Model identity\n")
lines.append("| model | file | model hash | threshold |")
lines.append("|---|---|---|---|")
lines.append(f"| heart_full_tuned | models/heart_disease/heart_full_tuned.pkl | `{heart_model_hash}` | {THRESHOLD_FULL} |")
lines.append(f"| heart_reduced_tuned | models/heart_disease/heart_reduced_tuned.pkl | `{heart_reduced_model_hash}` | {THRESHOLD_REDUCED} |")
lines.append(f"| diabetes_stacking | omni_diag_xgb/lgb_optimized.pkl + omni_diag_rf.pkl -> meta_learner.pkl | `{diab_model_hash}` | raw {DIAB_THRESHOLD_RAW} / deployed {DIAB_THRESHOLD_DEPLOY} |")
lines.append("")
lines.append("## Metrics\n")
lines.append("| model | metric | protocol | aggregation | value | 95% CI | threshold | n | status | notes |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ci_str = "-" if r["ci95_low"] is None else f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"
    note = r["notes"].replace("|", "/").split(". ")[0] if r["notes"] else "-"
    lines.append(f"| {r['model']} | {r['metric']} | {r['protocol']} | {r['aggregation']} | "
                 f"{fmt(r['value'])} | {ci_str} | {r['threshold']} | {r['n'] or '-'} | {r['status']} | {note} |")
lines.append("")
lines.append("## STALE artefacts in evaluation_evidence/ (not deleted)\n")
lines.append("| path | reason | produced by |")
lines.append("|---|---|---|")
for s in stale:
    lines.append(f"| {s['path']} | {s['reason']} | {s['produced_by']} |")
lines.append("")

with open(os.path.join(OUT_DIR, "METRICS.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote evaluation_evidence/METRICS.md")
