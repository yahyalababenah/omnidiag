"""AUDIT 4 — standalone heart-disease evaluation. Read-only; touches no app code."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
ROOT = "/home/yahia/Desktop/Projects/Heart_Disease_Project"
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from models.advanced_feature_engineering import (engineer_heuristic_features,
                                                 engineer_medical_features)

OUT = os.path.join(ROOT, "evaluation_evidence")
os.makedirs(OUT, exist_ok=True)
lines = []
def P(s=""):
    print(s); lines.append(str(s))

# --- Rebuild the EXACT training split (train_v5_xgb.py L95-142) ---
df = pd.read_csv(os.path.join(ROOT, "data/heart_disease/processed/final_ready_data.csv"))
P(f"Dataset: data/heart_disease/processed/final_ready_data.csv  shape={df.shape}")
dfe = engineer_medical_features(engineer_heuristic_features(df))
X = dfe.drop(columns=["HeartDisease"]); y = dfe["HeartDisease"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
P(f"Split: train_test_split(test_size=0.2, random_state=42, stratify=y) -> train={len(Xtr)} test={len(Xte)}")

model = joblib.load(os.path.join(ROOT, "models/heart_disease/omni_diag_xgb_optimized.pkl"))
Xte = Xte[list(model.feature_names_in_)]
P(f"Model: models/heart_disease/omni_diag_xgb_optimized.pkl ({type(model).__name__}, {model.n_features_in_} features)")
P(f"Feature order: {list(model.feature_names_in_)}")
P("")

proba1 = model.predict_proba(Xte)[:, 1]   # P(raw class 1)

def report(tag, y_true, y_score, thr, pos_label_desc):
    P("="*72); P(f"{tag}   (threshold = {thr})")
    P(f"positive class = {pos_label_desc}")
    yp = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, yp); auc = roc_auc_score(y_true, y_score)
    sens = tp/(tp+fn) if tp+fn else float("nan")
    spec = tn/(tn+fp) if tn+fp else float("nan")
    P(f"confusion_matrix [[tn fp][fn tp]] = [[{tn} {fp}][{fn} {tp}]]")
    P(f"accuracy    = {acc*100:.2f}%")
    P(f"roc_auc     = {auc:.4f}")
    P(f"sensitivity = {sens*100:.2f}%")
    P(f"specificity = {spec*100:.2f}%")
    P(classification_report(y_true, yp, target_names=["Neg","Pos"], digits=4))
    return dict(tag=tag, thr=thr, acc=acc, auc=auc, sens=sens, spec=spec, cm=cm)

res = {}
# Orientation A: raw CSV label as-is (what training/metrics.json measured)
res["raw_0.5"]  = report("A. RAW CSV LABEL, argmax (what metrics.json recorded)", yte, proba1, 0.5, "raw HeartDisease==1")
res["raw_0.42"] = report("B. RAW CSV LABEL, claimed threshold 0.420", yte, proba1, 0.42, "raw HeartDisease==1")
# Orientation B: clinical orientation used by the API (_LABELS_INVERTED = True)
y_clin = 1 - yte.values
proba_dis = model.predict_proba(Xte)[:, 0]   # ModelLoader.predict() confidence
res["clin_0.5"]  = report("C. CLINICAL ORIENTATION (API _LABELS_INVERTED), argmax 0.5 == PRODUCTION", y_clin, proba_dis, 0.5, "has disease")
res["clin_0.42"] = report("D. CLINICAL ORIENTATION, claimed threshold 0.420", y_clin, proba_dis, 0.42, "has disease")

# Confusion matrix PNGs for the two that matter
for key, fname, title in [("raw_0.5","heart_disease_confusion_matrix_raw.png","Heart — raw label, thr 0.5 (metrics.json basis)"),
                          ("clin_0.5","heart_disease_confusion_matrix.png","Heart — production (clinical orientation, argmax)")]:
    fig, ax = plt.subplots(figsize=(4.5,4))
    ConfusionMatrixDisplay(res[key]["cm"], display_labels=["Neg","Pos"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140); plt.close(fig)

CLAIM = dict(acc=0.8017, auc=0.856, sens=0.8046, spec=0.8313)
P("="*72); P("COMPARISON vs CLAIMED METRICS (acc 80.17%, AUC 0.856, sens 80.46%, spec 83.13%, thr 0.420)")
P(f"{'variant':<12}{'acc':>9}{'Δacc pp':>10}{'auc':>9}{'sens':>9}{'Δsens pp':>10}{'spec':>9}{'Δspec pp':>10}")
for k, r in res.items():
    P(f"{k:<12}{r['acc']*100:>8.2f}%{(r['acc']-CLAIM['acc'])*100:>10.2f}{r['auc']:>9.4f}"
      f"{r['sens']*100:>8.2f}%{(r['sens']-CLAIM['sens'])*100:>10.2f}{r['spec']*100:>8.2f}%{(r['spec']-CLAIM['spec'])*100:>10.2f}")
P("")
P("NOTE: ROC-AUC is threshold-independent; it is identical for A/B and for C/D")
P("      (AUC under the inverted orientation is the mirror of the raw one).")

with open(os.path.join(OUT, "heart_disease_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote evaluation_evidence/heart_disease_report.txt")
