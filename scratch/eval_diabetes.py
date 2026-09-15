"""AUDIT 4 — standalone diabetes stacking-ensemble evaluation. Read-only."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
ROOT = "/home/yahia/Desktop/Projects/Heart_Disease_Project"
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from features.diabetes_features import DiabetesFeatureEngineer

OUT = os.path.join(ROOT, "evaluation_evidence"); os.makedirs(OUT, exist_ok=True)
lines = []
def P(s=""):
    print(s); lines.append(str(s))

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/diabetes.yaml")))
THR = float(cfg["model"]["inference_threshold"])
CSV = os.path.join(ROOT, "data/diabetes/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
CONT = ["BMI","MentHlth","PhysHlth","GenHlth","Age","Education","Income"]

P(f"Dataset: data/diabetes/raw/...BRFSS2015.csv")
df = pd.read_csv(CSV); P(f"shape={df.shape}")
y = df["Diabetes_binary"]; X = df.drop(columns=["Diabetes_binary"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
P(f"Split (reproducing preprocess_diabetes.run_diabetes_preprocessing): "
  f"test_size=0.2, random_state=42, stratify=y -> train={len(Xtr)} test={len(Xte)}")

# Reproduce training-time scaler (fit on TRAIN only) — the leak-free path
sc_fresh = StandardScaler().fit(Xtr[CONT])
# Production scaler shipped on disk
sc_prod = joblib.load(os.path.join(ROOT, "models/diabetes/preprocessors/standard_scaler.pkl"))
P("")
P("Scaler check — shipped standard_scaler.pkl vs one refit on this train split:")
P(f"  max |mean_ delta|  = {np.abs(np.asarray(sc_prod.mean_) - sc_fresh.mean_).max():.3e}")
P(f"  max |scale_ delta| = {np.abs(np.asarray(sc_prod.scale_) - sc_fresh.scale_).max():.3e}")
P("  -> shipped scaler IS the train-fitted one (no leakage) if deltas ~0")

fe = DiabetesFeatureEngineer(cfg)
def prep(Xraw, scaler):
    d = Xraw.copy()
    d[CONT] = scaler.transform(d[CONT])
    d = fe.engineer_heuristic(d)
    d = fe.engineer_medical(d)
    return d

Xte_p = prep(Xte, sc_prod)

base = {}
for b in cfg["model"]["ensemble"]["base_models"]:
    m = joblib.load(os.path.join(ROOT, b["weights_path"]))
    base[b["name"]] = m
    P(f"Loaded base model '{b['name']}': {b['weights_path']} ({type(m).__name__})")
meta = joblib.load(os.path.join(ROOT, cfg["model"]["ensemble"]["meta_learner"]["weights_path"]))
P(f"Loaded meta-learner: {cfg['model']['ensemble']['meta_learner']['weights_path']} "
  f"({type(meta).__name__}) coef_={meta.coef_[0].round(4).tolist()} intercept_={meta.intercept_.round(4).tolist()}")
P("")

probas = {}
for name, m in base.items():
    aligned = Xte_p[[str(c) for c in m.feature_names_in_]]
    probas[name] = m.predict_proba(aligned)[:, 1]
    P(f"  base '{name}' AUC = {roc_auc_score(yte, probas[name]):.4f}")

Xmeta = np.column_stack([probas[n] for n in base.keys()])
p_stack = meta.predict_proba(Xmeta)[:, 1]
p_vote = Xmeta.mean(axis=1)

def report(tag, y_true, score, thr):
    P("="*72); P(f"{tag}   (threshold = {thr})")
    yp = (score >= thr).astype(int)
    cm = confusion_matrix(y_true, yp); tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, yp); auc = roc_auc_score(y_true, score)
    sens = tp/(tp+fn); spec = tn/(tn+fp)
    P(f"confusion_matrix [[tn fp][fn tp]] = [[{tn} {fp}][{fn} {tp}]]")
    P(f"accuracy    = {acc*100:.2f}%")
    P(f"roc_auc     = {auc:.4f}")
    P(f"sensitivity = {sens*100:.2f}%")
    P(f"specificity = {spec*100:.2f}%")
    P(classification_report(y_true, yp, target_names=["Neg","Pos"], digits=4))
    return dict(acc=acc, auc=auc, sens=sens, spec=spec, cm=cm)

r_stack = report("STACKING ENSEMBLE @ production threshold (configs/diabetes.yaml)", yte, p_stack, THR)
r_s05   = report("STACKING ENSEMBLE @ 0.5 (for reference)", yte, p_stack, 0.5)
r_vote  = report("VOTING ENSEMBLE (fallback path) @ production threshold", yte, p_vote, THR)

fig, ax = plt.subplots(figsize=(4.5,4))
ConfusionMatrixDisplay(r_stack["cm"], display_labels=["Neg","Pos"]).plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title(f"Diabetes — stacking @ thr {THR}", fontsize=9); fig.tight_layout()
fig.savefig(os.path.join(OUT, "diabetes_confusion_matrix.png"), dpi=140); plt.close(fig)

CLAIM = dict(acc=0.7518, auc=0.831, sens=0.9170, spec=0.5463)
P("="*72)
P("COMPARISON vs CLAIMED (acc 75.18%, AUC 0.831, sens 91.70%, spec 54.63%, thr 0.275)")
P(f"{'variant':<22}{'acc':>9}{'Δacc pp':>10}{'auc':>9}{'sens':>9}{'Δsens pp':>10}{'spec':>9}{'Δspec pp':>10}")
for k, r in [("stacking@0.275", r_stack), ("stacking@0.5", r_s05), ("voting@0.275", r_vote)]:
    P(f"{k:<22}{r['acc']*100:>8.2f}%{(r['acc']-CLAIM['acc'])*100:>10.2f}{r['auc']:>9.4f}"
      f"{r['sens']*100:>8.2f}%{(r['sens']-CLAIM['sens'])*100:>10.2f}{r['spec']*100:>8.2f}%{(r['spec']-CLAIM['spec'])*100:>10.2f}")

with open(os.path.join(OUT, "diabetes_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote evaluation_evidence/diabetes_report.txt")
