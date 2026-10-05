"""
Diabetes module audit — three questions, same as the heart module.

  1. Is the label direction correct?              (checked already: yes)
  2. Does anything leak between train and test?
  3. Is the shipped threshold (0.275) defensible?

Question 3 has two parts that must not be confused:
  3a. WHERE was it selected?  train_diabetes_ensemble.py:933 passes y_true=y_test.
  3b. On WHAT prevalence?     The file is the pre-balanced 50/50 Kaggle variant.
      Real BRFSS diabetes prevalence is roughly 14%. A probability threshold
      tuned on a 50% base rate does not transfer to a 14% one.
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

SEED = 42
FN, FP_ = 2.0, 1.0
T = "Diabetes_binary"
P = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
         colsample_bytree=0.8, random_state=SEED, eval_metric="logloss",
         n_jobs=-1, tree_method="hist")

d = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X, y = d.drop(columns=[T]), d[T].astype(int)


def cm(y_, p_, t):
    return confusion_matrix(y_, (p_ >= t).astype(int), labels=[0, 1]).ravel()


def cost_threshold(y_, p_):
    best, bc = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 200):
        tn, fp, fn, tp = cm(y_, p_, t)
        c = FN * fn + FP_ * fp
        if c < bc:
            bc, best = c, float(t)
    return best


def report(y_, p_, t, tag):
    tn, fp, fn, tp = cm(y_, p_, t)
    print(f"  {tag:<34} thr {t:.4f}   sens {tp/(tp+fn)*100:5.2f}%   "
          f"spec {tn/(tn+fp)*100:5.2f}%   FN {fn:5d}   FP {fp:5d}")
    return dict(thr=t, sens=tp/(tp+fn), spec=tn/(tn+fp), fn=int(fn), fp=int(fp))


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

print("=" * 82)
print("1. DUPLICATE ROWS ACROSS THE SPLIT")
print("=" * 82)
key = list(X.columns)
tr_keys = set(map(tuple, X_tr[key].values))
dup_rows = X_te[key].apply(tuple, axis=1).isin(tr_keys)
print(f"  test rows whose exact feature vector also appears in train: "
      f"{dup_rows.sum()} / {len(X_te)}  ({dup_rows.mean()*100:.2f}%)")
print("  All 21 features are binary or coarse ordinal, so identical patients are")
print("  expected by chance, not only by true duplication. Still, these rows are")
print("  memorised, not predicted. Scores below are reported both ways.")

model = xgb.XGBClassifier(**P).fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]
print(f"\n  test ROC-AUC, all rows              {roc_auc_score(y_te, p_te):.4f}")
m = ~dup_rows.values
print(f"  test ROC-AUC, de-duplicated rows    "
      f"{roc_auc_score(y_te[m], p_te[m]):.4f}   (n={m.sum()})")

print("\n" + "=" * 82)
print("2. THRESHOLD: selected on test (shipped) vs out-of-fold train (correct)")
print("=" * 82)
oof = cross_val_predict(xgb.XGBClassifier(**P), X_tr, y_tr, cv=cv,
                        method="predict_proba")[:, 1]
t_oof = cost_threshold(y_tr, oof)
t_test = cost_threshold(y_te, p_te)

print("  evaluated on the held-out test set:")
r_ship = report(y_te, p_te, 0.275, "shipped 0.275 (from y_test)")
r_test = report(y_te, p_te, t_test, "re-derived on y_test")
r_oof = report(y_te, p_te, t_oof, "out-of-fold on training  <- correct")
print(f"\n  selection-bias gap: {abs(t_test-t_oof):.4f} in threshold units")
print(f"  cost at shipped 0.275   : {FN*r_ship['fn']+FP_*r_ship['fp']:.0f}")
print(f"  cost at out-of-fold thr : {FN*r_oof['fn']+FP_*r_oof['fp']:.0f}")

print("\n" + "=" * 82)
print("3. PREVALENCE — the larger problem")
print("=" * 82)
print(f"  training file prevalence : {y.mean()*100:.1f}%  (pre-balanced 50/50 variant)")
print( "  real BRFSS prevalence    : ~14%")
print("\n  What the same threshold does as the base rate falls, holding the model")
print("  fixed and resampling the test set to each prevalence:\n")
print(f"  {'prevalence':>11}{'flagged':>10}{'sens':>9}{'spec':>9}{'PPV':>9}"
      f"{'  per 1000 screened'}")
pos, neg = np.where(y_te.values == 1)[0], np.where(y_te.values == 0)[0]
rng = np.random.default_rng(SEED)
for prev in [0.50, 0.30, 0.20, 0.14, 0.10]:
    n_neg = len(neg)
    n_pos = int(n_neg * prev / (1 - prev))
    idx = np.concatenate([rng.choice(pos, min(n_pos, len(pos)), replace=False), neg])
    yy, pp = y_te.values[idx], p_te[idx]
    tn, fp, fn, tp = cm(yy, pp, t_oof)
    ppv = tp / (tp + fp) if tp + fp else np.nan
    flag = (tp + fp) / len(yy)
    print(f"  {prev*100:>10.0f}%{flag*100:>9.1f}%{tp/(tp+fn)*100:>8.1f}%"
          f"{tn/(tn+fp)*100:>8.1f}%{ppv*100:>8.1f}%"
          f"{'   ' + str(int(flag*1000)) + ' referred, ' + str(int((1-ppv)*flag*1000)) + ' needlessly'}")
print("\n  Sensitivity and specificity are stable (they are prevalence-independent).")
print("  PPV is not. At a realistic base rate most people the system flags do not")
print("  have diabetes, and the threshold was never chosen with that in mind.")
