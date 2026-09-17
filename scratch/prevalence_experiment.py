"""
Does class balance change what the model can do, or only what its numbers mean?
==============================================================================
Run this yourself. It answers three questions with measurements, not assertions.

  Q1  Holding SAMPLE SIZE fixed, does training prevalence change ranking
      ability (ROC-AUC) or the prevalence-free operating characteristics
      (sensitivity, specificity)?
  Q2  Does it change the probability scale (calibration) and everything built
      on that scale (accuracy, PPV, optimal threshold)?
  Q3  Does a Bayes prior-shift correction recover what balancing broke, without
      retraining?

Design notes that make this honest:
  * One test set, never resampled, held at the NATURAL prevalence of the source
    file. Every model is judged on exactly the same patients.
  * Training sets are size-matched. Prevalence is the only variable that moves;
    otherwise a prevalence effect and a sample-size effect are confounded.
    (My earlier quick check did confound them — this one does not.)
  * Repeated over several seeds, reported as mean +/- sd, because a single split
    on one dataset is an anecdote.
  * `scale_pos_weight` is included as a third arm: it is the standard way to get
    the benefit of balancing without resampling, and it keeps every row.

DATA
  Preferred : diabetes_binary_health_indicators_BRFSS2015.csv   (253,680 rows,
              natural prevalence ~14%) — the full file from the same Kaggle page.
  Fallback  : diabetes_binary_5050split_health_indicators_BRFSS2015.csv
              (70,692 rows, artificially 50/50). The script still runs, but it
              can only DOWN-sample positives, so the size-matched training sets
              are smaller and the natural-prevalence test set is synthetic.
              It will say so loudly.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

TARGET = "Diabetes_binary"
SEEDS = [0, 1, 2, 3, 4]
TRAIN_PREVALENCES = [0.14, 0.25, 0.40, 0.50]
FN_COST, FP_COST = 2.0, 1.0
XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
           colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
           tree_method="hist")

FULL = "diabetes_binary_health_indicators_BRFSS2015.csv"
HALF = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"


def locate(name):
    import glob
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(name)
    return hits[0] if hits else None


def load():
    p = locate(FULL)
    if p:
        return pd.read_csv(p), True
    p = locate(HALF)
    if p:
        print("!" * 74)
        print("! Full unbalanced file not found. Falling back to the 50/50 file.")
        print("! Training sets must be built by DISCARDING negatives, so they are")
        print("! smaller, and the 'natural prevalence' test set is synthetic.")
        print("! Conclusions about direction hold; absolute numbers are weaker.")
        print("!" * 74)
        return pd.read_csv(p), False
    raise FileNotFoundError("Put one of the two BRFSS csv files next to this script.")


def at_prevalence(X, y, prev, n_target, rs):
    """Sample n_target rows at the requested prevalence, or as close as the
    available pool allows. Returns the realised prevalence too."""
    pos = np.where(y.values == 1)[0]
    neg = np.where(y.values == 0)[0]
    want_pos = int(round(n_target * prev))
    want_neg = n_target - want_pos
    if want_pos > len(pos) or want_neg > len(neg):
        scale = min(len(pos) / max(want_pos, 1), len(neg) / max(want_neg, 1))
        want_pos, want_neg = int(want_pos * scale), int(want_neg * scale)
    i = np.concatenate([rs.choice(pos, want_pos, replace=False),
                        rs.choice(neg, want_neg, replace=False)])
    rs.shuffle(i)
    return X.iloc[i], y.iloc[i], want_pos / (want_pos + want_neg)


def cost_threshold(y, p):
    best, bc = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 200):
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int),
                                          labels=[0, 1]).ravel()
        c = FN_COST * fn + FP_COST * fp
        if c < bc:
            bc, best = c, float(t)
    return best


def evaluate(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int),
                                      labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, p),
                sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                accuracy=(tp + tn) / len(y),
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                brier=brier_score_loss(y, p),
                mean_pred=float(p.mean()), threshold=thr)


def prior_shift(p, prev_train, prev_deploy):
    """Bayes correction for a change in the class prior.
       odds_out = odds_in * [pi_dep/(1-pi_dep)] / [pi_tr/(1-pi_tr)]"""
    r = (prev_deploy / (1 - prev_deploy)) / (prev_train / (1 - prev_train))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (r * p) / (1 - p + r * p)


def main():
    df, is_full = load()
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    natural = float(y.mean()) if is_full else 0.14
    print(f"\nrows {len(df)}   file prevalence {y.mean()*100:.2f}%   "
          f"deployment prevalence used: {natural*100:.2f}%")

    records = []
    for seed in SEEDS:
        rs = np.random.default_rng(seed)
        X_pool, X_te, y_pool, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)

        if not is_full:                      # synthesise a natural-prevalence test set
            X_te, y_te, _ = at_prevalence(X_te, y_te, natural,
                                          int(len(X_te) * 0.6), rs)

        # size-matched training budget: the largest n every prevalence can supply
        npos, nneg = int((y_pool == 1).sum()), int((y_pool == 0).sum())
        budget = min(min(int(npos / p), int(nneg / (1 - p)))
                     for p in TRAIN_PREVALENCES)

        for prev in TRAIN_PREVALENCES:
            X_tr, y_tr, realised = at_prevalence(X_pool, y_pool, prev, budget, rs)
            m = xgb.XGBClassifier(**XGB, random_state=seed).fit(X_tr, y_tr)
            p_te = m.predict_proba(X_te)[:, 1]
            thr = cost_threshold(y_tr, m.predict_proba(X_tr)[:, 1])

            r = evaluate(y_te, p_te, thr)
            records.append(dict(seed=seed, arm=f"resampled @ {prev:.0%}",
                                train_prev=realised, n_train=len(y_tr),
                                corrected=False, **r))

            pc = prior_shift(p_te, realised, natural)
            thr_c = float(prior_shift(np.array([thr]), realised, natural)[0])
            rc = evaluate(y_te, pc, thr_c)
            records.append(dict(seed=seed, arm=f"resampled @ {prev:.0%}",
                                train_prev=realised, n_train=len(y_tr),
                                corrected=True, **rc))

        # third arm: keep every row, rebalance through the loss instead
        X_tr, y_tr, realised = at_prevalence(X_pool, y_pool, natural, budget, rs)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = xgb.XGBClassifier(**XGB, random_state=seed,
                              scale_pos_weight=spw).fit(X_tr, y_tr)
        p_te = m.predict_proba(X_te)[:, 1]
        thr = cost_threshold(y_tr, m.predict_proba(X_tr)[:, 1])
        records.append(dict(seed=seed, arm="scale_pos_weight @ natural",
                            train_prev=realised, n_train=len(y_tr),
                            corrected=False, **evaluate(y_te, p_te, thr)))
        print(f"  seed {seed} done   (train budget {budget} rows, "
              f"test {len(y_te)} rows at {y_te.mean()*100:.1f}%)")

    res = pd.DataFrame(records)
    res.to_csv("prevalence_experiment_raw.csv", index=False)

    cols = ["roc_auc", "sensitivity", "specificity", "accuracy", "ppv",
            "brier", "threshold", "mean_pred"]
    agg = (res.groupby(["arm", "corrected"])[cols]
              .agg(["mean", "std"]).round(4))

    print("\n" + "=" * 78)
    print("Q1  PREVALENCE-FREE METRICS  (should be flat across arms)")
    print("=" * 78)
    raw = res[~res.corrected]
    t1 = raw.groupby("arm")[["roc_auc", "sensitivity", "specificity"]].agg(["mean", "std"])
    print(t1.round(4).to_string())

    print("\n" + "=" * 78)
    print("Q2  PREVALENCE-DEPENDENT METRICS  (should move with training prevalence)")
    print("=" * 78)
    t2 = raw.groupby("arm")[["accuracy", "ppv", "brier", "mean_pred", "threshold"]]\
            .agg(["mean", "std"])
    print(t2.round(4).to_string())
    print(f"\n  true prevalence of the test set: {natural:.4f}")
    print("  compare with mean_pred: a well-calibrated model matches it.")

    print("\n" + "=" * 78)
    print("Q3  DOES THE BAYES CORRECTION REPAIR IT?  (no retraining)")
    print("=" * 78)
    comp = res.groupby(["arm", "corrected"])[["brier", "mean_pred", "sensitivity",
                                              "specificity", "accuracy"]].mean()
    print(comp.round(4).to_string())
    print("\n  sensitivity/specificity identical before and after => the correction")
    print("  is a monotone relabelling: it changes the number, not who is flagged.")

    json.dump(dict(source="full unbalanced" if is_full else "50/50 fallback",
                   deployment_prevalence=natural, seeds=SEEDS,
                   train_prevalences=TRAIN_PREVALENCES,
                   summary=json.loads(agg.to_json())),
              open("prevalence_experiment_summary.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=140)
    order = sorted(raw.arm.unique())
    for a, metric, title in zip(ax, ["roc_auc", "accuracy", "brier"],
                                ["ROC-AUC (prevalence-free)",
                                 "Accuracy (prevalence-dependent)",
                                 "Brier (calibration)"]):
        mu = [raw[raw.arm == o][metric].mean() for o in order]
        sd = [raw[raw.arm == o][metric].std() for o in order]
        a.bar(range(len(order)), mu, yerr=sd, capsize=3, color="#4C78A8")
        if metric == "brier":
            cor = res[res.corrected]
            mu2 = [cor[cor.arm == o][metric].mean() if (cor.arm == o).any() else np.nan
                   for o in order]
            a.bar(range(len(order)), mu2, width=0.35, color="#54A24B",
                  label="after Bayes correction")
            a.legend(fontsize=7)
        a.set_xticks(range(len(order)))
        a.set_xticklabels(order, fontsize=6.5, rotation=15)
        a.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig("prevalence_experiment.png")
    print("\nwritten: prevalence_experiment_raw.csv, "
          "prevalence_experiment_summary.json, prevalence_experiment.png")


if __name__ == "__main__":
    main()
