"""
Generate every diabetes evaluation artefact from one re-runnable script.

    backend/.venv/bin/python scratch/generate_diabetes_evidence.py

Prerequisite: scratch/diabetes_oof_threshold.py has been run (it writes
evaluation_evidence/diabetes/oof_threshold.json and threshold_scan_oof.csv,
which this script reads rather than re-fitting fold models).

Read-only with respect to models/ and configs/. Deterministic (seed 42).

Deployment prevalence is emulated by importance weights on the held-out test
set (negatives weight 1, positives weight odds(pi)/odds(test)), so no rows
are discarded and no random subsample enters the point estimates.

final_metrics_table.{csv,md,json} is the single source for every diabetes
number quoted in documentation.
"""
import json
import os
import sys
import time

from _diabetes_common import (EVIDENCE_DIR, FN_COST, FP_COST, OLD_THRESHOLD,
                              ROOT, SEED, confusion, load_config, load_models,
                              load_split, make_preparer, stacking_proba)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve, roc_auc_score, roc_curve)

from backend.prevalence_correction import (apply_prevalence_correction,
                                           invert_prevalence_correction)

N_BOOT = 2000
PREVALENCE_GRID = [0.50, 0.40, 0.30, 0.237, 0.20, 0.14, 0.10]
N_CAL_BINS = 10

# Reference palette (dataviz skill): slot 1 = before, slot 2 = after.
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
C_TEXT, C_MUTED, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.grid": True,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})

REL = lambda name: os.path.join("evaluation_evidence", "diabetes", name)  # noqa: E731
written = []


def out(name):
    path = os.path.join(EVIDENCE_DIR, name)
    written.append(name)
    return path


# ── metric helpers ────────────────────────────────────────────────────────────
def prevalence_weights(y, pi):
    """Weights that make the weighted positive rate of y equal pi."""
    y = np.asarray(y)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    w_pos = (pi / (1 - pi)) * (n_neg / n_pos)
    return np.where(y == 1, w_pos, 1.0)


def ppv_at(se, sp, pi):
    return se * pi / (se * pi + (1 - sp) * (1 - pi))


def npv_at(se, sp, pi):
    return sp * (1 - pi) / (sp * (1 - pi) + (1 - se) * pi)


def brier(y, p, w=None):
    return float(np.average((np.asarray(p) - np.asarray(y)) ** 2, weights=w))


def ece(y, p, w, bins=N_CAL_BINS):
    """Expected calibration error, equal-width bins, weighted."""
    y, p, w = map(np.asarray, (y, p, w))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot, err = w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if w[m].sum() > 0:
            err += w[m].sum() / tot * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(err)


def calibration_points(y, p, w, bins=N_CAL_BINS):
    y, p, w = map(np.asarray, (y, p, w))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            rows.append((np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m]), int(m.sum())))
    return np.array(rows)


def point_metrics(y, p_raw, t_raw, pi_dep):
    # int(round(...)), matching main()'s `pct` -- plain int() truncates
    # (23.7 -> 23) instead of rounding (23.7 -> 24), which only ever
    # differed from round() by chance when pi_dep*100 was a whole number
    # (e.g. the old 0.14 -> 14 in both conventions). A pi_dep whose
    # percentage isn't a whole number (e.g. 0.237 -> 23.7) exposes the
    # mismatch as a KeyError further down where callers look up the key
    # main() computed with round().
    pct = int(round(pi_dep * 100))
    tn, fp, fn, tp = confusion(y, p_raw, t_raw)
    se, sp = tp / (tp + fn), tn / (tn + fp)
    ppv50 = tp / (tp + fp)
    return {
        "sensitivity": se, "specificity": sp,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "ppv_test_50": ppv50, "npv_test_50": tn / (tn + fn),
        f"ppv_at_{pct}": ppv_at(se, sp, pi_dep),
        f"npv_at_{pct}": npv_at(se, sp, pi_dep),
        "flagged_at_deploy": se * pi_dep + (1 - sp) * (1 - pi_dep),
        "cost_per_1000_test": 1000 * (FN_COST * fn + FP_COST * fp) / len(y),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def md_table(df, floatfmt=".4f"):
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ── plotting helpers ──────────────────────────────────────────────────────────
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.grid(False)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n{cm[i, j]/cm[i].sum()*100:.1f}% of row",
                    ha="center", va="center", fontsize=9, color="#ffffff" if dark else C_TEXT)
    ax.set_xticks([0, 1], ["Pred. negative", "Pred. positive"])
    ax.set_yticks([0, 1], ["Actual negative", "Actual positive"])
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    t0 = time.time()
    oof_path = os.path.join(EVIDENCE_DIR, "oof_threshold.json")
    if not os.path.exists(oof_path):
        sys.exit("Run scratch/diabetes_oof_threshold.py first (oof_threshold.json missing).")
    oof = json.load(open(oof_path))
    scan = pd.read_csv(os.path.join(EVIDENCE_DIR, "threshold_scan_oof.csv"))

    cfg = load_config()
    pi_tr = float(cfg["model"]["prevalence_train"])
    pi_dep = float(cfg["model"]["prevalence_deploy"])
    t_cfg = float(cfg["model"]["inference_threshold"])

    T_OLD = OLD_THRESHOLD
    T_NEW = float(oof["threshold_new_raw"])
    # What production actually applies: the config value mapped back to raw.
    T_NEW_EFFECTIVE = float(invert_prevalence_correction(t_cfg, pi_tr, pi_dep))
    assert abs(T_NEW_EFFECTIVE - T_NEW) < 1e-5, (
        f"configs/diabetes.yaml inference_threshold={t_cfg} maps to raw {T_NEW_EFFECTIVE}, "
        f"but the OOF threshold is {T_NEW}. Update the config first.")

    X_tr, X_te, y_tr, y_te = load_split(cfg)
    prep = make_preparer(cfg)
    base, meta = load_models(cfg)
    y = y_te.values.astype(int)
    p_raw = stacking_proba(prep(X_te), base, meta)
    p_cor = apply_prevalence_correction(p_raw, pi_tr, pi_dep)
    w_dep = prevalence_weights(y, pi_dep)
    pct = int(round(pi_dep * 100))

    # Decision invariance on the real test set, at the value production uses.
    dec_raw = p_raw >= T_NEW
    dec_prod = p_cor >= t_cfg
    n_decision_diff = int((dec_raw != dec_prod).sum())

    # Duplicates across the split (data property) and their effect on the shipped model.
    tr_keys = set(map(tuple, X_tr.values))
    dup = X_te.apply(tuple, axis=1).isin(tr_keys).values
    auc_all = roc_auc_score(y, p_raw)
    auc_dedup = roc_auc_score(y[~dup], p_raw[~dup])

    thresholds = {"old": (T_OLD, "0.275 (selected on y_test)"),
                  "new": (T_NEW, f"{T_NEW:.4f} (selected on train OOF)")}
    pm = {k: point_metrics(y, p_raw, t, pi_dep) for k, (t, _) in thresholds.items()}

    # 1 + 2. confusion matrices and classification reports ────────────────────
    for key, (t, label) in thresholds.items():
        tn, fp, fn, tp = confusion(y, p_raw, t)
        cm = np.array([[tn, fp], [fn, tp]])
        tag = f"{key}_{t:.4f}"
        plot_cm(cm, f"Stacking ensemble, test n={len(y):,}\nthreshold {label}", out(f"confusion_matrix_{tag}.png"))
        m = pm[key]
        with open(out(f"confusion_matrix_{tag}.txt"), "w") as f:
            f.write(f"Diabetes stacking ensemble — held-out test set (n={len(y)}, prevalence {y.mean():.2%})\n")
            f.write(f"threshold (raw, training prior)      : {t:.6f}\n")
            f.write(f"threshold (deployment prior, pi={pi_dep}): {apply_prevalence_correction(t, pi_tr, pi_dep):.6f}\n")
            f.write(f"source of threshold                  : {label}\n\n")
            f.write("                 pred_neg   pred_pos\n")
            f.write(f"actual_neg      {tn:9d}  {fp:9d}\nactual_pos      {fn:9d}  {tp:9d}\n\n")
            f.write(f"sensitivity {m['sensitivity']:.4f}   specificity {m['specificity']:.4f}\n")
            f.write(f"clinical cost 2*FN + FP = {int(FN_COST*fn + FP_COST*fp)}\n")
        with open(out(f"classification_report_{tag}.txt"), "w") as f:
            f.write(f"threshold {label}\n\n")
            f.write("A) Held-out test set as sampled (50/50):\n")
            f.write(classification_report(y, (p_raw >= t).astype(int), target_names=["No diabetes", "Diabetes"], digits=4))
            f.write(f"\nB) Same rows re-weighted to deployment prevalence {pi_dep:.0%} "
                    "(importance weights; recall per class is unchanged, precision is not):\n")
            f.write(classification_report(y, (p_raw >= t).astype(int), target_names=["No diabetes", "Diabetes"],
                                          digits=4, sample_weight=w_dep))

    # 3. ROC and PR ───────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, p_raw)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot(fpr, tpr, color=C_BEFORE, label=f"Stacking ensemble (AUC {auc_all:.4f})")
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Chance")
    for key, color, name in [("old", C_BEFORE, "old 0.275"), ("new", C_AFTER, f"new {T_NEW:.4f}")]:
        style = dict(ms=12, mfc="none", mec=color, mew=2) if key == "old" else dict(ms=6, color=color)
        ax.plot(1 - pm[key]["specificity"], pm[key]["sensitivity"], "o", ls="", **style,
                label=f"Operating point, {name}")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC — held-out test set (prevalence-free)", loc="left")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out("roc_curve.png")); plt.close(fig)

    ap50 = average_precision_score(y, p_raw)
    ap_dep = average_precision_score(y, p_raw, sample_weight=w_dep)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150, sharey=True)
    for ax, w, pi, ap, ttl in [(axes[0], None, y.mean(), ap50, "Test set as sampled (50%)"),
                               (axes[1], w_dep, pi_dep, ap_dep, f"Re-weighted to deployment ({pi_dep:.0%})")]:
        pr, rc, _ = precision_recall_curve(y, p_raw, sample_weight=w)
        ax.plot(rc, pr, color=C_BEFORE, label=f"Stacking (AP {ap:.3f})")
        ax.axhline(pi, color=C_MUTED, lw=1, ls="--", label=f"No-skill = prevalence {pi:.0%}")
        for key, color in [("old", C_BEFORE), ("new", C_AFTER)]:
            se, sp = pm[key]["sensitivity"], pm[key]["specificity"]
            style = dict(ms=12, mfc="none", mec=color, mew=2) if key == "old" else dict(ms=6, color=color)
            ax.plot(se, ppv_at(se, sp, pi), "o", ls="", **style,
                    label=f"{'old 0.275' if key == 'old' else f'new {T_NEW:.4f}'}: PPV {ppv_at(se, sp, pi):.1%}")
        ax.set_title(ttl, loc="left"); ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("Precision (PPV)")
    fig.suptitle("Precision–recall: same model, same rows, different base rate", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out("pr_curve.png")); plt.close(fig)

    # 4. Threshold decision log ───────────────────────────────────────────────
    best = scan.loc[scan.total_cost.idxmin()]
    top = scan.nsmallest(10, "total_cost")
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(scan.threshold, scan.total_cost, color=C_BEFORE, label="OOF training cost (2·FN + FP)")
    ax.axvline(T_NEW, color=C_AFTER, lw=1.5, label=f"Selected {T_NEW:.4f}")
    ax.axvline(T_OLD, color=C_MUTED, lw=1, ls="--", label="Old 0.275 (from y_test)")
    ax.set_xlabel("Threshold (raw, training prior)"); ax.set_ylabel("Cost on 56,553 OOF rows")
    ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Threshold scan on out-of-fold training predictions", loc="left")
    fig.tight_layout(); fig.savefig(out("threshold_cost_curve_oof.png")); plt.close(fig)
    old_row = scan.iloc[(scan.threshold - T_OLD).abs().idxmin()]
    with open(out("threshold_decision_log.md"), "w") as f:
        f.write("# Diabetes threshold decision log\n\n")
        f.write("| Item | Value |\n|---|---|\n")
        f.write("| Data used for selection | training split only, 56,553 rows, out-of-fold stacking probabilities |\n")
        f.write("| OOF procedure | `cross_val_predict`, `StratifiedKFold(5, shuffle=True, random_state=42)`; "
                "clones of the three shipped base models, then a clone of the shipped meta-learner on the OOF base matrix |\n")
        f.write("| Selection function | `find_optimal_clinical_threshold` (models/train_diabetes_ensemble.py), unchanged |\n")
        f.write("| Cost function | `Cost = 2.0·FN + 1.0·FP` |\n")
        f.write(f"| Range scanned | `np.linspace(0.01, 0.99, 200)` — {len(scan)} thresholds, step {scan.threshold.diff().median():.4f} |\n")
        f.write("| Tie rule | first threshold reaching the minimum (strict `<`) |\n")
        f.write(f"| **Selected (raw)** | **{T_NEW:.6f}** |\n")
        f.write(f"| Selected, deployment prior π={pi_dep} | {apply_prevalence_correction(T_NEW, pi_tr, pi_dep):.6f} (written to configs/diabetes.yaml) |\n")
        f.write(f"| OOF cost at selected | {best.total_cost:.0f} (FN {int(best.fn)}, FP {int(best.fp)}) |\n")
        f.write(f"| OOF cost at grid point nearest 0.275 ({old_row.threshold}) | {old_row.total_cost:.0f} |\n")
        f.write(f"| Old threshold re-derived on y_test today | {oof['threshold_old_reproduced_on_y_test']:.4f} |\n")
        f.write(f"| Sensitivity check: shipped meta-learner on OOF base matrix | {oof['threshold_new_raw_sensitivity_check_shipped_meta']:.4f} |\n\n")
        f.write("## Why this threshold won\n\n")
        f.write("It is the grid point with the lowest 2·FN + FP on predictions that no model had seen during fitting. "
                "With FN weighted 2 and FP weighted 1, the optimum under perfect calibration is where "
                f"P(diabetes | x) = 1/(1+2) = 0.333 on the training prior; the empirical optimum is {T_NEW:.4f}. "
                "The gap is a property of this model's calibration on the OOF rows, which is why the threshold is "
                "chosen empirically rather than set to 1/3. Whether the curve is flat near the minimum can be read "
                "from the ten best grid points below.\n\n")
        f.write("## Ten lowest-cost thresholds (OOF)\n\n")
        f.write(md_table(top[["threshold", "total_cost", "fn", "fp", "recall", "precision"]]))
        f.write("\n\n![cost curve](threshold_cost_curve_oof.png)\n")

    # 5. Calibration ─────────────────────────────────────────────────────────
    cal = {
        f"brier_raw_at_{pct}": brier(y, p_raw, w_dep),
        f"brier_corrected_at_{pct}": brier(y, p_cor, w_dep),
        "brier_raw_at_test_50": brier(y, p_raw),
        f"ece_raw_at_{pct}": ece(y, p_raw, w_dep),
        f"ece_corrected_at_{pct}": ece(y, p_cor, w_dep),
        "ece_raw_at_test_50": ece(y, p_raw, np.ones_like(p_raw)),
        f"mean_prob_raw_at_{pct}": float(np.average(p_raw, weights=w_dep)),
        f"mean_prob_corrected_at_{pct}": float(np.average(p_cor, weights=w_dep)),
        "deployment_prevalence": pi_dep,
        "method": "importance-weighted held-out test set; 10 equal-width bins",
    }
    json.dump(cal, open(out("calibration.json"), "w"), indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150, sharey=True)
    for ax, (pts_p, ttl) in zip(axes, [
        ((p_raw, None), "Reference: raw probabilities on the 50/50 test set"),
        (None, f"At deployment prevalence {pi_dep:.0%}: before vs after correction"),
    ]):
        ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Perfect calibration")
        ax.set_title(ttl, loc="left", fontsize=8.5); ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    pts = calibration_points(y, p_raw, np.ones_like(p_raw))
    axes[0].plot(pts[:, 0], pts[:, 1], "-o", ms=5, color=C_BEFORE, mec=C_SURFACE,
                 label=f"Raw (Brier {cal['brier_raw_at_test_50']:.3f})")
    pts_b = calibration_points(y, p_raw, w_dep)
    pts_a = calibration_points(y, p_cor, w_dep)
    axes[1].plot(pts_b[:, 0], pts_b[:, 1], "-o", ms=5, color=C_BEFORE, mec=C_SURFACE,
                 label=f"Before: raw (Brier {cal[f'brier_raw_at_{pct}']:.3f})")
    axes[1].plot(pts_a[:, 0], pts_a[:, 1], "-o", ms=5, color=C_AFTER, mec=C_SURFACE,
                 label=f"After: Bayes-corrected (Brier {cal[f'brier_corrected_at_{pct}']:.3f})")
    axes[0].set_ylabel("Observed fraction with diabetes")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out("calibration_curve.png")); plt.close(fig)

    # 6. PPV collapse table ──────────────────────────────────────────────────
    rows = []
    for key in ("old", "new"):
        se, sp = pm[key]["sensitivity"], pm[key]["specificity"]
        for pi in PREVALENCE_GRID:
            flagged = se * pi + (1 - sp) * (1 - pi)
            ppv = ppv_at(se, sp, pi)
            rows.append({
                "threshold": "0.275 (old)" if key == "old" else f"{T_NEW:.4f} (new)",
                "prevalence": pi, "flagged_pct": 100 * flagged,
                "sensitivity_pct": 100 * se, "specificity_pct": 100 * sp,
                "ppv_pct": 100 * ppv, "npv_pct": 100 * npv_at(se, sp, pi),
                "per_1000_referred": 1000 * flagged,
                "per_1000_true_positive": 1000 * se * pi,
                "per_1000_needless_referral": 1000 * (1 - sp) * (1 - pi),
                "per_1000_missed": 1000 * (1 - se) * pi,
            })
    ppv_df = pd.DataFrame(rows)
    ppv_df.to_csv(out("ppv_collapse_table.csv"), index=False, float_format="%.4f")
    with open(out("ppv_collapse_table.md"), "w") as f:
        f.write("# PPV versus prevalence (sensitivity and specificity held at their test-set values)\n\n")
        f.write("PPV = Se·π / (Se·π + (1−Sp)(1−π)).  Per-1000 columns are expected counts per 1000 people screened.\n\n")
        show = ppv_df.copy(); show["prevalence"] = (show.prevalence * 100).round(0).astype(int).astype(str) + "%"
        f.write(md_table(show, ".1f"))
        f.write("\n")

    # 7. Before/after ──────────────────────────────────────────────────────────
    # "before" = shipped state before this change: threshold 0.275, raw probability shown.
    # "after"  = OOF threshold, Bayes-corrected probability shown.
    def state(key, p_shown):
        m = pm[key]
        return {
            "threshold_used_raw": thresholds[key][0],
            "threshold_shown_to_user": thresholds[key][0] if key == "old" else t_cfg,
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "ppv_test_50": m["ppv_test_50"], f"ppv_at_{pct}": m[f"ppv_at_{pct}"],
            f"npv_at_{pct}": m[f"npv_at_{pct}"], "flagged_at_deploy": m["flagged_at_deploy"],
            "cost_per_1000_test": m["cost_per_1000_test"],
            f"brier_shown_prob_at_{pct}": brier(y, p_shown, w_dep),
            f"ece_shown_prob_at_{pct}": ece(y, p_shown, w_dep),
            f"mean_shown_prob_at_{pct}": float(np.average(p_shown, weights=w_dep)),
            "roc_auc": auc_all,
        }
    before, after = state("old", p_raw), state("new", p_cor)
    ba = {"before": before, "after": after,
          "delta": {k: after[k] - before[k] for k in before if isinstance(before[k], float)},
          "notes": {"before": "threshold 0.275 chosen on y_test; raw (50/50-prior) probability displayed",
                    "after": "threshold chosen on training OOF; probability and threshold on 14% prior",
                    "decisions_changed_by_correction_alone_on_test": n_decision_diff}}
    json.dump(ba, open(out("before_after.json"), "w"), indent=2)
    pct_keys = [("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                (f"ppv_at_{pct}", f"PPV at {pct}%"), (f"npv_at_{pct}", f"NPV at {pct}%"),
                ("flagged_at_deploy", f"Flagged at {pct}%")]
    cal_keys = [(f"brier_shown_prob_at_{pct}", "Brier (shown prob.)"),
                (f"ece_shown_prob_at_{pct}", "ECE (shown prob.)"),
                (f"mean_shown_prob_at_{pct}", f"Mean shown prob.\n(true = {pi_dep:.2f})")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150,
                             gridspec_kw={"width_ratios": [1.3, 1, 0.8]})
    for ax, keys, scale, xlabel in [(axes[0], pct_keys, 100, "%"), (axes[1], cal_keys, 1, "value (lower is better, except mean)")]:
        for i, (k, lab) in enumerate(keys):
            b, a = before[k] * scale, after[k] * scale
            ax.plot([b, a], [i, i], color=C_GRID, lw=3, zorder=1)
            ax.plot(b, i, "o", ms=13, mfc="none", mec=C_BEFORE, mew=2, zorder=3)
            ax.plot(a, i, "o", ms=7, color=C_AFTER, zorder=2)
            ax.annotate(f"{b:.1f} → {a:.1f}" if scale == 100 else f"{b:.3f} → {a:.3f}",
                        (max(a, b), i), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7, color=C_MUTED)
        ax.set_yticks(range(len(keys)), [l for _, l in keys]); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.margins(x=0.35)
        if scale == 100:
            ax.set_xlim(0, 135); ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_title("Decision metrics", loc="left")
    axes[1].set_title("Probability quality at deployment prevalence", loc="left")
    axes[2].bar([0, 1], [before["cost_per_1000_test"], after["cost_per_1000_test"]],
                color=[C_BEFORE, C_AFTER], width=0.6, edgecolor=C_SURFACE, linewidth=2)
    for i, v in enumerate([before["cost_per_1000_test"], after["cost_per_1000_test"]]):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=C_TEXT)
    axes[2].set_xticks([0, 1], ["Before", "After"])
    axes[2].set_title("Cost per 1000 test rows\n(2·FN + FP)", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C_BEFORE, mew=2, ms=11,
                          label="Before (0.275 from y_test, raw prob.)"),
               plt.Line2D([], [], marker="o", ls="", color=C_AFTER, ms=7,
                          label=f"After ({T_NEW:.4f} from OOF, corrected prob.)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out("before_after.png")); plt.close(fig)

    # 8. Final table with bootstrap CIs ────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    n = len(y)
    boot = {k: [] for k in ("roc_auc", "roc_auc_dedup", "pr_auc_test_50", f"pr_auc_at_{pct}",
                            f"brier_raw_at_{pct}", f"brier_corrected_at_{pct}", "brier_raw_at_test_50",
                            f"ece_corrected_at_{pct}")}
    for key in ("old", "new"):
        for mname in ("sensitivity", "specificity", "accuracy", "f1", "ppv_test_50", "npv_test_50",
                      f"ppv_at_{pct}", f"npv_at_{pct}", "flagged_at_deploy", "cost_per_1000_test"):
            boot[f"{mname}@{key}"] = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        yb, pb, cb = y[i], p_raw[i], p_cor[i]
        wb = prevalence_weights(yb, pi_dep)
        boot["roc_auc"].append(roc_auc_score(yb, pb))
        nd = ~dup[i]
        boot["roc_auc_dedup"].append(roc_auc_score(yb[nd], pb[nd]))
        boot["pr_auc_test_50"].append(average_precision_score(yb, pb))
        boot[f"pr_auc_at_{pct}"].append(average_precision_score(yb, pb, sample_weight=wb))
        boot[f"brier_raw_at_{pct}"].append(brier(yb, pb, wb))
        boot[f"brier_corrected_at_{pct}"].append(brier(yb, cb, wb))
        boot["brier_raw_at_test_50"].append(brier(yb, pb))
        boot[f"ece_corrected_at_{pct}"].append(ece(yb, cb, wb))
        for key, (t, _) in thresholds.items():
            m = point_metrics(yb, pb, t, pi_dep)
            for mname in ("sensitivity", "specificity", "accuracy", "f1", "ppv_test_50", "npv_test_50",
                          f"ppv_at_{pct}", f"npv_at_{pct}", "flagged_at_deploy", "cost_per_1000_test"):
                boot[f"{mname}@{key}"].append(m[mname])

    point = {"roc_auc": auc_all, "roc_auc_dedup": auc_dedup, "pr_auc_test_50": ap50,
             f"pr_auc_at_{pct}": ap_dep, f"brier_raw_at_{pct}": cal[f"brier_raw_at_{pct}"],
             f"brier_corrected_at_{pct}": cal[f"brier_corrected_at_{pct}"],
             "brier_raw_at_test_50": cal["brier_raw_at_test_50"],
             f"ece_corrected_at_{pct}": cal[f"ece_corrected_at_{pct}"]}
    for key in ("old", "new"):
        for mname, v in pm[key].items():
            if f"{mname}@{key}" in boot:
                point[f"{mname}@{key}"] = v

    src = {
        "roc_auc": REL("roc_curve.png"), "roc_auc_dedup": REL("final_metrics_table.json"),
        "pr_auc_test_50": REL("pr_curve.png"), f"pr_auc_at_{pct}": REL("pr_curve.png"),
        f"brier_raw_at_{pct}": REL("calibration.json"), f"brier_corrected_at_{pct}": REL("calibration.json"),
        "brier_raw_at_test_50": REL("calibration.json"), f"ece_corrected_at_{pct}": REL("calibration.json"),
    }
    table = []
    for k, v in point.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        if "@" in k:
            mname, key = k.split("@")
            thr = f"{thresholds[key][0]:.4f} raw" + ("" if key == "old" else f" / {t_cfg:.4f} deployed")
            if mname.startswith(("ppv_at", "npv_at", "flagged")):
                s = REL("ppv_collapse_table.csv")
            elif mname in ("sensitivity", "specificity", "cost_per_1000_test"):
                s = REL(f"confusion_matrix_{key}_{thresholds[key][0]:.4f}.txt")
            else:
                s = REL(f"classification_report_{key}_{thresholds[key][0]:.4f}.txt")
        else:
            mname, thr, s = k, "threshold-free", src[k]
        table.append({"metric": mname, "threshold_set": k.split("@")[1] if "@" in k else "-",
                      "value": float(v), "ci95_low": float(lo), "ci95_high": float(hi),
                      "threshold": thr, "evidence_file": s})
    extra = [
        {"metric": "threshold_raw_old (from y_test)", "value": T_OLD},
        {"metric": "threshold_raw_new (from train OOF)", "value": T_NEW},
        {"metric": f"threshold_deployed (pi={pi_dep})", "value": t_cfg},
        {"metric": "oof_auc_stacking_train", "value": oof["oof_auc_stacking"]},
        {"metric": "test_rows_duplicated_in_train_pct", "value": 100 * dup.mean()},
        {"metric": "roc_auc_drop_from_duplicates", "value": auc_all - auc_dedup},
        {"metric": "test_cost_change_pct_new_vs_old", "value": oof["test_cost_change_pct"]},
        {"metric": "decisions_changed_by_prevalence_correction", "value": n_decision_diff},
        {"metric": "n_test", "value": n}, {"metric": "n_train", "value": len(y_tr)},
    ]
    for e in extra:
        e.update({"threshold_set": "-", "ci95_low": np.nan, "ci95_high": np.nan,
                  "threshold": "-", "evidence_file": REL("oof_threshold.json")
                  if "threshold" in e["metric"] or "oof" in e["metric"] or "cost" in e["metric"]
                  else REL("final_metrics_table.json")})
    final = pd.DataFrame(table + extra)
    final.to_csv(out("final_metrics_table.csv"), index=False, float_format="%.6f")
    json.dump({"generated_by": "scratch/generate_diabetes_evidence.py", "bootstrap_resamples": N_BOOT,
               "bootstrap_seed": SEED, "ci": "percentile 2.5/97.5, resampling test rows",
               "rows": json.loads(final.to_json(orient="records"))},
              open(out("final_metrics_table.json"), "w"), indent=2)
    with open(out("final_metrics_table.md"), "w") as f:
        f.write("# Diabetes module — final numbers (single source)\n\n")
        f.write(f"Held-out test n={n}; bootstrap {N_BOOT} resamples (seed {SEED}), percentile 95% CI. "
                f"`*_at_{pct}` metrics use the test set re-weighted to prevalence {pi_dep}.\n\n")
        f.write(md_table(final))
        f.write("\n")

    with open(out("MANIFEST.txt"), "w") as f:
        f.write(f"generated by scratch/generate_diabetes_evidence.py in {time.time()-t0:.0f}s\n")
        f.write("inputs from scratch/diabetes_oof_threshold.py: oof_threshold.json, threshold_scan_oof.csv, "
                "oof_predictions_train.csv\n")
        for w in sorted(set(written)):
            f.write(w + "\n")
    print(final.to_string())
    print(json.dumps(cal, indent=2))
    print("decisions changed by correction:", n_decision_diff)
    print(f"done in {time.time()-t0:.0f}s; wrote {len(set(written))} files")


if __name__ == "__main__":
    main()
