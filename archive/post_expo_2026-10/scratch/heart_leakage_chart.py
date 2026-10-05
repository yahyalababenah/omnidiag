"""Leaked-vs-clean ROC-AUC chart for evaluation_evidence/heart/leakage_comparison.json.
Numbers sourced from that file. Run with the main .venv (matplotlib only needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yahia/Desktop/Projects/Heart_Disease_Project"
BLUE = "#4C78A8"
ORANGE = "#E45756"
BG = "#fafafa"

fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=140)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

labels = ["ROC-AUC\n(full-fit / nested LOSO)", "Held-out estimate\n(best CV / external Tehran)"]
leaked = [0.9571, 0.9186]
clean = [0.8130, 0.7634]

y = [1, 0]
h = 0.32
for yi, lk, cl, lab in zip(y, leaked, clean, labels):
    ax.barh(yi + h/2, lk, height=h, color=ORANGE, label="LEAKED (svm, ca/thal, pre-split fit)" if yi == 1 else None)
    ax.barh(yi - h/2, cl, height=h, color=BLUE, label="CLEAN (shipped xgboost, fold-isolated preprocessing)" if yi == 1 else None)
    ax.text(lk + 0.01, yi + h/2, f"{lk:.3f}", va="center", fontsize=9, color=ORANGE)
    ax.text(cl + 0.01, yi - h/2, f"{cl:.3f}", va="center", fontsize=9, color=BLUE)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel("ROC-AUC")
fig.suptitle("Heart disease model: leaked vs. clean evaluation", fontsize=12, y=0.97)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=1, fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25)

fig.text(0.5, 0.015,
         "LEAKED figures are not reproducible from the clean pipeline (different features, preprocessing fit before split).\n"
         "Kept only to document the leak found and fixed -- see evaluation_evidence/heart/leakage_comparison.json.",
         ha="center", fontsize=7.5, color="#666")

fig.tight_layout(rect=[0, 0.08, 1, 0.86])
out = f"{ROOT}/evaluation_evidence/heart/figures/leakage_comparison.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)
