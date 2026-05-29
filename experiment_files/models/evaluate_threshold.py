"""
OmniDiag v5.1 — Decision Threshold Tuning
===========================================
Loads the trained XGBoost model and test data, then evaluates F1-Score,
Precision, and Recall across thresholds from 0.20 to 0.80 (step 0.05).

Identifies the optimal threshold that maximizes Recall while maintaining
acceptable Precision for clinical triage (early disease detection).

Usage:
    python experiment_files/models/evaluate_threshold.py
"""

import pandas as pd
import pickle
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
)

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_loader import load_config, resolve_path
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_medical_features,
)

print("=" * 60)
print("🎯 OmniDiag v5.1 — Decision Threshold Tuning")
print("=" * 60)

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_dir = os.path.dirname(resolve_path(cfg, "model", "weights_path"))
target_col = cfg["disease"]["target_column"]
model_path = os.path.join(model_dir, os.path.basename(cfg["model"]["weights_path"]))

# Output directory for plots
threshold_plot_dir = os.path.join(model_dir, "threshold_plots")
os.makedirs(threshold_plot_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load data & apply feature engineering
# ------------------------------------------------------------------
data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
print(f"\n📂 Loading clean data from: {data_path}")
df = pd.read_csv(data_path)

print("\n🧬 Applying feature engineering...")
df_fe = engineer_heuristic_features(df)
df_fe = engineer_medical_features(df_fe)

X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]
print(f"   Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

# ------------------------------------------------------------------
# 2. Train/test split (same as training)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️ Test set: {X_test.shape[0]} samples")
print(f"   Positive class: {(y_test == 1).sum()} ({((y_test == 1).sum() / len(y_test)) * 100:.0f}%)")
print(f"   Negative class: {(y_test == 0).sum()} ({((y_test == 0).sum() / len(y_test)) * 100:.0f}%)")

# ------------------------------------------------------------------
# 3. Load model
# ------------------------------------------------------------------
print(f"\n📂 Loading model from: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Get probability scores
y_proba = model.predict_proba(X_test)[:, 1]
print(f"   Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

# ------------------------------------------------------------------
# 4. Evaluate across thresholds (0.20 to 0.80, step 0.05)
# ------------------------------------------------------------------
thresholds = np.arange(0.20, 0.85, 0.05)
results = []

print("\n" + "=" * 60)
print("📊 Threshold Analysis (F1-Score, Precision, Recall)")
print("=" * 60)
print(f"   {'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
print(f"   {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*22}")

best_f1 = 0
best_threshold_f1 = 0.5

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    
    results.append({
        'threshold': thresh,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    })
    
    marker = " ◀ BEST" if f1 > best_f1 else ""
    if f1 > best_f1:
        best_f1 = f1
        best_threshold_f1 = thresh
    
    print(f"   {thresh:>8.2f}   | {precision:>8.4f}   | {recall:>8.4f}   | {f1:>8.4f}   | {tn:3d} {fp:3d} {fn:3d} {tp:3d}{marker}")

# ------------------------------------------------------------------
# 5. Find optimal threshold for clinical triage
#    Clinical triage prioritizes RECALL (don't miss sick patients)
#    while maintaining Precision >= 0.70 (don't overwhelm with false alarms)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔍 Optimal Threshold Search for Clinical Triage")
print("=" * 60)
print("   Criteria: Maximize Recall while Precision >= 0.70")

valid_thresholds = [r for r in results if r['precision'] >= 0.70]
if valid_thresholds:
    best_clinical = max(valid_thresholds, key=lambda r: r['recall'])
    print(f"\n   🏆 Optimal threshold: {best_clinical['threshold']:.2f}")
    print(f"      Precision: {best_clinical['precision']:.4f}")
    print(f"      Recall:    {best_clinical['recall']:.4f}")
    print(f"      F1-Score:  {best_clinical['f1']:.4f}")
    print(f"      Confusion Matrix: TN={best_clinical['tn']} FP={best_clinical['fp']} "
          f"FN={best_clinical['fn']} TP={best_clinical['tp']}")
else:
    print("   ⚠️  No threshold achieves Precision >= 0.70")
    best_clinical = max(results, key=lambda r: r['f1'])
    print(f"   📌 Falling back to best F1 threshold: {best_clinical['threshold']:.2f}")
    
print(f"\n📌 Best F1-score threshold: {best_threshold_f1:.2f} (F1={best_f1:.4f})")
print(f"📌 Default (0.50) threshold: F1={results[len(results)//2]['f1']:.4f}")

# ------------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------------
threshold_results_path = os.path.join(threshold_plot_dir, "threshold_results.json")
output = {
    "version": "5.1.0",
    "analysis_date": __import__('datetime').datetime.now().isoformat(),
    "test_samples": int(X_test.shape[0]),
    "thresholds": [
        {
            "threshold": r['threshold'],
            "precision": round(r['precision'], 6),
            "recall": round(r['recall'], 6),
            "f1": round(r['f1'], 6),
            "tn": r['tn'],
            "fp": r['fp'],
            "fn": r['fn'],
            "tp": r['tp'],
        }
        for r in results
    ],
    "best_f1_threshold": round(float(best_threshold_f1), 2),
    "best_f1_score": round(float(best_f1), 6),
    "clinical_triage_threshold": round(float(best_clinical['threshold']), 2),
    "clinical_triage_recall": round(float(best_clinical['recall']), 6),
    "clinical_triage_precision": round(float(best_clinical['precision']), 6),
}
with open(threshold_results_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n💾 Saved threshold results to: {threshold_results_path}")

# ------------------------------------------------------------------
# 7. Generate Precision-Recall curve plot
# ------------------------------------------------------------------
print("\n🎨 Generating Precision-Recall curve...")

precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, 'b-', linewidth=2, label=f'XGBoost (AP={avg_precision:.3f})')
plt.axvline(x=best_clinical['recall'], color='green', linestyle='--', alpha=0.7,
            label=f"Clinical triage (Recall={best_clinical['recall']:.2f})")
plt.axhline(y=best_clinical['precision'], color='green', linestyle='--', alpha=0.7,
            label=f"Clinical triage (Precision={best_clinical['precision']:.2f})")
plt.scatter([best_clinical['recall']], [best_clinical['precision']], 
            color='green', s=100, zorder=5)

# Mark default threshold
default_idx = np.argmin(np.abs(thresholds_pr - 0.50))
if default_idx < len(precision_curve):
    plt.scatter([recall_curve[default_idx]], [precision_curve[default_idx]], 
                color='red', s=100, zorder=5, label='Default (0.50)')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve — OmniDiag v5.1 XGBoost', fontsize=14)
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)

pr_curve_path = os.path.join(threshold_plot_dir, "precision_recall_curve_v5.1.png")
plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Precision-Recall curve saved: {pr_curve_path}")

# ------------------------------------------------------------------
# 8. F1-Score vs Threshold plot
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
threshold_vals = [r['threshold'] for r in results]
f1_vals = [r['f1'] for r in results]
precision_vals = [r['precision'] for r in results]
recall_vals = [r['recall'] for r in results]

plt.plot(threshold_vals, f1_vals, 'b-o', label='F1-Score', linewidth=2)
plt.plot(threshold_vals, precision_vals, 'r--s', label='Precision', linewidth=1.5)
plt.plot(threshold_vals, recall_vals, 'g--^', label='Recall', linewidth=1.5)
plt.axvline(x=best_clinical['threshold'], color='green', linestyle=':', alpha=0.7,
            label=f"Clinical triage ({best_clinical['threshold']:.2f})")
plt.axvline(x=0.50, color='red', linestyle=':', alpha=0.5, label='Default (0.50)')

plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('F1-Score / Precision / Recall vs Threshold — OmniDiag v5.1', fontsize=14)
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.xticks(thresholds, rotation=45)

f1_plot_path = os.path.join(threshold_plot_dir, "f1_vs_threshold_v5.1.png")
plt.savefig(f1_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ F1 vs Threshold plot saved: {f1_plot_path}")

# ------------------------------------------------------------------
# 9. Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("📋 Threshold Tuning Summary")
print("=" * 60)
print(f"   Average Precision (AP): {avg_precision:.4f}")
print(f"   Best F1 threshold:      {best_threshold_f1:.2f} (F1={best_f1:.4f})")
print(f"   Clinical triage threshold: {best_clinical['threshold']:.2f}")
print(f"      → Recall:    {best_clinical['recall']:.4f}")
print(f"      → Precision: {best_clinical['precision']:.4f}")
print(f"      → F1-Score:  {best_clinical['f1']:.4f}")
print(f"   Plots saved to: {threshold_plot_dir}")
print("=" * 60)
