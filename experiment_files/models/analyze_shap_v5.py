"""
OmniDiag v5.1 — SHAP Analysis for 16-Feature XGBoost Model
===========================================================
Verifies that ALL 5 engineered features have meaningful SHAP importance:
    - Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio (heuristic)
    - RPP, Exercise_Risk_Index (medical)

Note: Age_Bins and Global_Risk_Score were tested but did NOT improve
accuracy — removed per A/B diagnostic (diagnose_v5_drop.py).

Generates:
    1. Summary bar plot (global feature importance by mean |SHAP|)
    2. Beeswarm plot (feature effect distribution)
    3. Waterfall plot for a random positive patient
    4. SHAP importance ranking table (console output)

Usage:
    python experiment_files/models/analyze_shap_v5.py
"""

import pandas as pd
import pickle
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_loader import load_config, resolve_path
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_medical_features,
)

print("=" * 60)
print("🔬 OmniDiag v5.1 — SHAP Analysis")
print("=" * 60)

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_dir = os.path.dirname(resolve_path(cfg, "model", "weights_path"))
target_col = cfg["disease"]["target_column"]
model_path = os.path.join(model_dir, os.path.basename(cfg["model"]["weights_path"]))

# Output directory for SHAP plots
shap_plot_dir = os.path.join(model_dir, "shap_plots")
os.makedirs(shap_plot_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load data & apply ALL feature engineering (same as training pipeline)
# ------------------------------------------------------------------
data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
print(f"\n📂 Loading clean data from: {data_path}")
df = pd.read_csv(data_path)
print(f"   Raw shape: {df.shape}")

print("\n🧬 Applying feature engineering (2 paths)...")
df_fe = engineer_heuristic_features(df)   # Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio
df_fe = engineer_medical_features(df_fe)  # RPP, Exercise_Risk_Index

# Drop target (keep y for waterfall plot patient selection)
X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]
print(f"   Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"   Features: {list(X.columns)}")

# Verify no NaN
assert X.isnull().sum().sum() == 0, "❌ NaN values in SHAP input!"
print("   ✅ 0% missing values")

# ------------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------------
print(f"\n📂 Loading model from: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"   Model type: {type(model).__name__}")

# ------------------------------------------------------------------
# 3. Compute SHAP values
# ------------------------------------------------------------------
print("\n🔄 Computing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"   SHAP values shape: {shap_values.shape}")
print(f"   Base value: {explainer.expected_value:.4f}")

# ------------------------------------------------------------------
# 4. Compute mean |SHAP| importance ranking
# ------------------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\n" + "=" * 60)
print("📊 SHAP Feature Importance Ranking (by mean |SHAP|)")
print("=" * 60)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    marker = " ⭐" if row['feature'] in ['RPP', 'Exercise_Risk_Index', 'Age_BP_Interaction', 'HR_Age_Ratio', 'Chol_Age_Ratio'] else ""
    print(f"   {i:2d}. {row['feature']:25s}  {row['mean_abs_shap']:.6f}{marker}")

# ------------------------------------------------------------------
# 5. Check ALL engineered features specifically
# ------------------------------------------------------------------
engineered = [
    'Age_BP_Interaction', 'HR_Age_Ratio', 'Chol_Age_Ratio',
    'RPP', 'Exercise_Risk_Index',
]
print("\n" + "=" * 60)
print("🔍 Engineered Feature Importance Check")
print("=" * 60)
all_nonzero = True
for feat in engineered:
    val = importance_df.loc[importance_df['feature'] == feat, 'mean_abs_shap'].values
    if len(val) == 0:
        print(f"   ⚠️  {feat:25s}  NOT FOUND in SHAP output!")
        all_nonzero = False
    elif val[0] > 0:
        rank = importance_df[importance_df['feature'] == feat].index[0] + 1
        print(f"   ✅ {feat:25s}  mean |SHAP| = {val[0]:.6f}  (rank #{rank})")
    else:
        print(f"   ❌ {feat:25s}  mean |SHAP| = 0.000000  — ZERO IMPORTANCE!")
        all_nonzero = False

if all_nonzero:
    print("\n   ✅ All engineered features have non-zero SHAP importance!")
else:
    print("\n   ⚠️  Some engineered features have zero importance — check feature engineering pipeline.")

# ------------------------------------------------------------------
# 6. Generate SHAP plots
# ------------------------------------------------------------------
print("\n🎨 Generating SHAP plots...")

# 6a. Summary bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
bar_path = os.path.join(shap_plot_dir, "shap_summary_bar_v5.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Summary bar plot saved: {bar_path}")

# 6b. Beeswarm plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
beeswarm_path = os.path.join(shap_plot_dir, "shap_beeswarm_v5.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Beeswarm plot saved: {beeswarm_path}")

# 6c. Waterfall plot (first positive patient in test set)
# Pick a patient from the positive class for a meaningful waterfall
positive_indices = np.where(y == 1)[0]
if len(positive_indices) > 0:
    sample_idx = positive_indices[0]  # first positive patient
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=list(X.columns)
        ),
        show=False,
        max_display=16
    )
    plt.tight_layout()
    waterfall_path = os.path.join(shap_plot_dir, "shap_waterfall_v5.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Waterfall plot saved: {waterfall_path}")
else:
    print("   ⚠️  No positive samples found — skipping waterfall plot.")

# ------------------------------------------------------------------
# 7. Summary report
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("📋 SHAP Analysis Summary")
print("=" * 60)
print(f"   Model: XGBoost v5.1 (Optuna-optimized, 300 trials)")
print(f"   Features: {X.shape[1]}")
print(f"   Samples analyzed: {X.shape[0]}")
print(f"   Top 5 features by SHAP importance:")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"      {i}. {row['feature']} ({row['mean_abs_shap']:.6f})")
print(f"\n   Engineered features all active: {'✅ YES' if all_nonzero else '❌ NO — needs debugging'}")
print(f"   Plot directory: {shap_plot_dir}")
print("=" * 60)
