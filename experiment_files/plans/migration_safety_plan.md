# Migration Safety Plan — OmniDiag Multi-Disease Refactoring

## 1. File Inventory & Destination

Below is the exact mapping of every existing file to its new location. **No file is left behind.**

### Data Files (`data/`)

| Current Path | Destination Path | Action |
|---|---|---|
| `data/raw/heart.csv` | `data/heart_disease/raw/heart.csv` | COPY |
| `data/raw/Z-Alizadeh sani dataset.xlsx` | `data/heart_disease/raw/Z-Alizadeh sani dataset.xlsx` | COPY |
| `data/processed/final_ready_data.csv` | `data/heart_disease/processed/final_ready_data.csv` | COPY |
| `data/processed/data_heuristic.csv` | `data/heart_disease/processed/data_heuristic.csv` | COPY |
| `data/processed/data_clinical.csv` | `data/heart_disease/processed/data_clinical.csv` | COPY |
| `data/processed/merged_heart_data.csv` | `data/heart_disease/processed/merged_heart_data.csv` | COPY |
| `data/processed/.gitkeep` | `data/heart_disease/processed/.gitkeep` | COPY |
| `data/interim/cv_scores.csv` | `data/heart_disease/interim/cv_scores.csv` | COPY |
| `data/interim/gridsearch_rf.csv` | `data/heart_disease/interim/gridsearch_rf.csv` | COPY |
| `data/interim/kmeans_summary.csv` | `data/heart_disease/interim/kmeans_summary.csv` | COPY |
| `data/interim/supervised_metrics.csv` | `data/heart_disease/interim/supervised_metrics.csv` | COPY |

### Model Files (`models/`)

| Current Path | Destination Path | Action |
|---|---|---|
| `models/omni_diag_xgb_optimized.pkl` | `models/heart_disease/omni_diag_xgb_optimized.pkl` | COPY |
| `models/final_model.pkl` | `models/heart_disease/final_model.pkl` | COPY |
| `models/metadata.json` | `models/heart_disease/metadata.json` | COPY |
| `models/metrics.json` | `models/heart_disease/metrics.json` | COPY |
| `models/grid_best.json` | `models/heart_disease/grid_best.json` | COPY |
| `models/preprocessors/label_encoders.pkl` | `models/heart_disease/preprocessors/label_encoders.pkl` | COPY |
| `models/preprocessors/standard_scaler.pkl` | `models/heart_disease/preprocessors/standard_scaler.pkl` | COPY |
| `models/xgboost_weights/omni_diag_xgb_optimized.pkl` | `models/heart_disease/xgboost_weights/omni_diag_xgb_optimized.pkl` | COPY |
| `models/xgboost_weights/omni_diag_xgb.pkl` | `models/heart_disease/xgboost_weights/omni_diag_xgb.pkl` | COPY |
| `models/tabnet_weights/omni_diag_model.zip` | `models/heart_disease/tabnet_weights/omni_diag_model.zip` | COPY |
| `models/tabnet_weights/omni_diag_tabnet_heuristic.zip` | `models/heart_disease/tabnet_weights/omni_diag_tabnet_heuristic.zip` | COPY |

### Scripts (remain in place, only modified)

| Current Path | Status |
|---|---|
| `models/advanced_feature_engineering.py` | MODIFIED — delegates to `HeartDiseaseFeatureEngineer` |
| `models/train_best_xgb.py` | MODIFIED — reads params from config |
| `models/explain_xgb.py` | MODIFIED — uses config-driven paths |
| `models/live_demo.py` | MODIFIED — uses config + dynamic loader |
| `models/optimize_xgboost.py` | MODIFIED — uses config paths |
| `models/ensemble_predict.py` | RETAINED — untouched (archived module) |
| `models/train_xgboost.py` | RETAINED — untouched (baseline trainer) |
| `models/train_tabnet_heuristic.py` | RETAINED — untouched (archived module) |
| `models/train_model.py` | RETAINED — untouched |
| `models/experiments_log.md` | RETAINED — untouched |
| `data_pipeline/clean_data.py` | MODIFIED — paths updated to `data/heart_disease/` |
| `data_pipeline/merge_data.py` | MODIFIED — paths updated to `data/heart_disease/` |
| `data_pipeline/harmonization.py` | RETAINED — untouched |
| `backend/main.py` | MODIFIED — uses dynamic `OmniDiagRouter` |
| `ui/app.py` | RETAINED — untouched (V2 UI, separate concern) |

---

## 2. Migration Logic: COPY, Never MOVE

**Every single file operation will be a COPY, not a MOVE.**

This means:
- The original file at its old path remains **completely untouched** during the copy phase.
- A duplicate is created at the new path.
- Only after **you explicitly confirm** that the new structure works correctly will we consider cleaning up old paths.

**There is no `mv` (move) command. Only `cp` (copy) and `mkdir -p` (create directories).**

This guarantees that even if something goes wrong, your original `data/processed/`, `models/`, etc. are fully intact and the project continues to work exactly as it does right now.

---

## 3. Risk Mitigation: Instant Rollback

If the new architecture fails during testing, recovery is instantaneous:

### Recovery Plan A: Revert Script Changes (30 seconds)
```bash
git checkout -- models/advanced_feature_engineering.py
git checkout -- models/train_best_xgb.py
git checkout -- models/explain_xgb.py
git checkout -- models/live_demo.py
git checkout -- models/optimize_xgboost.py
git checkout -- data_pipeline/clean_data.py
git checkout -- data_pipeline/merge_data.py
git checkout -- backend/main.py
```
This restores all modified scripts to their current state.

### Recovery Plan B: Delete New Structure (10 seconds)
```bash
rm -rf configs/ features/ backend/router.py backend/model_loader.py backend/schemas.py
```
The new directories are additive — removing them returns the project to exactly its current state.

### Recovery Plan C: Full Rollback (if old files were ever deleted)
Since we are using **COPY not MOVE**, the old files still exist at their original paths. Simply revert the script changes (Plan A) and the project works exactly as before.

---

## 4. Cleanup Policy: Zero Deletions Without Confirmation

**I will not delete any of your existing files during this refactoring — period.**

The only files that will be deleted are:
- **None.** Zero. Zilch.

The new architecture is designed to be **additive**. The old `data/processed/`, `models/`, etc. directories remain fully intact. The modified scripts will be updated to point to the new paths, but the old files still exist at their original locations.

If, after full testing and your approval, you want to eventually clean up the old duplicate files, that would be a separate, optional step that you explicitly authorize.

---

## 5. Confirmation Protocol: Step-by-Step

Here is the exact sequence of operations with confirmation gates:

### Phase 1: Create New Structure (No data touched)
```
Step 1.1: Create configs/ + heart_disease.yaml
  → YOU CONFIRM: config file looks correct

Step 1.2: Create features/ package (__init__.py, base_features.py, heart_disease_features.py)
  → YOU CONFIRM: feature engineer code is correct

Step 1.3: Create backend/ package files (router.py, model_loader.py, schemas.py)
  → YOU CONFIRM: router code is correct
```

### Phase 2: Copy Data (Original data untouched)
```
Step 2.1: Create data/heart_disease/raw/ + copy raw files
  → YOU CONFIRM: files exist in both old and new locations

Step 2.2: Create data/heart_disease/processed/ + copy processed files
  → YOU CONFIRM: files exist in both old and new locations

Step 2.3: Create data/heart_disease/interim/ + copy interim files
  → YOU CONFIRM: files exist in both old and new locations
```

### Phase 3: Copy Models (Original models untouched)
```
Step 3.1: Create models/heart_disease/ + copy all model artifacts
  → YOU CONFIRM: files exist in both old and new locations
```

### Phase 4: Update Scripts
```
Step 4.1: Modify data_pipeline scripts to use new paths
  → YOU CONFIRM: pipeline runs successfully on new paths

Step 4.2: Modify model scripts to use config-driven paths
  → YOU CONFIRM: model loads and predicts correctly

Step 4.3: Modify backend/main.py to use dynamic router
  → YOU CONFIRM: API responds correctly
```

### Phase 5: Verification
```
Step 5.1: Run full data pipeline → verify output matches old pipeline
Step 5.2: Load model from new path → verify prediction matches old model
Step 5.3: Start FastAPI → hit /api/v4/heart_disease/predict → verify response
Step 5.4: Run SHAP explainer → verify plots are generated
  → YOU CONFIRM: everything works
```

**Only after all 5 phases are confirmed by you would we even discuss cleanup of old files — and that would be a separate, optional task.**

---

## Summary

| Concern | Answer |
|---------|--------|
| **Will files be moved or copied?** | **COPIED.** Originals remain untouched. |
| **Will any files be deleted?** | **NO.** Not a single file. |
| **Can I recover instantly?** | **YES.** `git checkout` reverts scripts; old paths still have all data. |
| **What if the new structure fails?** | Delete the new `configs/`, `features/`, `backend/router.py` — project is back to original. |
| **Do I get to approve each phase?** | **YES.** You confirm each step before the next begins. |
