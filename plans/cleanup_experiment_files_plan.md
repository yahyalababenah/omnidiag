# experiment_files/ Cleanup Plan

**Approved by user** — Keep the following, remove everything else.

## ✅ KEEP (9 items)

| # | Path | Reason |
|---|------|--------|
| 1 | `experiment_files/models/train_best_xgb.py` | Production XGBoost training script (Optuna Trial 83 params) |
| 2 | `experiment_files/models/experiments_log.md` | Documents A/B test results: heuristic 88.98% won over clinical 88.57% |
| 3 | `experiment_files/models/explain_xgb.py` | SHAP explainer for production model |
| 4 | `experiment_files/models/live_demo.py` | Live prediction + SHAP waterfall demo |
| 5 | `experiment_files/data_pipeline/clean_data.py` | Full preprocessing pipeline (imputation, encoding, scaling) |
| 6 | `experiment_files/data_pipeline/merge_data.py` | Merges heart.csv with Z-Alizadeh Sani dataset |
| 7 | `experiment_files/comprehensive_analysis_report.md` | 237-line analysis report with model performance docs |
| 8 | `experiment_files/production_audit_and_action_plan.md` | 306-line security audit and action plan |
| 9 | `experiment_files/plans/` | Planning documents (user requested to keep) |

## ❌ REMOVE (everything else)

### Scripts
- `experiment_files/scripts/finalize_project.py` — V2 legacy, trains on old heart.csv
- `experiment_files/scripts/make_notebooks.py` — V2 legacy, generates old notebooks

### V2 Models
- `experiment_files/models/train_model.py` — V2 TabNet training
- `experiment_files/models/train_xgboost.py` — V2 baseline XGBoost (superseded)
- `experiment_files/models/metrics.json` — V2 metrics (best_model: svm)
- `experiment_files/models/metadata.json` — V2 metadata (old column names)
- `experiment_files/models/grid_best.json` — V2 Random Forest grid search
- `experiment_files/models/optimize_xgboost.py` — Buggy (uses clinical not heuristic), superseded

### Failed Experiments
- `experiment_files/models/train_tabnet_heuristic.py` — Collapsed to 68.16% accuracy
- `experiment_files/models/ensemble_predict.py` — Degraded performance, missing dependency

### V2 Artifact Folders
- `experiment_files/models/heart_disease/` — Old model weights directory
- `experiment_files/models/preprocessors/` — Old preprocessor artifacts
- `experiment_files/models/tabnet_weights/` — TabNet weight files
- `experiment_files/models/xgboost_weights/` — Old XGBoost weights

### UI & Deployment
- `experiment_files/ui/app.py` — V2 Streamlit UI (replaced by React)
- `experiment_files/deployment/ngrok_setup.txt` — Empty file
- `experiment_files/deployment/README.md` — Placeholder

### Reports & Data (V2 duplicates)
- `experiment_files/reports/evaluation_metrics.txt` — V2 text dump
- `experiment_files/reports/figures/` — V2 figures (dendrogram, ROC curve)
- `experiment_files/data/` — Duplicate of root `data/heart_disease/`

### V2 Archive
- `experiment_files/v2_archive/` — Old notebooks, src code, CatBoost info

### Other
- `experiment_files/data_pipeline/harmonization.py` — Exploratory (superseded by merge_data.py)
- `experiment_files/digest.txt` — 2810-line aggregated text dump
- `experiment_files/Figure_1.png` — Standalone figure

## Execution Steps (for Code mode)

1. Remove all files/folders listed under ❌ REMOVE using `rm -rf` for directories and `rm` for individual files
2. Verify the kept files remain intact
3. Run `git status` to confirm only the intended deletions are staged
