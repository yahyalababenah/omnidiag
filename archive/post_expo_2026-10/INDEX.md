# Post-expo archive — 2026-10-05

Moved out of the working tree after the AI Expo (4–5 October 2026). Kept for
provenance, with history preserved (`git log --follow <new path>`). Nothing the
build or the running app reads lives here.

Every file keeps its original path under this folder:
`<original path>` → `archive/post_expo_2026-10/<original path>`.

Do not cite a number from these files without re-checking it against the
shipped code: several describe models that are no longer served.

## Archived

| Original path | Why archived |
|---|---|
| `AUDIT_REPORT.md` | Pre-evaluation technical audit (2026-09-15); a point-in-time snapshot |
| `WEAKNESS_REGISTER.md` | Pre-expo work list; the HM-*/P-* IDs cited in code comments resolve here. One entry was added after archiving (PART 11, NM-1, 2026-10-05). **Current findings are recorded in `docs/phase9/FINDINGS_REGISTER.md`** |
| `omnidiag_v2_roadmap.md` | June 2026 roadmap describing XGBoost + Flower |
| `plans/OMNIDIAG_TECHNICAL_DOCS.md` | Describes heart_full_tuned.pkl / XGBoost; the source of audit findings C-3, C-4, C-5 |
| `plans/omnidiag_architecture_overview.md` | "Validation Accuracy 88.98%" is wrong (audit C-5) |
| `plans/omnidiag_implementation_details.md` | Wrong CV figure and hyperparameters (audit C-5, C-7) |
| `plans/omnidiag_problem_analysis_report.md` | Emergency-medicine framing that contradicts the proposal (CRITERION_01 F7) |
| `plans/heart_disease_dataset_correction.md` | Completed plan |
| `plans/counterfactual_medical_constraints_plan.md` | Completed plan |
| `plans/omnidiag_feature_2.1_plan.md` | Completed plan |
| `plans/readme_overhaul_plan.md` | Completed plan |
| `plans/test_suite_errors_and_fixes.md` | Old fix log |
| `plans/testing_checklist.md` | Old checklist |
| `plans/testing_prompt_for_claude.md` | Old prompt |
| `scratch/build_metrics.py`, `scratch/regen_heart_evidence.py`, `scratch/heart_leakage_chart.py` | Generated heart evidence for the archived XGBoost model; `evaluation_evidence/` still names the old `scratch/` paths in `generated_by` (left unedited on purpose) |
| `scratch/_diabetes_common.py`, `scratch/diabetes_oof_threshold.py`, `scratch/generate_diabetes_evidence.py`, `scratch/prevalence_experiment.py`, `scratch/audit_diabetes.py`, `scratch/eval_diabetes.py` | BRFSS diabetes evidence and audit scripts (cited by `docs/DIABETES_AUDIT_REPORT.md` and `evaluation_evidence/diabetes/`) |
| `scratch/eval_heart.py`, `scratch/check_bc.py`, `scratch/check_c_cold.py`, `scratch/inspect_pkls.py`, `scratch/latency.py`, `scratch/latency2.py`, `scratch/profile_cf.py`, `scratch/cf_shapes.py`, `scratch/shap_additivity.py`, `scratch/demo_cases.py` | One-off investigations |
| `scratch/golden_*.json` (5), `scratch/cf_raw.json`, `scratch/latency_results.json` | Outputs of the scripts above |
| `models/metrics.json` | v5.1 XGBoost metrics, 16 features; read by no code |
| `models/train_diabetes_ensemble.py`, `models/train_diabetes_lgb.py`, `models/train_diabetes_xgb.py` | BRFSS training scripts; no runtime consumer. They import `experiment_files.data_pipeline.preprocess_diabetes`, which is no longer tracked (still on the author's disk) |
| `models/retrain_rf_sklearn_180.py` | No references |
| `models/advanced_feature_engineering.py` | XGBoost-era feature functions; only importer was `features/heart_disease_features.py` |
| `features/heart_disease_features.py` | `HeartDiseaseFeatureEngineer`: no importer, no config `features.module` names it (only the git-ignored `configs/heart_disease.yaml.bak`, which the router does not load) |
| `backend/federated/__init__.py`, `client.py`, `aggregator.py` | Flower/FedAvg prototype, superseded; no importer. The GLORE replacement is documented in `docs/fl_report/` |
| `scripts/simulate_federated.py` | Flower simulation; no references |
| `docs/OmniDiag_Proposal_Defense.md` | Expo proposal defence; presents BRFSS as the diabetes module (state of 2026-09-26) |
| `docs/CRITERION_01_AUDIT.md` | Pre-expo proposal audit |
| `docs/code_cleanup_report.md` | 2026-06-28 report on another branch |
| `configs/diabetes.yaml` | BRFSS diabetes module retired (gate B3, 2026-10-05). Its absence from `configs/` is what unregisters it; `backend/retired_diseases.py` answers 410 for it. Kept for the numbers the docs cite: prevalence_train / prevalence_deploy, inference_threshold 0.108184 |
| `configs/diabetes.yaml.kaggle` | Orphan variant of the same config; never loaded (the router reads `.yaml`/`.yml` only) |

## Moved elsewhere (not archived)

| Original path | New path | Why |
|---|---|---|
| `scratch/verify_live.py` | `scripts/verify_live.py` | Live-Space check referenced by `requirements.txt` |
| `scratch/golden_master.py` | `scripts/golden_master.py` | Golden-master capture/diff tool |
| `scratch/ui_verify.py` | `scripts/ui_verify.py` | Playwright UI check (docs/FEATURE_VERIFICATION.md) |
| `scratch/verify_randomize.mjs` | `scripts/verify_randomize.mjs` | Randomize-button check (docs/FEATURE_VERIFICATION.md) |
| `experiment_files/data_pipeline/build_uci_sites.py` | `scripts/data/build_uci_sites.py` | Produces `data/heart_disease/processed/uci_heart_by_site.csv`, the shipped heart model's training data |
| `experiment_files/data_pipeline/translate_zalizadeh.py` | `scripts/data/translate_zalizadeh.py` | Produces the Tehran external-validation CSV and mapping |

## Removed from git

| Path | How |
|---|---|
| `diagnose_features.py` | `git rm`; it loaded heart files that no longer exist. Recoverable from history |
| `experiment_files/` (14 remaining files) | `git rm --cached`; already in `.gitignore`, still on the author's disk |
