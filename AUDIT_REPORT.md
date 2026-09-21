# OmniDiag — Pre-Evaluation Technical Audit

**Scope:** verification only. No application code was modified, nothing was committed.
**Date:** 2026-09-15 · **Branch:** `deploy/v2-platform` @ `698974b`
**Method:** every finding below was produced by reading the file or by running the code.
Evaluation scripts live in `scratch/` (untracked); their raw output is in `evaluation_evidence/`.

**Environment used for all runs:** `backend/.venv` (Python 3.13.5, scikit-learn 1.9.0,
xgboost 3.3.0, lightgbm 4.6.0, shap 0.52.0), backend served locally on `127.0.0.1:8901`
with `DATABASE_URL=sqlite+aiosqlite:///./omnidiag_dev.db`.

---

## Verdict at a glance

| Claimed metric | Verified? | Measured here |
|---|---|---|
| Heart accuracy 80.17% | ✅ **OK** — exact | 80.17% |
| Heart ROC-AUC 0.856 | ✅ **OK** — exact | 0.8564 |
| Heart sensitivity 80.46% | ❌ **CRITICAL** — not reproducible | 71.11% (production orientation) |
| Heart specificity 83.13% | ❌ **CRITICAL** — not reproducible | 85.53% (production orientation) |
| Heart threshold 0.420 | ❌ **CRITICAL** — does not exist in code | production uses argmax (0.5) |
| Diabetes accuracy 75.18% | ❌ **CRITICAL** — measured at the wrong threshold | 73.12% at the production 0.275 |
| Diabetes ROC-AUC 0.831 | ✅ **OK** | 0.8305 |
| Diabetes sensitivity 91.70% | ✅ **OK** | 91.55% |
| Diabetes specificity 54.63% | ✅ **OK** | 54.68% |

**16 CRITICAL · 14 WARNING.** The two that will end the evaluation if a judge finds them first
are **C-1** (a tracked artifact says the best model is an SVM with ROC-AUC 0.957) and
**C-8** (the heart-disease scaler and imputer are fit on the full dataset before the split).

---

# AUDIT 1 — Stale or contradictory metric artifacts

## Full inventory of files containing performance numbers

| File | Tracked? | Numbers it contains | Matches claim? |
|---|---|---|---|
| `models/metrics.json` | ✅ | acc 0.801653, AUC 0.856433, CM {tn 32, fp 13, fn 11, tp 65} | ✅ acc + AUC |
| `models/heart_disease/metrics.json` | ✅ | AUC 0.957, **best_model "svm"**, cv 0.9186 | ❌ **CRITICAL** |
| `models/heart_disease/metadata.json` | ✅ | **best_model "svm"**, feature list incl. **`ca`, `thal`** | ❌ **CRITICAL** |
| `models/heart_disease/grid_best.json` | ✅ | best_score 0.8182, 16 features, v5.0 params | ⚠️ 0.8182 ≠ 0.8017 |
| `models/heart_disease/diagnostic_v5_ab.json` | ✅ | acc 0.801653, AUC 0.857018, **18 features** | ⚠️ 18 ≠ 16 features |
| `models/heart_disease/threshold_plots/threshold_results.json` | ✅ | best_f1_threshold **0.3**, triage 0.2; grid 0.20–0.80 step 0.05 | ❌ no 0.420 |
| `models/diabetes/ensemble_metrics.json` | ✅ | acc 0.75182, AUC 0.83108, sens 0.91696, spec 0.54625, thr 0.275 | ⚠️ internally inconsistent (see C-6) |
| `models/diabetes/metrics.json` | ✅ | acc 0.75175, AUC 0.83096, recall 0.79544 | ⚠️ single-XGB, near-identical numbers |
| `models/diabetes/metrics_lgb.json` | ✅ | acc 0.75140, AUC 0.82978 | ⚠️ unlabelled provenance |
| `plans/OMNIDIAG_TECHNICAL_DOCS.md` | ✅ | **the source of every claimed number** | ❌ **CRITICAL** (C-3, C-4, C-5) |
| `plans/omnidiag_architecture_overview.md:74` | ✅ | "Validation Accuracy 88.98%" | ❌ **CRITICAL** (C-5) |
| `plans/omnidiag_implementation_details.md:24,221` | ✅ | "88.98% 5-fold CV"; params 898 / lr 0.0136 | ❌ **CRITICAL** (C-5, W-1) |
| `plans/omnidiag_problem_analysis_report.md:220,522` | ✅ | 91.7% / 54.6% | ✅ consistent |
| `experiment_files/models/experiments_log.md` | ✅ | TabNet 87.35%, XGB 88.16%, Optuna 88.98% | ⚠️ pre-production era |
| `omnidiag_diabetes_artifacts/.../ensemble_metrics.json` | ignored | byte-identical duplicate | ⚠️ duplicate |
| `README.md` | ✅ | **no performance numbers at all** | ✅ **OK** |

`configs/*.yaml`, `k8s/*`, `helm/*`, `deploy/*`, `.github/workflows/ci.yml`, `frontend/*.json`,
`docs/*.md`, `omnidiag_v2_roadmap.md`, `plans/heart_disease_dataset_correction.md` and the
diabetes notebook contain **no** model performance claims. `configs/heart_disease.yaml.bak`,
`configs/diabetes.yaml.kaggle` and the `.py.bak` files are clean of SVM / `ca` / `thal`.

---

### C-1 — CRITICAL — `models/heart_disease/metrics.json` claims an SVM at ROC-AUC 0.957

This is the file the audit brief asked about specifically. **Entire contents:**

```json
{
  "roc_auc_full_fit": 0.9570601851851852,
  "best_model": "svm",
  "best_cv_roc_auc": 0.9185570987654321
}
```
`models/heart_disease/metrics.json:1-5`

- **Named a model type that is not in production.** Production is a single `XGBClassifier`
  (`configs/heart_disease.yaml:83`, verified by loading the pkl).
- **Numbers contradict the claim by ~10 AUC points.** 0.957 vs the claimed 0.856.
- `roc_auc_full_fit` is, by its own field name, a model scored on the data it was fit on.

**Is it loaded / served / merely present?** — **Merely PRESENT, and TRACKED.**
Verified three ways:
1. No code reference. `grep -rn "metrics\.json|metadata\.json"` across `backend/`, `frontend/`,
   `configs/` returns only `models/train_diabetes_xgb.py:205` and `scripts/retrain.py:66,119`
   (both training-side writers, neither on the request path).
2. No static file serving. `grep -rn "StaticFiles|FileResponse|mount\("` over `backend/` → **zero hits**.
3. No endpoint exposes it. Every route was enumerated; `/api/v4/diseases` →
   `router.get_disease_info()` (`backend/router.py:71-101`) reads only the YAML config, and
   `GET /metrics` (`backend/main.py:280-286`) is the Prometheus scrape endpoint, unrelated.

So it cannot corrupt a prediction — but it is in the repo a judge will clone, committed
`2026-05-25` (`bede19b`), and it directly contradicts the number you intend to state publicly.

### C-2 — CRITICAL — `models/heart_disease/metadata.json` carries the leaked `ca` / `thal` features **and** "svm"

```json
  "all_columns": [ "age","sex","cp","trestbps","chol","fbs","restecg",
                   "thalach","exang","oldpeak","slope","ca","thal" ],
  "target": "target",
  "best_model": "svm",
  "best_cv_roc_auc": 0.9185570987654321
```
`models/heart_disease/metadata.json:19-34` (`ca` at :14 and :30, `thal` at :15 and :31)

This is the **only tracked JSON/YAML/py/jsx file in the repo that names `ca` or `thal`.** It is
the old Cleveland-schema metadata (lowercase `age`/`cp`/`thalach`, target `target`) left behind
after the dataset correction. Same verification as C-1: **merely PRESENT, tracked, not loaded,
not served.** The only other `thal*` hit anywhere is prose in
`plans/OMNIDIAG_TECHNICAL_DOCS.md:469`.

### C-3 — CRITICAL — the claimed heart sensitivity / specificity / threshold have a single, unverifiable source

Every claimed heart number that is **not** accuracy or AUC traces to one document:

```
| Precision | 0.8391 |
| Recall | 0.8046 |
| F1-Score | 0.8215 |
| Optimal Threshold | 0.420 |
```
`plans/OMNIDIAG_TECHNICAL_DOCS.md:411-414`

```
| Sensitivity | 80.46% | 91.70% |
| Specificity | 83.13% | 58.70% |
| Inference Threshold | 0.420 | 0.275 |
```
`plans/OMNIDIAG_TECHNICAL_DOCS.md:965-967`

These numbers appear in **no** model artifact, **no** config, and **no** source file. See
AUDIT 4 for the reproduction attempt — none of the four possible orientations produces them.

### C-4 — CRITICAL — that document describes the *wrong dataset*, including `ca` and `thal`, as the production input

```
**Dataset:** Cleveland Heart Disease (UCI Repository), processed via transform_heart11.py
**Input features:** 12 base clinical features (age, sex, chest pain type, resting BP,
cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina,
ST depression, slope, num major vessels, thalassemia)
```
`plans/OMNIDIAG_TECHNICAL_DOCS.md:398-399`

"num major vessels" is `ca`; "thalassemia" is `thal`. Production trains on
`data/heart_disease/processed/final_ready_data.csv` (605 rows, 11 base features, **no `ca`/`thal`** —
see AUDIT 2). The doc also states "Training Samples ~303" (`:970`) where production uses 605.

### C-5 — CRITICAL — "CV Accuracy 88.98%" is attributed to a file that does not contain it

```
**Performance metrics ([`models/metrics.json`](models/metrics.json)):**
| CV Accuracy (mean) | 88.98% |
| CV Accuracy (std) | 0.0169 |
```
`plans/OMNIDIAG_TECHNICAL_DOCS.md:403,409-410`

`models/metrics.json` has **no** cross-validation field of any kind (full contents dumped in
AUDIT 5). 88.98% is the pre-production Optuna result from
`experiment_files/models/experiments_log.md:15`, measured on the old dataset. It is repeated as
current in `plans/omnidiag_architecture_overview.md:74` and
`plans/omnidiag_implementation_details.md:24,221`. A judge who opens `models/metrics.json`
because the doc told them to will find 80.17% and a missing CV block.

### C-6 — CRITICAL — `ensemble_metrics.json` mixes two thresholds in one metric block

```json
  "stacking_ensemble": {
    "test_accuracy": 0.7518212037626424,
    "test_roc_auc": 0.8310834223894875,
    "optimal_threshold": 0.27499999999999997,
    "sensitivity_at_optimal": 0.9169613806761918,
    "specificity_at_optimal": 0.5462517680339463,
```
`models/diabetes/ensemble_metrics.json:7-11`

`test_accuracy` is computed at argmax 0.5; `sensitivity_at_optimal`/`specificity_at_optimal` at
0.275. Quoting all three together is arithmetically impossible on a balanced test set:
`(91.70 + 54.63) / 2 = 73.17%`, not 75.18%. Measured accuracy at 0.275 is **73.12%** (AUDIT 4).

### W-1 — WARNING — `grid_best.json` and `diagnostic_v5_ab.json` disagree with the shipped model

- `grid_best.json:7` `"best_score": 0.8182` — 1.65 pp above the 0.8017 you claim.
  Its own note (`:8`) explains it is a v5.0 baseline, but the field name reads as a result.
- `diagnostic_v5_ab.json:5-24` lists **18** features (adds `Age_Bins`, `Global_Risk_Score`) and
  AUC 0.857018. The shipped model has **16** features. It is an A/B artifact presented like a result.

### W-2 — WARNING — three near-identical diabetes metric files invite a "which one is it?" question

`models/diabetes/metrics.json` (0.75175 / 0.83096), `metrics_lgb.json` (0.75140 / 0.82978) and
`ensemble_metrics.json` (0.75182 / 0.83108) all sit in the same directory, all tracked, none
labelled with which model produced it. The differences are in the third decimal place.

---

# AUDIT 2 — Data leakage verification

## Is `ca` / `thal` present anywhere on the production path? — **NO. Verified clean.**

| Layer | Source of truth | `ca` / `thal`? |
|---|---|---|
| YAML config | `configs/heart_disease.yaml:39-50` | ✅ absent |
| Pydantic request schema | `backend/schemas.py:45-55` (`HeartDiseaseInput`) | ✅ absent |
| Feature engineering | `models/advanced_feature_engineering.py:17-29, 157-195` | ✅ absent |
| Trained `.pkl` feature order | `omni_diag_xgb_optimized.pkl` → `feature_names_in_` | ✅ absent |
| Training CSV | `data/heart_disease/processed/final_ready_data.csv` header | ✅ absent |

**The exact production feature list — source of truth is the pkl's `feature_names_in_`
(16 features, order-significant), read directly off disk:**

```
 1 Age                  7 RestingECG          13 HR_Age_Ratio
 2 Sex                  8 MaxHR               14 Chol_Age_Ratio
 3 ChestPainType        9 ExerciseAngina      15 RPP
 4 RestingBP           10 Oldpeak             16 Exercise_Risk_Index
 5 Cholesterol         11 ST_Slope
 6 FastingBS           12 Age_BP_Interaction
```

`booster.feature_names` agrees exactly. `models/metrics.json:5-22` and `grid_best.json:11-28`
list the same 16 in the same order. The 11 base features match `HeartDiseaseInput`
(`backend/schemas.py:45-55`) one-for-one; the 5 engineered ones are appended by
`_engineer_features()`.

> ⚠️ Naming nit: `configs/heart_disease.yaml:13` and `backend/schemas.py:8` both say
> **"12 clinical features"**. The schema accepts **11**. Cosmetic, but a judge counting fields
> will notice.

The old leaked dataset **is still on disk** at `data/heart_disease/raw/heart11.csv`
(header: `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target`).
It is not referenced by `configs/heart_disease.yaml:24-26`, which lists only `heart.csv` and
`Z-Alizadeh sani dataset.xlsx`. Present but unused — see W-3.

## Training pipeline

- **Dataset:** `data/heart_disease/processed/final_ready_data.csv` — 605 rows × 12 columns.
- **Training script:** `experiment_files/models/train_v5_xgb.py`.
- **Split** (`experiment_files/models/train_v5_xgb.py:140-142`):
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  ```
  `sklearn.model_selection.train_test_split`, `random_state=42`, **stratified** on the target.
  → 484 train / 121 test. Reproduced exactly in AUDIT 4.
- **Target-derived features:** ✅ **none.** All five engineered features are pure functions of
  input columns (`models/advanced_feature_engineering.py:24-28, 177, 191`); the target is
  dropped at `train_v5_xgb.py:122` before any of them is touched.

### C-8 — CRITICAL — the heart-disease scaler **and** imputer are fit on the full dataset before the split

`experiment_files/data_pipeline/clean_data.py` builds `final_ready_data.csv` from all 605 rows
**before** `train_v5_xgb.py` ever splits it.

```python
 44     imputed_values = imputer.fit_transform(impute_data)
```
`experiment_files/data_pipeline/clean_data.py:44` — an `IterativeImputer` wrapping a
`RandomForestRegressor` (`:41-42`), **fit on all 605 rows** across
`['Age','Sex_Num','RestingBP','MaxHR','Cholesterol']` (`:38`).

```python
 76     scaler = StandardScaler()
 77     df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```
`experiment_files/data_pipeline/clean_data.py:76-77` — `fit_transform` on **all 605 rows**.

```python
 60     for col in categorical_cols:
 61         le = LabelEncoder()
 62         df[col] = le.fit_transform(df[col])
```
`experiment_files/data_pipeline/clean_data.py:60-62` — encoders also fit on the full set (lower
severity: no target information).

The saved CSV is then split at `train_v5_xgb.py:140-142`. **Test-set statistics — means,
standard deviations, and a full RandomForest imputation model — are baked into the training
features.** The reported 80.17% / 0.856 are therefore optimistically biased by an unmeasured
amount. This is the single most likely question from a competent judge, and the pipeline
docstring does not acknowledge it.

**Contrast — diabetes gets this right**, which makes the heart pipeline harder to defend:
```python
245     log.info("Fitting StandardScaler on TRAINING set continuous features...")
246     scaler = StandardScaler()
247     X_train_cont_scaled = scaler.fit_transform(X_train[continuous_features])
249     log.info("Transforming TEST set with TRAINING-derived scaler...")
250     X_test_cont_scaled = scaler.transform(X_test[continuous_features])
```
`experiment_files/data_pipeline/preprocess_diabetes.py:245-250`, split at `:216-222`.
I verified the shipped `models/diabetes/preprocessors/standard_scaler.pkl` **is** the
train-only-fitted scaler: refitting on the reproduced train split gives
`max |mean_ delta| = 0.000e+00`, `max |scale_ delta| = 0.000e+00`
(`evaluation_evidence/diabetes_report.txt`). ✅ **OK, no diabetes leakage.**

### W-3 — WARNING — `eval_set` is the test set

```python
153     model.fit(
154         X_train, y_train,
155         eval_set=[(X_test, y_test)],
156         verbose=False
157     )
```
`experiment_files/models/train_v5_xgb.py:153-157`

No `early_stopping_rounds` is set, so this does not currently alter the fitted model — it only
logs. But handing the held-out test set to `fit()` reads as leakage on sight, and would become
real leakage the moment anyone adds early stopping.

### W-4 — WARNING — the leaked-feature dataset is still committed

`data/heart_disease/raw/heart11.csv` (1025 rows, with `ca` and `thal`) remains tracked. Unused
by the production config, but it is the dataset `plans/OMNIDIAG_TECHNICAL_DOCS.md:398`
incorrectly describes as production.

---

# AUDIT 3 — Model loading correctness

## Config → router → files actually opened on disk

**Routing:** `backend/router.py:236-241` selects the loader by the presence of `model.ensemble`:
`configs/diabetes.yaml:103` has it → `EnsembleModelLoader`; `configs/heart_disease.yaml` does
not → `ModelLoader`. Loader objects are built at **import time**; model bytes load **lazily**
on first request.

**Heart disease** — `configs/heart_disease.yaml:86` → `ModelLoader._resolve_weights_path()`
(`backend/model_loader.py:535-571`) → opens:
| File | Loaded | Verified contents |
|---|---|---|
| `models/heart_disease/omni_diag_xgb_optimized.pkl` | `model_loader.py:75` | `XGBClassifier`, 16 features |
| `models/heart_disease/preprocessors/label_encoders.pkl` | `model_loader.py:186` | dict of 5 `LabelEncoder` |
| `models/heart_disease/preprocessors/standard_scaler.pkl` | `model_loader.py:186` | `StandardScaler` |

The preprocessor loader globs `*.pkl` in the directory (`model_loader.py:182-186`). The
`.bak.v4.pkl` backups sitting there are excluded only because `filename.endswith(".pkl")` is
checked against keys — `standard_scaler.bak.v4` lands under a distinct key and is never read.
Fragile but currently harmless.

**Diabetes** — `configs/diabetes.yaml:103-117` → `EnsembleModelLoader.base_models`
(`backend/ensemble_loader.py:97-118`):
| Role | File loaded | Verified |
|---|---|---|
| XGBoost | `models/diabetes/omni_diag_xgb_optimized.pkl` | ✅ `XGBClassifier`, 26 feats |
| LightGBM | `models/diabetes/omni_diag_lgb_optimized.pkl` | ✅ `LGBMClassifier`, 26 feats |
| RandomForest | `models/diabetes/omni_diag_rf.pkl` | ✅ `RandomForestClassifier`, 26 feats |
| Meta-learner | `models/diabetes/meta_learner.pkl` | ✅ `LogisticRegression`, 3 inputs |

✅ **OK — the optimized artifacts are loaded, not the generic ones.** The older
`xgb_model.pkl` / `lgb_model.pkl` / `rf_model.pkl` / `voting_ensemble.pkl` are present but
unreferenced except `voting_ensemble.pkl` as `fallback_weights_path` (`configs/diabetes.yaml:89`),
which `EnsembleModelLoader` never consults. Confirmed materially different: the generic
`xgb_model.pkl` has a **different column order** (`...CholCheck, Smoker...` vs
`...CholCheck, BMI, Smoker...`) — loading it would silently misalign features.

## Silent fallback on load failure — **none for model loading**

✅ Base model missing → `FileNotFoundError` raised, `backend/ensemble_loader.py:109-112`.
✅ Meta-learner missing → `FileNotFoundError` raised, `backend/ensemble_loader.py:131-134`.
✅ Base model fails to predict → `RuntimeError` raised, with an explicit comment rejecting the
   neutral-0.5 fallback (`backend/ensemble_loader.py:301-313`).
✅ Heart weights missing → `FileNotFoundError` raised, `backend/model_loader.py:566-571`.

### C-9 — CRITICAL — a failing base model silently contributes **zeroed SHAP** and is dropped from the explanation

The one place an exception is swallowed into a default value:

```python
617                 except Exception as e:
618                     log.warning(
619                         f"SHAP explanation for '{name}' failed: "
620                         f"{type(e).__name__}: {e}. Skipping."
621                     )
622                     log.warning(traceback.format_exc())
623                     per_model_shap[name] = {
624                         "values": [0.0] * len(feature_names),
625                         "base_value": 0.0,
626                     }
627                     failed_models.add(name)
```
`backend/ensemble_loader.py:617-627`, followed by `shap_weights[failed] = 0.0` and
re-normalisation at `:654-663`.

`except Exception` catches **everything** — including the `joblib.load()` on line 536 that
re-reads the model from disk. So a corrupt or missing pkl during `/explain` produces a
**200 response with a confident-looking explanation built from the surviving models**, while
`/predict` on the identical input would raise. The only signal is a WARNING in the log. The
response carries no field indicating degradation (`shap_weights` is computed but then stripped
by the response model — see C-14). A judge who kills one model file mid-demo gets a plausible
chart, not an error.

### W-5 — WARNING — a broken config makes a disease silently vanish

```python
215                 except Exception as e:
216                     log.error("Error loading %s: %s", filename, e)
```
`backend/router.py:213-216`

A malformed YAML removes the disease from the registry; the API then answers 404 "not
registered" rather than surfacing the parse error. Startup still reports success.

### W-6 — WARNING — dead 501 branch contradicts its own error text

`backend/router.py:160-169` raises `COUNTERFACTUALS_NOT_SUPPORTED` with the hint *"Heart disease
uses a single XGBoost model without a counterfactual generator."* But `ModelLoader` **does**
define `generate_counterfactuals` (`backend/model_loader.py:423`), so `hasattr` is always True
and the branch is unreachable. Verified live: heart counterfactuals return 3 scenarios.

## Feature-engineering order — INFERENCE vs TRAINING, side by side

Both diseases use **scale-then-engineer**, and both match. ✅ **OK.**

**Heart disease**
```python
# TRAINING — experiment_files/models/train_v5_xgb.py:97,105-106
 97  df = pd.read_csv(data_path)          # final_ready_data.csv: ALREADY encoded + scaled
105  df_fe = engineer_heuristic_features(df)
106  df_fe = engineer_medical_features(df_fe)
```
```python
# INFERENCE — backend/model_loader.py:298-300
298  df = pd.DataFrame([patient_data])
299  df = self._apply_preprocessors(df)   # encode → scale first
300  df = self._engineer_features(df)     # then engineer from scaled values
```
Identical in `explain()` at `backend/model_loader.py:339-341`, and inherited by
`generate_counterfactuals()` which calls `self.predict()` (`model_loader.py:479`).

**Diabetes**
```python
# TRAINING — models/train_diabetes_ensemble.py:829-835
829  X_train, X_test, y_train, y_test = load_preprocessed_data(config)  # already scaled
833  X_train_fe = engineer_features(X_train, config)
835  X_test_fe  = engineer_features(X_test,  config)
```
```python
# INFERENCE — backend/ensemble_loader.py:296-298
296  df = pd.DataFrame([patient_data])
297  df = self._apply_preprocessors(df)   # scale first (matches training pipeline)
298  df = self._engineer_features(df)     # then engineer from scaled values
```
Identical in `explain()` (`:489-490`) and in the counterfactual `pipeline_fn` (`:390-391`).

Column *order* is additionally re-aligned per model against `feature_names_in_` before every
call, with a missing column raising rather than being NaN-filled
(`backend/ensemble_loader.py:196-273`). ✅ Good defensive design.

### W-7 — WARNING — every artifact was pickled under scikit-learn 1.6.1 and is being unpickled under 1.9.0

```
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version
1.6.1 when using version 1.9.0. This might lead to breaking code or invalid results.
```
Emitted on load of `meta_learner.pkl` and the RF. Predictions are numerically sane here, but
`requirements.txt` should pin the training-time versions before anyone re-runs this on a clean
machine.

---

# AUDIT 4 — Reproducing the numbers

Scripts: `scratch/eval_heart.py`, `scratch/eval_diabetes.py` (untracked).
Full output: `evaluation_evidence/heart_disease_report.txt`, `evaluation_evidence/diabetes_report.txt`.
Confusion matrices: `evaluation_evidence/heart_disease_confusion_matrix.png`,
`heart_disease_confusion_matrix_raw.png`, `diabetes_confusion_matrix.png`.
Nothing was tuned to make anything match.

## Heart disease — 121-sample held-out test split, reproduced exactly

Because `ModelLoader._LABELS_INVERTED = True` (`backend/model_loader.py:44`) flips the class
axis at inference, I evaluated **all four** orientation × threshold combinations so no reading
is left untested.

| Variant | Accuracy | Δ acc | ROC-AUC | Sensitivity | Δ sens | Specificity | Δ spec |
|---|---|---|---|---|---|---|---|
| A. raw label, thr 0.5 | 80.17% | **−0.00** | 0.8564 | 85.53% | +5.07 | 71.11% | −12.02 |
| B. raw label, thr 0.420 | 80.99% | +0.82 | 0.8564 | 90.79% | +10.33 | 64.44% | −18.69 |
| **C. clinical orientation, argmax — THIS IS PRODUCTION** | **80.17%** | **−0.00** | **0.8564** | **71.11%** | **−9.35** | **85.53%** | **+2.40** |
| D. clinical orientation, thr 0.420 | 77.69% | −2.48 | 0.8564 | 77.78% | −2.68 | 77.63% | −5.50 |

Production confusion matrix (variant C): `[[tn 65, fp 11], [fn 13, tp 32]]`.

**Deviations greater than 0.5 pp:**

### C-10 — CRITICAL — claimed heart sensitivity 80.46% is off by 9.35 pp
Production (variant C) measures **71.11%**. No orientation or threshold produces 80.46%.
The closest value in the whole sweep is variant D's 77.78% — still 2.68 pp away, and at a
threshold the code does not use.

### C-11 — CRITICAL — claimed heart specificity 83.13% is off by 2.40 pp
Production measures **85.53%**. Note the direction: the true specificity is *better* than
claimed, while sensitivity is far worse. The claimed pair (80.46 / 83.13) describes a more
balanced classifier than the one that is actually deployed.

### C-12 — CRITICAL — the claimed threshold 0.420 exists nowhere in the codebase
`grep -rn` for `0.42` across `backend/`, `frontend/src/`, `configs/`, `features/` returns
**zero** threshold hits (the only matches are a "SHAP 0.42+" version badge in `README.md:19`
and `plans/omnidiag_implementation_details.md:35`).

- `configs/heart_disease.yaml` has **no `inference_threshold` key at all** (compare
  `configs/diabetes.yaml:87`, which does).
- `ModelLoader.predict()` uses the model's own argmax:
  ```python
  301         raw_pred = int(self.model.predict(df)[0])
  ```
  `backend/model_loader.py:301` — i.e. an implicit 0.5 cutoff.
- The frontend already documents this, in writing:
  ```js
  18  * `ModelLoader.predict()` (backend/model_loader.py) has no configurable
  19  * threshold at all — it just takes the model's own argmax (the sklearn
  20  * default 0.5 cutoff) and none of the API endpoints (/predict, /explain,
  21  * /schema, /api/v4/diseases) expose that number.
  26  export const HEART_DISEASE_DEFAULT_THRESHOLD = 0.5;
  ```
  `frontend/src/constants/thresholds.js:18-26`
- The threshold-tuning artifact it is supposedly derived from disagrees:
  `models/heart_disease/threshold_plots/threshold_results.json` reports
  `best_f1_threshold: 0.3` and `clinical_triage_threshold: 0.2`, and its grid contains
  0.40 and 0.45 but **never 0.42**.
- The doc also misdescribes the script: it claims *"grid search over [0.1, 0.9] with stride
  0.01"* (`plans/OMNIDIAG_TECHNICAL_DOCS.md:416`), but
  `experiment_files/models/evaluate_threshold.py:4-5` documents **0.20 to 0.80, step 0.05**.

**If asked "show me where 0.420 is applied," there is no answer.** The UI displays 50.0%.

**What *is* defensible:** accuracy 80.17% and ROC-AUC 0.8564 reproduce to the decimal, from the
exact training split, against the shipped pkl — subject to the leakage in C-8.

## Diabetes — 14,139-sample held-out test split, reproduced

| Variant | Accuracy | Δ acc | ROC-AUC | Sensitivity | Δ sens | Specificity | Δ spec |
|---|---|---|---|---|---|---|---|
| **stacking @ 0.275 — THIS IS PRODUCTION** | **73.12%** | **−2.06** | **0.8305** | **91.55%** | **−0.15** | **54.68%** | **+0.05** |
| stacking @ 0.5 | 75.23% | +0.05 | 0.8305 | 78.38% | −13.32 | 72.08% | +17.45 |
| voting @ 0.275 (unused fallback) | 71.55% | −3.63 | 0.8311 | 93.78% | +2.08 | 49.34% | −5.29 |

Production confusion matrix: `[[tn 3866, fp 3204], [fn 597, tp 6472]]`.
Base model AUCs: XGB 0.8310, LGBM 0.8298, RF 0.8286.

### C-13 — CRITICAL — claimed diabetes accuracy 75.18% is measured at a threshold production does not use
At the production threshold of 0.275 (`configs/diabetes.yaml:87`, applied at
`backend/ensemble_loader.py:328`), accuracy is **73.12%** — a **2.06 pp** overstatement.
75.18% is the accuracy at argmax 0.5, where sensitivity would be 78.38%, not 91.70%.

**You cannot claim 75.18% accuracy and 91.70% sensitivity simultaneously.** They come from two
different operating points on the same ROC curve. The correct, defensible statement is:
**"73.1% accuracy at 91.6% sensitivity / 54.7% specificity, threshold 0.275"** — a deliberate
screening trade-off, which is a *stronger* story than the inconsistent one.

✅ **OK:** sensitivity (−0.15 pp), specificity (+0.05 pp) and ROC-AUC (−0.0005) all reproduce
well inside tolerance.

---

# AUDIT 5 — Config and documentation drift

## Hyperparameters: YAML vs the actual `.pkl` vs the grid-search file

### C-7 — CRITICAL — `configs/heart_disease.yaml` publishes hyperparameters the deployed model does not have

```yaml
 89   best_params:
 90     n_estimators: 898
 91     max_depth: 5
 92     learning_rate: 0.013594126498405943
 93     subsample: 0.963733407970185
 94     colsample_bytree: 0.5454718650965412
 95     eval_metric: "logloss"
 96     random_state: 42
```
`configs/heart_disease.yaml:89-96`

| Param | YAML config | **Shipped `.pkl` (authoritative)** | `grid_best.json` | `models/metrics.json` |
|---|---|---|---|---|
| n_estimators | **898** | **1124** | 1124 | 1124 |
| learning_rate | **0.013594** | **0.006309** | 0.006309 | 0.006309 |
| subsample | **0.963733** | **0.833105** | 0.833105 | 0.833105 |
| colsample_bytree | **0.545472** | **0.479243** | 0.479243 | 0.479243 |
| max_depth | 5 | 5 | 5 | 5 |
| min_child_weight | *absent* | **3** | 3 | 3 |
| gamma | *absent* | **1.445995** | 1.445995 | 1.445995 |
| reg_alpha | *absent* | **0.190203** | 0.190203 | 0.190203 |
| reg_lambda | *absent* | **0.141671** | 0.141671 | 0.141671 |

**Which one does the runtime use?** — **The `.pkl`.** `grep -rn "best_params"` across `backend/`
returns **zero hits**; the YAML block is inert at inference. Everything except the YAML agrees.
The config is a stale v5.0-era copy (identical to `configs/heart_disease.yaml.bak:81-83`) and is
the **only** disagreeing source — and it is the file a judge will open first, because it is
labelled *"the single source of truth"* at `configs/heart_disease.yaml:4`.

The stale values have already propagated into prose:
> "898 estimators, `max_depth=5`, `learning_rate=0.0136`, `colsample_bytree=0.5455`"
> — `plans/omnidiag_implementation_details.md:24`

### ✅ OK — diabetes XGBoost hyperparameters match exactly
`configs/diabetes.yaml:91-100` ≡ `XGB_BEST_PARAMS` (`models/train_diabetes_ensemble.py:58-72`)
≡ the loaded `omni_diag_xgb_optimized.pkl`, all nine values. RandomForest matches `RF_PARAMS`
(`models/train_diabetes_ensemble.py:91-101`): 500 estimators, max_depth 12,
min_samples_leaf 4, max_features sqrt.

### W-8 — WARNING — `omni_diag_lgb_optimized.pkl` contains the *un*-optimized fallback params
```python
 74  # LightGBM params — used as fallback/default; overridden by full Optuna (100 trials) in main()
 75  LGB_BEST_PARAMS = {
 76      "n_estimators": 950, "max_depth": 5, "learning_rate": 0.035,
 84      "num_leaves": 64, ...
```
`models/train_diabetes_ensemble.py:74-88`

The shipped pkl carries **exactly these values** (n_estimators 950, num_leaves 64, lr 0.035,
max_depth 5, reg_alpha 0.5, reg_lambda 2.0, subsample 0.9, colsample 0.85). So the artifact
named `..._optimized.pkl` did **not** come from the 100-trial Optuna run that
`plans/OMNIDIAG_TECHNICAL_DOCS.md:110` and `models/train_diabetes_ensemble.py:809` advertise.
If a judge asks "show me the LightGBM Optuna study," there is no matching artifact.

### W-9 — WARNING — `ensemble_metrics.json` records meta-learner coefficients that are **not** the shipped ones

```json
    "meta_learner_coefficients": {
      "XGBoost": 0.6073617646404844,
      "LightGBM": 0.9793098333766345,
      "RandomForest": 0.7113493936964314
    },
```
`models/diabetes/ensemble_metrics.json:16-20`

The deployed `models/diabetes/meta_learner.pkl` has
`coef_ = [0.4362, 3.8572, 0.9019]`, `intercept_ = [-2.6573]` (read directly off the object;
the copy in `omnidiag_diabetes_artifacts/` is byte-identical, md5 `76fc7132…`). Different
training run. The shipped weights are functionally correct — they reproduce the claimed
sensitivity/specificity to 0.15 pp — but the JSON documents a different model, and those
coefficients are what `/explain` uses as SHAP weights (`backend/ensemble_loader.py:635`).

## Documentation drift: numbers, branches, repo URLs

| Location | States | Actual | Verdict |
|---|---|---|---|
| `plans/OMNIDIAG_TECHNICAL_DOCS.md:7` | repo `github.com/yahyoha/omnidiag` | `github.com/yahyalababenah/omnidiag` | ❌ **C-15** |
| `…DOCS.md:65,109,129,185,233,248,261` | 7 branch README links under `yahyoha/` | all wrong org | ❌ **C-15** |
| `…DOCS.md:987` | "reflect the state of `feature/omni-platform-final`" | current branch is `deploy/v2-platform` | ⚠️ W-10 |
| `…DOCS.md:190` | "`feature/omni-platform-final`: Current production branch" | superseded | ⚠️ W-10 |
| `…DOCS.md:966` | Diabetes specificity **58.70%** | 54.68% measured; 54.63% in the artifact | ❌ **C-16** |
| `…DOCS.md:455, :867` | "lower specificity (58.7%)" ×2 | same | ❌ **C-16** |
| `…DOCS.md:409` | CV accuracy 88.98% "from models/metrics.json" | field does not exist | ❌ C-5 |
| `…DOCS.md:398-399` | Cleveland dataset, `ca`/`thal` features | 605-row merged set, neither | ❌ C-4 |
| `…DOCS.md:970` | "Training Samples ~303" | 605 (484 train / 121 test) | ⚠️ W-10 |
| `…DOCS.md:416` | threshold grid "[0.1,0.9] stride 0.01" | 0.20–0.80 step 0.05 | ❌ C-12 |
| `plans/omnidiag_implementation_details.md:24` | 898 est. / lr 0.0136 | 1124 / 0.006309 | ❌ C-7 |
| `plans/omnidiag_architecture_overview.md:74` | "Validation Accuracy 88.98%" | 80.17% test | ❌ C-5 |
| `README.md` | no metric claims anywhere | — | ✅ **OK** |
| `configs/heart_disease.yaml:13`, `schemas.py:8` | "12 clinical features" | 11 | ⚠️ W-10 |

### C-15 — CRITICAL — the technical document points every reader at the wrong GitHub org
`git remote -v` → `origin https://github.com/yahyalababenah/omnidiag.git`. The document's
repository link and all seven branch-README links use `github.com/yahyoha/…`. Judges given this
document cannot reach the code.

### C-16 — CRITICAL — the document states diabetes specificity as 58.70% in three places
Your own claim is 54.63% and the measured value is 54.68%. `plans/OMNIDIAG_TECHNICAL_DOCS.md`
says **58.70%** at `:966`, and repeats "58.7%" at `:455` and `:867`. Two different numbers for
the same metric in the same document, neither flagged.

### W-10 — WARNING — assorted stale figures
Branch attribution (`feature/omni-platform-final` vs `deploy/v2-platform`), "~303 training
samples", and the "12 clinical features" count. Individually cosmetic; collectively they signal
an unmaintained document.

---

# AUDIT 6 — Latency profiling

Backend started locally (`backend/.venv`, uvicorn, port 8901). Measurements are wall-clock from
an HTTP client (`scratch/latency.py`, `scratch/latency2.py`); in-process breakdown from
`scratch/profile_cf.py`.

## Endpoint latency — cold (first call) vs warm

| Endpoint | Cold | Warm (repeat input) | Warm (**new** patient) | Why warm is fast |
|---|---|---|---|---|
| `heart /predict` | 0.131 s | 0.011 s | **0.035–0.055 s** | model cached in process |
| `heart /explain` | 0.228 s | 0.027 s | **0.025–0.038 s** | TreeExplainer cached |
| `heart /counterfactuals` | **10.36 s** | 0.010 s | **9.9–11.0 s** | warm is a **cache hit**, not real work |
| `diabetes /predict` | 0.486 s | 0.014 s | **0.093–0.099 s** | 3 models + meta cached |
| `diabetes /explain` | 1.504 s | 1.446 s | **1.55–1.57 s** | ⚠️ never gets faster |
| `diabetes /counterfactuals` | **9.10 s** | 0.011 s | **9.1–9.3 s** | warm is a **cache hit** |
| `/parse-notes` | 0.015 s | 0.010 s | 0.010 s | pure regex |

> The "warm" column in the middle is **misleading as a performance claim**: `/predict` caches
> for 300 s (`backend/main.py:378`) and `/counterfactuals` for 3600 s (`backend/main.py:520`).
> The right-hand column — a distinct patient each call — is the real cost. **Quote the
> right-hand column.** A judge typing a new patient will wait ~10 s for What-If.

### C-17 — CRITICAL — `/counterfactuals` takes ~10 s per unique patient
The cold number and the true warm number are the same, because the only thing that makes it
fast is the response cache.

### C-18 — CRITICAL — `diabetes /explain` re-reads 97.7 MB from disk on **every single request**
```python
536                     fresh_model = joblib.load(shap_model_paths[name])
```
`backend/ensemble_loader.py:532-536` — inside the per-model loop, with the comment
*"Freshly reload model for SHAP (avoids cached-object issues with sklearn 1.8.0 + SHAP TreeExplainer)"*.

Measured: the three base models total **97.7 MB** (`omni_diag_rf.pkl` alone is 93.9 MB), and a
new `shap.TreeExplainer` is constructed per model per request (`:589`). This is the entire
1.55 s, and it never amortises. On a container with cold page cache it will be far worse — and
`heart /explain`, which caches its explainer (`backend/model_loader.py:158-161`), is **40×
faster** at 0.038 s, proving the cost is the reload, not SHAP itself.

## Counterfactuals: candidate sample size and model invocations per request

**Heart disease** — `backend/model_loader.py:457`
```python
457         for _ in range(800):
```
- **Candidate sample size: 800.**
- Each surviving candidate calls `self.predict(cf)` (`:479`), which runs `model.predict()` **and**
  `model.predict_proba()` (`:301-302`) → **2 model invocations per candidate**.
- **Up to 1,600 model invocations + 800 full preprocess→engineer pipeline runs per request.**
- Measured warm `predict()` cost: **21.7 ms** → 800 × 21.7 ms ≈ 17.3 s of pure predict work
  (HTTP-observed 9.9–11.0 s; candidates with no mutated feature `continue` early at `:475-476`).

**Diabetes** — `backend/ensemble_loader.py:441`
```python
441             n_samples=100,
442             n_counterfactuals=3,
```
- **Candidate sample size: 100** (reduced from the generator default of 500,
  `backend/counterfactual_generator.py:145`).
- Each candidate runs through `predict_fn` (`ensemble_loader.py:395-420`): 3 base
  `predict_proba` + 1 meta `predict_proba` → **4 invocations per candidate**.
- **≈ 400 model invocations + 1 baseline `predict()` per request.**

### Where the time actually goes

| Component | Measured | Share of the ~9.3 s diabetes CF request |
|---|---|---|
| **RandomForest `predict_proba`, 1 row** | **71.3 ms** | **~87%** — 100 candidates × 71.3 ms ≈ 7.1 s |
| XGBoost `predict_proba`, 1 row | 7.7 ms | ~8% |
| LightGBM `predict_proba`, 1 row | 1.3 ms | ~1.4% |
| Candidate generation + preprocess/engineer | 2.1 ms/row | ~2% |

**Model inference dominates — candidate generation is negligible (~2%).** The single
RandomForest (500 trees × depth 12, 93.9 MB) is ~87% of the cost. `n_samples` is *not* the main
lever; the RF is.

For `/explain`, the split is inverted: ~100% of the 1.55 s is the 97.7 MB disk reload +
explainer construction (C-18), not SHAP value computation.

## Instantiation: startup vs lazy, and SHAP background sample size

| Object | When | Evidence |
|---|---|---|
| `OmniDiagRouter` + loader objects | **startup** (import time) | `backend/main.py:101` |
| Heart model bytes | **lazy**, first request | `backend/model_loader.py:69-75` |
| Heart TreeExplainer | **lazy**, then **cached** for process life | `backend/model_loader.py:158-161` |
| Diabetes base models + meta | **lazy**, first request, then cached | `backend/ensemble_loader.py:99-118, 126-137` |
| **Diabetes TreeExplainers** | **per request, never cached** | `backend/ensemble_loader.py:589` |
| Preprocessors | lazy, cached | `model_loader.py:175-186`, `ensemble_loader.py:146-177` |

**SHAP background sample size: zero — no background dataset is supplied anywhere.**
```python
161                 self._explainer = shap.TreeExplainer(self.model)
```
`backend/model_loader.py:161`, and `shap.TreeExplainer(fresh_model)` at
`backend/ensemble_loader.py:589`. Both are constructed with **no `data=` argument**, so SHAP
falls back to `feature_perturbation="tree_path_dependent"` using the trees' internal node
coverage. That is a legitimate and fast choice — but if a judge asks *"what is your SHAP
background distribution?"* the honest answer is **"none — tree-path-dependent, not
interventional."** Be ready for the follow-up about whether that is appropriate for
correlated clinical features (it is a known approximation).

### W-11 — WARNING — the first request of the day pays the model-load cost
Because loading is lazy, the very first `/predict` after a container start absorbs it
(diabetes: 2.5 s in-process to load 3 models + meta). The frontend already compensates with a
120 s timeout and a "waking up" message (`frontend/src/api.js:4,48-52`). Fine for a demo,
worth a sentence if asked about production readiness.

---

# AUDIT 7 — Cross-layer contract check

## Backend response schemas vs what the frontend reads

| Endpoint | Backend emits | Frontend reads | Verdict |
|---|---|---|---|
| `/predict` (heart) | `prediction`, `confidence`, `diagnosis` | same 3 | ✅ OK |
| `/predict` (diabetes) | + `model_contributions`, `ensemble_type`, `inference_threshold`, `ensemble_variance`, `model_agreement` | `inference_threshold` (`constants/thresholds.js:41`) | ✅ OK |
| `/explain` (both) | `chart_data`, `text_explanation`, `base_value`, `prediction`, `confidence`, `diagnosis`, `ensemble_variance`, `model_agreement`, `counterfactuals` | `chart_data`, `text_explanation`, `base_value`, `confidence`, `prediction` | ✅ OK |
| `/counterfactuals` (heart) | `{scenario_id, probability, changes: [...]}` | that exact shape | ✅ OK |
| **`/counterfactuals` (diabetes)** | `{scenario, changes: {...}, new_probability, risk_reduction, feasibility}` | **the heart shape** | ❌ **C-19** |
| `/parse-notes` | `extracted_features`, `mapped_features`, `field_count` | all 3 | ✅ OK |
| `/batch` | `disease`, `total`, `succeeded`, `failed`, `results[]` | all | ✅ OK |

### C-19 — CRITICAL — the two `/counterfactuals` endpoints emit **different shapes**, and the frontend only understands one

Captured live from the running backend (`scratch/cf_shapes.py`):

```jsonc
// HEART — POST /api/v4/heart_disease/counterfactuals  (P-002)
{ "scenario_id": 1,
  "probability": 0.2556,
  "changes": [ { "feature": "Oldpeak", "original_value": 2.3,
                 "counterfactual_value": 0.0, "direction": "decrease" } ] }

// DIABETES — POST /api/v4/diabetes/counterfactuals  (D-002)
{ "scenario": "If BMI drops to 27.7, blood pressure is controlled and physical health days drop to 1",
  "changes": { "HighBP": 0.0, "BMI": 27.677, "PhysHlth": 1.0 },
  "new_probability": 0.2239,
  "risk_reduction": "49%",
  "feasibility": "low" }
```
Produced by `backend/model_loader.py:519-523` and `backend/counterfactual_generator.py:241-247`
respectively.

The renderer branches on a field only the heart shape has:
```js
166           const isBackendFormat = s.scenario_id !== undefined;
171           if (!isBackendFormat) {                     // ← diabetes lands HERE
```
`frontend/src/components/WhatIfScenarioCard.jsx:166,171`

A diabetes scenario has **no `scenario_id`**, so it is routed into the *mock* renderer, which
then reads fields that do not exist on it:

| Frontend reads | Heart | **Diabetes** |
|---|---|---|
| `s.scenario_id` (:166) | ✅ | ❌ **undefined** |
| `s.feature` (:185,188,198) | — | ❌ **undefined** |
| `s.current` (:201) | — | ❌ **undefined** |
| `s.proposed` (:205) | — | ❌ **undefined** |
| `s.riskReduction` (:209,130) | — | ❌ **undefined** |
| `s.description` (:213) | — | ❌ **undefined** |
| `s.probability` (:131) | ✅ | ❌ **undefined** (it is `new_probability`) |
| `c.feature` / `c.original_value` / `c.counterfactual_value` (:232,247,259) | ✅ array | ❌ `changes` is an **object**, not an array |

**Fields the frontend reads that the backend does not currently emit for diabetes:**
`scenario_id`, `feature`, `current`, `proposed`, `riskReduction`, `description`, `probability`.

**Consequences, in order of visibility:**
1. Every diabetes What-If card renders *"If **undefined** drops from **undefined** → **undefined**
   → Risk reduces by **undefined**%"* with a **"-0%"** badge (`getReductionPct` falls through
   both branches at `WhatIfScenarioCard.jsx:129-135` and returns `0`).
2. The backend's own good prose — the ready-made `scenario` string and the correct
   `risk_reduction: "49%"` — is computed and then thrown away.
3. **PDF export throws.** `frontend/src/components/PDFReport.jsx:193-195` does
   `(cf.changes ?? []).map(...)` — `changes` is a plain object for diabetes, which has no
   `.map`, so this is a `TypeError` at render time. The comment directly above it
   (`PDFReport.jsx:191`) documents only the heart shape as "Real API shape".

This is precisely the failure mode the brief warned about, and it is **live on the diabetes
module right now**.

### C-14 — CRITICAL — `ExplainResponse` silently strips the ensemble's per-model SHAP attribution

```python
438  @app.post("/api/v4/{disease}/explain", response_model=ExplainResponse, ...)
```
`backend/main.py:438`

`EnsembleModelLoader.explain()` computes and returns `per_model_shap` and `shap_weights`
(`backend/ensemble_loader.py:712-713`), but `ExplainResponse` (`backend/schemas.py:133-173`)
declares neither, so **FastAPI filters both out before the response is sent.** Verified live —
the emitted key set is exactly:
`['base_value','chart_data','confidence','counterfactuals','diagnosis','ensemble_variance','model_agreement','prediction','text_explanation']`

Consequences: (a) the explainability story a judge is most likely to probe — *"show me each base
model's contribution"* — is computed at cost and then discarded; (b) it makes C-9 unobservable
over the wire, since `shap_weights` would have been the only signal that a model was dropped.

### W-12 — WARNING — `ExplainResponse.counterfactuals` is typed to the diabetes shape and never populated
`backend/schemas.py:170-173` declares `Optional[List[Counterfactual]]`, where `Counterfactual`
(`schemas.py:91-129`) has `changes: Dict[str, float]`, `new_probability`, `risk_reduction`,
`feasibility` — the *diabetes* shape, which the frontend cannot read (C-19) and which the heart
endpoint does not produce. `/explain` always returns `null` for it. Three inconsistent
definitions of "counterfactual" now exist in the codebase.

## SHAP label-inversion handling — ✅ **OK, applied exactly once, verified mathematically**

`ModelLoader._LABELS_INVERTED = True` (`backend/model_loader.py:44`). Every site that touches
the class axis:

| Site | What it does | Verdict |
|---|---|---|
| `model_loader.py:307-308` | `has_disease = (raw_pred == 0)`; `confidence = raw_proba[0]` | ✅ once |
| `model_loader.py:371-373` | negates `shap_values.values` **and** `base_values` | ✅ once |
| `model_loader.py:380-382` | re-derives prediction/confidence inside `explain()` | ⚠️ W-13 |
| `model_loader.py:432` (counterfactuals) | delegates to `self.predict()` | ✅ inherits |
| `backend/shap_service.py` | **no sign manipulation at all** — pure presentation | ✅ no double-flip |
| `backend/ensemble_loader.py` | no inversion (diabetes labels are not inverted) | ✅ correct |

**Proof it is applied exactly once** (`scratch/shap_additivity.py`, against the live API): if the
flip were applied twice or missed, `sigmoid(base_value + Σ shap_values)` would equal
`1 − confidence` instead of `confidence`.

| Case | `sigmoid(base + Σshap)` | `explain.confidence` | `predict.confidence` | Additive? | Agree? |
|---|---|---|---|---|---|
| P-001 | 0.357080 | 0.357080 | 0.357080 | ✅ | ✅ |
| P-002 | 0.716182 | 0.716182 | 0.716182 | ✅ | ✅ |
| P-003 | 0.229599 | 0.229599 | 0.229599 | ✅ | ✅ |

Additivity holds to 1e-9 and `/explain` agrees with `/predict` exactly. The negative
`base_value` (−0.5167) is the expected consequence, documented at `model_loader.py:366-370`.

### W-13 — WARNING — `explain()` hardcodes the inversion instead of reading the flag
```python
380             has_disease = (raw_pred == 0)
382             result["confidence"] = float(raw_proba[0])
```
`backend/model_loader.py:380-382` — unlike `predict()` (`:307-308`), this does **not** consult
`self._LABELS_INVERTED`. Setting the flag to `False` would flip `/predict` while leaving
`/explain` inverted, silently desynchronising the decision from its explanation — the exact
failure the flag's own docstring (`:40-43`) says it exists to prevent. Not a live bug; a trap.

## Demo patients A / B / C / D — measured live

`scratch/demo_cases.py`, every case from `frontend/src/mockPatients.js` POSTed to the running API.

**Immutable list** (`backend/counterfactual_generator.py:59-69`), applies to diabetes only:
`Age, AnyHealthcare, CholCheck, Education, HeartDiseaseorAttack, Income, NoDocbcCost, Sex, Stroke`

| Case | Patient | Documented risk | **Measured risk** | Pred | CF status | **Scenarios** | Immutable features in this case |
|---|---|---|---|---|---|---|---|
| **A** | D-001 Noor Sabbagh | 9.6% | **7.9%** | 0 Neg | `not_applicable` | **0** ⚠️ | all 9 |
| **B** | D-002 Karim Yaghi | 43.2% | **43.5%** | 1 Pos | `generated` | **3** ✅ | all 9 |
| **C** | D-003 Samir Abu-Ghazaleh | 75.8% | **85.0%** ❌ | 1 Pos | `no_valid_counterfactuals` | **0** ❌ | all 9 |
| **D** | D-004 Hala Mansour | 33.8% | **35.1%** | 1 Pos | `generated` | **3** ✅ | all 9 |
| — | P-001 Ahmed Al-Rashid (heart) | — | 35.7% | 0 Neg | `not_applicable` | **0** ⚠️ | Age, Sex |
| — | P-002 Fatima Hassan (heart) | — | 71.6% | 1 Pos | `success` | **3** ✅ | Age, Sex |
| — | P-003 Khalid Othman (heart) | — | 23.0% | 0 Neg | `not_applicable` | **0** ⚠️ | Age, Sex |

All 21 diabetes input fields are present in every diabetes case, so all 9 immutable features
appear in all four. For heart, `IMMUTABLE_FEATURES` does not apply — `ModelLoader` uses its own
hardcoded `MUTABLE` dict (`backend/model_loader.py:445-452`: `RestingBP`, `Cholesterol`,
`MaxHR`, `Oldpeak`, `FastingBS`, `ExerciseAngina`), so `Age` and `Sex` are immutable by omission.

**Cases flagged with fewer than 3 scenarios: A, C, P-001, P-003.**
A, P-001 and P-003 are **expected and correct** — those patients are below threshold, so
`not_applicable` with 0 scenarios is the right answer, and the UI renders a green "Low Clinical
Risk" card (`WhatIfScenarioCard.jsx:87-112`). **Case C is a real failure.**

### C-20 — CRITICAL — Case C is your flagship severe-patient demo and it produces **zero** counterfactuals

`mockPatients.js:14-16` bills D-003 as *"severe but fully mutable risk profile"*, and
`:23-31` explains at length that `HeartDiseaseorAttack` and `Stroke` were deliberately set to 0
precisely so the What-If engine would have "real, actionable ground to work with."
Measured here: **`no_valid_counterfactuals`, 0 scenarios, after 9.3 s of computation.**

Root cause: at **85.0%** the patient is far above the 0.275 threshold, and 100 random candidate
perturbations (`backend/ensemble_loader.py:441`) never reach it. Because the prediction is
positive with an empty array, the UI falls to state 4 and renders **mock BMI/RestingBP/
Cholesterol placeholder data with a "Coming Soon" badge**
(`WhatIfScenarioCard.jsx:13-14,119-120`) — on a *diabetes* patient, showing heart-disease
feature names. If a judge clicks Case C, that is what they will see.

### C-21 — CRITICAL — the demo-case docstring states probabilities that do not reproduce

`frontend/src/mockPatients.js:8-11` asserts the figures are *"verified live against the deployed
Space's EnsembleModelLoader.predict(), so the printed probabilities below are actual production
output, not estimates."*

| Case | Documented | Measured here | Δ |
|---|---|---|---|
| A | 9.6% | 7.9% | −1.7 pp |
| B | 43.2% | 43.5% | +0.3 pp |
| **C** | **75.8%** | **85.0%** | **+9.2 pp** |
| D | 33.8% | 35.1% | +1.3 pp |

Case C is 9.2 pp off — far beyond the floating-point jitter the file's own caveat
(`mockPatients.js:33-44`) describes, and beyond the 0.78–0.85 band commit `dd7095e` was written
to hit. B and D are within jitter; A and C are not. The last two commits
(`698974b`, `dd7095e`) were specifically tuning Case C, so the number in the docstring appears
to predate the final state of the data — or the deployed Space differs materially from this
checkout. **Either way, the numbers a judge reads in the code do not match what the code does.**

---

# Git hygiene

| Check | Result |
|---|---|
| Current branch | `deploy/v2-platform` @ `698974b` |
| Default branch on origin | `origin/HEAD → origin/main1` |
| Local HEAD vs origin | **in sync** — `git rev-list --left-right --count HEAD...@{u}` → `0  0` |
| Uncommitted | `backend/auth/routes.py` (+13/−2), `scripts/simulate_federated.py` (+74/−20) |
| Untracked | only this audit's own output (`AUDIT_REPORT.md`, `evaluation_evidence/`, `scratch/`) |
| `.env` ever committed? | ✅ **No.** Not tracked, ignored at `.gitignore:184` |
| Secrets in history? | Only `k8s/secrets.yaml` + `helm/…/secrets.yaml`, both **placeholder-only** (`CHANGE_ME`) |
| Model weights in history? | ⚠️ Yes — 20 `.pkl` paths added historically, via **Git LFS** (`.gitattributes:1`) |
| Repo size | `.git` **713 MB** on disk (45.4 MB packed + LFS objects); working tree ~2.6 GB incl. `.venv` |
| Largest **tracked** files | `frontend/package-lock.json` 0.30 MB · `plans/OMNIDIAG_TECHNICAL_DOCS.md` 0.07 MB · `README.md` 0.05 MB — **no large binaries tracked on this branch** |
| Largest blob ever added | `data/diabetes/raw/…BRFSS2015.csv` 6.1 MB |

### C-22 — CRITICAL — a live Hugging Face write token is embedded in `.git/config`

```
hf  https://yahyoha:hf_KueNwzRH…[REDACTED — full value is in .git/config locally]@huggingface.co/spaces/yahyoha/omnidiag.git
```
(`git remote -v`)

This is a **real, non-placeholder credential with push access to the deployment Space**, stored
in plaintext. It is local to `.git/config` (not committed, not pushed), but it is exposed to
anything that reads the working directory, appears in `git remote -v` output, and will be
visible if you screen-share a terminal or hand over a machine during the evaluation.

**Rotate this token now**, and re-add the remote without inline credentials
(use a git credential helper or `HF_TOKEN`).

Related, lower severity: `.env` (untracked, correctly ignored) holds a real
`DEEPSEEK_API_KEY=sk-573…` (35 chars) and a `JWT_SECRET_KEY` that still begins with `change…`.
Neither was ever committed. ✅

### W-14 — WARNING — no model weights exist on the branch you are demoing

`git ls-tree -r --name-only deploy/v2-platform | grep '\.pkl'` → **0 files.**

| Branch | `.pkl` files on HEAD |
|---|---|
| `deploy/v2-platform` (current) | **0** |
| `feature/omni-platform-final` | 0 |
| `main1` | 1 |
| `production-final-v3` | 3 |

`*.pkl` is ignored at `.gitignore:18`. **A judge who clones this repo and runs it gets
`FileNotFoundError` on the first prediction** — every model is local-only. Have the artifacts on
a USB drive, in LFS on the branch you hand over, or a documented download step. This is the
most likely way a live demo fails for a reason unrelated to your work.

Also present but untracked/ignored: `OmniDiag_Full_Backup.zip` (92 MB), `mlruns.db` (872 KB),
`omnidiag_dev.db` (1.1 MB), `omnidiag_diabetes_artifacts/` (a full duplicate model tree),
`.coverage`, plus 5 `.bak`/`.kaggle` sibling files inside `configs/`, `backend/`, `features/` and
`experiment_files/`. None affect runtime; all add noise to a repo walkthrough.

### W-15 — WARNING — the diabetes training script cannot run as committed
```python
 66  DATA_PATH = os.path.join(
 67      _PROJECT_ROOT,
 68      "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
 69  )
```
`experiment_files/data_pipeline/preprocess_diabetes.py:66-69` points at the **project root**;
the file actually lives at `data/diabetes/raw/`. `models/train_diabetes_ensemble.py:829` calls
it with no override, so "reproduce your training" fails at step one with `FileNotFoundError`.
(My AUDIT 4 script reads the correct path directly.)

---

# Prioritized remediation

## MUST FIX BEFORE THE EVALUATION

| # | Action | Why |
|---|---|---|
| 1 | **Rotate the HF token** (C-22) and strip credentials from `.git/config` | A live write credential to your deployment is sitting in plaintext. |
| 2 | **Restate the heart metrics** (C-10, C-11, C-12) as **80.17% accuracy, 0.856 ROC-AUC, 71.1% sensitivity, 85.5% specificity, decision rule = argmax (0.5)** — or implement and verify a 0.420 threshold before claiming it | You cannot defend 80.46 / 83.13 / 0.420; none of them exists in the code or reproduces from the model. |
| 3 | **Restate the diabetes accuracy** (C-13, C-6) as **73.1% at threshold 0.275**, alongside 91.6% sensitivity / 54.7% specificity | 75.18% + 91.70% are two different operating points. A judge can catch this with mental arithmetic: `(91.7+54.6)/2 ≈ 73.2`. |
| 4 | **Delete or quarantine `models/heart_disease/metrics.json` and `metadata.json`** (C-1, C-2) | Tracked files asserting an SVM at 0.957 AUC and listing `ca`/`thal`. Both are unused — deleting them costs nothing and removes your two worst exhibits. |
| 5 | **Fix the diabetes counterfactual contract** (C-19) | Every diabetes What-If card currently renders `undefined`, and PDF export throws a `TypeError`. Live, visible, on the module you are showcasing. Cheapest fix: branch on `s.scenario !== undefined` in `WhatIfScenarioCard.jsx:166` and read `new_probability` / `risk_reduction` / object-form `changes`. |
| 6 | **Fix or replace demo Case C** (C-20, C-21) | Your flagship severe patient shows heart-disease mock data with a "Coming Soon" badge. Either soften the profile until CFs generate, or raise `n_samples` for it. |
| 7 | **Prepare an answer for the heart-disease leakage** (C-8) | The scaler *and* a RandomForest imputer are fit on all 605 rows pre-split. Either re-run the pipeline fit-on-train-only and re-report, or state the limitation before a judge finds it. Diabetes is already clean — say so. |
| 8 | **Correct `configs/heart_disease.yaml:89-96`** (C-7) | The file labelled "single source of truth" is the only place disagreeing with the deployed model on four hyperparameters. |
| 9 | **Get the model weights onto the demo path** (W-14) | Zero `.pkl` files on `deploy/v2-platform`. A clean clone cannot make a single prediction. |
| 10 | **Fix the repo URL in the technical document** (C-15) and the 58.70% specificity (C-16) | Judges given that document cannot reach the code, and it contradicts your own specificity claim three times. |

## SHOULD FIX BEFORE THE EVALUATION (cheap, high credibility)

| # | Action | Why |
|---|---|---|
| 11 | Remove the 88.98% CV claim (C-5) from three docs, or relabel it "pre-production, superseded" | It is attributed to a file that has no CV field. |
| 12 | Fix the dataset description at `…DOCS.md:398-399` (C-4) | It names `ca` and `thal` as production inputs. They are not. |
| 13 | Cache the diabetes TreeExplainers (C-18) | One-line-ish fix turning 1.55 s into ~0.04 s, and removes a 97.7 MB disk read per `/explain`. |
| 14 | Add `per_model_shap` + `shap_weights` to `ExplainResponse` (C-14) | You compute the best explainability answer you have and then throw it away. |
| 15 | Delete `data/heart_disease/raw/heart11.csv` (W-4) and the `.bak`/`.kaggle` files | Removes the last on-disk trace of the leaked-feature dataset. |
| 16 | Fix `preprocess_diabetes.py:66-69` (W-15) | "Reproduce your training" currently fails immediately. |
| 17 | Have a prepared answer on SHAP background (AUDIT 6) | "None — tree-path-dependent, not interventional" is correct; say it confidently. |

## CAN WAIT

| # | Item |
|---|---|
| 18 | Narrow the `except Exception` in the SHAP loop (C-9) and the config loader (W-5) |
| 19 | Make `explain()` read `_LABELS_INVERTED` instead of hardcoding it (W-13) |
| 20 | Speed up counterfactuals (C-17) — the RandomForest is ~87% of the cost, not `n_samples` |
| 21 | Consolidate the three diabetes metric JSONs (W-2); reconcile `grid_best.json` / `diagnostic_v5_ab.json` (W-1) |
| 22 | Re-train and re-ship LightGBM from the advertised Optuna study, or rename the artifact (W-8) |
| 23 | Regenerate `ensemble_metrics.json` so its meta-learner coefficients match the shipped pkl (W-9) |
| 24 | Pin scikit-learn to the training version in `requirements.txt` (W-7) |
| 25 | Fix "12 clinical features" → 11 (AUDIT 2); remove the dead 501 branch (W-6); branch/sample-count drift (W-10) |
| 26 | Unify the three definitions of "counterfactual" across the codebase (W-12) |
| 27 | Prune `OmniDiag_Full_Backup.zip`, `omnidiag_diabetes_artifacts/`, `mlruns.db` from the working tree |

---

## Nothing was changed

No application code, config, model artifact, or document was modified. No commits were made.
New files created, all untracked: `AUDIT_REPORT.md`, `evaluation_evidence/` (5 files),
`scratch/` (7 files). `backend/auth/routes.py` and `scripts/simulate_federated.py` were already
modified before this audit began and were left untouched.

## Nothing was left UNVERIFIED

Every finding was produced by reading the file or executing the code. Two scope notes:

- **Deployed-Space parity.** All measurements are from this checkout on this machine. The
  Hugging Face Space may hold different artifacts — relevant to C-21, where `mockPatients.js`
  cites Space-measured probabilities that do not reproduce here. Which one is wrong is
  **UNVERIFIED**: confirming it requires querying the live Space, which is outside a read-only
  local audit.
- **Magnitude of the heart-disease leakage.** C-8 is confirmed as a fact of the pipeline. How
  many points of the 80.17% it is worth is **UNVERIFIED** — quantifying it requires re-running
  `clean_data.py` fit-on-train-only and retraining, which would modify artifacts.

---

## تحديث لاحق — 21-09-2026

**كل رقم وجدول أعلاه يعكس حالة النظام بتاريخ التدقيق (2026-09-15) ولم يُعدَّل — هذا سجل
تاريخي لما وُجد فعلاً وقتها، لا وصفاً للحالة الحالية.** منذ ذلك التاريخ استُبدل نموذج القلب
بالكامل (Pipeline ذاتي الاحتواء يشمل التصحيح/التوسيع/الترميز داخلياً)، وصُحِّح انتشار السكري
المنشور. النتيجة: كل بند في "Verdict at a glance" (أعلاه) الخاص بالقلب — بما فيها C-8 (تسريب
المُعايِر/المُستوفي على كامل الـ605 صفاً قبل التقسيم) وC-10/C-11/C-12 (العتبة 0.420 غير
موجودة، argmax 0.5 هو المستخدَم فعلياً) — **لم يعد قابلاً للتكرار على النموذج المنشور حالياً**،
لأنه نموذج مختلف كلياً ببنية مختلفة، لا لأن الأخطاء المكتشَفة أُصلحت في نفس النموذج القديم.

**الأرقام الحالية الصحيحة** (مصدرها `evaluation_evidence/heart/hpo_threshold_summary.json`
للأداء الداخلي والعتبة، و`evaluation_evidence/heart/final_metrics.json` لرقم طهران الخارجي —
راجع أيضاً `evaluation_evidence/diabetes/final_metrics_table.json` للسكري):

| البند | كان (بتاريخ هذا التدقيق) | الآن (2026-09-21) |
|---|---|---|
| بيانات القلب | `final_ready_data.csv`، 605 صف، Cleveland فقط | UCI 4 مواقع مدموجة، 920 مريضاً (Cleveland 303 + Hungarian 294 + Switzerland 123 + Long Beach VA 200) |
| عتبة القلب | لا توجد في الكود — argmax 0.5 فعلياً | **0.3695**، مكتوبة في `configs/heart_disease.yaml` ومطبَّقة فعلياً |
| LOSO ROC-AUC (القلب) | غير مقاس بهذا الشكل | **81.30%** (nested، تقدير أمين) |
| تحقق طهران الخارجي (القلب) | 72.17% (نموذج مخفَّض غير مضبوط، على checkpoint قبل الضبط) | **76.34%** (النموذج المخفَّض بعد الضبط — 7 ميزات مشتركة، عتبة 0.4237) — أفضل تقريب مُتاح؛ النموذج الكامل المنشور فعلياً (11 ميزة، عتبة 0.3695) لا يمكن تقييمه على طهران مباشرة لغياب 4 من ميزاته فيها. التفاصيل في `WEAKNESS_REGISTER.md` HM-6 (**RESOLVED**) |
| معالجة القلب المسبقة | `label_encoders.pkl` + `standard_scaler.pkl` منفصلان، fit على كامل البيانات قبل التقسيم (C-8) | لا ملفات معالجة منفصلة إطلاقاً — الـPipeline يتضمّن `IterativeImputer`/`StandardScaler`/`OrdinalEncoder` داخلياً |
| انتشار السكري المنشور | 0.14 (افتراض أمريكي/BRFSS مؤقت) | **0.237** (انتشار السكري الفعلي في الأردن) |
| عتبة السكري المصحَّحة | 0.059776 | **0.108184** |

**لم يُعَد تشغيل أي تدقيق جديد كامل** — هذا الجدول إشارة إلى مصدر الحقيقة الحالي فقط، وليس
تدقيقاً موازياً لِما ورد أعلاه. التفاصيل الكاملة في `docs/DIABETES_AUDIT_REPORT.md` (السكري)
و`WEAKNESS_REGISTER.md` بنود `HM-1` إلى `HM-6` (القلب).
