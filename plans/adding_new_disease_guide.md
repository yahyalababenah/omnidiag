# Adding a New Disease to OmniDiag — Architectural Guide

## Purpose

This document explains how to add a new disease prediction module to OmniDiag without breaking existing functionality. It describes the architecture, the responsibilities of each file, and the order of operations. This is a reference for developers who understand the project structure.

---

## Overview: The Config-Driven Architecture

OmniDiag uses a **config-driven, zero-touch integration** pattern. The core backend (`router.py`, `model_loader.py`, `main.py`) is fully generic — it never references any specific disease by name. Instead, it discovers diseases by scanning the `configs/` directory at startup.

Adding a new disease requires:
- **6 new files** (data, config, feature engineer, training script, model artifacts)
- **1 existing file modified** (`backend/schemas.py` — add one class + one registry entry)
- **Zero changes** to routing, model loading, API entry point, or SHAP service

---

## Part 1: Backend Architecture

### How the Backend Discovers a New Disease

When the backend starts:
1. `OmniDiagRouter.__init__()` scans `configs/` for all `.yaml` files
2. For each YAML file, it reads `disease.name` and registers the disease
3. It creates a `ModelLoader` for that disease (lazy — model is loaded on first request)
4. The FastAPI routes `/api/v4/{disease}/predict` and `/api/v4/{disease}/explain` become available automatically

### Backend Files You Must NOT Modify

| File | Why It Must Stay Untouched |
|------|---------------------------|
| `backend/router.py` | Auto-discovers all configs. No disease names are hardcoded. |
| `backend/model_loader.py` | Works with any config. Loads the feature engineer dynamically via `importlib` using the module/class names from the YAML config. Applies preprocessing then runs inference — all generically. |
| `backend/main.py` | Routes are parameterized by `{disease}`. No disease-specific logic. |
| `backend/shap_service.py` | Pure transformation logic — works with any SHAP values array from any model. |
| `features/base_features.py` | Abstract parent class. Never modified. Ensures all feature engineers implement the same interface. |
| `features/__init__.py` | Package initializer. No disease-specific imports. |

---

## Part 2: Step-by-Step — What to Create and Modify

### Step 1: Prepare Raw Data

Create a directory structure under `data/`:

```
data/{disease_name}/
├── raw/          # Place your CSV/XLSX files here
├── processed/    # Will hold cleaned CSVs and feature-engineered CSVs
└── interim/      # Optional: validation scores, grid search results
```

Rules:
- Each disease gets its own data directory to avoid column name collisions.
- Raw data files are committed or documented so the pipeline is reproducible.
- The directory name should match `disease.name` in the YAML config.

---

### Step 2: Create a Preprocessing Pipeline

Write a script: `data_pipeline/clean_{disease}_data.py`

This script must perform these operations in order:

1. **Load** raw data from `data/{disease}/raw/`
2. **Merge** if there are multiple raw data sources (optional — create `merge_{disease}_data.py` if needed)
3. **Encode** categorical variables into integers (label encoding)
4. **Impute** missing values (use MissForest, median, or domain-appropriate method)
5. **Scale** numerical features (use StandardScaler)

**Critical requirement:** The preprocessors (label encoders + scaler) must be saved as `.pkl` files to `models/{disease}/preprocessors/`. The backend uses these exact preprocessors at inference time to transform raw patient input the same way training data was transformed.

After running this script:
- `data/{disease}/processed/final_clean_data.csv` — cleaned dataset
- `models/{disease}/preprocessors/label_encoders.pkl`
- `models/{disease}/preprocessors/standard_scaler.pkl`

---

### Step 3: Create a Feature Engineer

Create a file: `features/{disease}_features.py`

This class must:
- Inherit from `BaseFeatureEngineer` (defined in `features/base_features.py`)
- Implement `engineer_heuristic()` — creates statistical interaction features (ratios, multiplications, polynomials). These are model-agnostic mathematical transformations that help the model discover non-linear patterns.
- Implement `engineer_clinical()` — creates medically-derived risk scores using fixed coefficients from domain literature.

The class receives the YAML config dictionary in its constructor. It is instantiated dynamically by `ModelLoader` at inference time using the `features.module` and `features.class` values from the config.

After creating the class, run it against `final_clean_data.csv` to produce two datasets for A/B testing:
- `data/{disease}/processed/data_heuristic.csv` — with heuristic features
- `data/{disease}/processed/data_clinical.csv` — with clinical features

---

### Step 4: Select and Train a Model

Write a training script: `models/train_{disease}_xgb.py`

**Model Selection Guide:**

| Model | Best For | SHAP Support | Config `type` | Config `explainer_type` |
|-------|----------|-------------|---------------|------------------------|
| XGBoost | Tabular data, mixed types, <100K rows, fast inference | TreeExplainer | `xgboost` | `tree` |
| Random Forest | Tabular data, when interpretability is paramount | TreeExplainer | `random_forest` | `tree` |
| TabNet | Tabular data with deep learning, GPU, >100K rows | DeepExplainer | `tabnet` | `deep` |
| Neural Network | Non-tabular data (images, sequences) | DeepExplainer | `deep_learning` | `deep` |

**Training procedure:**
1. Load `data_heuristic.csv` (the winning approach from heart disease A/B testing)
2. Separate features from target (target column name comes from `config['disease']['target_column']`)
3. Train/test split using parameters from the config
4. Train the model with hyperparameters from `config['model']['best_params']`
5. Evaluate on the test split
6. Save the trained model using `joblib.dump()` to the path specified in `config['model']['weights_path']`

**Optional — Hyperparameter Optimization:** Create an Optuna-based optimization script that searches for the best hyperparameters. Store the winning trial's parameters in the YAML config under `model.best_params`.

---

### Step 5: Create the YAML Config File

Create: `configs/{disease}.yaml`

This is the **single source of truth** for the disease module. It must contain these sections:

| Section | Required Fields |
|---------|----------------|
| `disease` | `name` (must match filename), `display_name`, `description`, `target_column`, `version` |
| `data` | `raw_path`, `processed_path`, `interim_path`, `raw_files`, `merged_file`, `final_clean_file`, `heuristic_file`, `clinical_file` |
| `features` | `module` (path to feature engineer class), `class` (class name), `categorical_columns`, `numerical_columns`, `imputation_features`, `heuristic_features` (list of name+formula+description), `clinical_features` |
| `model` | `type`, `explainer_type`, `weights_path`, `fallback_weights_path`, `preprocessors_path`, `best_params` |
| `training` | `test_size`, `random_state`, `optimization_trials`, `optimization_direction`, `optimization_metric` |
| `explainer` | `enabled`, `type`, `plots_output_dir` |
| `api` | `endpoint_prefix`, `version` |

**Critical rule:** `disease.name` must exactly match the config filename (without `.yaml`). For example, if the file is `configs/diabetes.yaml`, then `disease.name` must be `diabetes`.

---

### Step 6: Register the Pydantic Input Schema

Modify: `backend/schemas.py`

This file contains:
- One Pydantic input class per disease (e.g., `HeartDiseaseInput`)
- A `DISEASE_SCHEMA_REGISTRY` dictionary mapping disease names to their input classes
- A `get_schema_for_disease()` function used by the API endpoints for request validation

**What to do:**
1. Add a new Pydantic class for your disease. Each field corresponds to a column in the dataset. Each field should include:
   - The correct Python type (`int`, `float`, `str`)
   - Validation ranges where applicable (e.g., `ge=20, le=120` for Age)
   - A description string
   - An example value in `json_schema_extra` for API documentation
2. Register the new class in `DISEASE_SCHEMA_REGISTRY` by adding one line: `"{disease}": {DiseaseName}Input`

This is the **only existing file you modify**. It is safe because it uses a dictionary registry pattern — adding a new entry has zero side effects on existing entries.

---

### Step 7: Save Model Artifacts

After training, ensure this directory structure exists:

```
models/{disease}/
├── preprocessors/
│   ├── label_encoders.pkl       # From Step 2 preprocessing
│   └── standard_scaler.pkl      # From Step 2 preprocessing
└── model.pkl                     # From Step 4 training
```

All paths are defined in the YAML config. The `ModelLoader` resolves them relative to the project root.

---

### Step 8: Verify the Integration

1. **Start the API server:** `uvicorn backend.main:app --reload`
2. **Check auto-discovery:** `GET /api/v4/diseases` — the new disease should appear in the list alongside heart_disease.
3. **Test prediction:** `POST /api/v4/{disease}/predict` with valid patient data matching the Pydantic schema.
4. **Test explanation:** `POST /api/v4/{disease}/explain` — should return SHAP chart data and text explanation.

**If prediction fails (404), check:**
- The YAML config filename matches `disease.name` inside the file.
- The YAML file has valid syntax (no tabs, correct indentation).
- All referenced paths exist (model weights, preprocessors).

**If prediction returns wrong values, check:**
- The preprocessors were saved from the same training pipeline.
- The column names in the Pydantic schema match those in the training data.
- The `explainer_type` matches the model type (`tree` for XGBoost/RF, `deep` for neural networks).

---

## Part 3: Frontend Changes

### Overview

The frontend consists of:
- `frontend/src/api.js` — **Already dynamic.** Uses `api.predict(disease, data)` and `api.explain(disease, data)` where `disease` is a parameter. No changes needed.
- `frontend/src/components/ShapBarChart.jsx` — **Already generic.** Renders any SHAP chart data regardless of disease. No changes needed.
- `frontend/src/App.jsx` — Currently has two modes (Engineering Mode + Clinical EMR Mode) but **no disease selector**. For multi-disease support, a disease selector must be added.
- `frontend/src/components/EngineeringMode.jsx` — **Heart disease-specific.** Hardcodes the 11 heart disease input fields (`FIELD_META`) and calls `api.predict('heart_disease', form)`.
- `frontend/src/components/ClinicalEmrMode.jsx` — **Heart disease-specific.** Hardcodes `api.predict('heart_disease', patient.data)` and the vitals display for heart disease metrics.
- `frontend/src/mockPatients.js` — **Heart disease-specific.** Contains 3 mock patients with heart disease data fields.

### What Must Change in the Frontend

**Option A — Per-Disease Components (Recommended for 2-3 diseases)**

Create a separate component for each disease's input form and display logic:

| File | Change |
|------|--------|
| `frontend/src/components/EngineeringMode.jsx` | Add a disease selector dropdown at the top. When the disease changes, swap the input form fields to match that disease's schema. The `api.predict()` call must use the selected disease name as the first argument. |
| `frontend/src/components/ClinicalEmrMode.jsx` | Add a disease selector. The vitals cards and patient data fields must match the selected disease. Mock patient data must include fields for all diseases. |
| `frontend/src/mockPatients.js` | Add mock patients for the new disease with appropriate clinical profiles and data fields. |
| `frontend/src/App.jsx` | No changes needed if the disease selector is inside the mode components. |

The API calls must change from:
```js
api.predict('heart_disease', form)
```
To:
```js
api.predict(selectedDisease, form)
```

**Option B — Dynamic Form Rendering (Recommended for 3+ diseases or a platform)**

Build a single generic form component that fetches the disease schema from the backend and renders input fields dynamically. This is more complex but scales to unlimited diseases without frontend code changes.

Components that need updating:

| File | What to Do |
|------|------------|
| `frontend/src/App.jsx` | Add a persistent disease selector in the sidebar so the selected disease persists across mode switches. |
| `frontend/src/components/EngineeringMode.jsx` | Make the input form dynamic — load the field definitions from the backend's schema or the YAML config, rather than hardcoding `FIELD_META` for heart disease. Replace the hardcoded `api.predict('heart_disease', form)` with `api.predict(selectedDisease, form)`. |
| `frontend/src/components/ClinicalEmrMode.jsx` | Make the vitals display dynamic — the number and type of vital cards should depend on the selected disease. Replace hardcoded `api.predict('heart_disease', patient.data)` with `api.predict(selectedDisease, patient.data)`. |
| `frontend/src/mockPatients.js` | Add mock patients for the new disease. Restructure or add a `disease` field to each patient so the UI knows which disease to predict. |

**Files that need NO changes:**

| File | Why |
|------|-----|
| `frontend/src/api.js` | Already uses `api.predict(disease, data)` — the disease is a parameter. |
| `frontend/src/components/ShapBarChart.jsx` | Renders any chart data generically. |
| `frontend/src/index.css` | No disease-specific styles. |
| `frontend/vite.config.js` | No disease-specific build configuration. |

---

## Part 4: Complete File Checklist

| Step | Action | New Files | Modified Files |
|------|--------|-----------|----------------|
| 1 | Create data directories | `data/{disease}/raw/`, `processed/`, `interim/` | None |
| 2 | Preprocess data | `data_pipeline/clean_{disease}_data.py` (+ `merge_{disease}_data.py` if multiple sources) | None |
| 3 | Create feature engineer | `features/{disease}_features.py` | None |
| 4 | Train model | `models/train_{disease}_xgb.py` (+ `models/optimize_{disease}_xgb.py` if optimizing) | None |
| 5 | Create config | `configs/{disease}.yaml` | None |
| 6 | Register schema | None | `backend/schemas.py` (add one class + one registry line) |
| 7 | Save artifacts | `models/{disease}/model.pkl`, `models/{disease}/preprocessors/*.pkl` | None |
| 8 | Update frontend | Per-disease component or generic form | `frontend/src/components/EngineeringMode.jsx`, `ClinicalEmrMode.jsx`, `mockPatients.js` |
| 9 | Verify | Start server, test endpoints | None |

**Summary:** 6-8 new files created, 1-3 existing files modified (backend schema + frontend components).

---

## Part 5: Failure Modes and Prevention

| Problem | Likely Cause | How to Prevent |
|---------|-------------|----------------|
| `404 Disease Not Found` | Config filename doesn't match `disease.name` inside the file | Verify: `configs/{x}.yaml` → `disease.name` must be `x` |
| `Model weights not found` | Path in config doesn't match `joblib.dump()` destination | Train script should read the path from config and write to it |
| `Schema validation error` on ALL diseases | Syntax error in newly added class in `schemas.py` | Test before starting server: `python -c "from backend.schemas import *"` |
| `SHAP explainer type mismatch` | Using `TreeExplainer` on a neural network or vice versa | Set `explainer_type: tree` for XGBoost/RF, `explainer_type: deep` for TabNet/NN |
| `Prediction probabilities are wrong` | Preprocessors not applied or wrong transformation order | Verify encoding happens BEFORE scaling in `ModelLoader._apply_preprocessors()` |
| `Column not found during encoding` | Pydantic schema field name doesn't match training column name | Ensure field names in the Pydantic class match the CSV column names exactly |
| Frontend shows heart disease fields for new disease | The form component hardcodes heart disease fields | Replace hardcoded fields with a disease selector that switches field sets |
