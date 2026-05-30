# Diabetes Module Integration Plan — OmniDiag v4

> **Phase**: 1 (Audit) + 2 (Plan) — STOPPED before any code changes per user instruction
> **Current Branch**: `production-final-v3` — **Will create new branch** for integration
> **Zero-Code-Change Rule**: No modifications to heart disease module files

---

## Phase 1: Audit & Evaluation — Results

### 1.1 What EXISTS in `omnidiag_diabetes_artifacts/`

| Artifact | Status | Notes |
|----------|--------|-------|
| `configs/diabetes.yaml` | ✅ Present | Simpler version than workspace — `inference_threshold: 0.275` confirmed |
| `data/diabetes/interim/false_negatives_profile.csv` | ✅ Present | 100 FN cases, sorted by `probability_distance_from_threshold` |
| `models/diabetes/ensemble_metrics.json` | ✅ Present | Stacking: 75.18% accuracy, 83.11% ROC-AUC |
| `models/diabetes/preprocessors/feature_names.json` | ✅ Present | 21 base features (14 binary + 7 continuous) |
| `models/diabetes/preprocessors/meta_feature_names.json` | ✅ Present | 5 engineered (26 total), 3 meta features |
| **All `.pkl` model files** | ✅ Confirmed by user | `xgb_model.pkl`, `lgb_model.pkl`, `rf_model.pkl`, `xgb_models.pkl`, `lgb_models.pkl`, `rf_models.pkl`, `meta_learner.pkl`, `voting_ensemble.pkl`, `standard_scaler.pkl` |

### 1.2 Training Quality Assessment

```
┌──────────────────────┬──────────┬──────────┐
│ Model                │ Accuracy │ ROC-AUC  │
├──────────────────────┼──────────┼──────────┤
│ Single XGBoost       │ 75.05%   │ 82.58%   │
│ Stacking Ensemble    │ 75.18%   │ 83.11%   │  ← BEST
│ Voting Ensemble      │ 75.08%   │ 83.02%   │
└──────────────────────┴──────────┴──────────┘

Clinical Threshold:  0.275 (FN penalty ×2)
Sensitivity:        91.70%
Specificity:        54.63%
F1 at threshold:    77.36%
Meta-learner coeffs: XGBoost=0.61, LightGBM=0.98, RF=0.71
```

### 1.3 Config Delta Analysis

| Field | Workspace Config (`configs/diabetes.yaml`) | Kaggle Config (`omnidiag_.../diabetes.yaml`) | Action |
|-------|---------------------------------------------|-----------------------------------------------|--------|
| `disease.name` | `"diabetes"` | `"Diabetes"` | Keep workspace |
| `disease.version` | `"1.1.0"` | Missing | Keep workspace |
| `model.inference_threshold` | `0.5` | `0.275` | **UPDATE → 0.275** |
| `model.weights_path` | `omni_diag_xgb_optimized.pkl` | Missing | **UPDATE → ensemble paths** |
| `model.ensemble` | Full ensemble block | Missing | Keep workspace (verify paths) |
| `model.best_params` | XGBoost params | Missing | Keep workspace |
| `features` | Full spec with formulas | Partial | Keep workspace |
| `explainer` | Full SHAP config | Missing | Keep workspace |
| `api` | API config | Missing | Keep workspace |

---

## Phase 2: Execution Plan

### Step 0 — Git Branch
```bash
git checkout -b feature/diabetes-module-integration
```

### Step 1 — Copy Kaggle Artifacts to Workspace

| Source | Destination | Action |
|--------|-------------|--------|
| `omnidiag_.../configs/diabetes.yaml` | `configs/diabetes.yaml` | **Merge** — take `inference_threshold: 0.275`, keep workspace structure |
| `omnidiag_.../models/diabetes/ensemble_metrics.json` | `models/diabetes/ensemble_metrics.json` | **Overwrite** |
| `omnidiag_.../data/diabetes/interim/false_negatives_profile.csv` | `data/diabetes/interim/false_negatives_profile.csv` | **Copy** |
| `omnidiag_.../models/diabetes/*.pkl` | `models/diabetes/*.pkl` | **Copy all 9 pickle files** |
| `omnidiag_.../models/diabetes/preprocessors/*.json` | `models/diabetes/preprocessors/*.json` | **Copy** |

### Step 2 — Update `configs/diabetes.yaml`

Change exactly **one line**:
```yaml
# Line 87: BEFORE
  inference_threshold: 0.5
# Line 87: AFTER
  inference_threshold: 0.275
```

And verify the following paths in the `ensemble` block match actual files:
```yaml
ensemble:
  base_models:
    - name: "xgboost"
      weights_path: "models/diabetes/xgb_model.pkl"    # Verify exists
    - name: "lightgbm"
      weights_path: "models/diabetes/lgb_model.pkl"    # ✅ Confirmed
    - name: "random_forest"
      weights_path: "models/diabetes/rf_model.pkl"     # Verify exists
  meta_learner:
    weights_path: "models/diabetes/meta_learner.pkl"   # Verify exists
```

### Step 3 — Verify Backend Auto-Discovery

1. Start server: `uvicorn backend.main:app --reload`
2. Test: `GET /api/v4/diseases` → expect `["heart_disease", "diabetes"]`
3. Test: `GET /api/v4/diabetes` → expect disease info from config

### Step 4 — Test Predict Endpoint

```bash
curl -X POST http://localhost:8000/api/v4/diabetes/predict \
  -H "Content-Type: application/json" \
  -d '{
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1,
    "Fruits": 1, "Veggies": 0, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "DiffWalk": 0, "Sex": 0,
    "BMI": 28.0, "MentHlth": 5, "PhysHlth": 3, "GenHlth": 3,
    "Age": 8, "Education": 4, "Income": 7
  }'
```

Expected response includes:
- `"prediction": 1` or `0` (using threshold 0.275)
- `"predicted_probability": <float>`
- `"inference_threshold": 0.275`

### Step 5 — Test Explain Endpoint

```bash
curl -X POST http://localhost:8000/api/v4/diabetes/explain \
  -H "Content-Type: application/json" \
  -d '<same patient data as above>'
```

Expected response includes:
- `"chart_data": [...]` (SHAP values for 26 features)
- `"text_explanation": "..."`

### Step 6 — Frontend Verification

1. Navigate to the OmniDiag frontend
2. Select "Diabetes" from disease dropdown
3. Verify all 21 input fields render correctly
4. Submit a prediction → verify result appears

### Step 7 — Rollback Plan

```bash
# If anything breaks:
git checkout production-final-v3          # Revert to heart-disease-only
git branch -D feature/diabetes-module-integration  # Delete feature branch
```

### Files NOT to Modify (Enforced)

| File | Reason |
|------|--------|
| `backend/model_loader.py` | Heart Disease only |
| `configs/heart_disease.yaml` | Heart Disease only |
| `features/heart_disease_features.py` | Heart Disease only |

---

## Summary Diagram

```mermaid
flowchart TB
    subgraph "Artifacts Import"
        A[omnidiag_diabetes_artifacts/] --> B[models/diabetes/*.pkl]
        A --> C[configs/diabetes.yaml\nthreshold: 0.275]
        A --> D[ensemble_metrics.json]
        A --> E[false_negatives_profile.csv]
    end

    subgraph "Config Merge"
        C --> F[configs/diabetes.yaml\nKeep workspace structure\nUse threshold=0.275]
    end

    subgraph "Backend Verify"
        F --> G[router.py auto-discovers\ndiabetes.yaml]
        G --> H[EnsembleModelLoader loads\n9 pickle files]
        H --> I[POST /predict returns\n{diagnosis, proba, threshold}]
        H --> J[POST /explain returns\nSHAP chart_data]
    end

    subgraph "Frontend Verify"
        K[React dynamic form\nrenders 21 fields] --> L[Submit → predict]
        L --> M[Display result\nwith threshold info]
    end

    B --> H
    D -.->|reference| H
    E -.->|reference| I
```

## Blocker Status

~~❌ BLOCKED: Missing `.pkl` model files in artifacts~~  
✅ **RESOLVED**: User confirmed ALL `.pkl` files present at `omnidiag_diabetes_artifacts/models/diabetes/`
