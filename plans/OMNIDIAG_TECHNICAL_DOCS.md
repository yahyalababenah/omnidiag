# OmniDiag Platform — Production-Grade Clinical Decision Support System

**IEEE AI Expo Technical Documentation**

**Version:** 4.0.0  
**Last Updated:** June 2026  
**Repository:** [`github.com/yahyoha/omnidiag`](https://github.com/yahyoha/omnidiag)  

---

## Table of Contents

1. [Architectural Evolution & Design Rationale](#1-architectural-evolution--design-rationale)  
   1.1 [Phase I — Monolithic Streamlit Prototype (scikit-learn)](#11-phase-i--monolithic-streamlit-prototype-scikit-learn)  
   1.2 [Phase II — Clinical Threshold Optimization & Stacking Ensemble (Streamlit)](#12-phase-ii--clinical-threshold-optimization--stacking-ensemble-streamlit)  
   1.3 [Phase III — Complete Rewrite: FastAPI + React + OmniDiagRouter](#13-phase-iii--complete-rewrite-fastapi--react--omnidiagrouter)  
   1.4 [Phase IV — Diabetes Module Integration & Multi-Disease Routing](#14-phase-iv--diabetes-module-integration--multi-disease-routing)  
   1.5 [Phase V — UI/UX Polish & Clinical Firewall](#15-phase-v--uiux-polish--clinical-firewall)  
   1.6 [Phase VI — Production Hardening & Hugging Face Deployment](#16-phase-vi--production-hardening--hugging-face-deployment)  
   1.7 [Phase VII — Current Production State](#17-phase-vii--current-production-state)  

2. [Current System Architecture](#2-current-system-architecture)  
   2.1 [Decoupled Three-Layer Architecture](#21-decoupled-three-layer-architecture)  
   2.2 [Dynamic Configuration Routing](#22-dynamic-configuration-routing)  
   2.3 [API Surface & Request Lifecycle](#23-api-surface--request-lifecycle)  
   2.4 [Disease Module Registration Protocol](#24-disease-module-registration-protocol)  

3. [Machine Learning Methodology](#3-machine-learning-methodology)  
   3.1 [Heart Disease — Single XGBoost Classifier](#31-heart-disease--single-xgboost-classifier)  
   3.2 [Diabetes — Stacking Ensemble (XGBoost + LightGBM + Random Forest)](#32-diabetes--stacking-ensemble-xgboost--lightgbm--random-forest)  
   3.3 [Feature Engineering Pipeline](#33-feature-engineering-pipeline)  
   3.4 [Inference Threshold Tuning & Clinical Sensitivity Optimization](#34-inference-threshold-tuning--clinical-sensitivity-optimization)  
   3.5 [XGBoost 3.x Compatibility Layer](#35-xgboost-3x-compatibility-layer)  

4. [Explainable AI (XAI) & The Clinical Firewall](#4-explainable-ai-xai--the-clinical-firewall)  
   4.1 [SHAP Attribution Engine](#41-shap-attribution-engine)  
   4.2 [Ensemble SHAP Aggregation Strategy](#42-ensemble-shap-aggregation-strategy)  
   4.3 [DiCE-Inspired Counterfactual Generator](#43-dice-inspired-counterfactual-generator)  
   4.4 [The Clinical Firewall — Five-Layer Safety System](#44-the-clinical-firewall--five-layer-safety-system)  

5. [Clinical UI/UX Engineering](#5-clinical-uiux-engineering)  
   5.1 [Dual-Mode Interface Architecture](#51-dual-mode-interface-architecture)  
   5.2 [Schema-Driven Dynamic Form Engine](#52-schema-driven-dynamic-form-engine)  
   5.3 [Semantic Clinical Badges & Cognitive-Load Reduction](#53-semantic-clinical-badges--cognitive-load-reduction)  
   5.4 [Three-State Counterfactual Visualization](#54-three-state-counterfactual-visualization)  
   5.5 [Graceful Degradation & Error Boundaries](#55-graceful-degradation--error-boundaries)  

6. [Commercial Viability & Scalability](#6-commercial-viability--scalability)  
   6.1 [Zero-Code-Change Disease Addition](#61-zero-code-change-disease-addition)  
   6.2 [Deployment Architecture](#62-deployment-architecture)  
   6.3 [EMR Integration Pathways](#63-emr-integration-pathways)  
   6.4 [Multi-Tenancy & Horizontal Scaling](#64-multi-tenancy--horizontal-scaling)  
   6.5 [Regulatory Compliance Trajectory](#65-regulatory-compliance-trajectory)  

---

## 1. Architectural Evolution & Design Rationale

The OmniDiag platform underwent seven distinct architectural phases over an eight-month development cycle, each documented by a dedicated git branch with its own README reflecting the system state at that point. Each phase addressed specific limitations of its predecessor, resulting in progressive decoupling, increased clinical robustness, and zero-code-change disease addition capability. The following subsections use the actual branch READMEs as primary source material.

### 1.1 Phase I — Monolithic Streamlit Prototype (scikit-learn)

**Timeline:** Months 1–2  
**Primary Branch:** [`main1`](.git/logs/refs/heads/main1)  
**README:** [`main1/README.md`](https://github.com/yahyoha/omnidiag/blob/main1/README.md) — Streamlit badge, scikit-learn badge, Arabic UI  
**Objective:** Validate feasibility of ML-driven heart disease prediction with an interactive, clinically usable interface.

The initial prototype was developed as part of an advanced data science training program in collaboration with Sprite and Microsoft. It employed a monolithic **Streamlit** application with a scikit-learn ML pipeline evaluating multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVC) using Stratified K-Fold Cross-Validation and `GridSearchCV` for Random Forest hyperparameter tuning.

```
┌──────────────────────────────────────────────────┐
│            Phase I — Streamlit Monolith          │
├──────────────────────────────────────────────────┤
│  Streamlit UI (Arabic Language)                  │
│  ┌────────────────────────────────────────────┐  │
│  │  scikit-learn Pipeline                     │  │
│  │  · ColumnTransformer                       │  │
│  │  · StandardScaler + OneHotEncoder          │  │
│  │  · RandomForest / LR / SVC                 │  │
│  │  · GridSearchCV + Stratified K-Fold CV     │  │
│  ├────────────────────────────────────────────┤  │
│  │  KMeans + PCA Clustering                   │  │
│  │  Hierarchical Clustering Dendrograms       │  │
│  ├────────────────────────────────────────────┤  │
│  │  SHAP Explanations (embedded in UI)        │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Key characteristics from the `main1` README:**
- **Frameworks:** Streamlit, scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
- **Models evaluated:** Logistic Regression, Decision Tree, Random Forest, SVC
- **Additional ML:** KMeans clustering, PCA dimensionality reduction, hierarchical clustering
- **Language:** Arabic web interface for real-time risk assessment
- **Deployment:** Streamlit Cloud (`streamlit.app` subdomain)
- **SHAP:** Present but embedded in the UI layer

**Limitations identified:**
- Monolithic Streamlit architecture precluded programmatic API access for EMR integration
- Single-model constraint prevented multi-disease support without code duplication
- SHAP explainability was embedded within the UI layer, violating separation of concerns
- No configuration-driven dispatch — each new disease required a new Streamlit page
- Deployment on Streamlit Cloud limited scalability and domain customization

### 1.2 Phase II — Clinical Threshold Optimization & Stacking Ensemble (Streamlit)

**Timeline:** Months 2–3  
**Primary Branch:** `origin/V2-advanced-model` (remote-only)  
**README:** [`V2-advanced-model/README.md`](https://github.com/yahyoha/omnidiag/blob/V2-advanced-model/README.md) — "Clinical AI Approach (V2)"  
**Objective:** Improve clinical sensitivity through stacking ensemble architecture and aggressive threshold tuning.

Phase II introduced two critical innovations that persisted throughout the project's evolution: **clinical threshold tuning** and **stacking ensemble** architecture. The V2 README explicitly frames the engineering problem: *"In medical diagnostics, a False Negative (sending a sick patient home) is fatal, whereas a False Positive (requesting further tests for a healthy patient) is an acceptable clinical precaution."*

**Key characteristics from the `V2-advanced-model` README:**
- **Model:** Stacking Ensemble (`RandomForest` + `GradientBoosting` + `SVC` → `LogisticRegression` meta-learner)
- **Clinical threshold:** 30% decision boundary (down from default 50%)
- **Resulting sensitivity (Recall):** 96% for the disease class
- **Accuracy:** 89.67%
- **Data imputation:** `KNNImputer` for physiological missing values
- **Architecture:** Modular Python (`src/` backend), but still Streamlit-based with `@st.cache_data` lazy loading
- **Rationale:** Explicit rejection of "data-hungry boosting frameworks" (XGBoost, CatBoost) due to severe overfitting on the 918-record Cleveland dataset

**Engineering significance:** The V2 stacking ensemble established the ensemble architecture pattern later adapted for the diabetes module. The clinical threshold tuning at 30% foreshadowed the diabetes module's 0.275 inference threshold (see §3.4). However, V2 remained a single-disease Streamlit application with no multi-disease routing or REST API.

### 1.3 Phase III — Complete Rewrite: FastAPI + React + OmniDiagRouter

**Timeline:** Months 3–5  
**Primary Branch:** [`V3-advanced-DL`](.git/logs/refs/heads/V3-advanced-DL)  
**README:** [`V3-advanced-DL/README.md`](https://github.com/yahyoha/omnidiag/blob/V3-advanced-DL/README.md) — First use of "OmniDiag" name, FastAPI + React badges  
**Objective:** Complete architectural decoupling — replace monolithic Streamlit with decoupled FastAPI backend and React frontend, introduce config-driven disease routing.

This phase represents the single largest architectural transformation in the project's history. The entire codebase was rewritten from a Streamlit monolith into a three-layer decoupled architecture. The project was renamed to **OmniDiag: Multi-Disease Diagnostic Platform**.

```
┌────────────────────────────────────────────────────────────┐
│            Phase III — V3-advanced-DL (Complete Rewrite)   │
├────────────────────────────────────────────────────────────┤
│  React + Vite Frontend (Dual-Mode UI)                     │
│  ┌───────────────────┐  ┌───────────────────────────────┐  │
│  │ Engineering Mode  │  │ Clinical EMR Mode            │  │
│  │ · Full controls   │  │ · Mock patient records       │  │
│  │ · Raw JSON view   │  │ · Color-coded risk badges    │  │
│  └─────────┬─────────┘  └──────────┬────────────────────┘  │
│            │ HTTP REST              │                        │
├────────────┼────────────────────────┼────────────────────────┤
│  FastAPI Server                     │                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  OmniDiagRouter                                     │   │
│  │  · Scans configs/*.yaml                             │   │
│  │  · Lazy model loading                               │   │
│  │  · Dynamic loader dispatch                          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  ModelLoader → XGBoost + SHAP TreeExplainer          │   │
│  │  Clinical NLP Generator (EN/AR bilingual)            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

**Key characteristics from the `V3-advanced-DL` README:**
- **Backend:** FastAPI with CORS middleware, Pydantic schema validation, lazy model loading
- **Frontend:** React 18 + Vite + Tailwind CSS + Recharts (dual-mode: Engineering + Clinical EMR)
- **Routing:** [`OmniDiagRouter`](backend/router.py:27) — YAML config scanning, auto-discovery, dynamic loader dispatch
- **Model:** XGBoost (single model — notably reverted from the V2 stacking ensemble)
- **Explainability:** SHAP `TreeExplainer` with 3-layer interpretability stack (numerical, visual, linguistic)
- **API versioning:** `/api/v4/` namespace with legacy `/api/v3/predict` endpoint
- **Infrastructure:** Docker containerization, MLflow experiment tracking, Optuna hyperparameter optimization
- **License:** MIT

**Engineering rationale for reverting to single XGBoost:** The V3 phase abandoned the stacking ensemble from V2 in favor of a single XGBoost model. This decision was motivated by:
1. XGBoost's superior performance on the V5 engineered feature space (12 base + 5 interaction terms), where interaction features captured the non-linear patterns that previously required ensemble diversity
2. Deployment simplicity of a single model — critical for the new FastAPI architecture where lazy loading and cold-start latency were primary concerns
3. The ability to use `shap.TreeExplainer` directly on a single tree ensemble, producing cleaner SHAP explanations without ensemble aggregation complexity

**Bilingual NLP feature (later removed):** The `_generate_clinical_summary()` method at [`model_loader.py:231`](backend/model_loader.py:231) (since removed) automatically translated SHAP-driven clinical findings into both English and Arabic:

> *"The primary factors increasing risk are Oldpeak (2.4), Age (58), and ST_Slope (Flat). A protective factor reducing risk is ChestPainType (ATA)."*
> *"العوامل الرئيسية التي تزيد من الخطر هي Oldpeak (2.4) و Age (58) و ST_Slope (Flat). العامل الوقائي الذي يقلل الخطر هو ChestPainType (ATA)."*

This feature was removed in later phases as the system evolved from single-disease bilingual tool to multi-disease platform, where language localization was deferred to a config-driven localization system (see V3 roadmap: V4.1 Dynamic Linguistic Engine).

### 1.4 Phase IV — Diabetes Module Integration & Multi-Disease Routing

**Timeline:** Months 5–6  
**Primary Branches:** [`feature/diabetes-module-integration`](.git/logs/refs/heads/feature/diabetes-module-integration) → [`feature/omni-platform-final`](.git/logs/refs/heads/feature/omni-platform-final)  
**README:** [`feature/omni-platform-final/README.md`](https://github.com/yahyoha/omnidiag/blob/feature/omni-platform-final/README.md) — First multi-disease CDSS documentation  
**Objective:** Add diabetes prediction as a second disease module, reintroduce stacking ensemble architecture, and implement the Clinical Firewall.

This phase marked the transition from single-disease to multi-disease architecture. Two branches capture this transition:
- [`feature/diabetes-module-integration`](https://github.com/yahyoha/omnidiag/tree/feature/diabetes-module-integration): Introduces diabetes config, ensemble loader, and feature engineer — README still describes single-disease heart disease
- [`feature/omni-platform-final`](https://github.com/yahyoha/omnidiag/tree/feature/omni-platform-final): Current production branch with full multi-disease support — README documents both CAD and DM

```
┌──────────────────────────────────────────────────────────────┐
│             Phase IV — Multi-Disease Architecture            │
├──────────────────────────────────────────────────────────────┤
│  configs/heart_disease.yaml ──┐                              │
│  configs/diabetes.yaml ───────┤                              │
│                               ▼                              │
│  OmniDiagRouter._load_all_configs()                          │
│       │                                                      │
│       ├── model.ensemble=false → ModelLoader (XGBoost)      │
│       │                          └─ HeartDiseaseFeatureEng   │
│       │                                                     │
│       └── model.ensemble=true  → EnsembleModelLoader        │
│                                   ├─ XGBoost base           │
│                                   ├─ LightGBM base          │
│                                   ├─ RandomForest base      │
│                                   ├─ LogisticRegression ML  │
│                                   └─ DiabetesFeatureEngineer│
│                                                                
│  CounterfactualGenerator ── Clinical Firewall (5-layer)       │
└──────────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

1. **Reintroduction of stacking ensemble:** The CDC BRFSS 2015 diabetes dataset (21 base features) demanded ensemble methods. [`EnsembleModelLoader`](backend/ensemble_loader.py:41) implemented a stacking ensemble combining XGBoost, LightGBM, and Random Forest with a Logistic Regression meta-learner — reusing the V2 ensemble design pattern but with different base learners optimized for the BRFSS feature space.

2. **Dynamic loader dispatch ([`_create_loader()`](backend/router.py:202)):** Each YAML config's `model.ensemble` flag determines loader instantiation — `ModelLoader` for single XGBoost (CAD) or `EnsembleModelLoader` for stacking ensemble (DM). No routing code changes required.

3. **Inference threshold tuning:** The diabetes module adopted an inference threshold of **0.275** (mirroring V2's 30% clinical threshold), achieving **91.7% sensitivity** — a direct application of the clinical thresholding philosophy established in Phase II.

4. **Feature engineer polymorphism:** Feature engineering was refactored into a class hierarchy inheriting from [`BaseFeatureEngineer`](features/base_features.py), with [`HeartDiseaseFeatureEngineer`](features/heart_disease_features.py:36) and [`DiabetesFeatureEngineer`](features/diabetes_features.py:23) implementing disease-specific heuristics.

5. **Schema-driven frontend:** The `/api/v4/{disease}/schema` endpoint was introduced to serve JSON Schemas dynamically. The frontend's [`DynamicClinicalForm.jsx`](frontend/src/components/DynamicClinicalForm.jsx:54) renders disease-specific forms using a component type resolution algorithm ([`schemaFieldParser.js`](frontend/src/utils/schemaFieldParser.js:58)) with zero hardcoded disease logic.

6. **License change:** MIT → All Rights Reserved

### 1.5 Phase V — UI/UX Polish & Clinical Firewall

**Timeline:** Months 6–7  
**Primary Branch:** [`feature/ui-ux-polish`](.git/logs/refs/heads/feature/ui-ux-polish)  
**README:** [`feature/ui-ux-polish/README.md`](https://github.com/yahyoha/omnidiag/blob/feature/ui-ux-polish/README.md) — Same content as V3-advanced-DL, heart disease only  
**Objective:** Refine clinical safety mechanisms, optimize frontend UX for clinical environments.

The `feature/ui-ux-polish` branch focused on frontend refinements while maintaining the same README as V3-advanced-DL, indicating it was a leaf branch focused on UI improvements.

1. **Clinical Firewall (see §4.4):** A five-layer safety system embedded within the [`CounterfactualGenerator`](backend/counterfactual_generator.py:121) that prevents clinically invalid explanations. The firewall enforces immutable features (e.g., sex, family history), directional medical constraints (e.g., physical activity can only increase), and illegal flip detection.

2. **Semantic clinical badges:** Raw binary predictions were replaced with contextual risk indicators — "Elevated Risk," "Low Risk," "Moderate Risk" — using `getFieldStatus()` in [`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:141).

3. **Three-state counterfactual visualization ([`WhatIfScenarioCard.jsx`](frontend/src/components/WhatIfScenarioCard.jsx:15)):** Loading spinner, green low-risk card, or interactive scenario cards — with mock fallback for graceful degradation.

### 1.6 Phase VI — Production Hardening & Hugging Face Deployment

**Timeline:** Month 7  
**Primary Branches:** [`production-final-v3`](.git/logs/refs/heads/production-final-v3), [`hf-deploy`](.git/logs/refs/heads/hf-deploy)  
**README:** [`production-final-v3/README.md`](https://github.com/yahyoha/omnidiag/blob/production-final-v3/README.md) — Same content as V3, HF Spaces config  
**Objective:** Containerize for Hugging Face Spaces deployment, implement CI/CD, ensure model compatibility.

The `production-final-v3` branch diverged with 131 unique commits focused on deployment and stability:

1. **Docker containerization ([`Dockerfile`](Dockerfile)):** Python 3.10-slim with model weight download from Hugging Face Hub
2. **XGBoost 3.x compatibility monkey-patch:** `save_raw()` patch at [`model_loader.py:94`](backend/model_loader.py:94) and [`ensemble_loader.py:513`](backend/ensemble_loader.py:513) for breaking serialization changes between XGBoost 2.x and 3.x
3. **CI/CD pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):** GitHub Actions with parallel backend validation (import verification + API smoke test) and frontend build
4. **`hf-deploy` branch:** Hugging Face Spaces-specific configuration (Space metadata header, `app_port: 8000`)

### 1.7 Phase VII — Current Production State

**Primary Branch:** [`feature/omni-platform-final`](.git/logs/refs/heads/feature/omni-platform-final) (current)  
**README:** [`README.md`](https://github.com/yahyoha/omnidiag/blob/feature/omni-platform-final/README.md) — Full multi-disease CDSS with CI/CD  
**License:** All Rights Reserved  
**Deployment:** Hugging Face Spaces (Docker) + Vercel (frontend SPA)

The current branch represents the culmination of all prior phases:
- **FastAPI + React decoupled architecture** from Phase III
- **Multi-disease support** (CAD + DM) from Phase IV
- **Stacking ensemble architecture** pattern from Phase II, adapted for diabetes
- **Clinical Firewall** and **semantic UI badges** from Phase V
- **Production deployment infrastructure** from Phase VI
- **Bilingual NLP removed** — deferred to config-driven localization (see Future Roadmap)

---

## 2. Current System Architecture

### 2.1 Decoupled Three-Layer Architecture

OmniDiag employs a three-layer decoupled architecture with strict separation of concerns between the presentation layer, API/application layer, and ML engine layer. No layer has direct knowledge of the internal implementation of any other layer.

```
┌─────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                    │
│  React SPA (Vite + Tailwind CSS + Recharts)            │
│  ┌─────────────────┐  ┌──────────────────────────────┐ │
│  │ EngineeringMode │  │ ClinicalEmrMode              │ │
│  │ · Full controls  │  │ · Mock patient selection    │ │
│  │ · Raw JSON view  │  │ · Color-coded vital cards   │ │
│  │ · All features   │  │ · AI Diagnostic Panel       │ │
│  └────────┬────────┘  └──────────────┬───────────────┘ │
│           │                           │                  │
│           └──────────┬────────────────┘                  │
│                      │  HTTP/REST (VITE_API_BASE)        │
├──────────────────────┼──────────────────────────────────┤
│            API LAYER (FastAPI + Uvicorn)                 │
│  ┌──────────────────────────────────────────────────────┐│
│  │  OmniDiagRouter                                     ││
│  │  · Config scanning (configs/*.yaml)                  ││
│  │  · Lazy model loading                                ││
│  │  · Automatic dispatch (single vs ensemble)           ││
│  │  · Disease registry management                       ││
│  ├──────────────────────────────────────────────────────┤│
│  │  /api/v4/diseases         — List available diseases  ││
│  │  /api/v4/{d}/schema       — Get JSON Schema          ││
│  │  /api/v4/{d}/predict      — Run inference            ││
│  │  /api/v4/{d}/explain      — SHAP + counterfactuals   ││
│  │  /api/v4/{d}/counterfactuals — Standalone CF         ││
│  └──────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────┤
│                   ML ENGINE LAYER                        │
│  ┌──────────────┐  ┌────────────────┐                   │
│  │ ModelLoader  │  │ EnsembleLoader │                   │
│  │ · XGBoost    │  │ · XGBoost      │                   │
│  │ · SHAP Tree  │  │ · LightGBM     │                   │
│  │ · FeatureEng │  │ · RandomForest │                   │
│  │              │  │ · Meta-Learner │                   │
│  │              │  │ · Weighted SHAP│                   │
│  └──────────────┘  └────────────────┘                   │
│  ┌──────────────────────────────────────────────────────┐│
│  │ CounterfactualGenerator  (DiCE-inspired)             ││
│  │ · Clinical Firewall (5 layers)                       ││
│  │ · Directional constraints                            ││
│  │ · Diversity selection (Jaccard lambda=0.5)            ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

### 2.2 Dynamic Configuration Routing

The core architectural innovation enabling multi-disease support without code modification is the **Dynamic Configuration Routing** system implemented in [`backend/router.py`](backend/router.py:30).

**Mechanism:**

1. **Config scanning:** On initialization, [`OmniDiagRouter._load_all_configs()`](backend/router.py:171) scans the [`configs/`](configs/) directory for all `*.yaml` files using `os.listdir()` and `yaml.safe_load()`.

2. **Loader resolution:** [`_create_loader(config)`](backend/router.py:202) inspects each config for the `model.ensemble` key. If `true`, an [`EnsembleModelLoader`](backend/ensemble_loader.py:41) is instantiated; otherwise, a single [`ModelLoader`](backend/model_loader.py:41) is used. Both loaders share the same public interface (`predict()`, `explain()`, `generate_counterfactuals()`), enabling polymorphic dispatch.

3. **Lazy model loading:** Model weights are loaded on first inference request, not at router initialization. This is achieved via `@property` decorators with memoization guards (e.g., [`ModelLoader.model`](backend/model_loader.py:62) checks `self._model is None` before loading from disk).

4. **Disease-to-schema mapping:** [`DISEASE_SCHEMA_REGISTRY`](backend/schemas.py:257) maps disease name strings to Pydantic `BaseModel` subclasses. [`get_schema_for_disease()`](backend/schemas.py:263) performs a dictionary lookup and raises `HTTPException(404)` for unknown diseases.

**Code flow for a prediction request:**

```
POST /api/v4/heart_disease/predict
  -> main.py: router.get_loader("heart_disease")
  -> router.py: configs["heart_disease"] lookup
  -> router.py: _get_loader() lazy-init or cache-hit
  -> model_loader.py: predict(patient_data)
  -> model_loader.py: _engineer_features() -> heuristic -> clinical
  -> model_loader.py: _apply_preprocessors() -> StandardScaler
  -> model_loader.py: model.predict_proba() -> XGBoost inference
  -> Return {prediction, probability, ...}
```

### 2.3 API Surface & Request Lifecycle

All endpoints are defined in [`backend/main.py`](backend/main.py:36) and share a consistent lifecycle pattern:

| Endpoint | Method | Purpose | Request Body | Response |
|---|---|---|---|---|
| `/` | GET | Health check | --- | `{status, version, timestamp}` |
| `/api/v4/diseases` | GET | List registered diseases | --- | `[{disease, display_name, description}]` |
| `/api/v4/{disease}/schema` | GET | Get JSON Schema for disease form | --- | `{title, type, properties, required}` |
| `/api/v4/{disease}/predict` | POST | Run inference | `{field: value, ...}` | `{prediction, probability, threshold, ...}` |
| `/api/v4/{disease}/explain` | POST | SHAP + optional counterfactuals | `{field: value, ...}` | `{chart_data, text_explanation, base_value, counterfactuals?, ...}` |
| `/api/v4/{disease}/counterfactuals` | POST | Standalone counterfactual generation | `{field: value, ...}` | `{counterfactuals: [...]}` |
| `/api/v3/predict` | POST | Legacy endpoint (backward compat) | Legacy format | `{prediction}` |

**Request lifecycle for the explain endpoint ([`main.py`](backend/main.py:139)):**

1. FastAPI receives `POST /api/v4/{disease}/explain` with JSON body
2. `patient: dict` is passed directly (no typed Pydantic model — generic dict enables multi-disease support)
3. [`router.get_loader(disease)`](backend/router.py:227) retrieves or initializes the appropriate loader
4. [`loader.explain(patient_data)`](backend/router.py:119) executes the full XAI pipeline (see §4.1)
5. The result is serialized through Pydantic's [`ExplainResponse`](backend/schemas.py:134) model
6. FastAPI returns the validated JSON response with OpenAPI-compliant schema

**CORS configuration ([`main.py`](backend/main.py:45)):** A whitelist of allowed origins is maintained, with the production Hugging Face Spaces domain and local development servers explicitly enumerated.

### 2.4 Disease Module Registration Protocol

Adding a new disease to OmniDiag requires zero modifications to the core API code. The registration protocol involves:

1. **Create YAML config** in [`configs/`](configs/) — defines model type, weights path, feature names, preprocessing parameters, and display metadata
2. **Add Pydantic model** to [`backend/schemas.py`](backend/schemas.py:257) — registers in `DISEASE_SCHEMA_REGISTRY`
3. **Create feature engineer** (optional) — subclass [`BaseFeatureEngineer`](features/base_features.py) for disease-specific feature transformations
4. **Place model weights** in [`models/{disease_name}/`](models/) — path resolved via [`resolve_file_path()`](configs/config_loader.py:72)

This protocol is documented in [`plans/adding_new_disease_guide.md`](plans/adding_new_disease_guide.md).

---

## 3. Machine Learning Methodology

### 3.1 Heart Disease — Single XGBoost Classifier

**Dataset:** Cleveland Heart Disease (UCI Repository), processed via [`experiment_files/data_pipeline/transform_heart11.py`](experiment_files/data_pipeline/transform_heart11.py)  
**Input features:** 12 base clinical features (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, num major vessels, thalassemia)  
**Engineered features:** 5 interaction terms — Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio, Rate-Pressure Product (RPP), Exercise_Risk_Index  
**Model architecture:** `XGBClassifier` with tuned hyperparameters stored in [`models/heart_disease/grid_best.json`](models/heart_disease/grid_best.json)

**Performance metrics ([`models/metrics.json`](models/metrics.json)):**

| Metric | Value |
|---|---|
| Test Accuracy | 80.17% |
| ROC-AUC | 85.64% |
| CV Accuracy (mean) | 88.98% |
| CV Accuracy (std) | 0.0169 |
| Precision | 0.8391 |
| Recall | 0.8046 |
| F1-Score | 0.8215 |
| Optimal Threshold | 0.420 |

**Threshold selection rationale:** The default XGBoost threshold of 0.5 was suboptimal for the Cleveland dataset due to class imbalance (approximately 46% positive). [`experiment_files/models/evaluate_threshold.py`](experiment_files/models/evaluate_threshold.py) performed grid search over [0.1, 0.9] with stride 0.01, maximizing the F1-score. The optimal threshold of 0.420 balances precision (0.8391) and recall (0.8046), minimizing false negatives in a clinical context.

**Training pipeline:** [`experiment_files/models/train_v5_xgb.py`](experiment_files/models/train_v5_xgb.py) executes the full training cycle — data loading, 5-fold cross-validation, hyperparameter grid search via [`optimize_xgb_v5.py`](experiment_files/models/optimize_xgb_v5.py), and final model serialization to `omni_diag_xgb_optimized.pkl`.

### 3.2 Diabetes — Stacking Ensemble (XGBoost + LightGBM + Random Forest)

**Dataset:** CDC BRFSS 2015 Diabetes Health Indicators ([`data/diabetes/raw/`](data/diabetes/raw/))  
**Input features:** 21 base behavioral and demographic indicators (BMI, age, income bracket, education level, healthcare access, physical activity, fruit/vegetable consumption, smoking status, alcohol consumption, mental health days, physical health days, sleep hours, general health status, etc.)  
**Engineered features:** 5 composite indices — BMI_Age_Interaction, Health_Index, Lifestyle_Score, SES_Composite, Diabetes_Clinical_Risk  
**Model architecture:** Stacking ensemble with 3 base learners and a Logistic Regression meta-learner

**Base learners ([`models/diabetes/`](models/diabetes/)):**

| Base Model | File | Purpose |
|---|---|---|
| XGBoost | `omni_diag_xgb_optimized.pkl` | Non-linear interaction capture, gradient-boosted decision trees |
| LightGBM | `omni_diag_lgb_optimized.pkl` | Efficient categorical feature handling, leaf-wise tree growth |
| Random Forest | `omni_diag_rf.pkl` | Robust variance reduction, ensemble bagging |

**Meta-learner:** Logistic Regression with `meta_feature_names.json` defining the stacking feature space (predicted probabilities from each base model). The meta-learner coefficients ([`models/diabetes/ensemble_metrics.json`](models/diabetes/ensemble_metrics.json)) reflect the relative contribution of each base model:

| Base Model | Meta-Learner Coefficient |
|---|---|
| XGBoost | 0.5198 |
| LightGBM | 0.4630 |
| Random Forest | 0.0172 |

The dominance of XGBoost and LightGBM coefficients indicates that tree-based gradient boosting methods extract the majority of predictive signal from the BRFSS dataset, with Random Forest contributing marginal complementary information.

**Performance metrics ([`models/diabetes/ensemble_metrics.json`](models/diabetes/ensemble_metrics.json)):**

| Metric | Single XGBoost | Stacking Ensemble | Voting Ensemble |
|---|---|---|---|
| Accuracy | 0.7541 | **0.7518** | 0.7532 |
| ROC-AUC | 0.8253 | **0.8311** | 0.8309 |
| Sensitivity | 0.8726 | **0.917** | 0.901 |
| Specificity | 0.6357 | 0.587 | 0.605 |
| Inference Threshold | 0.5 | **0.275** | 0.5 |

**Ensemble selection rationale:** The stacking ensemble was selected over single XGBoost and voting ensemble despite comparable accuracy because of its superior **sensitivity (91.7%)** — a critical clinical requirement for a screening tool where false negatives carry higher risk than false positives. The lower specificity (58.7%) is acceptable in a screening context, as positive cases would proceed to confirmatory diagnostic testing.

### 3.3 Feature Engineering Pipeline

Feature engineering is implemented as a three-stage pipeline invoked within both [`ModelLoader`](backend/model_loader.py:210) and [`EnsembleModelLoader`](backend/ensemble_loader.py:201):

```
Stage 1: Heuristic Features
  +-- engineer_heuristic(df) -> domain-specific interaction terms
  |    HeartDisease: Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio, RPP, Exercise_Risk_Index
  |    Diabetes: BMI_Age_Interaction, Health_Index, Lifestyle_Score, SES_Composite

Stage 2: Clinical Features
  +-- engineer_clinical(df) -> composite clinical risk indices
  |    HeartDisease: Thalach deviation, ST-segment stress differential
  |    Diabetes: Diabetes_Clinical_Risk (polyuria, polydipsia, fatigue, blurred vision proxies)

Stage 3: Medical Features (future expansion)
  +-- engineer_medical(df) -> placeholder for biomarker/diagnostic inputs
```

Each stage is implemented as a method on the disease-specific `FeatureEngineer` class, inheriting from [`BaseFeatureEngineer`](features/base_features.py). The method pattern uses `pd.DataFrame` transformations with a `self._log_feature()` utility for debugging and audit trail logging.

### 3.4 Inference Threshold Tuning & Clinical Sensitivity Optimization

The diabetes module employs a **clinically optimized inference threshold** of 0.275, significantly lower than the default 0.5. This threshold was determined through systematic evaluation ([`experiment_files/models/evaluate_threshold.py`](experiment_files/models/evaluate_threshold.py)) prioritizing sensitivity over specificity.

**Clinical rationale:** For diabetes screening applications, the cost of a false negative (undiagnosed diabetic patient) substantially exceeds the cost of a false positive (patient referred for confirmatory testing). The 0.275 threshold achieves 91.7% sensitivity, meaning approximately 92 of every 100 diabetic patients would be correctly identified — compared to 87.3% at the default 0.5 threshold.

**Configuration in [`configs/diabetes.yaml`](configs/diabetes.yaml):**
```yaml
inference_threshold: 0.275
```

The threshold is applied within [`EnsembleModelLoader.predict()`](backend/ensemble_loader.py:246) — predicted probabilities above the threshold are classified as positive, and the threshold value is included in the API response for audit transparency.

**Historical context:** This approach was pioneered in Phase II (V2 Streamlit prototype), where a 30% threshold achieved 96% sensitivity for heart disease prediction. The diabetes module inherits this clinical-first philosophy.

### 3.5 XGBoost 3.x Compatibility Layer

During the production deployment phase, a critical compatibility issue was identified between models trained with XGBoost 1.x/2.x serialization format and the XGBoost 3.x runtime used in the Docker container.

**Problem:** XGBoost 3.x introduced a breaking change in `save_raw()` where `base_score` values could be serialized as bracket-wrapped strings (e.g., `"0.5"` -> `"[0.5]"`), causing model loading failures with `XGBoostError: base_score must be a number`.

**Solution ([`backend/model_loader.py`](backend/model_loader.py:94) and [`backend/ensemble_loader.py`](backend/ensemble_loader.py:513)):**

A monkey-patch is applied to `xgboost.Booster.save_raw()` that intercepts the raw byte output, detects if the `base_score` field contains bracket-wrapped content, and strips the brackets before returning:

```python
def _patched_save_raw(self, raw_format="ubj"):
    raw = self.__original_save_raw(raw_format)
    raw_str = raw.decode("utf-8", errors="replace")
    if "base_score" in raw_str and "[" in raw_str:
        import re
        raw_str = re.sub(
            r'"base_score":\s*"\[([^\]]+)\]"',
            r'"base_score": \1', raw_str
        )
        raw_str = re.sub(
            r'"base_score":\s*\[([^\]]+)\]',
            r'"base_score": \1', raw_str
        )
        return raw_str.encode("utf-8")
    return raw
```

This patch is applied conditionally — only when XGBoost version >= 3.0.0 is detected.

---

## 4. Explainable AI (XAI) & The Clinical Firewall

### 4.1 SHAP Attribution Engine

SHAP (SHapley Additive exPlanations) is integrated at the model-loader level, ensuring every prediction is accompanied by feature-level attribution.

**Single-model SHAP ([`ModelLoader.explain()`](backend/model_loader.py:299)):**

1. The `TreeExplainer` from the SHAP library is instantiated as a lazy property on [`ModelLoader.explainer`](backend/model_loader.py:146)
2. `explainer.shap_values(df)` computes SHAP values for the input DataFrame
3. The raw SHAP `Explanation` object is passed to [`generate_shap_explanation()`](backend/shap_service.py:14) for serialization

**Output transformation ([`shap_service.py`](backend/shap_service.py)):**

```python
chart_data: List[FeatureImpact] = [
    {"feature": name, "impact": float(impacts[i]), "value": float(values[i])}
    for i, name in enumerate(feature_names)
]
text_explanation: str = f"The model's baseline prediction (log-odds) was {base_value:.4f}..."
```

The `text_explanation` field provides a natural-language description of the top contributing features, enabling clinicians to understand the prediction without specialized ML training.

**Zero-image philosophy:** SHAP waterfall, beeswarm, and summary plots are deliberately excluded from the API response. The platform provides quantitative feature attribution (chart_data) rather than static images, enabling the frontend to render interactive visualizations ([`ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx:27)) that clinicians can explore dynamically.

### 4.2 Ensemble SHAP Aggregation Strategy

For stacking ensembles, SHAP computation is non-trivial because the meta-learner operates on base-model probability features rather than the original clinical features. [`EnsembleModelLoader.explain()`](backend/ensemble_loader.py:427) implements a **weighted SHAP averaging** strategy:

1. **Per-model SHAP computation:** For each base model (XGBoost, LightGBM, Random Forest), a fresh `TreeExplainer` is loaded and SHAP values are computed independently ([`ensemble_loader.py`](backend/ensemble_loader.py:500))

2. **Weight calculation ([`ensemble_loader.py`](backend/ensemble_loader.py:590)):** Weights are derived from the meta-learner's learned coefficients, normalized to sum to 1:
   ```python
   raw_weights = {
       "xgb": meta_learner.coef_[0][0],    # 0.5198
       "lgb": meta_learner.coef_[0][1],    # 0.4630
       "rf":  meta_learner.coef_[0][2],    # 0.0172
   }
   normalized = {k: v / sum(raw_weights.values()) for k, v in raw_weights.items()}
   ```

3. **Weighted averaging:** Each feature's SHAP value is computed as the weighted sum across base models, with the meta-learner coefficients serving as attribution weights.

4. **Robust fallback ([`ensemble_loader.py`](backend/ensemble_loader.py:594)):** If any base model fails to generate SHAP values, the remaining models are averaged without the failed model, and the response includes a `model_agreement` field.

**Response fields specific to ensemble explanations ([`ExplainResponse`](backend/schemas.py:134)):**

| Field | Type | Description |
|---|---|---|
| `ensemble_variance` | Optional[float] | Variance of predicted probabilities across base models |
| `model_agreement` | Optional[str] | Agreement level: "Full Agreement", "Majority", "Split Decision" |

### 4.3 DiCE-Inspired Counterfactual Generator

The [`CounterfactualGenerator`](backend/counterfactual_generator.py:121) implements a counterfactual search algorithm inspired by the Diverse Counterfactual Explanations (DiCE) framework (Mothilal et al., 2020), adapted for clinical settings with additional safety constraints.

**Algorithm ([`generate()`](backend/counterfactual_generator.py:156)):**

```
Input: patient_data (dict of feature: value)
       prediction_fn (callable returning probability dict)
       pipeline_fn (callable applying feature engineering)
       num_cfs (int, default=5)

1. Sample N random perturbations of mutable features within clinical bounds
   N = num_cfs * 60 (oversampling factor for diversity selection)

2. Apply directional constraints -- reject samples violating monotonic constraints

3. Filter: keep only candidates where prediction changes from baseline

4. Compute changes, proximity scores, and feasibility for valid candidates

5. Select diverse subset using greedy algorithm with lambda = 0.5
   (trade-off between proximity to original and diversity among counterfactuals)

6. Build clinical scenario strings for each selected counterfactual

7. Return list of Counterfactual objects
```

**Oversampling factor rationale:** The factor of 60 was determined empirically. Given the feature dimensionality (12-21 base features) and the constraint satisfaction rate (approximately 1-5% of random samples survive all filters), 300 samples per counterfactual target ensures a sufficiently large candidate pool for diversity selection.

**Diversity selection algorithm ([`_select_diverse()`](backend/counterfactual_generator.py:386)):**

The algorithm implements a greedy forward selection that balances proximity and diversity:

```
Score(cf) = lambda * Proximity(cf, original) + (1-lambda) * Diversity(cf, selected)

where Proximity = 1 / (1 + normalized_euclidean_distance)
      Diversity = min over selected of Jaccard_distance(cf, selected_cf)
      lambda = 0.5 (equal weighting)
```

### 4.4 The Clinical Firewall — Five-Layer Safety System

The Clinical Firewall is a multi-layer constraint system embedded within the [`CounterfactualGenerator`](backend/counterfactual_generator.py) that prevents the generation of medically invalid counterfactual explanations.

#### Layer 1: Immutable Features ([`counterfactual_generator.py`](backend/counterfactual_generator.py:59))

Features that cannot be altered by any clinical intervention are locked against modification:

```python
IMMUTABLE_FEATURES: Set[str] = {
    "sex", "diabetes_sex", "Age", "age", "Age_Category",
    "family_history", "HeartDisease", "Cholesterol",
}
```

**Clinical rationale:** Sex, age, and family history are immutable characteristics. Cholesterol is locked because medication-induced changes would produce misleading counterfactuals that conflate lifestyle intervention with pharmacological treatment.

#### Layer 2: Directional Constraints ([`counterfactual_generator.py`](backend/counterfactual_generator.py:111))

Monotonic constraints enforce physiologically valid directions of change:

```python
DIRECTIONAL_CONSTRAINTS: Dict[str, Set[int]] = {
    "PhysicalActivity": {1},          # can only increase (never decrease)
    "Fruits": {1},                    # can only increase
    "Veggies": {1},                   # can only increase
    "Smoking": {0},                   # can only decrease or stay same
    "HvyAlcoholConsump": {0},         # can only decrease or stay same
    "exercise_angina": {0},           # can only decrease
    "ST_Slope": {1},                  # can only improve
}
```

#### Layer 3: Post-Generation Illegal Flip Detection ([`counterfactual_generator.py`](backend/counterfactual_generator.py:323))

After candidate generation, [`_compute_changes()`](backend/counterfactual_generator.py:323) explicitly audits each candidate against immutable and directional constraints, providing double-validation safety.

#### Layer 4: Deterministic Clinical Phrasing ([`counterfactual_generator.py`](backend/counterfactual_generator.py:507))

Counterfactual scenarios are rendered as structured clinical narratives rather than raw feature deltas. Three scenario templates are used:
- **Risk-reduction scenario:** Probabilistic framing for quantitative changes
- **Lifestyle scenario:** Behavioral framing for modifiable risk factors
- **Monitoring scenario:** Surveillance framing when no modifiable factors exist

#### Layer 5: Clinical Feasibility Assessment ([`counterfactual_generator.py`](backend/counterfactual_generator.py:466))

Each counterfactual is assigned a feasibility rating using a scoring heuristic:

```python
def _assess_feasibility(self, changes: Dict[str, float]) -> str:
    score = 0
    change_count = len(changes)
    if change_count <= 2:  score += 2
    elif change_count <= 4: score += 1
    if any(f in changes for f in ["BMI", "Weight"]): score -= 1
    if any(f in changes for f in ["PhysicalActivity", "Smoking"]): score += 1
    if score >= 2: return "high"
    if score >= 1: return "medium"
    return "low"
```

The feasibility rating is surfaced in the API response and rendered in the frontend as a visual indicator (high = green, medium = amber, low = red).

---

## 5. Clinical UI/UX Engineering

### 5.1 Dual-Mode Interface Architecture

The frontend implements two distinct user modes, each optimized for a specific stakeholder group. Mode switching is handled by a top-level toggle in [`App.jsx`](frontend/src/App.jsx:13) with the selected mode persisted in React state.

**Engineering Mode ([`EngineeringMode.jsx`](frontend/src/components/EngineeringMode.jsx:16)):**

Designed for ML engineers and system integrators:
- Full manual parameter input via [`DynamicClinicalForm`](frontend/src/components/DynamicClinicalForm.jsx:129)
- Raw prediction display with probability threshold indicators
- SHAP feature importance bar chart ([`ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx:27))
- Counterfactual "What-If" scenarios ([`WhatIfScenarioCard.jsx`](frontend/src/components/WhatIfScenarioCard.jsx:15))
- Raw JSON response viewer for debugging

**Clinical EMR Mode ([`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:37)):**

Designed for clinicians:
- Mock patient selection from a predefined cohort ([`mockPatients.js`](frontend/src/mockPatients.js))
- Schema-driven vital cards with color-coded risk indicators
- AI Diagnostic Panel with SHAP explanations
- Counterfactual intervention suggestions formatted as clinical recommendations

### 5.2 Schema-Driven Dynamic Form Engine

The form engine eliminates the need for hardcoded disease-specific UI components. The pipeline operates as follows:

```
GET /api/v4/{disease}/schema
  -> JSON Schema with "properties", "required", "descriptions"
  -> parseSchema() in schemaFieldParser.js
  -> FieldMetadata[] with component types, validation rules, defaults
  -> DynamicClinicalForm.jsx renders categorized accordion cards
```

**Component type resolution ([`schemaFieldParser.js`](frontend/src/utils/schemaFieldParser.js:58)):**

```javascript
function resolveComponentType(schemaField) {
  if (schemaField.enum && schemaField.enum.length <= 3) return "segmented";
  if (schemaField.type === "boolean") return "toggle";
  if (schemaField.enum) return "select";
  if (schemaField.type === "integer" && schemaField.minimum !== undefined) return "slider";
  if (schemaField.type === "number") return "number";
  return "text";
}
```

**Field categorization ([`featureCategorizer.js`](frontend/src/utils/featureCategorizer.js)):**

Fields are grouped into clinical categories (Vital Signs, Lifestyle, Medical History, Lab Results, Demographics) using keyword-based mapping. Each category renders as a collapsible accordion card.

**Form features:**
- **Randomize button:** Generates clinically plausible random values for all fields
- **Reset button:** Restores all fields to schema-defined defaults
- **Validation:** Enforced at field level using schema-defined constraints with inline error messages

### 5.3 Semantic Clinical Badges & Cognitive-Load Reduction

A key UI/UX insight during development was that raw binary classifications ("Positive"/"Negative") introduce unnecessary cognitive load in clinical contexts where probabilistic risk is the relevant decision variable.

**Clinical risk badges ([`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:141)):**

The `getFieldStatus()` function maps each patient field to a risk category with associated color coding:

| Field Category | Example Fields | Visual Indicator |
|---|---|---|
| Normal range | 120/80 BP, BMI 22 | Green "Normal" badge |
| Borderline | Pre-hypertension, BMI 27 | Amber "Monitor" badge |
| Elevated risk | High BP, BMI 32 | Red "Elevated" badge |

**Prediction display transformation:**

Instead of displaying a raw binary output, the prediction is rendered as:
- **"Elevated Risk"** (red) — probability > threshold
- **"Low Risk"** (green) — probability < threshold with wide margin
- **"Moderate Risk"** (amber) — probability near threshold (within +/-0.1)

### 5.4 Three-State Counterfactual Visualization

[`WhatIfScenarioCard.jsx`](frontend/src/components/WhatIfScenarioCard.jsx:15) implements a three-state rendering pattern:

**State 1 — Loading:** A spinner with shimmer animation is displayed while the API call is in progress. The [`OmniDiagApi`](frontend/src/api.js:10) client implements a 120-second timeout with `AbortController`.

**State 2 — Low Risk (no counterfactuals needed):** When the predicted probability is below the clinical threshold, a green card is rendered indicating no counterfactual scenarios are needed.

**State 3 — Scenarios Available:** When counterfactuals are returned, each is rendered as an interactive card showing the clinical scenario text, feature changes, new probability, risk reduction percentage, and feasibility indicator.

**Mock fallback:** When the API is unavailable, mock scenarios with a "Coming Soon" badge ensure functional demonstrations.

### 5.5 Graceful Degradation & Error Boundaries

The frontend implements multiple layers of error handling:

1. **[`ErrorBoundary.jsx`](frontend/src/components/ErrorBoundary.jsx):** React error boundary with "Reload" button
2. **[`SchemaErrorFallback.jsx`](frontend/src/components/SchemaErrorFallback.jsx):** Specific fallback for schema loading failures
3. **[`FormSkeleton.jsx`](frontend/src/components/FormSkeleton.jsx):** Skeleton loader preventing layout shift
4. **API error handling ([`api.js`](frontend/src/api.js:25)):** `_fetch()` with `AbortController` timeout
5. **Cached schema persistence ([`useDiseaseSchema.js`](frontend/src/hooks/useDiseaseSchema.js:28)):** Stale-while-revalidate caching for offline-capable form rendering

---

## 6. Commercial Viability & Scalability

### 6.1 Zero-Code-Change Disease Addition

The platform's most significant commercial advantage is the ability to add new disease modules without modifying core application code.

**Registration checklist for a new disease:**

1. Create `configs/{disease_name}.yaml` — ~50 lines of YAML configuration
2. Add Pydantic model to `backend/schemas.py` — ~30 lines of field definitions
3. (Optional) Create feature engineer in `features/` — ~100 lines of domain logic
4. Place model weights in `models/{disease_name}/` — pre-trained model artifact
5. Restart the API server — zero code changes to routing, prediction, or UI logic

**Commercial implication:** A deployment covering 50 disease modules requires only 50 YAML configurations and 50 Pydantic models. The core API, ML engine, and frontend remain unchanged. This reduces marginal cost per disease from weeks of development to hours of configuration.

### 6.2 Deployment Architecture

**Current production deployment:**

| Component | Platform | Configuration |
|---|---|---|
| Backend API | Hugging Face Spaces (Docker) | [`Dockerfile`](Dockerfile), Python 3.10-slim, 2 vCPU, 8 GB RAM |
| Frontend SPA | Vercel | [`vercel.json`](frontend/vercel.json), static export, CDN-served |
| Model weights | Hugging Face Hub | Downloaded at container startup via startup script |
| CI/CD | GitHub Actions | [`ci.yml`](.github/workflows/ci.yml) — lint, build, deploy |

**Docker image structure ([`Dockerfile`](Dockerfile)):**

```
FROM python:3.10-slim
  -> Install system dependencies (gcc, libgomp)
  -> Copy requirements.txt & install Python deps
  -> Copy backend/, configs/, features/, models/ (weights via startup.sh)
  -> EXPOSE 7860 (Hugging Face Spaces convention)
  -> CMD ["bash", "startup.sh"]
```

**Resource considerations:** The stacking ensemble for diabetes requires loading four model artifacts (XGBoost, LightGBM, Random Forest, meta-learner) plus corresponding SHAP explainers. With lazy loading, only requested disease modules consume memory. Peak memory usage per disease module is approximately 500 MB (model weights + SHAP explainer).

### 6.3 EMR Integration Pathways

**Pathway 1 — FHIR-based middleware:** A FHIR-to-OmniDiag adapter would transform FHIR R4 resources into disease-specific input format.

**Pathway 2 — SMART-on-FHIR app launch:** OmniDiag can be embedded as a SMART-on-FHIR application, receiving patient context via the SMART launch endpoint.

**Pathway 3 — REST callback integration:** EMR systems can integrate OmniDiag as an asynchronous decision support service via `POST /api/v4/{disease}/predict`.

**Auth integration:** For production EMR deployment, the API would require OAuth 2.0 integration with the EMR's identity provider and HIPAA-compliant audit logging.

### 6.4 Multi-Tenancy & Horizontal Scaling

**Database persistence:** Current operation is fully stateless (models loaded from filesystem). Multi-tenancy would require:
- PostgreSQL for tenant configuration, model metadata, and audit logs
- Tenant-specific model weight storage (object storage like S3)
- Per-tenant inference thresholds and clinical firewall configurations

**Horizontal scaling strategy:**
- API layer (FastAPI) is stateless and horizontally scalable behind a load balancer
- ML engine layer requires shared memory or model-serving sidecar to avoid redundant memory per replica
- Redis-backed caching for frequent inference queries and schema responses

### 6.5 Regulatory Compliance Trajectory

| Regulatory Milestone | Current Status | Required Enhancements |
|---|---|---|
| HIPAA compliance (US) | Not implemented | PHI encryption, access audit logs, BAA |
| GDPR compliance (EU) | Not implemented | Anonymization, right-to-deletion API, DPA |
| FDA 510(k) clearance | Not pursued | Clinical validation study, locked algorithm |
| CE-IVDR (EU) | Not pursued | Clinical performance study, post-market surveillance |

**Current risk management features:**
- Clinical Firewall (SS4.4) provides algorithmic safety guarantees
- Dual-mode interface separates engineering testing from clinical use
- API responses include threshold values and probability estimates
- Graceful degradation patterns prevent silent failures

**Limitations requiring disclosure:**
- Models trained on limited datasets (Cleveland: ~300 samples; BRFSS: ~70K with class imbalance)
- Diabetes ensemble has reduced specificity (58.7%)
- No continuous monitoring or drift detection implemented
- Counterfactual explanations based on learned patterns, not causal inference

---

## Appendix A: Clinical Data Dictionary & Feature Semantics

The OmniDiag Diabetes Risk Assessment module operates on the **CDC BRFSS 2015 Health Indicators** dataset — a balanced binary classification benchmark of ~70,000 survey responses covering 21 base clinical, behavioural, and demographic variables. This appendix provides the formal clinical semantics for every feature, matching the structure defined in [`DiabetesInput`](backend/schemas.py:183) and the [`diabetes.yaml`](configs/diabetes.yaml) configuration.

The front-end SHAP tooltip system in [`ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx) maps these raw feature names to plain-English clinical descriptions using the [`medicalDictionary.js`](frontend/src/utils/medicalDictionary.js) utility, ensuring that clinicians never see opaque column names in the explainability interface.

### A.1 Base Features (CDC BRFSS 2015)

| Feature Name | Data Type | Clinical Definition | Value Scale / Range |
|---|---|---|---|
| `HighBP` | Integer | Self-reported or clinically measured high blood pressure (hypertension) diagnosis. | `0` = No high blood pressure; `1` = High blood pressure |
| `HighChol` | Integer | Self-reported or clinically measured high cholesterol diagnosis. | `0` = No high cholesterol; `1` = High cholesterol |
| `CholCheck` | Integer | Has had a blood cholesterol check within the past 5 years. Access-to-care screening indicator. | `0` = Not checked; `1` = Checked |
| `BMI` | Float | Body Mass Index — standard weight (kg) / height² (m²) metric for obesity classification. | Range: `10.0` – `100.0` (clinical caps); WHO cut-offs: <18.5 underweight, 18.5–24.9 normal, 25–29.9 overweight, ≥30 obese |
| `Smoker` | Integer | Smoked at least 100 cigarettes over lifetime. Chronic tobacco exposure indicator. | `0` = Non-smoker; `1` = Smoker |
| `Stroke` | Integer | Ever been told by a healthcare professional that they had a stroke. Cerebrovascular history. | `0` = No stroke history; `1` = Stroke history |
| `HeartDiseaseorAttack` | Integer | Ever diagnosed with coronary heart disease (CHD) or suffered a myocardial infarction (MI). Cardiovascular comorbidity. | `0` = No CHD/MI; `1` = History of CHD or MI |
| `PhysActivity` | Integer | Performed physical activity or exercise (excluding job-related) in the past 30 days. Lifestyle behaviour. | `0` = No physical activity; `1` = Any physical activity |
| `Fruits` | Integer | Consumes fruit one or more times per day. Dietary quality indicator. | `0` = Less than 1/day; `1` = 1+ servings/day |
| `Veggies` | Integer | Consumes vegetables one or more times per day. Dietary quality indicator. | `0` = Less than 1/day; `1` = 1+ servings/day |
| `HvyAlcoholConsump` | Integer | Heavy alcohol consumption — defined as >14 drinks/week for adult men or >7 drinks/week for adult women (BRFSS definition). | `0` = Not heavy drinker; `1` = Heavy drinker |
| `AnyHealthcare` | Integer | Has any form of health insurance or healthcare coverage. Access-to-care indicator. | `0` = No healthcare coverage; `1` = Any coverage |
| `NoDocbcCost` | Integer | Indicates that at some point in the past 12 months care was needed but not obtained due to cost. Socioeconomic barrier indicator. | `0` = Could see doctor; `1` = Could not see doctor due to cost |
| `GenHlth` | Integer | Self-reported general health rating on a five-point ordinal scale. The primary subjective health perception measure in BRFSS. | `1` = Excellent; `2` = Very Good; `3` = Good; `4` = Fair; `5` = Poor |
| `MentHlth` | Integer | Number of days in the past 30 days during which mental health was assessed as "not good" due to stress, depression, or emotional problems. | Scale: `0` – `30` days |
| `PhysHlth` | Integer | Number of days in the past 30 days during which physical health was assessed as "not good" due to illness or injury. | Scale: `0` – `30` days |
| `DiffWalk` | Integer | Self-reported serious difficulty walking or climbing stairs. Functional mobility limitation indicator. | `0` = No difficulty; `1` = Serious difficulty |
| `Sex` | Integer | Biological sex assigned at birth — used as a demographic covariate in epidemiological risk models. | `0` = Female; `1` = Male |
| `Age` | Integer | 13-level age category following standard BRFSS 2015 coding. Continuous age is not available in the public dataset. | `1` = 18–24; `2` = 25–29; `3` = 30–34; `4` = 35–39; `5` = 40–44; `6` = 45–49; `7` = 50–54; `8` = 55–59; `9` = 60–64; `10` = 65–69; `11` = 70–74; `12` = 75–79; `13` = 80+ years |
| `Education` | Integer | Highest level of educational attainment. Used as a component of the SES composite index. | `1` = Never attended / kindergarten only; `2` = Grades 1–8 (elementary); `3` = Grades 9–11 (some high school); `4` = High school graduate or GED; `5` = Some college or technical school; `6` = College graduate (4+ years) |
| `Income` | Integer | Annual household income bracket. Used as a component of the SES composite index. | `1` = <$10K; `2` = $10K–$15K; `3` = $15K–$20K; `4` = $20K–$25K; `5` = $25K–$35K; `6` = $35K–$50K; `7` = $50K–$75K; `8` = ≥$75K |

### A.2 Target Variable

| Variable Name | Data Type | Clinical Definition | Value Scale / Range |
|---|---|---|---|
| `Diabetes_binary` | Integer | Indicator of diagnosed diabetes (BRFSS definition: ever told by a physician that they have diabetes, excluding prediabetes and gestational diabetes). | `0` = No diabetes; `1` = Diabetes diagnosis |

### A.3 Auto-Computed Engineered Features

These five features are computed server-side by [`DiabetesFeatureEngineer`](features/diabetes_features.py:23) during the prediction pipeline and are never exposed for user input. They are documented here for model transparency.

| Feature Name | Data Type | Formula | Clinical Rationale | Expected Range |
|---|---|---|---|---|
| `BMI_Age_Interaction` | Float | `BMI × Age` | Captures the compounding effect of elevated BMI across the lifespan — older patients with high BMI face disproportionately higher metabolic risk. | ~10.0 – 1300.0 (may vary by Age category) |
| `Health_Index` | Float | `GenHlth × (MentHlth + PhysHlth)` | Composite morbidity index: amplifies subjective health perception by the total number of unhealthy days, weighting both chronic and acute burden. | `0` – `300` |
| `Lifestyle_Score` | Integer | `PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump` | Aggregate healthy-lifestyle indicator where higher values reflect healthier behavioural patterns. Range: –2 (all negative) to +3 (all positive). | `-2` – `3` |
| `SES_Composite` | Integer | `Education × Income` | Socioeconomic status proxy combining educational attainment and income. Higher values indicate greater socioeconomic advantage. | `1` – `48` |
| `Diabetes_Clinical_Risk` | Float | `exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)` | Logarithmic diabetes risk index based on epidemiological weights drawn from CDC literature. BMI and hypertension dominate the exponent. | ~1.0 – 100.0+ |

### A.4 Heart Disease Features

For completeness, the heart disease module (Cleveland UCI dataset) features are defined in the [`HeartDiseaseInput`](backend/schemas.py:24) schema. See Section 3.1 for the feature list and [`heart_disease.yaml`](configs/heart_disease.yaml) for the complete configuration.

### A.5 UI Tooltip Integration

The [`medicalDictionary.js`](frontend/src/utils/medicalDictionary.js) front-end utility provides case-insensitive lookup of any feature name to its full clinical description, used in the SHAP bar chart tooltips ([`ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx:76)) and the Clinical EMR Mode field labels ([`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:186)). This ensures that every feature attribution shown to a clinician is accompanied by a plain-English definition, removing the cognitive gap between raw column names and clinical reality.

---

## Appendix B: Key File Reference

| File | Purpose | Key Classes/Functions |
|---|---|---|
| [`backend/main.py`](backend/main.py) | FastAPI application entry point, route definitions | `app`, `health_check()`, `explain_disease()` |
| [`backend/router.py`](backend/router.py) | Dynamic disease routing and model dispatch | `OmniDiagRouter`, `_create_loader()`, `_get_loader()` |
| [`backend/schemas.py`](backend/schemas.py) | Pydantic models for API request/response | `HeartDiseaseInput`, `DiabetesInput`, `ExplainResponse`, `Counterfactual` |
| [`backend/model_loader.py`](backend/model_loader.py) | Single XGBoost model loading and inference | `ModelLoader`, XGBoost 3.x monkey-patch |
| [`backend/ensemble_loader.py`](backend/ensemble_loader.py) | Stacking ensemble inference and SHAP aggregation | `EnsembleModelLoader`, weighted SHAP averaging |
| [`backend/counterfactual_generator.py`](backend/counterfactual_generator.py) | DiCE-inspired counterfactual search | `CounterfactualGenerator`, Clinical Firewall |
| [`backend/shap_service.py`](backend/shap_service.py) | SHAP explanation serialization | `generate_shap_explanation()` |
| [`configs/heart_disease.yaml`](configs/heart_disease.yaml) | Heart disease module configuration | Model paths, feature names, version 5.0.0 |
| [`configs/diabetes.yaml`](configs/diabetes.yaml) | Diabetes module configuration | Ensemble model paths, threshold 0.275 |
| [`configs/config_loader.py`](configs/config_loader.py) | Centralized config loading utility | `load_config()`, `resolve_path()`, `resolve_file_path()` |
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | Main application shell | Dual-mode routing, sidebar, disease selector |
| [`frontend/src/components/EngineeringMode.jsx`](frontend/src/components/EngineeringMode.jsx) | Developer testing interface | Full parameter form, raw JSON view |
| [`frontend/src/components/ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx) | Clinical decision support interface | Patient selector, vital cards, AI panel |
| [`frontend/src/components/DynamicClinicalForm.jsx`](frontend/src/components/DynamicClinicalForm.jsx) | Schema-driven dynamic form renderer | `CategoryCard`, randomize/reset |
| [`frontend/src/components/ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx) | SHAP feature importance visualization | Recharts horizontal bar chart, medical tooltips |
| [`frontend/src/components/WhatIfScenarioCard.jsx`](frontend/src/components/WhatIfScenarioCard.jsx) | Counterfactual scenario display | 3-state rendering, mock fallback |
| [`frontend/src/utils/schemaFieldParser.js`](frontend/src/utils/schemaFieldParser.js) | JSON Schema -> FieldMetadata parser | `resolveComponentType()`, `parseSchema()` |
| [`frontend/src/api.js`](frontend/src/api.js) | API client with timeout handling | `OmniDiagApi` class, `_fetch()` with AbortController |
| [`features/heart_disease_features.py`](features/heart_disease_features.py) | Heart disease feature engineering | `HeartDiseaseFeatureEngineer` |
| [`features/diabetes_features.py`](features/diabetes_features.py) | Diabetes feature engineering | `DiabetesFeatureEngineer`, composite indices |
| [`Dockerfile`](Dockerfile) | Container definition | Python 3.10-slim, HF Spaces config |

## Appendix C: Model Performance Comparison

| Metric | Heart Disease (XGBoost) | Diabetes (Stacking Ensemble) |
|---|---|---|
| Accuracy | 80.17% | 75.18% |
| ROC-AUC | 85.64% | 83.11% |
| Sensitivity | 80.46% | 91.70% |
| Specificity | 83.13% | 58.70% |
| Inference Threshold | 0.420 | 0.275 |
| Base Features | 12 | 21 |
| Engineered Features | 5 | 5 |
| Training Samples | ~303 | ~35,000 (balanced subset) |

## Appendix D: Git Branch History

| Branch | Phase | README Summary |
|---|---|---|
| `main1` | I | Streamlit + scikit-learn prototype, Arabic UI, basic ML models |
| `origin/V2-advanced-model` | II | Streamlit + Stacking Ensemble (RF+GB+SVC->LR), 96% recall at 30% threshold |
| `V3-advanced-DL` | III | **Complete rewrite**: FastAPI + React + OmniDiagRouter + bilingual NLP, single XGBoost |
| `feature/diabetes-module-integration` | IV | Diabetes config + ensemble loader introduced |
| `feature/omni-platform-final` | IV (current) | Full multi-disease CDSS (CAD + DM), CounterfactualGenerator, Clinical Firewall |
| `feature/ui-ux-polish` | V | UI refinements: semantic badges, three-state counterfactuals, error boundaries |
| `production-final-v3` | VI | Production hardening: Docker, CI/CD, XGBoost 3.x patch |
| `hf-deploy` | VI | Hugging Face Spaces deployment configuration |

---

*Document prepared for IEEE AI Expo 2026. All metrics and architectural descriptions reflect the state of the `feature/omni-platform-final` branch as of June 2026. Git branch READMEs were consulted as primary source material for the architectural evolution narrative (Section 1).*
