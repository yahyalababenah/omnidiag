# OmniDiag — Implementation Details

**Version:** 4.0.0 API | **Last Updated:** 2026-05-31  
**Project:** Multi-Disease Diagnostic Platform with Explainable AI (XAI)

---

OmniDiag's technology stack was meticulously engineered to prioritize high-performance inference, clinical data security, and architectural scalability. The implementation leverages the following core technologies across its decoupled layers:

---

## 1. Core Programming Languages

| Language | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Primary backend language for ML inference, data preprocessing, feature engineering, and asynchronous API handling via FastAPI + Uvicorn. |
| **JavaScript (ES6+) / JSX** | ES2021 | Frontend SPA built with React 18 — dynamic form rendering, SHAP waterfall visualizations, and clinical EMR dashboard. |

---

## 2. AI & Machine Learning Frameworks

### 2.1 XGBoost (2.0+)
The core gradient-boosted decision tree framework for the Heart Disease module. Chosen for state-of-the-art performance on clinical tabular data with built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`). Hyperparameters tuned via Optuna Bayesian search (300 trials, 5-fold stratified CV): 898 estimators, `max_depth=5`, `learning_rate=0.0136`, `colsample_bytree=0.5455`. Achieves **88.98% 5-fold CV accuracy**.

### 2.2 LightGBM (4.0+)
Gradient-boosting framework used as a base model in the Diabetes stacking ensemble. Selected for its leaf-wise tree growth strategy that converges faster than XGBoost's level-wise approach on high-dimensional BRFSS data (21 features). Optimized via Optuna alongside XGBoost and Random Forest.

### 2.3 scikit-learn (1.3+)
Provides three critical components:
- **RandomForestClassifier**: Third base model in the diabetes stacking ensemble (180 estimators, entropy criterion).
- **LogisticRegression**: Meta-learner for the stacking ensemble, learns optimal per-input weighting of base model probabilities.
- **StandardScaler**: Z-score normalization fitted during training and serialized via `joblib` for inference-time preprocessing.

### 2.4 SHAP (0.42+)
Specifically utilizing `shap.TreeExplainer` to compute exact Shapley values for the Explainable AI (XAI) layer. For the Heart Disease single-model architecture, TreeExplainer computes exact values in polynomial time by exploiting tree structure. For the Diabetes ensemble, SHAP values from each base model are combined using meta-learner coefficients (stacking) or equal weights (voting), with a robust fallback mechanism that redistributes weights if any single base model's explainer fails.

### 2.5 Optuna
Advanced Bayesian hyperparameter optimization framework. Systematically tunes the XGBoost ensemble (Heart Disease: 300 trials) and all three base models + meta-learner (Diabetes: 100 trials per model). Uses Tree-structured Parzen Estimator (TPE) sampler with Median Pruner for early termination of unpromising trials.

### 2.6 Pandas (2.0+) & NumPy (1.24+)
Foundation of the preprocessing pipeline and feature engineering layer. Pandas DataFrames flow through the multi-stage pipeline: raw payload → feature engineering (heuristic, clinical, medical paths) → preprocessing (Label Encoding, StandardScaler) → inference. NumPy handles array operations for SHAP value computation, counterfactual perturbation sampling, and distance calculations.

---

## 3. Backend & API Infrastructure

### 3.1 FastAPI (0.100+)
The ASGI web framework powering the backend. Specifically chosen for:
- **Native asynchronous capabilities**: Enables high-throughput parallel inference via `async/await`.
- **Automatic OpenAPI documentation**: Generates interactive Swagger UI at `/docs` for debugging and clinical validation.
- **Pydantic integration**: Every disease has a dynamically generated Pydantic schema that validates incoming patient data at the boundary before any processing logic executes.

The API exposes four core endpoints under `/api/v4/{disease}`:
| Endpoint | Method | Purpose |
|---|---|---|
| `/schema` | GET | Returns JSON Schema for dynamic frontend form generation |
| `/predict` | POST | Runs inference, returns binary prediction + confidence |
| `/explain` | POST | Runs inference + SHAP explanation, returns chart data + text summary |
| `/counterfactuals` | POST | Generates DiCE-inspired what-if scenarios (Diabetes only) |

### 3.2 Pydantic (2.0+)
Handles rigorous, dynamic data validation. Each disease config dynamically maps to a Pydantic `BaseModel` that strictly validates incoming patient data — enforcing:
- **Range constraints**: e.g., `RestingBP` must be 80–220 mm Hg, `Age` 20–100 years — ensuring inputs fall within physically possible clinical bounds.
- **Literal types**: Categorical fields like `ChestPainType` accept only valid codes (`'TA'`, `'ATA'`, `'NAP'`, `'ASY'`).
- **Auto-computed features**: Five features (RPP, Exercise_Risk_Index, Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio) are computed server-side from 12 base inputs — the Pydantic schema accepts only the base fields and delegates computation to the feature engineer.

### 3.3 PyYAML (6.0+)
The engine driving the **Zero-Code-Change** architecture. Each `*.yaml` file in the `configs/` directory is parsed at startup by the dynamic `OmniDiagRouter`, which registers new diseases into the FastAPI router without any code changes. The config schema defines:
- Disease metadata (name, display name, target column)
- Data paths (raw, processed, interim)
- Feature engineering configuration (module, class, categorical/numerical columns, engineered feature formulas)
- Model configuration (type, explainer type, weights path, preprocessors path, hyperparameters)
- Training parameters (test size, random state, optimization trials)
- Ensemble configuration (base models, meta-learner, inference threshold)

### 3.4 Dynamic Router Architecture

The `OmniDiagRouter` is the central nervous system of the backend:

```
                    ┌─────────────────────┐
                    │   configs/*.yaml     │
                    │  (heart_disease,     │
                    │   diabetes, ...)      │
                    └──────────┬──────────┘
                               │ parses
                               ▼
                    ┌─────────────────────┐
                    │   OmniDiagRouter    │
                    │  (scans at startup) │
                    └──┬──────────────┬───┘
                       │              │
              ┌────────▼───┐   ┌──────▼────────┐
              │ ModelLoader│   │EnsembleModel- │
              │ (single    │   │Loader         │
              │  XGBoost)  │   │(stacking/     │
              │            │   │ voting)       │
              └────────────┘   └───────────────┘
                       │              │
                       └──────┬───────┘
                              ▼
                    ┌─────────────────────┐
                    │   FastAPI Router    │
                    │  /api/v4/{disease}/ │
                    │  predict | explain  │
                    └─────────────────────┘
```

Key design principles:
- **Zero hardcoded disease references**: Adding a new disease = drop a YAML file + implement a feature engineer class.
- **Lazy model loading**: Models are loaded on the first request, not at container startup — reducing cold-start latency.
- **Auto-detection**: If a config has a `model.ensemble` section, `EnsembleModelLoader` is used instead of `ModelLoader`.
- **CORS strict allowlist**: Production Vercel URLs only, configurable via `CORS_ALLOWED_ORIGINS` environment variable.

---

## 4. Frontend & Clinical UI

### 4.1 React 18 & Vite 5
Forms the foundation of the Single-Page Application (SPA). Vite provides highly optimized production builds (~400 KB gzipped) and rapid Hot Module Replacement during development. The app is structured as:
- **Context**: `DiseaseContext.jsx` — global state for selected disease, API client, and disease info.
- **Hooks**: `useDiseaseSchema.js` — fetches and caches the JSON Schema for the active disease; `useDiseaseForm.js` — manages form state and validation.
- **Utils**: `schemaToZod.js` — converts backend JSON Schema to Zod validation schemas client-side; `featureCategorizer.js` — categorizes fields into Vitals, Demographics, Lifestyle, etc.; `medicalDictionary.js` — maps feature names to Arabic/English clinical descriptions.

### 4.2 Tailwind CSS 3
Utility-first CSS framework used to design the responsive, color-coded **Clinical EMR Mode** dashboard. Custom clinical theme with:
- `clinical-bg` background, `clinical-border` borders, `primary-600` accent
- Vital sign cards with color thresholds: green (normal), yellow/warning (elevated), red (critical)
- Responsive sidebar with mobile overlay pattern
- Gradient risk badges (green `CheckCircle2` / red `AlertCircle`)

### 4.3 Recharts 3
Composable charting library leveraged to render the custom SHAP Waterfall visualizations. Key features:
- Horizontal bar chart with red (risk-increasing) / green (protective) color encoding
- Custom Y-axis ticks using SVG `foreignObject` for **MedicalTooltip** hover effects
- Dynamic chart height based on feature count
- "Show All N Features" toggle for truncated views
- Responsive container with `maxVisible` prop (default 10)

### 4.4 Dual-Mode Interface

The frontend implements two distinct views toggled via sidebar navigation:

**Engineering Mode** — for ML engineers and data scientists:
- Schema-driven dynamic form with accordion sections
- Manual parameter adjustment (selects, toggles, sliders, number inputs)
- Parallel `predict` + `explain` API calls
- Raw JSON inspection panels
- Randomize / Reset buttons for edge-case testing

**Clinical EMR Mode** — for physicians and clinical staff:
- Mock patient selector with realistic clinical profiles
- Color-coded vital signs monitor grid
- AI Diagnostic Panel with large prediction badge + confidence
- Bilingual toggle (English/Arabic) for MENA region
- Counterfactual What-If scenario cards with feasibility labels
- Cumulative risk-reduction progress bar

### 4.5 API Client
The `OmniDiagApi` class encapsulates all backend communication:
- 120-second `AbortController` timeout for HF Spaces cold starts
- Descriptive error messages for timeout vs. HTTP errors
- All four endpoints: `health()`, `listDiseases()`, `getSchema()`, `predict()`, `explain()`, `counterfactuals()`
- Configurable base URL via `VITE_API_BASE` environment variable (defaults to HF Spaces URL)

### 4.6 Deployment Configuration
- **Vercel**: Frontend deployed globally with SPA rewrites via `vercel.json`
- **Environment variable**: `VITE_API_BASE` configured per Vercel environment (production, preview, development)

---

## 5. Database & State Management (Stateless Design)

To ensure strict adherence to medical data privacy standards (e.g., HIPAA, GDPR), OmniDiag is architected as a **Stateless Inference API**:

- **No persistent database**: The system does not store patient data in any relational database (PostgreSQL, MySQL) or document store.
- **Immediate payload disposal**: Once the prediction and SHAP explanation are returned to the client, the patient payload is immediately discarded from memory.
- **Persistent configuration**: System configurations and schemas are stored as YAML artifacts in `configs/`.
- **Serialized preprocessors**: Label encoders and StandardScaler states are managed via `joblib` serialization and stored in model-specific preprocessor directories.
- **Mock patient data**: The frontend ships with pre-defined mock patients for demo/EHR-simulation purposes — no real PHI is ever transmitted.

---

## 6. Feature Engineering Pipeline

### 6.1 Abstract Base Class
All feature engineers inherit from `BaseFeatureEngineer`, providing a uniform interface:
- `engineer_heuristic(df)` — statistical interaction features
- `engineer_clinical(df)` — medically-weighted risk scores
- `engineer_medical(df)` — domain-specific biomarkers (cardiology only)

### 6.2 Heart Disease Features
| Feature | Formula | Category |
|---|---|---|
| `Age_BP_Interaction` | `Age × RestingBP` | Heuristic |
| `HR_Age_Ratio` | `MaxHR / Age` | Heuristic |
| `Chol_Age_Ratio` | `Cholesterol / Age` | Heuristic |
| `Clinical_Risk_Score` | `exp(Age × 0.048 + RestingBP × 0.015 + Cholesterol × 0.002)` | Clinical |
| `RPP` (Rate-Pressure Product) | `RestingBP × MaxHR` | Medical |
| `Exercise_Risk_Index` | `Oldpeak × ExerciseAngina(encoded)` | Medical |

### 6.3 Diabetes Features
| Feature | Formula | Category |
|---|---|---|
| `BMI_Age_Interaction` | `BMI × Age` | Heuristic |
| `Health_Index` | `GenHlth × (MentHlth + PhysHlth)` | Heuristic |
| `Lifestyle_Score` | `PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump` | Heuristic |
| `SES_Composite` | `Education × Income` | Heuristic |
| `Diabetes_Clinical_Risk` | `exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)` | Clinical |

---

## 7. Model Architectures

### 7.1 Single Model (Heart Disease)
- **Algorithm**: XGBoost Classifier
- **Features**: 12 base + 5 engineered = 17 total features
- **Optimization**: Optuna Bayesian search, 300 trials, 5-fold stratified CV
- **Best params**: `n_estimators=898`, `max_depth=5`, `learning_rate=0.0136`, `colsample_bytree=0.5455`, `subsample=0.9637`, `reg_alpha=3.52`, `reg_lambda=0.478`
- **Validation accuracy**: 88.98%
- **Model weight**: ~200–300 KB serialized `.pkl`

### 7.2 Stacking Ensemble (Diabetes)
- **Base models**: XGBoost (optimized), LightGBM (optimized), Random Forest (180 estimators, entropy)
- **Meta-learner**: Logistic Regression trained on out-of-fold predictions
- **Fallback**: Soft Voting (equal-weight average of base models)
- **Clinical threshold**: 0.275 — calibrated with FN penalty weight 2× for 91.7% sensitivity
- **Ensemble variance**: Standard deviation of base model probabilities reported as `model_agreement` label (high / moderate / low)

### 7.3 SHAP Explainability per Architecture

```
Single Model (Heart Disease):
    shap.TreeExplainer(model)(df) → Explanation object
                                    → generate_shap_explanation()
                                    → { chart_data, text_explanation, base_value }

Ensemble (Diabetes):
    Per-model: shap.TreeExplainer(base_model)(df) → per_model_shap_values
    Weighted combination:
        stacking: Σ(meta_learner_coefficient_i × per_model_shap_i)
        voting:   average(per_model_shap_i)
    Fallback: if any base model explainer fails, redistribute weights
    → generate_shap_explanation() → { chart_data, text_explanation, base_value,
                                       per_model_shap, shap_weights,
                                       ensemble_variance, model_agreement }
```

---

## 8. Counterfactual Engine (CIDI — Clinical Intervention Discovery Interface)

A DiCE-inspired algorithm that generates diverse "what-if" scenarios exclusively for the Diabetes module:

### Algorithm
1. Sample 500+ random perturbations of mutable patient features
2. Pass each perturbation through the full pipeline (engineer + preprocess + predict)
3. Filter perturbations that flip Positive → Negative (y_target = 0)
4. Score candidates by proximity (normalized L1 distance) with diversity penalty (Jaccard similarity, λ = 0.5)
5. Select top 3 most diverse counterfactuals

### Five-Layer Clinical Firewall
| Layer | Mechanism | Clinical Purpose |
|---|---|---|
| 1 | Immutable Features | Prevents changing biological/socioeconomic givens (sex, age, past medical history) |
| 2 | Directional Constraints | Only allows changes toward clinically safer values |
| 3 | Illegal Flip Detection | Post-generation hard-coded filter — defense in depth |
| 4 | Deterministic Clinical Phrasing | Unconditionally health-positive text in explanations |
| 5 | Feasibility Assessment | Labels scenarios as high (lifestyle), medium (medical), or low (major intervention) |

**Note**: Requires zero external dependencies beyond NumPy and Pandas — the original `dice-ml` library was replaced due to incompatibility with Pandas 3.0 (`LossySetitemError`).

---

## 9. Deployment, MLOps & Containerization

### 9.1 Docker Containerization
The backend API is fully containerized:
- **Base image**: `python:3.10-slim` — minimal attack surface
- **Non-root user**: Executes under `omnidiag` (UID 1000) for medical software security compliance
- **HEALTHCHECK**: `--interval=30s --timeout=10s --start-period=60s --retries=3` — necessary for Hugging Face Spaces orchestration
- **System dependencies**: `gcc`, `libgomp1` (scikit-learn/SHAP), `curl` (model download)
- **Image size**: ~400 MB (model weights excluded)

### 9.2 Decoupled Model Registry (Hugging Face Hub)
To maintain an ultra-lightweight Docker image:
- Heavy `.pkl` model weights are excluded from `.dockerignore` and the Docker build
- A `startup.sh` script fetches the latest weights from a Git-LFS Hugging Face Dataset Hub repository upon container initialization
- If weights are already cached on disk, the download is skipped
- This decoupling enables model updates without image rebuilds

### 9.3 Hugging Face Spaces
Hosts the containerized FastAPI backend, providing scalable serverless GPU/CPU environments with:
- Auto-scaling from idle to active (cold starts of 60–120s)
- Integrated health-check monitoring
- Environment variable configuration (`CORS_ALLOWED_ORIGINS`)

### 9.4 Vercel Edge Network
Hosts the React frontend, ensuring ultra-low latency global content delivery:
- SPA rewrites via `vercel.json`
- Environment variable `VITE_API_BASE` per environment
- Instant scaling with zero configuration

### 9.5 CI/CD Pipeline
GitHub Actions runs on every push and pull request:

```yaml
Backend CI:
  - Python 3.11 setup with pip cache
  - pip install -r requirements.txt
  - Verify all backend modules import successfully
  - Start Uvicorn, hit health endpoint, verify response

Frontend CI:
  - Node.js 20 setup with npm cache
  - npm ci (clean install)
  - npm run build (verifies production build succeeds)
```

---

## 10. Security & Compliance Architecture

| Concern | Implementation |
|---|---|
| **Data Privacy** | Stateless API — no patient data stored; immediate payload disposal after inference |
| **Container Security** | Non-root user (`omnidiag`); minimal base image (`python:3.10-slim`) |
| **CORS** | Strict allowlist — only Vercel production URLs + localhost; configurable via env var |
| **Input Validation** | Pydantic schemas with `ge`/`le` range constraints; Literal types for categoricals |
| **Model Access** | Weights fetched from HF Hub at startup; excluded from Docker image |
| **Dependency Pinning** | Minimum version pins in `requirements.txt` balance reproducibility with security patches |

---

## 11. Development & Experimentation Tooling

| Tool | Purpose |
|---|---|
| **Jupyter Notebooks** | Kaggle-based ensemble training notebook |
| **Training Scripts** | Heart Disease XGBoost v5 training, Diabetes XGBoost/LightGBM training, Diabetes ensemble training |
| **Optimization Scripts** | Optuna-based hyperparameter search for XGBoost v5 and Diabetes models |
| **Evaluation Scripts** | Clinical threshold calibration with FN/FP cost analysis |
| **Data Pipeline** | Data merging, cleaning, transformation, and preprocessing scripts for both Heart Disease and Diabetes datasets |
| **Experiment Logs** | Comprehensive experiment logs and analysis reports |
| **Feature Engineering Library** | Shared math library (`advanced_feature_engineering.py`) reused by both heart disease and diabetes modules |
