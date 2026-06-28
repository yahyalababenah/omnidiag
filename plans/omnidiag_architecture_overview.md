# OmniDiag System Architecture Overview

**Version:** 4.0.0 (API) | **Status:** Production-Ready  
**Frontend:** React 18 + Vite + Tailwind CSS (Vercel)  
**Backend:** FastAPI + Uvicorn (Hugging Face Spaces / Docker)  
**Last Updated:** 2026-05-31

---

OmniDiag operates on a highly decoupled, client-server topology engineered for clinical scalability, high availability, and Zero-Code-Change disease expansion. The architecture is logically partitioned into five sequential layers, each with a distinct responsibility:

---

## Layer 1: Dynamic Input Layer

Unlike rigid systems that hardcode API contracts per disease, OmniDiag's input layer is driven by a **dynamic YAML Config Engine**. When a request is initiated, the FastAPI backend resolves incoming clinical data against a dynamically generated Pydantic schema specific to the target disease.

**How it works:**

1. A dynamic router scans the `configs/` directory at startup and parses each `*.yaml` file into a disease descriptor.
2. On each incoming `POST /api/v4/{disease}/predict` or `explain` request, the router looks up the disease name in a registry and returns the appropriate Pydantic `BaseModel`.
3. The Pydantic model validates all input fields — enforcing type constraints, numerical ranges (`ge`/`le`), and required fields — before the data reaches any processing logic.

**Input composition:** Standard EMR data points including patient demographics (age, sex), vital signs (resting blood pressure, maximum heart rate), and laboratory results (fasting blood sugar, serum cholesterol, ECG findings). The schema is defined per-disease in YAML and exposed to the frontend via `GET /api/v4/{disease}/schema` for dynamic form generation.

**Key architectural decision:** Adding a new disease requires only a YAML config and a Pydantic schema registration — no new endpoints, no routing logic, no middleware changes.

---

## Layer 2: Server-Side Processing Pipeline

Upon validation, the raw JSON payload enters a deterministic, multi-stage processing pipeline before reaching the inference engine. This pipeline is invoked by a disease-specific Feature Engineer class that inherits from an abstract base class.

### Stage 2a: Feature Engineering

The payload passes through a plugin-based **Feature Engineer** class that computes advanced physiological heuristics server-side:

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `Age_BP_Interaction` | `Age × RestingBP` | Captures compound risk of vascular aging with hypertensive load |
| `HR_Age_Ratio` | `MaxHR / Age` | Cardio-fitness index; higher-than-expected MaxHR for age signals autonomic dysregulation |
| `Chol_Age_Ratio` | `Cholesterol / Age` | Normalizes lifetime lipid exposure; same cholesterol level carries different risk at age 30 vs. 70 |
| `RPP` (Rate-Pressure Product) | `RestingBP × MaxHR` | Myocardial oxygen consumption — a global cardiology standard for ischemic threshold assessment |
| `Exercise_Risk_Index` | `Oldpeak × ExerciseAngina` | Composite marker for exercise-induced ischemia; zero when no angina present |

For the diabetes module, the feature engineer computes analogous heuristics:

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `BMI_Age_Interaction` | `BMI × Age` | Compounding obesity risk across lifespan |
| `Health_Index` | `GenHlth × (MentHlth + PhysHlth)` | Composite morbidity score |
| `Lifestyle_Score` | `PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump` | Aggregate healthy-behaviour indicator |
| `SES_Composite` | `Education × Income` | Socioeconomic status proxy |
| `Diabetes_Clinical_Risk` | `exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)` | Logarithmic epidemiological risk index |

### Stage 2b: Preprocessing

The augmented feature set is passed through serialized preprocessors (loaded via `joblib` from a disease-specific preprocessors directory):

- **Categorical variables** undergo Label Encoding (e.g., `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`).
- **Numerical variables** are standardized to zero mean and unit variance via Z-score scaling (`StandardScaler`), ensuring alignment with the model's expected training distribution.
- The preprocessors are fitted during the training phase and serialized for runtime use — guaranteeing identical transformations across training and inference.

---

## Layer 3: The Predictive Model — Decoupled Engine

The core inference engine is a gradient-boosted decision tree model. Architecturally, the model weights are **decoupled from the application codebase** to maintain a lightweight, secure deployment footprint.

### Single-Model Architecture (Heart Disease)

- **Algorithm:** XGBoost Classifier, optimized via Optuna Bayesian search (300 trials, 5-fold stratified cross-validation).
- **Hyperparameters:** 898 estimators, `max_depth=5`, `learning_rate=0.0136`, `colsample_bytree=0.5455` — favoring low learning rates with high estimators for generalization.
- **Validation Accuracy:** 88.98% (5-fold CV) on the heuristic feature set.
- **Inference Speed:** Single-digit milliseconds per prediction.
- **Model Weight:** ~200–300 KB serialized `.pkl` file.

### Ensemble Architecture (Diabetes)

- **Algorithm:** Stacking ensemble of three heterogeneous base models — XGBoost, LightGBM, and Random Forest — with a Logistic Regression meta-learner.
- **Meta-Learner Role:** Learns optimal model weighting per input rather than applying a static average; if one model systematically underperforms on specific patient profiles, the meta-learner dynamically down-weights it and up-weights others.
- **Fallback Mode:** Soft Voting (equal-weight average) when the meta-learner is unavailable.
- **Clinical Threshold:** 0.275 — calibrated with a false-negative penalty weight of 2×, achieving 91.7% sensitivity for diabetes screening.
- **Model Agreement Signals:** The ensemble computes `ensemble_variance` (standard deviation of base model probabilities) and a qualitative `model_agreement` label (high / moderate / low) to inform clinical caution.

### Dynamic Weight Acquisition

To ensure the Docker container remains lightweight and secure:

1. Model weights (`.pkl` files) are **excluded from the Docker image**.
2. At container startup, a startup script downloads the latest weights from a dedicated Hugging Face Dataset Hub repository.
3. If weights are already cached on disk, the download is skipped.
4. Models are loaded lazily — on the first request to a disease — reducing cold-start time when multiple diseases are registered.

This decoupling offers three advantages: (a) smaller Docker images under 1 GB, (b) model updates without image rebuilds, and (c) granular access control between public and private repositories.

---

## Layer 4: The Output and Explainability Layer (XAI)

The system output is dual-faceted, returning both the diagnostic prediction and its mathematical justification in a single structured response.

### SHAP Attribution Pipeline

Immediately following inference, the preprocessed data is routed through `shap.TreeExplainer` (for single models) or a weighted average of per-model SHAP values (for ensembles):

1. **TreeExplainer** computes exact Shapley values in polynomial time by exploiting the tree structure — avoiding the exponential cost of model-agnostic approximation.
2. The raw `Explanation` object is transformed into a frontend-ready JSON structure with `chart_data` (sorted by absolute SHAP value), `base_value`, and `text_explanation`.
3. For ensembles, SHAP values from each base model are combined using meta-learner coefficients (stacking) or equal weights (voting), with a robust fallback that redistributes weights if any base model's explainer fails.

### API Response Payload

The response from `POST /api/v4/{disease}/explain` contains:

| Field | Description |
|---|---|
| `prediction` | Binary diagnosis (`0` = Negative, `1` = Positive) |
| `confidence` | Probability score from the model (or meta-learner for ensembles) |
| `chart_data` | Ranked array of exact Shapley values quantifying the specific risk contribution of every patient feature |
| `text_explanation` | A dynamically generated, natural-language clinical summary (bilingual: English/Arabic) that translates SHAP data into a readable medical justification |
| `base_value` | The expected model output before any feature contributions |

For ensembles, additional fields include `per_model_shap`, `shap_weights`, `ensemble_variance`, and `model_agreement`.

### Counterfactual Engine (CIDI)

The diabetes module extends explainability with a DiCE-inspired counterfactual generator that:

1. Samples 500+ random perturbations of mutable patient features.
2. Evaluates each through the full feature engineering + preprocessing + prediction pipeline.
3. Filters perturbations that flip the prediction from Positive to Negative.
4. Scores candidates by proximity (normalized L1 distance) with a diversity penalty (Jaccard similarity on changed feature sets, λ = 0.5).
5. Selects the top 3 most diverse counterfactuals — each targeting a different intervention pathway (e.g., one focused on BMI + diet, another on blood pressure + exercise, another on smoking cessation).

All counterfactuals pass through a **five-layer Clinical Firewall** that prevents clinically unsafe or biologically implausible proposals:

| Layer | Mechanism | Clinical Purpose |
|---|---|---|
| 1 | Immutable Features | Prevents changing biological/socioeconomic givens (sex, age, past medical history) |
| 2 | Directional Constraints | Only allows changes toward clinically safer values (e.g., `HighBP → 0`, `Smoker → 0`, `PhysActivity → 1`) |
| 3 | Illegal Flip Detection | Post-generation hard-coded filter — defense in depth |
| 4 | Deterministic Clinical Phrasing | Unconditionally health-positive text (e.g., "controls blood pressure," never "raises blood pressure") |
| 5 | Feasibility Assessment | Labels each scenario as high (lifestyle), medium (medical intervention), or low (major intervention) feasibility |

---

## Layer 5: The Dual-Mode Frontend Interface

The frontend is a modern Single-Page Application (SPA) built with **React 18**, **Vite**, and **Tailwind CSS**, deployed globally via Vercel. It implements a deliberate **Dual-Mode** interface to serve two distinct user personas:

### Engineering Mode

Designed for **ML engineers, data scientists, and model evaluators**. Key capabilities:

- **Manual parameter adjustment:** Every clinical feature is presented as a typed input (select dropdowns for categoricals, number inputs with validation bounds for continuous variables, toggle switches for binaries, sliders for bounded ranges).
- **Schema-driven dynamic form:** The form renders automatically from the JSON Schema returned by `GET /api/v4/{disease}/schema`, with fields categorized into collapsible accordion sections (Vitals, Demographics, Lifestyle, etc.).
- **Real-time inference:** Parallel `predict` and `explain` API calls with loading states, cold-start notifications, and error boundaries.
- **Raw JSON inspection:** Expandable JSON viewers for both prediction results and SHAP explanations.
- **SHAP waterfall visualization:** Recharts-based horizontal bar chart with red (risk-increasing) and green (protective) color encoding, interactive tooltips, and a "Show All N Features" toggle.
- **Randomize / Reset buttons** for stress-testing edge cases.

### Clinical EMR Mode

Designed for **physicians and clinical staff** to integrate into high-stress workflows. Key capabilities:

- **Patient selector:** Dropdown of pre-defined mock patients with realistic clinical profiles, medications, and admitting complaints.
- **Vitals monitor:** A grid of color-coded vital sign cards (green = normal, yellow/warning = elevated, red = critical) with medically relevant thresholds — enabling sub-2-second patient risk assessment.
- **AI Diagnostic Panel:** Large, color-coded diagnosis badges (green `CheckCircle2` / red `AlertCircle`) with confidence percentage and SHAP base value.
- **Clinical Insights section:** SHAP bar chart, natural-language feature impact summary, and expandable detailed feature impact table with ↑ Risk / ↓ Protective indicators.
- **Bilingual toggle:** English/Arabic language switch for the MENA region.
- **Counterfactual scenarios:** What-If cards with risk-reduction percentages, feasibility labels, and a cumulative risk-reduction progress bar.

### Graceful Degradation

All three diagnostic endpoints (`/predict`, `/explain`, `/counterfactuals`) are fired in parallel from the frontend using `Promise.all()`:

1. **Predict + Explain** are required — if either fails, an error is displayed.
2. **Counterfactuals** are optional — if unsupported or the backend is unavailable, mock placeholder data is rendered with a "Coming Soon" badge.
3. **Cold start handling:** A "Waking up the diagnostic engine..." message appears after 8 seconds of loading (Hugging Face Spaces idle/sleep policy).
4. **API timeout:** The API client implements a 120-second AbortController timeout with descriptive error messages.

---

## Summary: Zero-Code-Change Scalability

The defining architectural property of OmniDiag is that **adding a new disease requires no modifications to the core API surface, router logic, or frontend routing**. The complete workflow is:

1. **Create a YAML config** defining the disease name, features, model paths, and preprocessors.
2. **Implement a feature engineer** inheriting from the abstract base class.
3. **Register a Pydantic schema** in the schema registry.
4. **Train and save** model weights and preprocessors to the disease-specific model directory.

The dynamic router auto-discovers the configuration at startup (or on-demand via config reload), and the frontend renders the input form dynamically from the JSON Schema — no hardcoded references, no `if/elif` chains, no database migrations.
