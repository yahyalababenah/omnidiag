---
title: OmniDiag
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OmniDiag: Multi-Disease Clinical Decision Support System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](backend/main.py:1)
[![React 18](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react)](frontend/package.json)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)](docker-compose.yml:23)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis)](backend/cache.py:39)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600)](backend/model_loader.py)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-800080)](backend/monitoring/metrics.py:37)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow)](backend/monitoring/mlflow_tracker.py:63)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.51-E6522C?logo=prometheus)](deploy/prometheus.yml)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-HPA-326CE5?logo=kubernetes)](k8s/hpa.yaml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions)](.github/workflows/ci.yml)

---

**OmniDiag** is a config-driven, multi-disease clinical decision support system built for healthcare professionals. It serves per-disease models behind a unified FastAPI surface, with SHAP-based explainability, a DiCE-inspired counterfactual engine, a human-in-the-loop active learning pipeline, DeepSeek LLM clinical report generation, Prometheus metrics, and Kubernetes deployment with horizontal pod autoscaling.

## Who this is for, and what each module covers

**The system** is a clinical decision support system (CDSS) for **resident and general physicians**, in **outpatient clinics, teaching hospitals and government hospitals**. It supports a clinician's decision. It does not make one, and it does not diagnose.

**The heart module is part of that system with a narrower scope of its own.** It ranks **patients a clinician has already decided to refer for catheterisation** by the probability of a >50 % stenosis. It is **not a general-population screen**: every patient it learned from had already been selected for catheterisation by a clinician (UCI, 1982–1987), so it has never seen the people a screening test would be pointed at. The sentence is carried in the module's own config (`disease.scope_note`) and shown beside every result on screen and in the printed report, because a number without its scope is the failure this project is most exposed to.

**The diabetes module is now a second, separately scoped module: dysglycaemia screening on NHANES (Phase 9).** It replaces the BRFSS 2015 stacking ensemble (`configs/diabetes.yaml`, kept and marked superseded). Sections further down that describe the BRFSS ensemble — the 0.108184 threshold, the 0.237 prevalence correction, the `risk_bands` — describe the superseded module, not this one.

### The two modules, side by side

| | **Heart** (Phase 8) | **Dysglycaemia screening** (Phase 9, NHANES) |
|---|---|---|
| **Question answered** | Among patients already chosen for catheterisation, who has a >50 % stenosis? | Among adults with **no diabetes diagnosis and on no glucose-lowering drug**, who has HbA1c ≥ 5.7 %? |
| **Not** | A general-population screen | A diabetes diagnostic. Dysglycaemia is established by the laboratory result, never by this model |
| **Data** | UCI cohorts, 1982–1987 | NHANES 2007–2018, n = 26,904, prevalence 32.9 % |
| **Model** | Spline-GLM, Venn-Abers calibration | Explainable Boosting Machine, Platt calibration (contributions *are* the model, no SHAP) |
| **Inputs** | 11 accepted, 7 read | 20 measurements at a first visit; 6 mandatory, never imputed |
| **Decision** | refer / uncertain / no referral; Mondrian conformal by sex × class | referral / uncertain / no referral; conformal by **age band** (20–39, 40–59, 60+), α = 0.20 |
| **Threshold, risk bands** | None, deliberately | None, deliberately (the BRFSS bands are retired, not recomputed) |
| **Uncertain share** | about 40 % | 38.3 % (referral 33.9 %, no referral 27.8 %) |
| **Next step on screen** | Result panel per decision | `clinical_action_plan` on every prediction: HbA1c/OGTT (High), fasting or random glucose (Medium), none (Low) |

**Measured on the held-out 2017–2018 cycle** (train 2007–2014, calibrate 2015–2016; n = 4,099, prevalence 35.9 %; every figure is from the shipped bundle's `model_card.json`):

- **AUC 0.752** (95 % CI 0.738–0.766); 0.766 with NHANES exam weights. Calibration slope 0.952, intercept −0.090, ECE 0.025.
- **86 % of dysglycaemic patients are flagged** (referral or uncertain) — and therefore **14.1 % receive `no_referral`**: 9.8 % aged 20–39, 15.8 % aged 40–59, 14.3 % aged 60+. **Roughly one in seven patients who do have dysglycaemia is told there is no indication to test.** A `no_referral` here is not a clearance; the action plan says so on screen. At α = 0.15 the figure was 10.1 %, and the move to 0.20 was made knowingly: about 6 points fewer healthy patients sent for a test, about 4 points more missed positives.
- Specificity of `no_referral` is 35.5 %, and **60–77 % of healthy patients are still told to test.** This is a "test most people" screen, not a narrow one.

**Limits, each measured and each carried in `configs/diabetes_nhanes.yaml`:**

- **Discrimination is uneven.** AUC 0.757 (20–39), 0.691 (40–59), 0.619 (60+); 0.785 female, 0.713 male.
- **Not race-blind.** Race is never an input, but the twenty features recover NH Asian membership at AUC 0.785 and NH Black at 0.780. Calibration differs by group behind the pooled figure (mean predicted minus observed: NH White +0.106, NH Asian −0.097, NH Black −0.076).
- **The label is ambiguous by construction.** HbA1c ≥ 5.7 % is a hard cut on an assay with 0.1–0.2 point within-person variability; 24.5 % of the cohort sits inside that band. About 0.07 AUC is unattainable in principle. HbA1c is also shifted by red-cell indices independently of glucose, so the label is not a pure measure of glycaemia.
- **Glycaemic measurements are banned as inputs, permanently** (glucose, insulin, OGTT): they measure the same quantity as the label. A complete blood count is banned too — its AUC gain (+0.022) is an assay artefact, not risk information.
- **Monotone priors hold on main effects only.** Pairwise terms can reverse direction for some patients (for example GGT +5 U/L lowers predicted risk for 16.3 %). "The model can never say older implies lower risk" is **false** for the shipped model and is not claimed anywhere.
- **Conformal coverage is measured under a temporal split**, which breaks exchangeability, so no coverage guarantee is claimed.
- **No deployment prevalence.** The 0.237 Jordan figure is total diabetes prevalence and does not transfer to this target. Absolute probabilities need local recalibration before they are shown as anything but a relative ordering.
- **Not clinically reviewed.** No clinician has read these screens or the action plans; that is a declared next step, as for the heart module.
- **What-If** for this module uses its own lever policy (`backend/diabetes_what_if_levers.py`), including raising HDL to a ceiling of 60 mg/dL; the heart module's counterfactual generator is untouched.

### What changed in Phase 8 (heart module only)

The heart module now ships a **Spline-GLM on seven pre-stress-test inputs, calibrated with exact inductive Venn-Abers, with a Mondrian conformal decision by sex × class** (`models/heart_disease/heart_l3_glm_stack.pkl`, built at image build time and verified against a recorded fingerprint). It answers **refer / do not refer / uncertain** rather than thresholding a probability, and **uncertain counts as a referral everywhere in the system**. It publishes **no decision threshold and no risk bands**, deliberately: Gate 6 measured that the probability's meaning does not transport between hospitals, so a HIGH/MODERATE/LOW badge on it would claim a precision the model does not have. The declared cost is that **around 40 % of patients land in `uncertain`**.

The full research record — every decision, its alternatives, and the measurement behind it — is in the experiments repository, not here.

> **What the screenshots in the research repo do and do not show.** They show that the system behaves as designed — that the interface reports the model's decision faithfully. **They do not show that it is clinically understood.** No clinician has read these screens. Around 40 % of patients land in `uncertain`, and what a physician does with that has not been tested with any physician. **Clinical review is a declared next step before any real use.**

### The three answers, on screen

The heart module returns **one of three answers**, and the interface reports
whichever one the model gave. These are real screens, captured from this branch
against a local backend (Gate 8.6-b, 2026-09-27).

| Referral | No referral | Uncertain |
|---|---|---|
| <img src="docs/assets/screenshots/10_referral_result.png" alt="Referral: the result panel reads Refer — confirmatory testing recommended, with the calibrated interval 85.9%–87.1%" width="280"> | <img src="docs/assets/screenshots/10_no_referral_result.png" alt="No referral: the result panel reads No referral indicated, with the calibrated interval 11.8%–25.0%" width="280"> | <img src="docs/assets/screenshots/10_uncertain_result.png" alt="Uncertain: the result panel reads Uncertain — refer for further evaluation, in amber, with the calibrated interval 71.4%–73.3%" width="280"> |
| `Refer — confirmatory testing recommended` | `No referral indicated` | `Uncertain — refer for further evaluation` |

**"Uncertain" is a third decision, not a middle band of risk.** The conformal set
contains *both* labels, which means the model is declining to rank this patient
at the guaranteed error rate — it is not saying "moderate risk". It is amber
rather than red or green for that reason, it **counts as a referral everywhere in
the system**, and it is where about 40 % of patients land. The module publishes
no threshold and no risk bands, so there is no middle band for it to be.

The generated report says the same thing in prose, including for the uncertain
patient — [`20_uncertain_report.png`](docs/assets/screenshots/20_uncertain_report.png).

**These screens show that the interface reports the model faithfully. They do not
show that it is clinically understood** — the limit stated above applies to these
four images too.

### The measurements behind the model

Every figure below is copied from the experiments repository, where it was
produced; none is a live plot. Each caption names the gate that produced it, the
script that drew it, and the date of the commit it was taken from.

| | |
|---|---|
| <img src="docs/assets/research/fig_ladder_auc.png" alt="Forest plot of pooled leave-one-hospital-out AUC for the seven feature-set layers, each with its HKSJ interval" width="330"> | <img src="docs/assets/research/fig_models_forest.png" alt="Forest plot of pooled AUC for the six model families on two feature sets" width="330"> |
| **The seven-layer feature ladder**, pooled leave-one-hospital-out AUC with HKSJ intervals. Gate 3 · `scripts/p3_figures.py` · research repo `0e0047d`, 2026-09-25 | **The six model families**, on the shipped feature set and the wider one. Gate 4 · `scripts/p4_figures.py` · research repo `3c5d88a`, 2026-09-25 |
| <img src="docs/assets/research/fig_ablation_forest.png" alt="Forest plot comparing nine training-objective ablation arms by pooled AUC" width="330"> | <img src="docs/assets/research/fig_hf13_options.png" alt="Sensitivity by sex under four decision-rule options, showing the single-threshold gap and the Mondrian option" width="330"> |
| **Nine training objectives** compared on the same split. Gate 5 · `scripts/p5_figures.py` · research repo `0e0047d`, 2026-09-25 | **The sex-sensitivity gap (HF-13)** and the four decision rules considered; O3, the Mondrian rule, is what ships. Gate 7 · `scripts/p7_hf13_figure.py` · research repo `f2e17a3`, 2026-09-26 |
| <img src="docs/assets/research/fig_tehran_layers.png" alt="External Tehran cohort AUC across feature-set layers" width="330"> | <img src="docs/assets/research/fig_tehran_cp_only.png" alt="Chest-pain coding comparison between the UCI and Tehran cohorts" width="330"> |
| **External validation, Tehran cohort** — a different country, a different decade, a different recording convention. Gate 1 · `scripts/p3_figures.py` · research repo `0e0047d`, 2026-09-25 | **Why chest-pain coding matters** between the two cohorts. Gate 4 · `scripts/p4_figures.py` · research repo `3c5d88a`, 2026-09-25 |

The pooled figure is never the whole story here: the widest single-hospital
interval spans 0.40 AUC, and the 95 % prediction interval for a *new* hospital
runs from 0.36 to 0.97. Both are in the research repo's `results.md`, with the
per-hospital tables the plots summarise.


### What changed in Phase 9 (dysglycaemia module only)

The diabetes slot now ships an **Explainable Boosting Machine on twenty first-visit inputs, calibrated with Platt scaling, with a group-conditional conformal decision by age band** (`models/diabetes_nhanes/diabetes_nhanes_ebm.joblib`, fetched at image build time and checked against a recorded sha256). Like the heart module, it answers **referral / no referral / uncertain** rather than thresholding a probability, and **uncertain counts as a referral everywhere in the system**. It publishes **no decision threshold and no risk bands**, deliberately. The declared cost is that **38.3 % of patients land in `uncertain`**, and that **14.1 % of the patients who do have dysglycaemia receive `no_referral`**. Every decision the model made on the way there, and the measurement behind it, is in [`docs/phase9/`](docs/phase9/).

> **What the screenshots below do and do not show.** They show that the interface reports the model's decision faithfully. **They do not show that it is clinically understood.** No clinician has read these screens or their action plans; that is a declared next step before any real use, as for the heart module.

### The three answers, on screen (dysglycaemia)

The module returns **one of three answers**, and the interface reports whichever one the model gave.
These are real screens, captured from the merged branch against a local backend (2026-09-28, Clinical
EMR mode, demo patients N-003, N-001 and N-002).

| Referral | No referral | Uncertain |
|---|---|---|
| <img src="docs/assets/screenshots/30_nhanes_referral_result.png" alt="Referral: the result panel reads Refer — confirmatory testing recommended, with the calibrated interval 60.9%–68.4%" width="280"> | <img src="docs/assets/screenshots/30_nhanes_no_referral_result.png" alt="No referral: the result panel reads No referral indicated, with the calibrated interval 2.6%–4.8%, and states that it is not a clearance" width="280"> | <img src="docs/assets/screenshots/30_nhanes_uncertain_result.png" alt="Uncertain: the result panel reads Uncertain — refer for further evaluation, with the calibrated interval 32.7%–37.4%" width="280"> |
| `Refer — confirmatory testing recommended` · High urgency: HbA1c, or OGTT where HbA1c is unreliable | `No referral indicated` · Low urgency; the panel says **14.1 % of truly dysglycaemic patients receive this answer** | `Uncertain — refer for further evaluation` · Medium urgency: fasting or random glucose |

**"Uncertain" is a third decision, not a middle band of risk.** The conformal set contains *both*
labels: the model is declining to rank this patient at the guaranteed error rate. It counts as a
referral everywhere in the system, and it is where 38.3 % of patients land. Each panel carries a **next
clinical step** (`clinical_action_plan`) built from the decision, never from a risk band.

The generated report says the same thing in prose, including for the uncertain patient —
[`40_nhanes_uncertain_report.png`](docs/assets/screenshots/40_nhanes_uncertain_report.png). It is the
rule-based report, not the LLM one.

**These screens show that the interface reports the model faithfully. They do not show that it is
clinically understood** — the limit stated above applies to these images too. Three defects visible in
them are open, not hidden: the report quotes contributions as "SHAP +0.169" although this model has no
SHAP step; the report tells the clinician to "order an HbA1c test" for an uncertain patient while the
panel above it recommends a fasting or random glucose; and the uncertain panel's frame is green while its
title and next-step box are amber.

### How the dysglycaemia model explains itself

There is **no SHAP step in this module, and none is wanted**: the explanation is the model. An EBM is
additive by construction,

    logit(raw score) = intercept + Σ f_j(x_j) + Σ f_jk(x_j, x_k)

so each feature's contribution is read straight out of the fitted model, not estimated after the fact.
The identity is **asserted on every request** to within 1e-8 (measured at 2.7e-15 over the whole test
cycle) — an explanation that does not sum to the score is refused, not served. A pairwise term is split
evenly between its two features for the bar chart, the sum stays exact, and the unsplit terms are in
`term_contributions` for Engineering mode.

<img src="docs/assets/screenshots/32_nhanes_contributions.png" alt="Clinical Insights for N-002: horizontal contribution bars per feature, base value −1.1499, and the What-If scenario below" width="420">

*Clinical Insights for N-002. Red bars raise the estimate, green bars lower it; the base value is the
model intercept.* Two things to read it correctly:

- **The bars are on the raw log-odds scale; the probability shown is after Platt calibration.** They do
  not add up to the displayed percentage by eye.
- **Monotone priors hold on main effects only.** Pairwise terms can reverse direction for some patients,
  so a single bar is a statement about this patient, not a rule about the feature.

**What-If uses this module's own lever policy.** For N-002 it moves BMI 30 → 24.9, adiposity band
high → normal, SBP 126 → 120, HDL 46 → 60 and vigorous activity 0 → 1, and the estimate falls to 18.2 %.
It is the model's response to modifiable factors, not a predicted treatment effect, and HDL is a proxy
for the behaviour that raises it ([`diabetes_what_if_levers.py`](backend/diabetes_what_if_levers.py)).

<img src="docs/assets/screenshots/31_nhanes_whatif.png" alt="What-If scenario for N-002: BMI, adiposity band, SBP, HDL and activity move to their targets, post-intervention probability 18.2%" width="420">

### The measurements behind the dysglycaemia model

Every figure below is a static image copied from the Phase 9 research folder into
[`docs/phase9/figures/`](docs/phase9/figures/) and committed on 2026-09-28 (`7300042`); none is a live
plot, and the scripts that drew them are **not in this repository**. Each caption names the finding or
decision it supports, in [`FINDINGS_REGISTER.md`](docs/phase9/FINDINGS_REGISTER.md) and
[`DISCOVERY_RECORD.md`](docs/phase9/DISCOVERY_RECORD.md).

| | |
|---|---|
| <img src="docs/phase9/figures/fig1_cbc_artefact_control.png" alt="AUC with and without a complete blood count, under an HbA1c label and a glucose label" width="330"> | <img src="docs/phase9/figures/fig3_label_noise_ceiling.png" alt="Validation AUC rising from 0.7796 to 0.8476 as patients near the HbA1c cut are excluded" width="330"> |
| **The CBC gain is an assay artefact (F9-29).** Adding a blood count lifts AUC by +0.0218 against the HbA1c label and *lowers* it by 0.0039 against a glucose label on the same patients, so it is banned as an input | **About 0.07 AUC is locked in label noise (F9-30).** Dropping the 24.5 % of patients within ±0.20 of the 5.7 % cut takes validation AUC from 0.7796 to 0.8476 |
| <img src="docs/phase9/figures/fig6_every_lever.png" alt="Change in validation AUC for every modelling lever tried in Phase 9" width="330"> | <img src="docs/phase9/figures/fig4_calibration_layers.png" alt="ECE, calibration slope and AUC for raw EBM, isotonic and Platt calibration" width="330"> |
| **Every lever tried.** Against a pre-declared bar of +0.005, only the CBC crossed it, and it was the artefact; up-weighting the 60+ band made AUC worse | **Platt scaling is adopted (D9-04).** ECE 0.054 → 0.023 with AUC untouched at 0.7796; isotonic cost 0.0013 AUC |
| <img src="docs/phase9/figures/fig5_pooled_vs_subgroup_auc.png" alt="AUC of age alone and of the full model, overall and by age band" width="330"> | <img src="docs/phase9/figures/fig2_blank_field_vulnerability.png" alt="Fraction of decisions flipped and median risk shift when each single field is left blank" width="330"> |
| **Why 60+ AUC is low.** Age alone scores 0.716 overall and 0.517 inside the 60+ band, so the full model's validation AUC of 0.648 there is not a defect that re-weighting can fix | **Leaving one field blank (F9-28).** Up to 38 % of decisions flip when a single field is missing, which is why six fields are mandatory and never imputed (D9-06) |
| <img src="docs/phase9/figures/fig7_per_band_coverage.png" alt="Conformal coverage by age band and class, class-conditional against group-conditional" width="330"> | <img src="docs/phase9/figures/fig8_alpha_dial.png" alt="Worst per-band coverage against share of patients marked uncertain, for class-conditional and group-conditional conformal" width="330"> |
| **Pooled coverage hid two failing groups (F9-31).** Class-conditional coverage fell to 0.578 for dysglycaemic 20–39s and 0.574 for healthy 60+; group-conditional (D9-05) restores both | **The decisiveness dial.** Group-conditional keeps worst-band coverage far above class-conditional at every α. This plot marks α = 0.15, selected on validation; the shipped α was later moved to **0.20** (D9-07), for the reasons in the config |

The pooled figure is not the whole story here either: AUC is 0.757 for ages 20–39, 0.691 for 40–59 and
0.619 for 60+, and the twenty inputs recover NH Asian and NH Black membership at AUC 0.785 and 0.780
although race is never an input (see the limits above).

> **What is actually running in the deployed Space** (verified 2026-09-23, see
> [docs/FEATURE_VERIFICATION.md](docs/FEATURE_VERIFICATION.md)):
>
> - **Monitoring: Prometheus only.** Evidently drift detection and MLflow
>   tracking are implemented in this repository but are **not live**. Evidently
>   is installed in the image yet unusable — the code targets its 0.4 API and
>   the pin resolves to 0.7, which removed it; the cost of each way out is
>   costed in [docs/EVIDENTLY_COST.md](docs/EVIDENTLY_COST.md).
>
>   **MLflow, corrected 2026-09-26 (Gate 8.6).** It logged no runs because
>   *nothing wrote to it*: the only automatic caller was the retrain path, which
>   fails for heart and is a no-op for diabetes, and the script that builds the
>   shipped heart artifact logged nothing at all. Worse, the admin endpoint
>   reported `count: 0` whether the package was missing, the store was
>   unreachable, or the store was working and empty — so "MLflow is empty" could
>   not be told apart from "MLflow is not installed".
>
>   Both are fixed. `GET /admin/mlflow/runs` now returns a `status` of
>   `unavailable` / `unreachable` / `empty` / `ok` with the `tracking_uri` and a
>   reason, and [`scripts/train_heart_glm.py`](scripts/train_heart_glm.py) logs
>   one run per image build recording the artifact's provenance — the training
>   CSV's sha256, the bundle's sha256, the reproducibility fingerprint, the
>   conformal alpha, and the blank-input impact — and **deliberately no accuracy
>   figure**, because every performance number for this model is cross-fitted or
>   leave-one-hospital-out and belongs with its confidence interval, not beside
>   an artifact hash.
>
>   **What this still is not:** a tracking *server*. With no `MLFLOW_TRACKING_URI`
>   set, the store is a SQLite file baked into the image at build time, so it
>   holds exactly one run — the model in that image — and a new build replaces
>   it. There is no experiment comparison, and the retrain cycle that would
>   produce one is still broken (see the Known limitation below). `docker-compose`
>   and the k8s manifest do point at a real MLflow server, and the same code logs
>   there when one is reachable.
> - **Clinical notes: regex, English only.** The BioBERT path exists in the
>   code but is never called — the frontend always sends `use_bert: false` and
>   `transformers` is not part of the deployed dependency set. Arabic is not
>   supported: `/api/v4/parse-notes` reports `language.supported: false` for
>   any note containing Arabic and the UI says so explicitly.
> - **Patient Comparison compares two different patients**, side by side. It
>   is not a before/after view of one patient under an intervention; no such
>   view exists.

Three modules are currently registered: Coronary Artery Disease (`heart_disease`), Diabetes Risk Assessment on BRFSS (`diabetes`, superseded but still served) and Dysglycaemia Screening on NHANES (`diabetes_nhanes`). Adding a disease within a registered model family requires a YAML config, a Pydantic schema, a feature engineer, and model weights — no routing, middleware, auth, or API changes. Adding a new model family requires one backend class implementing the ModelBackend interface, registered once. Tree-based families use TreeExplainer; other families use a slower generic SHAP explainer.

---

## System Architecture

### Request Flow & MLOps Pipeline

```mermaid
flowchart LR
    subgraph Client["Frontend (React 18 + Vite)"]
        A[Engineering Mode] --> B[DynamicClinicalForm]
        C[Clinical EMR Mode] --> B
        B --> D[useDiseaseForm]
        D --> E[useDiseaseSchema]
        E --> F["GET /api/v4/{disease}/schema"]
        D --> G[buildZodSchema]
        G --> H[React Hook Form + Zod]
        H --> I[SchemaFieldFactory]
        I --> J[OmniDiagApi]
    end

    subgraph Server["FastAPI Backend (port 7860)"]
        K[SecurityHeadersMiddleware] --> L[AuditMiddleware]
        L --> M[CORSMiddleware]
        M --> N[SlowAPI RateLimiter]
        N --> O{RBAC require_role}
        O --> P[Redis cache lookup]
        P -->|HIT| Q[Return cached]
        P -->|MISS| R[OmniDiagRouter]
        R --> S[FeatureEngineer]
        S --> T[ModelLoader / EnsembleLoader]
        T --> U[SHAP TreeExplainer]
        U --> V{uncertainty_band}
        V -->|H ≥ 0.88| W[ReviewQueue]
        V --> X[record_prediction]
        X --> Y[Prometheus metrics]
        T --> Z[Persist Prediction]
    end

    subgraph MLOps["MLOps Stack"]
        AA[MLflow :5000]
        BB[Prometheus :9090]
        CC[Grafana :3001]
        DD[Evidently DriftMonitor]
        BB --> CC
        Y --> BB
        DD --> BB
    end

    J -.->|HTTPS| K
    Z --> PostgreSQL[("PostgreSQL 15")]
    P -.-> Redis[("Redis 7")]
```

*Simplified: `U[SHAP TreeExplainer]` only runs on `/explain` requests, not `/predict` — a plain `/predict` call returns straight from `T[ModelLoader]` to `X`/`Z` without touching SHAP.*

### Config-Driven Disease Registration

```mermaid
flowchart LR
    A[configs/heart_disease.yaml] --> B[OmniDiagRouter._load_all_configs]
    C[configs/diabetes.yaml] --> B
    B --> D{Register loader}
    D -->|CAD: self-contained sklearn Pipeline| E[ModelLoader]
    D -->|DM: stacking ensemble| F[EnsembleModelLoader]
    E --> G[pipeline.predict_proba - no feature engineering]
    F --> H[features/diabetes_features.py]
    H --> J[DiabetesFeatureEngineer]
    J --> K[3 engineering paths]
    K --> L[Heuristic: statistical interactions]
    K --> M[Clinical: risk score formulas]
    K --> N[Medical: domain-specific markers]
```

### Active Learning Lifecycle

```mermaid
flowchart LR
    A["POST /predict\n(prob = 0.53)"] --> B["prediction_entropy(p)\nH = 0.999"]
    B --> C{H ≥ 0.88?}
    C -->|Yes| D["ReviewQueue\nstatus=pending"]
    C -->|No| E[Return to client]
    D --> F["GET /review/queue\n(doctor reviews)"]
    F --> G["POST /review/{id}/annotate\nlabel=0"]
    G --> H["status=reviewed"]
    H --> I["run_retrain_pipeline()\nget_annotated_samples()"]
    I --> J["retrain_xgb()\nxgb_model= param\n20 boost rounds"]
    J --> K["Model backup .bak.pkl"]
    K --> L["ModelLoader.reload(disease)"]
    L --> M["log_model_info()\nMLflow run"]
```

---

## Architecture & Core Features

### Config-Driven Disease Routing

[`OmniDiagRouter`](backend/router.py) scans [`configs/`](configs/) at startup and auto-discovers all YAML configuration files. Each config names its model family in `model.family` (`sklearn_pipeline` for CAD, `stacking_ensemble` for DM), plus weights path, feature engineering module, and preprocessor artifacts. The router instantiates the [`ModelBackend`](backend/model_backends/base.py) registered for that family (the two built-in ones wrap [`ModelLoader`](backend/model_loader.py) and [`EnsembleModelLoader`](backend/ensemble_loader.py)); artifacts lazy-load on first request, and an unknown family fails at startup. See [docs/ADDING_A_MODEL_FAMILY.md](docs/ADDING_A_MODEL_FAMILY.md) for the steps, measured costs, and current limits.

Each disease config declares the model architecture, weights paths with fallback resolution, SHAP explainer type (`tree` or `deep`), and schema back-reference via [`DISEASE_SCHEMA_REGISTRY`](backend/schemas.py) (or a `schema: {module, class}` block in the YAML). A Python module path for a disease-specific [`BaseFeatureEngineer`](features/base_features.py:17) subclass and a preprocessor directory (`label_encoders.pkl` + `standard_scaler.pkl`) still apply to diabetes; heart_disease's config carries neither — its Pipeline is self-contained (see [Explainable Inference Core](#explainable-inference-core)).

### RBAC & Security Middleware

[`require_role()`](backend/auth/rbac.py:39) is a FastAPI dependency factory that enforces role membership before any route handler executes. It reads `current_user.roles` (loaded eagerly via `lazy="selectin"` on the `User.roles` relationship) and raises HTTP 403 with a structured payload listing required vs. held roles if the intersection is empty. Five roles are seeded at startup: `super_admin`, `admin`, `doctor`, `nurse`, `viewer`. `CLINICAL_ROLES = ("doctor", "nurse", "super_admin")` gates `batch`; retrain and audit endpoints require `("super_admin",)`. **`predict`, `explain` and `counterfactuals` are deliberately open to anonymous callers** — they depend on `get_optional_user` rather than `require_role`, so the public demo can be tried without an account. Results are persisted (and linkable to a patient record) only for authenticated users; everything that reads or writes stored clinical data is role-gated.

[`get_current_user()`](backend/auth/dependencies.py:31) resolves the authenticated identity in priority order: `Authorization: Bearer <JWT>` header → `access_token` HttpOnly cookie → `X-API-Key` header (hash compared against `users.api_key_hash` via `get_user_by_api_key()`). [`get_optional_user()`](backend/auth/dependencies.py:108) returns `None` instead of raising 401, used on endpoints that permit anonymous one-off predictions while persisting results only for authenticated users.

Three middleware layers wrap every request in [`main.py`](backend/main.py): [`SecurityHeadersMiddleware`](backend/middleware/security.py) sets the `Strict-Transport-Security` (HSTS) header when `ENFORCE_HTTPS=true` — this is a response header only; the middleware does not perform a server-side HTTP→HTTPS redirect; [`AuditMiddleware`](backend/middleware/audit.py) writes every authenticated request as an immutable `audit_logs` row (user, endpoint, method, status, IPv6-capable IP address, duration_ms); [`CORSMiddleware`](backend/main.py) enforces an allowlist from `CORS_ALLOWED_ORIGINS` and never accepts wildcard origins.

### Redis Caching & Rate Limiting

[`init_cache()`](backend/cache.py:39) selects the cache backend at startup: if `REDIS_URL` is set, it pings a `redis.asyncio` client and initialises a `RedisBackend` (fastapi-cache2); otherwise it falls through to `InMemoryBackend` silently — no `REDIS_URL` means no Redis dependency in development or test environments. [`predict_cache_key()`](backend/cache.py:76) produces a deterministic 16-character SHA-256 fingerprint over `{disease, sorted_features}` so identical patient inputs within any 5-minute window return cached responses. Schema responses carry a 24-hour TTL; prediction responses carry 300 seconds. [`cache_flush()`](backend/cache.py:114) enumerates `omnidiag:*` keys via `redis.keys()` on the Redis backend, or clears `backend._store` directly on the in-memory fallback.

Rate limiting is applied via [`SlowAPI`](backend/rate_limit.py:49) with per-route limits: clinical endpoints are capped at `LIMIT_CLINICAL = "30/minute"`, admin operations at `LIMIT_ADMIN = "60/minute"`, and the `/generate-report` endpoint at a hard 10 requests/minute enforced at the route level. A `viewer` role is seeded (see RBAC above) but no route currently gates on it — there is no viewer-specific rate limit yet.

### Human-in-the-Loop Active Learning

The active learning pipeline consists of three components. [`sampler.py`](backend/active_learning/sampler.py:26) computes binary entropy **around the module's own decision threshold**, not around 0.5: the probability is first mapped by [`centre_on_threshold()`](backend/active_learning/sampler.py) — the prior-shift map with the threshold `t` sent to 0.5, strictly increasing and the identity when `t = 0.5` — and then `H(q) = -q·log₂(q) - (1-q)·log₂(1-q)` is taken. A prediction with `H ≥ 0.88` ([`_DEFAULT_ENTROPY_THRESHOLD`](backend/active_learning/sampler.py)) is queued via [`should_queue_for_review(probability_corrected, decision_threshold)`](backend/active_learning/sampler.py). For diabetes (`t = 0.108184` on the prevalence-corrected scale) that is ≈5–22 %. **Heart no longer takes this path at all:** its module reports a decision rather than a probability against a cut-point, so a row is queued when the model itself answers `uncertain` — `decision == 'uncertain'`, not an entropy score. Routing it through the threshold rule was wrong in both directions and measurably so: with the 0.5 default it left a patient the model had called UNCERTAIN at p = 0.95 unqueued while queueing a confidently decided one at p = 0.52. A queued heart row records `decision_threshold = NULL`, because writing 0.5 there would make the audit trail claim a threshold the model does not have. The distinction matters: a 0.5-centred sampler on the corrected diabetes scale queued 4,789 of 14,139 test rows, every one a confident Positive at ≥ 5× the threshold, and **0 of the 849** rows within ±20 % of the threshold; the threshold-centred sampler queues 3,566 rows including **all 849**. Each queued row records `uncertainty_scale` and `decision_threshold` beside `uncertainty_score`, so scores from different releases are never compared blindly. [`uncertainty_band()`](backend/active_learning/sampler.py) maps the same centred value to `CERTAIN` / `CONFIDENT` / `BORDERLINE` / `UNCERTAIN`. The [`routes.py`](backend/active_learning/routes.py) module exposes `GET /api/v4/review/queue` (paginated, filterable by disease), `POST /api/v4/review/{id}/annotate` (writes `label` and transitions `status → reviewed`), `POST /api/v4/review/{id}/skip`, and `GET /api/v4/review/stats`.

[`run_retrain_pipeline()`](backend/active_learning/retrain.py:149) is the full async pipeline. [`get_annotated_samples()`](backend/active_learning/retrain.py:38) issues a raw SQL JOIN of `review_queue` and `predictions` filtered to `status='reviewed'` and `label IS NOT NULL`. [`retrain_xgb()`](backend/active_learning/retrain.py:84) loads the current `.pkl`, constructs an `xgb.DMatrix`, and calls `xgb.train()` with `xgb_model=model` for 20 incremental boost rounds at lr=0.05 — the existing tree structure is preserved and extended. The old model is renamed to a timestamped `.bak.pkl` before the new weights are written. On success, [`ModelLoader.invalidate()`](backend/model_loader.py) / [`EnsembleModelLoader.invalidate()`](backend/ensemble_loader.py) clears the live router's cached model object so the very next prediction lazy-reloads the new weights from disk — **no process restart required**. This hot-reload path was verified end-to-end (swap in a differently-shaped model file → confirm the next request immediately errors with a feature-mismatch specific to the *new* file, proving it was actually loaded). [`_log_to_mlflow()`](backend/active_learning/retrain.py:137) records the retrain run unconditionally, with a warning-only failure path if MLflow is unreachable — **but it is never reached in practice, because `retrain_xgb()` fails before it for both diseases** (see the Known limitation below). Until Gate 8.6 that made it the only automatic writer to an experiment that consequently stayed empty.

🚧 **Known limitation:** `retrain_xgb()` builds its training matrix directly from the raw predict-time `input_features` (`X = np.array([list(feat.values()) for feat in features_list])`) and always reads/writes the single hardcoded path `models/{disease}/omni_diag_xgb_optimized.pkl`. For **heart_disease**, that filename no longer exists at all — the shipped artifact is `models/heart_disease/heart_l3_glm_stack.pkl`, a `dict` bundle holding a Spline-GLM Pipeline with its calibration and conformal state, not a bare `XGBClassifier` — so `retrain_xgb()` fails immediately at its own `if not model_path.exists()` check, before the un-encoded-categorical-strings issue it was originally written to describe is ever reached. It is also built at image build time from the training CSV and verified against a recorded fingerprint, so retraining it from annotated rows is a larger question than swapping a file. No real annotated-sample retrain cycle for heart_disease currently completes, for a different reason than previously documented. For **diabetes**, that same hardcoded path is **not** one of the three files `EnsembleModelLoader` actually loads (`xgb_model.pkl`, `lgb_model.pkl`, `rf_model.pkl`) — so even a numerically successful run retrains a file the live ensemble never reads, a silent no-op. The hot-reload mechanism described above is implemented and tested; connecting it to a disease-aware, correctly-shaped retrain step is the next piece of work here.

### DeepSeek LLM Clinical Report Generation

[`generate_report()`](backend/llm/report_generator.py:100) calls the DeepSeek API using an `AsyncOpenAI` client pointed at `_DEEPSEEK_BASE_URL = "https://api.deepseek.com"` with `model="deepseek-chat"` and `max_tokens=600`. The key is read lazily via [`_get_api_key()`](backend/llm/report_generator.py:18) on each call — not at import time — so Hugging Face Space secrets injected after startup are picked up correctly. [`_format_shap()`](backend/llm/report_generator.py:57) sorts SHAP values by absolute magnitude and formats the top 5 as directional bullets (`↑ increases risk` / `↓ decreases risk`) for inclusion in the user prompt alongside disease, probability, risk band, decision threshold and patient features. The risk band is computed **once, on the server**, by classifying `probability_corrected` against the disease's own `risk_bands` — never from literal cut-points and never from a band supplied by the client. The same band drives the narrative, the recommended actions and the LLM prompt, and is returned as `risk_band`. The prompt states that the probability is calibrated to the deployment prevalence so the model does not read 11 % as "low".

> **heart_disease takes a different prompt.** It configures `risk_bands: null` on purpose (D-32: Gate 6 measured that its probability's meaning does not transport between hospitals), so it gets **no band at all** — `risk_band` is `null` and the recommended actions are keyed on the decision instead. Its prompt states the decision, the Venn-Abers interval, that the probability is calibrated to the training hospitals' mix rather than to the reading institution's population, and that there is no threshold and no band to report. Until Phase 8 it was handed the threshold-module prompt unchanged, which told it the opposite on all three points and supplied a 0.70/0.40 band this module does not define.

When `DEEPSEEK_API_KEY` is absent or the API call raises any exception, [`_rule_based_report()`](backend/llm/report_generator.py:71) generates the same four-section structure — Clinical Summary, Key Risk Drivers, Recommended Actions, Risk Stratification Note — using the SHAP rankings and a `HIGH` / `MODERATE` / `LOW` risk band derived from the probability. The response `source` field distinguishes `"llm"` (with `llm_model` and `latency_ms`) from `"rule_based"` (with `fallback_reason`) so callers can surface the provenance to clinicians.

### Clinical NLP Notes Parser

[`parse_clinical_note()`](backend/nlp/notes_parser.py:249) implements two-tier extraction from free-text clinical notes. [`_regex_extract()`](backend/nlp/notes_parser.py:97) runs first as the always-available baseline: it applies 20+ regex patterns (matched fresh via `re.search()` per call, not pre-compiled) across categories including age (with short-form aliases like `y/o`), BP systolic/diastolic, cholesterol, glucose, BMI, heart rate, creatinine, hemoglobin, oldpeak, and boolean flags for hypertension, diabetes, stroke, smoking, chest pain, exercise angina, edema, and anemia. If `use_bert=True` and HuggingFace Transformers is installed, `_bert_extract()` runs the `d4data/biomedical-ner-all` NER pipeline (lazy-loaded on first call, CPU inference, confidence threshold 0.7) and merges its output — BERT values win on overlapping keys. **This path is never taken in the deployment**: the frontend hard-codes `use_bert: false` and `transformers` is not installed in the Space image, so every extraction is regex. Describing the deployed parser as "BioBERT" is inaccurate. Every pattern is English; `language_support()` classifies the note by script and returns `supported: false` for anything containing Arabic (including mixed notes, where only the English abbreviations are read), which the UI surfaces instead of reporting "0 fields extracted". [`map_to_disease_schema()`](backend/nlp/notes_parser.py:236) applies disease-specific field name and value transformations: for `heart_disease`, `bp_systolic → RestingBP` (int); for `diabetes`, `cholesterol → HighChol` (binarised at 200 mg/dL). Missing `transformers` degrades silently to regex-only with no user-visible error.

### Explainable Inference Core

The inference pipeline order is **disease-specific**, matching how each model was trained (verified live for both diseases):

- **heart_disease (CAD)** — [`HeartGlmConformalBackend`](backend/model_backends/heart_glm_conformal.py) does **no manual preprocessing at all**: raw feature values are encoded by [`stack.encode_for_inference()`](backend/heart_glm/stack.py) and go straight into the bundle's `sklearn.Pipeline`. The shipped artifact (`models/heart_disease/heart_l3_glm_stack.pkl`) is built at image build time by [`scripts/train_heart_glm.py`](scripts/train_heart_glm.py) and holds a `ColumnTransformer` (`IterativeImputer` + `StandardScaler` + `SplineTransformer`) feeding a `LogisticRegression`, plus the Venn-Abers calibration arrays and the Mondrian conformal cells. There is no separate feature-engineering step and no preprocessor files for this disease. **The schema accepts eleven inputs and the model reads seven** (L3, D-25): `MaxHR`, `Oldpeak`, `ExerciseAngina` and `ST_Slope` are accepted for the record and not read at all — blanking any of them changes the answer by exactly nothing, measured, and the UI labels them. Of the seven it does read, `RestingBP`, `Cholesterol` and `FastingBS` are `Optional` and imputed, and a blank one is reported to the clinician as `data_completeness_warning`.
  > The previous model — a tuned `XGBClassifier` Pipeline in `models/heart_disease/heart_full_tuned.pkl` with a 0.3695 threshold — is **archived, not shipped**. Descriptions of it elsewhere in this file were corrected in Phase 8; `AUDIT_REPORT.md` and `WEAKNESS_REGISTER.md` keep their original text with dated addenda, because they are audit records.
- **diabetes** — [`EnsembleModelLoader.predict()`](backend/ensemble_loader.py:245): engineer heuristic + medical features **first**, then encode + scale. Here `engineer_clinical()` is a documented no-op ([diabetes_features.py:101](features/diabetes_features.py:101)); its formula is computed inside `engineer_medical()` instead, by deliberate design (documented in-code) so it participates in live inference. It does — `Diabetes_Clinical_Risk` is typically the top-ranked SHAP feature for this disease (confirmed live).

On explain requests, the pipeline appends a SHAP TreeExplainer step returning structured JSON: `shap_chart_data` sorted by absolute SHAP value descending, a text explanation of the top-3 features with direction labels, and the base expected log-odds value. The `shap_chart_data` column in `predictions` stores this JSON for offline audit retrieval via `GET /admin/audit-logs` or direct DB query.

#### XGBoost 3.x Compatibility Patch

XGBoost 3.x stores `base_score` as a bracket-wrapped string (e.g. `[5.85E-1]`) in its UBJSON serialisation. SHAP's `TreeExplainer` calls `save_raw()` and parses the output with `float()`, which fails on the bracketed format. [`ModelLoader`](backend/model_loader.py) applies a `save_raw()` monkey-patch at load time that strips the brackets from the UBJSON byte stream, enabling SHAP to read `base_score` correctly. The patch is applied only to XGBoost models and silently skipped for other types.

> **No longer applies to heart_disease.** It did while that module shipped an `XGBClassifier` inside a Pipeline. The Spline-GLM that ships now is additive on the logit scale, so its SHAP values are computed exactly in closed form by [`stack.shap_log_odds()`](backend/heart_glm/stack.py) — no TreeExplainer, and no patch.

### DiCE-Inspired Counterfactual Engine

[`CounterfactualGenerator`](backend/counterfactual_generator.py) implements a DiCE-inspired algorithm using random sampling with diversity selection — no external dependencies beyond NumPy and pandas (`dice-ml` was removed due to a `LossySetitemError` incompatibility with pandas ≥ 3.0):

1. **Sample** 500+ random perturbations of mutable patient features within clinical bounds
2. **Evaluate** each perturbation through the full engineering + preprocessing + prediction pipeline
3. **Filter** perturbations that flip the predicted class Positive → Negative
4. **Score** by proximity (normalised L1 distance) with a diversity penalty (Jaccard similarity of changed feature sets)
5. **Select** the top 3 most diverse counterfactuals

**Clinical Firewall** ([`_is_illegal_flip()`](backend/counterfactual_generator.py)) — discards candidates with clinically absurd transitions (e.g. advising a patient to start smoking, raise BP, or reduce physical activity). **Feasibility scoring** ([`_assess_feasibility()`](backend/counterfactual_generator.py)) classifies each scenario as `high` (lifestyle-only), `medium` (requires medical intervention), or `low` (unrealistically large or concurrent changes).

### Feature Engineering Pipeline

Disease-specific now, not uniform: **diabetes** implements a [`BaseFeatureEngineer`](features/base_features.py:17) subclass with three abstract methods executed in dependency order; **heart_disease does none of this** — its shipped Pipeline (see [Explainable Inference Core](#explainable-inference-core)) takes the 11 raw clinical fields directly, with no derived features at all. `features/heart_disease_features.py` and its heuristic/clinical/medical formulas below (`Age_BP_Interaction`, `Clinical_Risk_Score`, `RPP`, ...) describe the pre-Pipeline heart model and are **no longer called anywhere in live inference**.

**Heuristic (Statistical Interactions)** — Computes multiplicative interaction terms and ratios:
- DM: [`BMI_Age_Interaction`](features/diabetes_features.py:55), [`Health_Index`](features/diabetes_features.py:63), [`Lifestyle_Score`](features/diabetes_features.py:77), [`SES_Composite`](features/diabetes_features.py:88)

**Clinical (Risk Score Formulas)** — Computes exponentiated linear risk scores using clinical weights. Diabetes' live pipeline does not call `engineer_clinical()` directly (see [Explainable Inference Core](#explainable-inference-core)); the formula is real, but reached differently:
- DM: `Diabetes_Clinical_Risk = exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)` — despite the name, this is actually computed inside [`engineer_medical()`](features/diabetes_features.py:128), **not** `engineer_clinical()` (a documented no-op at [diabetes_features.py:101](features/diabetes_features.py:101)). This relocation is deliberate so the formula runs live — it's typically the top SHAP-ranked feature for diabetes predictions.

**Medical (Domain-Specific)** — Computes cardiology-validated composite markers:
- DM: `Diabetes_Clinical_Risk` (see Clinical section above — it lives in the medical-path method for pipeline-compatibility reasons, not because of aliasing)

### Prometheus + Evidently + Grafana Monitoring

> **Deployment status:** only Prometheus `/metrics` is live. The Evidently and
> Grafana pieces below describe code in this repository that is not running in
> the deployed Space — see [docs/EVIDENTLY_COST.md](docs/EVIDENTLY_COST.md).

[`metrics.py`](backend/monitoring/metrics.py:37) registers all Prometheus instruments at module import time using lazy-guarded `try/except` so the app starts even if `prometheus_client` is absent. Eight instruments are exposed at `GET /metrics`: `omnidiag_predictions_total{disease,prediction}` (Counter), `omnidiag_prediction_confidence{disease}` (Histogram, buckets 0.01–1.0 with fine resolution below 0.1 — the old 0.5–1.0 set left 7 of 9 buckets unreachable for prevalence-corrected diabetes probabilities, whose maximum on the test split is 0.653; series recorded before this change are not comparable with those after it), `omnidiag_request_duration_seconds{method,endpoint,status}` (Histogram, buckets 0.01–5.0 s), `omnidiag_active_requests` (Gauge), `omnidiag_drift_share{disease}` (Gauge, updated post-drift-run), `omnidiag_cache_hits_total{disease}`, `omnidiag_cache_misses_total{disease}`, and `omnidiag_batch_rows_total{disease,status}`.

[`DriftMonitor`](backend/monitoring/drift.py:48) wraps Evidently's `Report` with `DatasetDriftMetric` and `DatasetMissingValuesMetric` against a per-disease reference CSV baseline: heart_disease uses `data/heart_disease/processed/final_ready_data.csv`; diabetes uses the raw BRFSS export `data/diabetes/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (the paths intentionally differ — see [`_REFERENCE_PATHS`](backend/monitoring/drift.py:175)). A `threading.Lock` prevents concurrent report runs. [`get_monitor(disease)`](backend/monitoring/drift.py:181) returns per-disease singleton instances. After each run, [`record_drift()`](backend/monitoring/metrics.py:101) pushes the `drift_share` value to the Prometheus Gauge so Grafana dashboards reflect it without polling the admin API.

### MLflow Experiment Tracking

> **Deployment status, corrected 2026-09-27 (Gate 8.6-b).** The sentence that
> stood here — that `list_recent_runs()` returns an empty list because nothing
> logs to it — stopped being true in Gate 8.6 and is replaced, not appended,
> because this section describes what the code does now.
>
> `scripts/train_heart_glm.py` logs **one run per image build** into the store
> baked into the image, recording that artifact's provenance and no accuracy
> figure; the admin endpoint reports a `status` of `unavailable` / `unreachable`
> / `empty` / `ok` instead of `count: 0` for all three. What is still **not**
> live is a tracking *server*: with no `MLFLOW_TRACKING_URI` the store is a
> SQLite file inside the image, holding exactly that image's build, and a new
> build replaces it.
>
> The **research** history — the seven-layer ladder, the six model families, the
> nine ablation arms, the calibration candidates, HF-13 and the external Tehran
> cohort, 45 runs in all — lives in a permanent store in the experiments
> repository, not here and not in the image. Each of those runs is tagged
> `post_hoc: true` with the commit of the results file it was read from: they are
> **readings of saved results, not re-runs**, and none of them was tracked live.

[`log_model_info()`](backend/monitoring/mlflow_tracker.py:63) and [`log_drift_metrics()`](backend/monitoring/mlflow_tracker.py:110) in [`mlflow_tracker.py`](backend/monitoring/mlflow_tracker.py) log all runs under the `"OmniDiag"` experiment. [`ensure_experiment()`](backend/monitoring/mlflow_tracker.py:46) is idempotent — it creates the experiment on first call and returns the existing `experiment_id` on subsequent calls. Model runs log `disease` and `model_version` as MLflow **tags**, evaluation metrics as MLflow **metrics** (via `mlflow.log_metrics()` — a separate mechanism from tags), and `.pkl` artifacts from `models/{disease}/`. Drift runs log `drift_share`, `drifted_columns`, `total_columns`, and `sample_size` as metrics with a `run_type=drift` tag. The retraining pipeline calls `_log_to_mlflow()` automatically after each incremental update. [`list_recent_runs(n=20)`](backend/monitoring/mlflow_tracker.py:143) backs the admin dashboard endpoint.

### Federated Learning *(🚧 prototype SUPERSEDED — replacement designed, not implemented)*

> **Status, plainly.** The Flower prototype described below is **superseded and will not be developed further**: it transmits a pickled XGBoost model, and the heart module no longer ships an XGBoost model. Its `add_dp_noise()` is not wired into anything and offers no privacy guarantee.
>
> The replacement is **designed and not implemented**: federated learning, cross-silo, over horizontally partitioned data, using **GLORE** (distributed Newton with a per-site intercept, Wu et al. 2012, JAMIA) — chosen because naive FedAvg averages the intercept too, and prevalence across these four hospitals runs from 36 % to 94 % (calibrated intercept −0.67 to +2.37), so one averaged intercept fits none of them. GLORE keeps an intercept per site and shares only the slopes.
>
> **No federated result is claimed anywhere in this repository.** The design critique lives in the research repo (`gate_8_10a.md`); it is a critique, not a measurement, and nothing has been run. When it is run it will be a simulation across four historical cohorts in isolated processes on one machine — no patient rows leaving a site, aggregation mathematically identical to the centralised model — and **not** a trial between real institutions, and **not** differentially private. What is sent (gradients and Hessians) are statistical summaries that may leak information, as in any federated learning without differential privacy.

[`OmniDiagFLClient`](backend/federated/client.py:27) implements the hospital-site node. [`get_parameters()`](backend/federated/client.py:67) serialises the current XGBoost model via `pickle.dumps`; [`set_parameters()`](backend/federated/client.py:77) deserialises and applies the aggregated global model. [`fit()`](backend/federated/client.py:77) runs 10 incremental XGBoost rounds on local EHR data — raw patient records never leave the site. [`evaluate()`](backend/federated/client.py:99) computes local binary accuracy and returns it as a Flower metric dict.

[`start_fl_server()`](backend/federated/aggregator.py:62) configures a Flower `FedAvg` strategy with `fraction_fit=1.0` (all connected clients participate in each round) and a configurable `FL_MIN_CLIENTS` threshold. [`add_dp_noise()`](backend/federated/aggregator.py:96) is a standalone utility that clips a gradient vector to `max_grad_norm` (L2) and adds calibrated Gaussian noise scaled by `noise_multiplier=1.1`. It is **not yet wired** into `client.py`'s `get_parameters()`/`fit()` — those transmit the full pickled XGBoost model (`pickle.dumps(model)`), not a gradient vector, so differential privacy is implemented but not yet applied to weight transmission. Both files use `try: import flwr as fl` with a `log.error` fallback, so the rest of the backend starts without Flower installed; `flwr>=1.0.0` is now listed in `requirements.txt`.

### Schema-Driven Frontend Architecture

The frontend uses a pure schema-driven rendering pipeline with no hardcoded disease forms:

1. **Schema Fetching** — [`useDiseaseSchema`](frontend/src/hooks/useDiseaseSchema.js) fetches `GET /api/v4/{disease}/schema` and caches parsed results in three in-memory Maps. [`DiseaseContext`](frontend/src/context/DiseaseContext.jsx) pre-fetches schemas for all registered diseases at app init and persists the selected disease in `localStorage`.
2. **Schema Parsing** — [`parseSchema()`](frontend/src/utils/schemaFieldParser.js) converts JSON Schema properties into `FieldMetadata[]` with resolved component types: `segmented` (binary radio-group), `toggle` (yes/no switch), `select` (enum dropdown), `slider` (small-range integer), `number` (twin-bound slider + numeric input).
3. **Field Categorisation** — [`categorizeFields()`](frontend/src/utils/featureCategorizer.js) groups fields into ordered categories (Vitals & Signs, Lifestyle, Demographics, Medical History, Healthcare Access, Mental Health, General).
4. **Zod Validation** — [`buildZodSchema()`](frontend/src/utils/schemaToZod.js:27) mirrors Pydantic server-side rules as a Zod object schema for client-side validation without round-trips.
5. **Form Rendering** — [`DynamicClinicalForm`](frontend/src/components/DynamicClinicalForm.jsx) renders categorised accordion cards via [`SchemaFieldFactory`](frontend/src/components/SchemaFieldFactory.jsx), integrated with React Hook Form through [`useDiseaseForm`](frontend/src/hooks/useDiseaseForm.js).

---

## Database Schema

```mermaid
flowchart TD
    subgraph Auth["Auth & Identity"]
        R[roles\nid · name · description]
        U[users\nid · email · hashed_password\nfull_name · is_active\napi_key_hash · api_key_expires_at\ncreated_at · updated_at]
        UR[user_roles\nuser_id · role_id · assigned_at]
        R --- UR --- U
    end

    subgraph Clinical["Clinical Data"]
        P[patients\nid · mrn · full_name\ndate_of_birth · gender\ncontact_email · created_by\ncreated_at · deleted_at]
        PV[patient_visits\nid · patient_id · disease\nvisit_date · features JSON\nrisk_score · prediction · notes]
        PR[predictions\nid · patient_id · disease\ninput_features JSON\nprediction · confidence\nshap_chart_data JSON\ncreated_by · created_at]
        P --> PV
        P --> PR
    end

    subgraph HitL["Human-in-the-Loop"]
        RQ[review_queue\nid · prediction_id UNIQUE\nuncertainty_score\nreviewer_id · label\nreviewed_at · status\ncreated_at]
        PR --> RQ
        U --> RQ
    end

    subgraph Audit["Audit & Compliance"]
        AL[audit_logs\nid · user_id · endpoint\nmethod · status_code\nip_address · duration_ms\ncreated_at]
        U --> AL
    end

    U --> PR
    U --> P
```

Two Alembic migrations ship with the project:

| Revision | Description |
|---|---|
| `da87946a56a6` | Initial schema — roles, users, user_roles, patients, predictions, review_queue, audit_logs |
| `f1e2d3c4b5a6` | Adds `patient_visits`; adds `api_key_hash` + `api_key_expires_at` to users; fixes `audit_logs.id` type (BigInteger → VARCHAR 36) |

The `patients.deleted_at` nullable timestamp implements GDPR soft-delete — prediction history is preserved when a patient record is removed. The `review_queue.prediction_id` UNIQUE constraint enforces one-to-one cardinality with `predictions`. All primary keys use `VARCHAR(36)` UUIDs for cross-database portability between SQLite (development) and PostgreSQL (production).

---

## Repository Structure

```
.
├── backend/                              # FastAPI application layer
│   ├── main.py                           # App entry, middleware stack, lifespan, routes
│   ├── router.py                         # OmniDiagRouter — config-driven disease routing
│   ├── schemas.py                        # Pydantic models + DISEASE_SCHEMA_REGISTRY
│   ├── model_loader.py                   # Lazy XGBoost loader + SHAP TreeExplainer + XGB3 patch
│   ├── ensemble_loader.py                # Stacking ensemble (XGB/LGB/RF + LR meta-learner) — BRFSS diabetes
│   ├── model_backends/                   # One class per model family, registered once
│   │   ├── base.py                       # ModelBackend interface + capabilities flags
│   │   ├── heart_glm_conformal.py        # Heart: Spline-GLM + Venn-Abers + Mondrian conformal
│   │   ├── diabetes_ebm_conformal.py     # NHANES: EBM + Platt + age-band conformal + What-If
│   │   ├── stacking_ensemble.py          # BRFSS diabetes stacking ensemble
│   │   └── sklearn_pipeline.py, sklearn_generic.py
│   ├── heart_glm/                        # Heart bundle loader (stack.py) + reference scores
│   ├── diabetes_what_if_levers.py        # NHANES What-If lever policy (own module; imports nothing from backend)
│   ├── schemas_diabetes_nhanes.py        # NHANES input schema (20 fields, 6 mandatory)
│   ├── schemas_clinical_action.py        # clinical_action_plan built from the decision
│   ├── shap_service.py                   # SHAP value → JSON chart data transformation
│   ├── counterfactual_generator.py       # DiCE-inspired generator + Clinical Firewall
│   ├── database.py                       # Async SQLAlchemy engine + session factory
│   ├── cache.py                          # Redis / InMemoryBackend init + cache_get/set/flush
│   ├── rate_limit.py                     # SlowAPI limiter + LIMIT_CLINICAL/VIEWER/ADMIN
│   │
│   ├── auth/                             # Authentication & authorisation
│   │   ├── routes.py                     # /auth/register, /login, /refresh, /logout, /me
│   │   ├── jwt.py                        # encode_token(), decode_token()
│   │   ├── dependencies.py               # get_current_user, get_optional_user
│   │   ├── rbac.py                       # require_role() dependency factory
│   │   └── api_key.py                    # X-API-Key hash verification
│   │
│   ├── db_models/                        # SQLAlchemy ORM models
│   │   ├── user.py                       # User + user_roles association table
│   │   ├── role.py                       # Role (super_admin/admin/doctor/nurse/viewer)
│   │   ├── patient.py                    # Patient with soft-delete (deleted_at)
│   │   ├── patient_visit.py              # PatientVisit — longitudinal risk snapshots
│   │   ├── prediction.py                 # Prediction + shap_chart_data JSON
│   │   ├── review_queue.py               # ReviewQueue — active learning annotation store
│   │   └── audit_log.py                  # AuditLog — immutable HIPAA audit trail
│   │
│   ├── active_learning/                  # Human-in-the-loop pipeline
│   │   ├── sampler.py                    # prediction_entropy(), should_queue_for_review()
│   │   ├── routes.py                     # /review/queue, /review/{id}/annotate, /skip, /stats
│   │   ├── retrain.py                    # run_retrain_pipeline() — fetch → retrain → reload → log
│   │   └── diabetes_nhanes_candidate.py  # NHANES active learning: candidate only, never promoted
│   │
│   ├── monitoring/                       # MLOps observability
│   │   ├── metrics.py                    # Prometheus: 8 counters/histograms/gauges
│   │   ├── drift.py                      # DriftMonitor (Evidently) + get_monitor() singletons
│   │   ├── drift_stats.py                # Input drift on scipy alone: KS, chi-square, PSI
│   │   ├── mlflow_tracker.py             # log_model_info(), log_drift_metrics(), list_recent_runs()
│   │   └── routes.py                     # /admin/drift/*, /admin/mlflow/*
│   │
│   ├── llm/
│   │   └── report_generator.py           # generate_report() — DeepSeek API + rule-based fallback
│   │
│   ├── nlp/
│   │   └── notes_parser.py               # parse_clinical_note() — regex (English only); BioBERT path unused in deployment
│   │
│   ├── federated/                        # Federated learning (🚧 server manifest pending)
│   │   ├── aggregator.py                 # FedAvg strategy + add_dp_noise() (Gaussian DP)
│   │   └── client.py                     # OmniDiagFLClient — get_parameters/fit/evaluate
│   │
│   ├── admin/
│   │   └── routes.py                     # /admin/audit-logs, /admin/retrain
│   ├── patients/
│   │   └── routes.py                     # /patients/* CRUD (soft-delete aware)
│   └── middleware/
│       ├── audit.py                      # AuditMiddleware — writes every request to audit_logs
│       └── security.py                   # SecurityHeadersMiddleware — HTTPS enforcement
│
├── features/                             # Per-disease feature engineering
│   ├── base_features.py                  # BaseFeatureEngineer (ABC) — 3 abstract methods
│   ├── heart_disease_features.py         # CAD: heuristic / clinical / medical paths
│   └── diabetes_features.py             # DM: heuristic / clinical / medical paths
│
├── models/                               # Trained model artifacts
│   ├── advanced_feature_engineering.py   # Standalone feature computation functions
│   ├── heart_disease/                    # CAD bundle (heart_l3_glm_stack.pkl), built at image build time
│   ├── diabetes/                         # BRFSS DM ensemble weights + preprocessors (superseded)
│   └── diabetes_nhanes/                  # NHANES EBM bundle (.joblib), model_card.json, drift_reference.json
│
├── configs/                              # Disease YAML configurations
│   ├── heart_disease.yaml                # CAD v7.0.0 — Spline-GLM + Venn-Abers + Mondrian conformal
│   ├── diabetes.yaml                     # DM v1.1.0 — BRFSS stacking ensemble (superseded)
│   ├── diabetes_nhanes.yaml              # NHANES v2.0.0 — EBM + Platt + age-band conformal, alpha 0.20
│   └── config_loader.py                  # Centralised YAML loader
│
├── alembic/                              # Database migrations
│   ├── env.py                            # Async-to-sync URL strip; imports all ORM models
│   └── versions/
│       ├── da87946a56a6_initial_schema.py
│       └── f1e2d3c4b5a6_add_patient_visits_api_keys_fix_audit_id.py
│
├── k8s/                                  # Kubernetes manifests
│   ├── namespace.yaml                    # Namespace: omnidiag
│   ├── backend-deployment.yaml           # 2 replicas, port 7860, model-cache PVC
│   ├── postgres-statefulset.yaml         # postgres:15-alpine, 10 Gi PVC
│   ├── redis-deployment.yaml             # redis:7-alpine
│   ├── ingress.yaml                      # api.omnidiag.ai — TLS via cert-manager
│   ├── hpa.yaml                          # Min 2 / max 10 replicas, CPU 70% / Mem 80%
│   ├── pvc.yaml                          # 5 Gi ReadWriteMany model cache
│   └── secrets.yaml                      # Secret template (fill before apply)
│
├── helm/omnidiag/                        # Helm chart v2.0.0
│   ├── Chart.yaml
│   ├── values.yaml                       # Parameterises all K8s resources
│   └── templates/                        # deployment, ingress, hpa, secrets, service
│
├── deploy/                               # Monitoring provisioning
│   ├── prometheus.yml                    # Scrape config targeting backend:7860/metrics
│   └── grafana/
│       ├── datasources/prometheus.yml    # Grafana datasource provisioning
│       └── dashboards/omnidiag.json      # Pre-built dashboard (predictions, latency, drift)
│
├── tests/                                # 41 test modules (SQLite in-memory, no Redis needed)
│   ├── conftest.py                       # Fixtures: DB, cache, mocked router, seeded roles
│   ├── test_auth.py
│   ├── test_rbac.py
│   ├── test_clinical.py
│   ├── test_patients.py
│   ├── test_admin.py
│   ├── test_audit_log.py
│   ├── test_cache.py
│   ├── test_llm_report.py
│   ├── test_security.py
│   ├── test_database.py
│   ├── test_unit_active_learning.py
│   ├── test_unit_auth.py
│   ├── test_unit_counterfactuals.py
│   ├── test_diabetes_ebm_backend.py      # NHANES backend: each test names the finding it protects
│   ├── test_diabetes_gate93.py           # NHANES What-If levers (incl. named adiposity band)
│   ├── test_diabetes_frontend_contract.py, test_diabetes_monitoring.py, test_diabetes_active_learning.py
│   └── … heart GLM, What-If policy, drift and MLflow-state tests (full list: ls tests/)
│
├── frontend/                             # React 18 + Vite + Tailwind CSS
│   ├── src/
│   │   ├── api.js                        # OmniDiagApi class (120 s timeout for HF cold starts)
│   │   ├── App.jsx                       # Sidebar navigation, mode routing
│   │   ├── context/DiseaseContext.jsx    # Schema pre-fetch + localStorage persistence
│   │   ├── hooks/
│   │   │   ├── useDiseaseSchema.js       # Schema fetch + parse + 3-layer Map cache
│   │   │   └── useDiseaseForm.js         # React Hook Form + Zod integration
│   │   ├── components/
│   │   │   ├── EngineeringMode.jsx       # Manual parameter testing interface
│   │   │   ├── ClinicalEmrMode.jsx       # Doctor's patient record dashboard
│   │   │   ├── DynamicClinicalForm.jsx   # Schema-driven form engine
│   │   │   ├── SchemaFieldFactory.jsx    # Component resolver (toggle/select/slider/number)
│   │   │   ├── ShapBarChart.jsx          # Recharts horizontal SHAP bar chart
│   │   │   ├── WhatIfScenarioCard.jsx    # Counterfactual viewer (retries once on 502/503/504)
│   │   │   ├── ClinicalActionPlan.jsx    # Next clinical step from the decision (NHANES)
│   │   │   └── MedicalTooltip.jsx        # Dictionary-backed hover tooltip
│   │   └── utils/
│   │       ├── schemaFieldParser.js      # JSON Schema → FieldMetadata[]
│   │       ├── schemaToZod.js            # FieldMetadata[] → Zod validation schema
│   │       ├── featureCategorizer.js     # Field → category grouping
│   │       └── medicalDictionary.js      # Feature name → clinical description
│   └── mockPatients.js                   # Pre-defined patient records (CAD, BRFSS DM, NHANES)
│
├── docs/
│   ├── assets/screenshots/               # Real UI captures: heart (10_, 20_) and NHANES (30_ to 40_)
│   ├── assets/research/                  # Heart research figures copied from the experiments repo
│   └── phase9/                           # NHANES record: DISCOVERY_RECORD, FINDINGS_REGISTER, RESEARCH_FIDELITY_AUDIT, figures/
├── data/                                 # Raw and processed datasets per disease
├── experiment_files/                     # Training experiments and diagnostics
├── docker-compose.yml                    # 7 services: postgres, redis, backend, mlflow, prometheus, grafana, retrain
├── Dockerfile                            # Production container
├── requirements.txt                      # Python dependencies
├── alembic.ini                           # Alembic config (sqlite fallback for local dev)
└── pytest.ini                            # asyncio_mode=auto, testpaths=tests/
```

---

## Local Setup

### Backend (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set JWT_SECRET_KEY at minimum

# SQLite fallback is automatic when DATABASE_URL is unset
uvicorn backend.main:app --reload --port 7860
```

API docs available at `http://localhost:7860/docs`. Redis is optional — the cache auto-falls-back to `InMemoryBackend` when `REDIS_URL` is unset.

### Full Stack (Docker Compose)

```bash
docker compose up -d
docker compose exec backend alembic upgrade head
```

| Service | URL | Default credentials |
|---|---|---|
| FastAPI | http://localhost:7860/docs | `doctor@omnidiag.com` / `Doctor@123` |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | `admin` / `omnidiag_grafana` |

```bash
# On-demand retrain (profile-gated — does not start with up -d)
docker compose run --rm retrain --disease heart_disease --min-samples 10
```

### Frontend (Node.js 20+)

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### Federated Learning

```bash
# Central aggregation server
python -m backend.federated.aggregator --rounds 10 --min-clients 2

# Each hospital site — patient data stays local
python -m backend.federated.client \
  --server-address <central>:8080 \
  --disease heart_disease \
  --data-path /local/ehr_export.csv
```

---

## Deployment (Kubernetes + Helm)

The manifests below are the intended production deployment target; the live demo instance runs on Hugging Face Spaces via Docker (see the `sdk: docker` / `app_port: 7860` frontmatter at the top of this file), not on a Kubernetes cluster.

```bash
# Apply all K8s manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Or via Helm
helm install omnidiag ./helm/omnidiag -n omnidiag --create-namespace
helm upgrade omnidiag ./helm/omnidiag -n omnidiag --set image.tag=v2.1.0
```

The `hpa.yaml` autoscaler scales the backend between 2 and 10 replicas, targeting CPU utilisation at 70 % and memory at 80 %. The Ingress routes `api.omnidiag.ai` with TLS termination via cert-manager and Let's Encrypt. PostgreSQL runs as a StatefulSet with a 10 Gi PVC; model weights are shared across pods via a 5 Gi ReadWriteMany PVC mounted at `/app/models`.

---

## CI/CD Pipeline

The GitHub Actions workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request with two parallel jobs:

**Backend Validation** — Python 3.11 setup with pip caching; dependency install from `requirements.txt`; import verification (`OmniDiagRouter`, `ModelLoader`, `get_schema_for_disease`); API smoke test (starts uvicorn, verifies `GET /` returns 200, terminates).

**Frontend Build** — Node.js 20 with npm caching; `npm ci`; production build via `npm run build` (Vite compiles to `frontend/dist/`).

---

## API Reference

All endpoints are versioned under `/api/v4/`.

### System

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | None | Health check |
| `GET` | `/api/v4/diseases` | None | Registered diseases with metadata |
| `GET` | `/api/v4/{disease}/schema` | None | JSON Schema for input fields |
| `GET` | `/metrics` | None | Prometheus scrape endpoint |

### Auth

Mounted at `/auth` (**not** under `/api/v4`) — verified live against a running instance:

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/register` | Create account |
| `POST` | `/auth/login` | JWT login (sets HttpOnly cookie) |
| `POST` | `/auth/refresh` | Rotate access token |
| `POST` | `/auth/logout` | Clear cookie |
| `GET` | `/auth/me` | Current user + roles |

### Clinical

| Method | Path | Auth | Response |
|---|---|---|---|
| `POST` | `/api/v4/{disease}/predict` | Anonymous OK¹ | `PredictResponse` — heart: `{prediction, confidence, diagnosis, decision, conformal_set, decision_is_referral, probability_lower, probability_upper, output_type, probability_scale}` and **no `inference_threshold`**, plus `data_completeness_warning` when an input the model reads was missing and imputed; diabetes adds the scale audit fields below; `diabetes_nhanes` returns the same decision envelope as heart plus `clinical_action_plan`. `Cache-Hit` is a response **header**, not a body field |
| `POST` | `/api/v4/{disease}/explain` | Anonymous OK¹ | `{prediction, confidence, diagnosis, chart_data[], text_explanation, base_value}`; diabetes adds `base_value_raw, shap_scale, shap_reconstructed_probability_corrected, shap_additivity_gap, per_model_shap, shap_weights, ensemble_variance, model_agreement` |
| `POST` | `/api/v4/{disease}/counterfactuals` | Anonymous OK¹ | diabetes: `{status, baseline_probability, baseline_probability_corrected, probability_scale, counterfactuals[{scenario, changes{}, new_probability, new_probability_corrected, baseline_probability_corrected, risk_reduction, risk_reduction_relative_pct, risk_reduction_absolute_pp, probability_scale, feasibility}]}`; heart and `diabetes_nhanes`: `{status, baseline_probability, counterfactuals[{scenario_id, probability, changes[{feature, original_value, counterfactual_value, direction}]}]}`, and NHANES adds `best_achievable` when no scenario crosses the decision. The NHANES call takes about 0.3 s; the BRFSS diabetes call takes about 15 s locally |
| `POST` | `/api/v4/{disease}/batch` | `doctor`/`nurse`/`super_admin` | `{results[], summary{total, ok, errors}}` — CSV upload, max 500 rows |
| `POST` | `/api/v4/generate-report` | Anonymous OK¹ | body `{disease, probability_corrected, label, shap_values[], features{}}` (`probability` is a **deprecated** alias; `confidence_band` is advisory and ignored) → `{report, source, risk_band, llm_model?, latency_ms?, fallback_reason?}` |
| `POST` | `/api/v4/parse-notes` | Open (no auth dependency) | `{extracted_features{}, mapped_features{}, field_count}` — `mapped_features` holds the schema-ready dict when `disease` is supplied |

¹ Deliberately open so the public demo works without an account (`Depends(get_optional_user)`). Predictions are **persisted and linkable to a patient record only for authenticated users**; anonymous calls are stateless. Rate limits still apply per IP.

#### Probability scale (diabetes)

The diabetes ensemble was trained on the 50/50-resampled BRFSS file, so its raw output sits on a 50 % prior. `EnsembleModelLoader.predict()` maps it to the deployment prevalence (`configs/diabetes.yaml → prevalence_deploy`, currently 0.237 — Jordan's actual diabetes prevalence; a US/BRFSS placeholder of ~0.14 was used until 2026-09-21) with a Bayes prior-shift correction before anything is compared or returned. The rule the codebase follows ([`backend/probability_scale.py`](backend/probability_scale.py), enforced by the AST guard in [`tests/test_probability_scale_contract.py`](tests/test_probability_scale_contract.py)):

> An unsuffixed probability or threshold is on the scale the module reports — **corrected** for diabetes. A value on the model's own training-prior scale always carries a `_raw` suffix.

| Field | Scale | Notes |
|---|---|---|
| `confidence` | corrected | Same value as `probability_corrected`. Compare with `inference_threshold`, **never with 0.5** |
| `probability_corrected` / `probability_raw` | corrected / raw | Both returned for audit |
| `inference_threshold` / `inference_threshold_raw` | corrected / raw | 0.108184 / 0.280854 |
| `risk_bands` / `risk_bands_raw` | corrected / raw | high 0.4202 / 0.70, moderate 0.1716 / 0.40 |
| `prevalence_correction_applied`, `prevalence_train`, `prevalence_deploy` | — | `true`, 0.50, 0.237 |
| `model_contributions`, `ensemble_variance` | raw | Per-base-model outputs, uncorrected |

`GET /api/v4/diseases` returns `info.inference_threshold` and `info.risk_bands` on the same scale. Stored rows carry `predictions.probability_scale` (`'corrected'` | `'raw'` | `NULL` for rows written before the column existed); aggregates split by it (`GET /api/v4/admin/stats → avg_confidence_by_scale`, drift runs → `rows_by_probability_scale`) and treat `NULL` as its own `unknown` group. Heart applies no prevalence correction and has a single scale. It returns **no `inference_threshold` and no `risk_bands`** — its decision is a conformal set, not a probability against a cut-point — and `GET /api/v4/diseases` reports both as `null` for it, alongside `output_type: "conformal_decision"` and `probability_scale: "ivap_calibrated_training_mix"`. Consumers branch on `output_type`; an absent threshold is information, not a gap to fill with 0.5.

A legacy `POST /api/v3/predict` (tagged **Legacy** in Swagger) is still mounted and, unlike its v4 counterpart, requires `CLINICAL_ROLES`. It is retained for backward compatibility only — new integrations should use `/api/v4/{disease}/predict`.

### Active Learning (requires `doctor` / `nurse` / `super_admin`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v4/review/queue` | Paginated pending items, filterable by disease |
| `POST` | `/api/v4/review/{id}/annotate` | Submit expert label `{label: 0|1}` |
| `POST` | `/api/v4/review/{id}/skip` | Dismiss item |
| `GET` | `/api/v4/review/stats` | `{pending, reviewed, skipped}` |

> ⚠️ `POST /admin/retrain` currently fails for heart_disease (categorical encoding bug) and is a silent no-op for diabetes (writes to a file the live ensemble doesn't read) — see [Human-in-the-Loop Active Learning](#human-in-the-loop-active-learning) above.

### Admin (requires `super_admin`)

Two different prefixes are actually in use — verified live against a running instance: `/admin/*` (audit, retrain, cache) and `/api/v4/admin/*` (drift, MLflow).

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/audit-logs` | Paginated HIPAA audit trail |
| `POST` | `/admin/retrain` | Trigger retraining pipeline (see limitation above) |
| `POST` | `/admin/cache/flush` | Invalidate all cached responses (e.g. after a model update) |
| `GET` | `/api/v4/admin/drift/{disease}/status` | Latest Evidently drift metrics (JSON) |
| `POST` | `/api/v4/admin/drift/{disease}/run` | Trigger fresh drift computation |
| `GET` | `/api/v4/admin/drift/{disease}/report` | Full Evidently HTML report |
| `GET` | `/api/v4/admin/mlflow/runs` | Recent MLflow experiment runs |
| `POST` | `/api/v4/admin/mlflow/register` | Register model metrics in MLflow |

---

## Tech Stack

### Backend

| Component | Technology | Purpose |
|---|---|---|
| API Framework | [FastAPI 0.100+](backend/main.py:1) | Async Python web framework |
| Model v1 (CAD) | [XGBoost 2.x](backend/model_loader.py) | Single gradient-boosted classifier |
| Model v2 (DM) | [Stacking Ensemble](backend/ensemble_loader.py) | XGBoost + LightGBM + RF + LR meta |
| Explainability | [SHAP 0.42+](backend/monitoring/metrics.py:37) | TreeExplainer → structured JSON |
| Database ORM | [SQLAlchemy 2.0+](backend/database.py) | Async session + declarative base |
| Migrations | [Alembic 1.13+](alembic/env.py) | Version-controlled schema changes |
| Cache | [Redis 7 + fastapi-cache2](backend/cache.py:39) | Prediction caching + rate-limit store |
| Auth | [python-jose + passlib](backend/auth/jwt.py) | JWT + bcrypt |
| Rate Limiting | [SlowAPI](backend/rate_limit.py:49) | Per-route request caps |
| LLM | [OpenAI SDK → DeepSeek](backend/llm/report_generator.py:100) | Clinical report generation |
| NLP | [Transformers + regex](backend/nlp/notes_parser.py:249) | Clinical note feature extraction |
| Metrics | [prometheus-client 0.20+](backend/monitoring/metrics.py) | `/metrics` scrape endpoint |
| Drift | [Evidently 0.4+](backend/monitoring/drift.py:48) | Dataset drift detection — **code only, not live** |
| MLOps | [MLflow 2.10+](backend/monitoring/mlflow_tracker.py:63) | Experiment tracking — **code only, not live** |
| Federated | [Flower (flwr 1.0+)](backend/federated/aggregator.py:62) | Prototype, **superseded** — see [Federated Learning](#federated-learning-🚧-prototype-superseded--replacement-designed-not-implemented) |
| Validation | [Pydantic v2](backend/schemas.py) | Input/output model validation |
| Config | [PyYAML 6.0+](configs/config_loader.py) | Disease configuration files |

### Frontend

| Component | Technology | Purpose |
|---|---|---|
| Framework | [React 18.3.1](frontend/src/App.jsx) | UI component library |
| Build | [Vite 5.4.0](frontend/vite.config.js) | Dev server + production bundler |
| Styling | [Tailwind CSS 3.4.19](frontend/tailwind.config.js) | Utility-first CSS |
| Forms | [React Hook Form 7.76.1](frontend/src/hooks/useDiseaseForm.js) | Form state management |
| Validation | [Zod 4.4.3](frontend/src/utils/schemaToZod.js:27) | Client-side schema validation |
| Charts | [Recharts 3.8.1](frontend/src/components/ShapBarChart.jsx) | SHAP value bar chart |
| PDF Export | [@react-pdf/renderer 4.5.1](frontend/package.json) | Clinical report PDF generation |
| PWA | [Workbox 7.4.1](frontend/package.json) | Service worker + offline support |

### Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| Primary Database | [PostgreSQL 15-alpine](docker-compose.yml:23) | Relational datastore |
| Cache | [Redis 7-alpine](docker-compose.yml:43) | Prediction cache + rate limiting |
| Experiment Tracking | [MLflow on python:3.11-slim](docker-compose.yml:79) | Model and drift run registry |
| Metrics Scraper | [prom/prometheus:v2.51.0](docker-compose.yml:103) | Pulls from `/metrics` every 15 s |
| Dashboards | [grafana/grafana:10.4.0](docker-compose.yml:120) | Pre-provisioned OmniDiag dashboard |
| Container | [Docker + Compose v2](docker-compose.yml) | Local full-stack environment |
| Orchestration | [Kubernetes + Helm v2](helm/omnidiag/Chart.yaml) | Production autoscaling deployment |
| TLS | [cert-manager + Let's Encrypt](k8s/ingress.yaml) | Automatic certificate provisioning |
| CI/CD | [GitHub Actions](.github/workflows/ci.yml) | Backend smoke test + frontend build |

---

## License

Copyright (c) 2026 Yahya Lababenah. All Rights Reserved. — see [`LICENSE`](LICENSE) for full terms.
