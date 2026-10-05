# OmniDiag Architecture (Mermaid)

Modular monolith: one FastAPI process, feature modules, config-driven model-family registry.

## ١. الصورة الكبرى: Modular Monolith

عملية FastAPI واحدة (deployable واحد) مقسّمة داخلياً إلى وحدات (modules) بحدود واضحة؛ الأمراض تُضاف بملف YAML دون لمس باقي الكود.

```mermaid
flowchart TB
  subgraph CL["Clients"]
    direction LR
    SPA["React SPA - Vite, Tailwind, PWA<br/>(Vercel)"]
    EMR["EMR / API clients<br/>JWT cookie or X-API-Key"]
  end

  subgraph MW["Cross-cutting middleware (order of wrapping)"]
    direction LR
    CORS["CORS"] --> SEC["SecurityHeaders<br/>HTTPS enforce"] --> AUD["AuditMiddleware<br/>writes audit_logs"] --> MET["MetricsMiddleware<br/>Prometheus"] --> RL["slowapi rate limit<br/>per-endpoint"]
  end

  subgraph APP["FastAPI process = the monolith (backend/main.py)"]
    direction TB
    subgraph MODS["Feature modules (APIRouter each)"]
      direction LR
      AUTH["auth<br/>/auth"]
      ADM["admin<br/>/admin"]
      PAT["patients<br/>/api/v4/patients"]
      CLIN["clinical diagnosis<br/>/api/v4/{disease}/..."]
      AL["active_learning (HITL)<br/>/api/v4/review"]
      MON["monitoring<br/>/api/v4/admin/drift, mlflow"]
      NLP["nlp<br/>notes_parser"]
      LLM["llm<br/>report_generator"]
      FED["federated<br/>SUPERSEDED prototype"]
    end

    subgraph CORE["Shared core services"]
      direction LR
      ROUTER["OmniDiagRouter<br/>scans configs/*.yaml"]
      PS["probability_scale +<br/>prevalence_correction"]
      SHAP["shap_service"]
      CF["counterfactual_generator +<br/>diabetes_what_if_levers"]
      CACHE["cache.py<br/>Redis or in-memory"]
    end

    subgraph REG["Model-family registry (backend/model_backends)"]
      direction LR
      B1["glm_ivap_conformal<br/>HEART"]
      B2["stacking_ensemble<br/>DIABETES BRFSS"]
      B3["ebm_platt_conformal<br/>DIABETES NHANES"]
      B4["sklearn_pipeline /<br/>sklearn_generic"]
    end
  end

  subgraph DATA["State and artifacts"]
    direction LR
    DB[("SQL DB<br/>SQLite dev / PostgreSQL prod<br/>SQLAlchemy async + Alembic")]
    RED[("Redis cache")]
    ART[("models/ bundles<br/>built at image build")]
    MLF[("MLflow<br/>runs and registry")]
    CFG["configs/*.yaml<br/>single source of truth per disease"]
  end

  subgraph OBS["Observability"]
    PROM["Prometheus"] --> GRAF["Grafana"]
  end

  subgraph EXT["External"]
    direction LR
    DS["DeepSeek API<br/>(OpenAI-compatible)"]
    BERT["BioBERT NER<br/>HuggingFace, optional"]
  end

  %% Original Exact Connections
  CL --> MW
  MW --> MODS
  CLIN --> ROUTER
  ROUTER --> REG
  CLIN --> PS
  CLIN --> SHAP
  CLIN --> CF
  CLIN --> CACHE
  CACHE -.-> RED
  CFG --> ROUTER
  REG --> ART
  AUTH --> DB
  PAT --> DB
  ADM --> DB
  AL --> DB
  AUD -.-> DB
  MON --> MLF
  AL --> MLF
  MET --> PROM
  LLM -.-> DS
  NLP -.-> BERT
```
## ٢. خريطة الوحدات (Package Map)

كل مجلد = وحدة بمسؤولية واحدة. الوحدات تتحدث عبر استدعاءات داخلية (in-process) لا عبر شبكة، وهذا ما يجعله Monolith، والحدود النظيفة بينها هي ما يجعله Modular.

```mermaid
flowchart LR
  ROOT(["OmniDiag repo"])
  ROOT --> BE["backend/"]
  ROOT --> FE["frontend/src"]
  ROOT --> CF["configs/ + features/"]
  ROOT --> SC["scripts/ + models/ + experiment_files/"]
  ROOT --> OPS["Dockerfile, docker-compose, k8s, helm, deploy, .github"]
  ROOT --> TST["tests/ (about 60 files)"]
  ROOT --> DOC["docs/, evaluation_evidence/, alembic/"]
  BE --> m1["main.py - app wiring, lifespan,<br/>predict/explain/batch/report/parse-notes"]
  BE --> m2["router.py - disease dispatcher"]
  BE --> m3["model_backends/ - 5 families,<br/>auto-registered by file"]
  BE --> m4["heart_glm/ - stack.py, candidate.py,<br/>reference_scores.json"]
  BE --> m5["auth/ - jwt, hashing, rbac, api_key"]
  BE --> m6["admin/ - users, audit, cache flush,<br/>api-keys, stats, retrain"]
  BE --> m7["patients/ - EMR: patients, visits,<br/>predictions, export, notes"]
  BE --> m8["active_learning/ - sampler, routes,<br/>retrain, nhanes candidate"]
  BE --> m9["monitoring/ - drift, drift_stats,<br/>metrics, mlflow_tracker"]
  BE --> m10["nlp/ + llm/"]
  BE --> m11["middleware/ - audit, security"]
  BE --> m12["db_models/ - 8 ORM tables"]
  BE --> m13["federated/ - superseded"]
  CF --> c1["heart_disease.yaml<br/>diabetes.yaml<br/>diabetes_nhanes.yaml"]
  CF --> c2["heart_disease_features.py<br/>diabetes_features.py"]
  SC --> s1["train_heart_glm.py<br/>build_drift_reference.py<br/>retrain.py, seed_db.py"]
```

## ٣. رحلة طلب التنبؤ (Predict Request)

ما يحدث من الضغط على Predict حتى ظهور النتيجة: المصادقة، الكاش، النموذج، التسجيل، وإضافة الحالات غير المؤكدة لطابور المراجعة البشرية.

```mermaid
sequenceDiagram
  autonumber
  participant U as Clinician (React)
  participant MW as Middleware
  participant API as "POST /api/v4/{disease}/predict"
  participant AU as Auth + RBAC
  participant C as Cache (Redis or memory)
  participant R as OmniDiagRouter
  participant B as ModelBackend (per family)
  participant DB as SQL DB
  participant AL as Active-learning sampler

  U->>MW: JSON features
  MW->>API: rate limit, security headers, start timer
  API->>AU: JWT cookie or API key
  AU-->>API: user, roles must be doctor, nurse or super_admin
  API->>API: validate against schema from YAML
  API->>C: get(hash of disease + features)
  
  alt cache hit
    C-->>API: stored result (scale checked)
  else cache miss
    API->>R: predict in threadpool
    R->>B: engineer features, model, calibrate
    B-->>R: probability, decision, bounds
    R-->>API: result
    API->>C: set(result, ttl)
  end

  API->>API: record Prometheus counters (hit or miss)
  API->>DB: INSERT predictions (scale and threshold stored)
  API->>AL: uncertain?
  
  alt conformal decision is uncertain, or entropy near threshold
    AL->>DB: INSERT review_queue (pending)
  end

  API-->>MW: PredictResponse + Cache-Hit header
  MW->>DB: INSERT audit_logs (user, endpoint, status, ms)
  MW-->>U: result, SHAP available via /explain
```

## ٤. الإضافة بالإعدادات: Config-driven Registry

إضافة مرض = ملف YAML + ملف features. الـ Router لا يعرف أسماء الأمراض ولا العائلات. كل عائلة نموذج تسجّل نفسها بمجرد وجود ملفها في model_backends.

```mermaid
flowchart LR
  subgraph FS["Filesystem"]
    Y1["configs/heart_disease.yaml<br/>family: glm_ivap_conformal"]
    Y2["configs/diabetes.yaml<br/>family: stacking_ensemble"]
    Y3["configs/diabetes_nhanes.yaml<br/>family: ebm_platt_conformal"]
    FT["features/*_features.py"]
  end

  subgraph STARTUP["At startup"]
    SCAN["OmniDiagRouter._load_all_configs()<br/>one entry per YAML"]
    IMP["model_backends/__init__.py<br/>pkgutil imports every module"]
    REGI[("registry<br/>@register_backend('name')")]
    IMP --> REGI
  end

  subgraph IFACE["ModelBackend interface"]
    I1["predict_proba, feature_names, shap_values"]
    I2["predict, predict_batch, explain"]
    I3["BackendCapabilities:<br/>tree_shap, vectorized_batch,<br/>counterfactuals, explainer"]
  end

  Y1 --> SCAN
  Y2 --> SCAN
  Y3 --> SCAN
  SCAN -->|family key| GET["get_backend(family)"]
  REGI --> GET

  GET -->|unknown family fails at startup| ERR["UnknownModelFamilyError"]
  GET --> L1["HeartGlmConformal"]
  GET --> L2["StackingEnsemble"]
  GET --> L3["EbmPlattConformal"]
  GET -.-> L4["sklearn_pipeline / sklearn_generic"]

  L1 -.->|lazy load on first request| W1[("heart_l3_glm_stack.pkl")]
  L2 -.-> W2[("rf, xgb, lgb + meta_learner")]
  L3 -.-> W3[("diabetes_nhanes_ebm.joblib")]

  FT --> L1
  FT --> L2

  L1 --- I1
  L2 --- I1
  L3 --- I1
```

## ٥. نموذج القلب: Spline-GLM + Venn-Abers + Conformal

القرار ليس احتمالاً مقابل عتبة واحدة، بل مجموعة conformal تعطي referral / no_referral / uncertain، مع فاصل [p_lower, p_upper]. 7 مدخلات فقط، مدرَّب على 920 مريضاً من 4 مستشفيات.

```mermaid
flowchart LR
  IN["7 inputs (L3)<br/>Age, Sex, ChestPainType, RestingBP,<br/>Cholesterol, FastingBS, RestingECG"]
  PRE["Frozen preprocessing<br/>IterativeImputer, scaler,<br/>SplineTransformer, one-hot"]
  GLM["Spline-GLM<br/>logistic, ridge"]
  S["raw score<br/>(log-odds, exact additive SHAP)"]
  IVAP["IVAP<br/>inductive Venn-Abers"]
  P["calibrated p<br/>+ interval [p_lower, p_upper]"]
  MC["Mondrian conformal<br/>groups: Sex x class<br/>alpha = 0.10"]
  DEC{"decision"}
  R1["referral"]
  R2["no_referral"]
  R3["uncertain"]
  Q["review_queue<br/>(human in the loop)"]
  IN --> PRE --> GLM --> S
  S --> IVAP --> P
  S --> MC --> DEC
  DEC --> R1
  DEC --> R2
  DEC --> R3 --> Q
  subgraph LIM["Declared limits shown in config"]
    L1["Scope: patients already referred for catheterisation"]
    L2["No HIGH/MODERATE/LOW bands (risk_bands: null)"]
    L3["LOHO AUC 0.802, calibration does not transport between hospitals"]
  end
  DEC -.-> LIM
```

## ٦. الإنسان في الحلقة والتعلّم النشط (HITL + Active Learning)

النظام لا يتعلّم من نفسه تلقائياً: الحالات غير المؤكدة تذهب لطبيب، تسميته تُخزَّن، والإعادة التدريبية تنتج مرشَّحاً (candidate) يقرّر إنسان ترقيته.

```mermaid
flowchart TB
  P["Prediction"] --> Q1{"module type"}
  Q1 -->|"conformal (heart, NHANES)"| C1["decision == uncertain"]
  Q1 -->|"threshold (diabetes BRFSS)"| C2["threshold-centred entropy >= 0.88<br/>(prior-shift map sends threshold to 0.5)"]
  C1 --> QUEUE
  C2 --> QUEUE
  QUEUE[("review_queue<br/>status = pending<br/>uncertainty_score, scale, threshold")]
  QUEUE --> UI["ReviewQueuePanel (doctor)<br/>GET /api/v4/review/queue"]
  UI --> ACT{"doctor action"}
  ACT -->|"annotate: label 0/1 + notes"| REV["status = reviewed"]
  ACT -->|"skip"| SKP["status = skipped"]
  REV --> RT["super_admin: POST /admin/retrain<br/>min-samples gate"]
  RT --> FAM{"model.family"}
  FAM -->|"heart: glm_ivap_conformal"| H1["build_bundle() on 920 UCI rows<br/>+ reviewed rows<br/>(same code as shipped build)"]
  H1 --> H2["models/heart_disease/candidates/RUN_ID"]
  H2 --> H3["compare to SHIPPED model:<br/>conformal coverage +<br/>decision transition matrix"]
  H3 --> H4["log to MLflow"]
  H4 --> H5{{"HUMAN decides promotion<br/>(no code path promotes)"}}
  FAM -->|"diabetes legacy path"| D1["incremental XGBoost refit"]
  D1 --> D2["save and reload ModelLoader"]
  D2 --> D3["log to MLflow"]
  FAM -->|"NHANES"| N1["candidate only, never promoted<br/>row admitted only with lab provenance (HbA1c)"]
  SKP -.-> STATS["GET /review/stats"]
  REV -.-> STATS
```

## ٧. معالجة اللغة (NLP) والتقرير بالـ LLM

ملاحظة سريرية نصية تتحول إلى حقول جاهزة للنموذج؛ وبعد التنبؤ يُولَّد تقرير سريري بالـ LLM مع حارس يمنع المحتوى الممنوع وبديل قاعدي عند الفشل.

```mermaid
flowchart TB
  subgraph NLPX["NLP: POST /api/v4/parse-notes (20/min, no auth) or /patients/ID/notes"]
    N0["free-text note"] --> N1{"detect_script"}
    N1 -->|"Arabic"| N1a["language_support warning:<br/>patterns are English only"]
    N1 -->|"Latin"| N2["1. Regex baseline<br/>spelled numbers to digits, negation"]
    N2 --> N3["2. spaCy dependency parse<br/>numeric mentions, linking words,<br/>overrides regex (optional dep)"]
    N3 --> N4["3. BioBERT NER via HuggingFace<br/>lazy load, overrides on overlap"]
    N4 --> N5["map_to_disease_schema<br/>e.g. BRFSS age bucket"]
    N5 --> N6["pre-filled form<br/>missing fields omitted, user confirms"]
  end
  N6 --> PRED["/predict"]
  subgraph LLMX["LLM: POST /api/v4/generate-report (10/min)"]
    PRED --> L0["result: probability, decision, bounds, SHAP top 5"]
    L0 --> L1["band_for_report<br/>one band value used everywhere"]
    L1 --> L2{"DEEPSEEK_API_KEY set?"}
    L2 -->|"yes"| L3["DeepSeek chat via OpenAI SDK"]
    L3 --> L4{"forbidden_content(text)?"}
    L4 -->|"clean"| L5["source = llm"]
    L4 -->|"violations"| L6["rule-based template<br/>fallback_reason recorded"]
    L2 -->|"no"| L6
    L6 --> L7["source = rule_based"]
  end
  L5 --> OUT["ClinicalReportModal / PDFReport"]
  L7 --> OUT
```

## ٨. الأمن والصلاحيات والتدقيق

JWT في كوكي + Refresh، مفتاح API بديل للأنظمة الخارجية، صلاحيات RBAC، وسجل تدقيق لكل طلب.

```mermaid
flowchart TB
  subgraph AUTHF["Authentication"]
    REG["POST /auth/register"] --> HASH["password hashing"]
    LOG["POST /auth/login"] --> JWT["access JWT + refresh cookie"]
    REF["POST /auth/refresh"] --> JWT
    OUT["POST /auth/logout"]
    KEY["X-API-Key (hash stored)<br/>POST /admin/api-keys"]
  end
  subgraph AUTHZ["Authorisation (RBAC)"]
    DEP["get_current_active_user"]
    RR["require_role(...)"]
    ROLES["roles: doctor, nurse, super_admin<br/>CLINICAL_ROLES = doctor, nurse, super_admin<br/>ADMIN_ROLES = super_admin"]
    DEP --> RR --> ROLES
  end
  JWT --> DEP
  KEY --> DEP
  subgraph PROT["What each role reaches"]
    P1["predict, explain, patients, review: clinical roles"]
    P2["users, audit-logs, cache flush, retrain, drift run, mlflow: super_admin"]
  end
  RR --> P1
  RR --> P2
  subgraph DEF["Defence in depth"]
    D1["SecurityHeadersMiddleware + HTTPS enforcement"]
    D2["slowapi limits: LIMIT_CLINICAL, LIMIT_ADMIN, 10/min report, 20/min notes"]
    D3["AuditMiddleware: user, endpoint, method, status, IP, duration_ms"]
    D4["patient soft-delete (deleted_at)"]
  end
  D3 --> AL[("audit_logs")]
```

## ٩. المراقبة والـ Drift و MLflow

مراقبة تشغيلية (Prometheus/Grafana) ومراقبة انحراف المدخلات بإحصاءات KS و chi-square و PSI مقابل مرجع مجمَّد. الانحراف في المدخلات تنبيه للفحص وليس دليلاً على تدهور النموذج.

```mermaid
flowchart LR
  subgraph RUNTIME["Runtime metrics"]
    M1["MetricsMiddleware<br/>request duration, active requests"]
    M2["record_prediction<br/>predictions_total, confidence histogram"]
    M3["cache hits / misses"]
    M4["conformal decisions, batch rows"]
    M5["drift_share gauge"]
  end
  M1 --> ME["GET /metrics"]
  M2 --> ME
  M3 --> ME
  M4 --> ME
  M5 --> ME
  ME --> PR["Prometheus<br/>deploy/prometheus.yml"] --> GR["Grafana dashboards"]
  subgraph DRIFT["Input drift (Gate 8.7c)"]
    REFP[("drift_reference.json<br/>frozen profile<br/>built by build_drift_reference.py")]
    RECENT["recent predictions from DB"]
    PM["ProfileDriftMonitor<br/>KS, chi-square, PSI on scipy"]
    REFP --> PM
    RECENT --> PM
    PM --> ST["GET /drift/DISEASE/status"]
    PM --> RUN["POST /drift/DISEASE/run"]
    PM --> REP["GET /drift/DISEASE/report"]
    RUN --> M5
  end
  subgraph ML["MLflow"]
    TR["mlflow_tracker<br/>experiment OmniDiag"]
    TR --> RUNS["GET /mlflow/runs"]
    TR --> RG["POST /mlflow/register"]
  end
  RUN --> TR
  RET["retrain candidates"] --> TR
```

## ١٠. نموذج البيانات (Database)

ثمانية جداول؛ كل تنبؤ يُسجَّل مع مقياسه وعتبته، وطابور المراجعة مرتبط بالتنبؤ بعلاقة واحد لواحد.

```mermaid
erDiagram
  USERS ||--o{ USER_ROLES : has
  ROLES ||--o{ USER_ROLES : grants
  USERS ||--o{ PATIENTS : created_by
  PATIENTS ||--o{ PATIENT_VISITS : has
  PATIENTS ||--o{ PREDICTIONS : linked
  USERS ||--o{ PREDICTIONS : created_by
  PREDICTIONS ||--o| REVIEW_QUEUE : flagged
  USERS ||--o{ REVIEW_QUEUE : reviewer
  USERS ||--o{ AUDIT_LOGS : actor
  USERS {
    string id PK
    string email
    string hashed_password
    string api_key_hash
    bool is_active
  }
  ROLES {
    int id PK
    string name
  }
  PATIENTS {
    string id PK
    string mrn UK
    string full_name
    date date_of_birth
    datetime deleted_at
  }
  PATIENT_VISITS {
    string id PK
    string patient_id FK
  }
  PREDICTIONS {
    string id PK
    string disease
    float confidence
    string probability_scale
    string decision
    float probability_lower
    float probability_upper
  }
  REVIEW_QUEUE {
    string id PK
    string prediction_id UK
    float uncertainty_score
    float decision_threshold
    string status
    int label
    string notes
  }
  AUDIT_LOGS {
    string id PK
    string endpoint
    string method
    int status_code
    string ip_address
    int duration_ms
  }
```

## ١١. الواجهة الأمامية (React SPA)

الفورم يُبنى ديناميكياً من الـ schema القادم من الـ API (لا حقول مكتوبة يدوياً لكل مرض)، ولذلك تظهر الأمراض الجديدة بدون تعديل الواجهة.

```mermaid
flowchart TB
  MAIN["main.jsx"] --> PROV["Providers: Auth, Disease, Theme"]
  PROV --> APP["App.jsx - DiseaseSelector, theme toggle, view tabs"]
  APP --> V1["Engineering Mode"]
  APP --> V2["Clinical EMR Mode"]
  APP --> V3["Batch Prediction (CSV, max 500 rows)"]
  APP --> V4["Patient Comparison"]
  APP --> V5["Review Queue (clinical roles)"]
  APP --> V6["AdminDashboard"]
  subgraph SCH["Schema-driven form"]
    H1["useDiseaseSchema<br/>GET /api/v4/DISEASE/schema"]
    H2["schemaFieldParser + schemaToZod"]
    H3["DynamicClinicalForm + SchemaFieldFactory"]
    H4["ClinicalNotesInput (NLP prefill)"]
    H1 --> H2 --> H3
    H4 --> H3
  end
  V1 --> SCH
  V2 --> SCH
  subgraph RES["Result views"]
    R1["ThresholdBar + decision"]
    R2["ShapBarChart (useShapSnapshot)"]
    R3["WhatIfScenarioCard<br/>counterfactuals.js + /counterfactuals"]
    R4["ClinicalActionPlan"]
    R5["ClinicalReportModal + PDFReport"]
    R6["PatientRiskTimeline / PatientTimeline"]
  end
  SCH --> RES
  V5 --> RL["ReviewLabelBox<br/>annotate or skip"]
  APP --> PWA["PWA: InstallPrompt, OfflineBanner"]
  API["api.js - fetch wrapper, cookies"] --- SCH
  API --- RES
  API --- V5
  API --- V6
```

## ١٢. النشر والبنية التحتية (Deployment / CI)

نفس الصورة (image) تعمل محلياً وعلى Hugging Face Spaces وKubernetes. النماذج تُبنى أو تُحمَّل وقت بناء الصورة، فلا ملفات ثنائية في الريبو.

```mermaid
flowchart LR
  DEV["Developer"] --> GH["GitHub repo"]
  GH --> CI["CI: .github/workflows/ci.yml<br/>1. pip install, import backend<br/>2. start API check<br/>3. npm ci, npm run build"]
  GH --> DF["Dockerfile<br/>python 3.13-slim"]
  DF --> BUILD["Build time:<br/>train_heart_glm.py (CSV sha256 verified)<br/>download diabetes models from HF Hub"]
  BUILD --> IMG["Image: model baked in<br/>port 7860, instant start"]
  IMG --> HF["Hugging Face Space<br/>yahyoha/omnidiag"]
  IMG --> K8["Kubernetes<br/>k8s/*.yaml, helm chart,<br/>HPA, ingress, secrets"]
  IMG --> DC["docker-compose"]
  subgraph COMPOSE["docker-compose services"]
    PG[("postgres 15")]
    RD[("redis 7")]
    BE["backend"]
    MF["mlflow"]
    PRM["prometheus"]
    GRF["grafana"]
    RTN["retrain (profile: retrain)<br/>scripts/retrain.py"]
  end
  DC --> COMPOSE
  BE --> PG
  BE --> RD
  BE --> MF
  PRM --> BE
  GRF --> PRM
  RTN --> PG
  RTN --> MF
  FE["frontend build (vite dist)"] --> VC["Vercel<br/>SPA rewrite to index.html"]
  VC -->|"HTTPS API calls"| HF
```

## ١٣. التعلّم الاتحادي (Federated Learning): الحالة الفعلية

غير مُشغَّل في المنتج. النموذج الأولي (Flower + FedAvg + XGBoost) لا يستطيع التجميع، والبديل GLORE مُنفَّذ ومقاس في مستودع التجارب فقط.

```mermaid
flowchart LR
  subgraph OLD["backend/federated/ (SUPERSEDED, not used)"]
    O1["client.py<br/>pickle.dumps(XGBoost)"]
    O2["aggregator.py<br/>Flower FedAvg"]
    O3["add_dp_noise()<br/>never called, no epsilon"]
    O1 -->|"opaque bytes, cannot be averaged"| O2
    O2 -.-x O3
  end
  subgraph NEW["omnidiag_experiments (research only, not merged)"]
    G0["4 sites: Cleveland, Hungarian,<br/>Switzerland, Long Beach VA"]
    G1["each site computes locally<br/>gradient g and Hessian H"]
    G2["coordinator sums g, H<br/>arrow-block Newton step<br/>per-site intercept, shared slopes"]
    G3["7 iterations =<br/>centralised fit (error 2e-14)"]
    G0 --> G1 --> G2 --> G3
    G2 -->|"updated beta"| G1
  end
  NEW --> RES["Measured: LOSO AUC 0.779 vs 0.802 shipped (-0.023)<br/>no differential privacy, simulation on one machine"]
```

