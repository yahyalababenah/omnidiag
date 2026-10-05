# OmniDiag v2.0 - Development Roadmap (Task Tree)

> **Current MVP State:** Config-driven multi-disease CDSS with FastAPI backend, XGBoost + Stacking Ensemble models, SHAP explainability, DiCE counterfactual engine with Clinical Firewall, and a schema-driven React frontend. Deployed on Hugging Face Spaces + Vercel.
>
> **v2.0 Vision:** Evolve OmniDiag from a stateless demo into a production-grade, enterprise-ready clinical platform with persistent patient data, role-based access, MLOps infrastructure, and advanced AI capabilities.

---

## 🌳 Epic 1: Advanced AI & ML

### 🌿 Feature 1.1: NLP Clinical Notes Parser

- [ ] Task 1.1.1: Add `transformers`, `torch`, and `spacy` to `requirements.txt`; pin versions for reproducibility
- [ ] Task 1.1.2: Create `backend/nlp_extractor.py` implementing a `ClinicalNLPExtractor` class that wraps a BioBERT/ClinicalBERT model to extract structured feature values from free-text clinical notes
- [ ] Task 1.1.3: Add a new API endpoint `POST /api/v4/{disease}/predict-from-notes` that accepts a `{ "note": "..." }` body, runs it through `ClinicalNLPExtractor`, and routes to the existing `OmniDiagRouter.predict()`
- [ ] Task 1.1.4: Add a "Clinical Notes" textarea input panel in `ClinicalEmrMode.jsx` with a "Parse & Auto-fill" button that populates the form fields from NLP extraction results
- [ ] Task 1.1.5: Implement a confidence indicator per extracted field (high/medium/low) so the doctor can review uncertain extractions before confirming
- [ ] Task 1.1.6: Write unit tests in `tests/test_nlp_extractor.py` covering extraction accuracy on at least 5 sample clinical notes per disease

### 🌿 Feature 1.2: Time-Series Patient Risk Tracking

- [ ] Task 1.2.1: Design a time-series data schema: `PatientVisit(patient_id, timestamp, features: dict, risk_score: float, prediction: int)`
- [ ] Task 1.2.2: Create `features/time_series_features.py` implementing a `TimeSeriesFeatureEngineer` that computes temporal features: risk velocity (change per day), trend direction, volatility over the last N visits
- [ ] Task 1.2.3: Train a lightweight LSTM or Temporal Fusion Transformer model on sequences of patient visits (using existing tabular features as time steps) for each disease
- [ ] Task 1.2.4: Add `POST /api/v4/{disease}/predict-sequence` endpoint accepting a list of ordered patient visit records and returning a temporal risk trajectory
- [ ] Task 1.2.5: Create a `PatientRiskTimeline.jsx` component using Recharts `LineChart` to visualize risk score over time with color-coded risk zones (green < 30%, yellow 30–60%, red > 60%)
- [ ] Task 1.2.6: Add clinical event annotation markers on the timeline (e.g., "Started medication", "Surgery") sourced from patient history records

### 🌿 Feature 1.3: Active Learning & Human-in-the-Loop

- [ ] Task 1.3.1: Implement uncertainty sampling in `backend/active_learning.py`: compute entropy (`-p*log(p) - (1-p)*log(1-p)`) for each prediction and flag cases where confidence falls between 0.40–0.60 as "review needed"
- [ ] Task 1.3.2: Create a `review_queue` database table to persist uncertain predictions awaiting expert annotation
- [ ] Task 1.3.3: Add `GET /admin/review-queue` and `POST /admin/review-queue/{id}/label` endpoints for admins to retrieve and annotate uncertain cases
- [ ] Task 1.3.4: Build an `AnnotationQueue.jsx` panel in the Admin Dashboard showing uncertain cases with patient summaries, current model prediction, and a binary label selector (Correct / Incorrect)
- [ ] Task 1.3.5: Implement a feedback loop script `scripts/retrain_from_feedback.py` that appends annotated cases to the training set and triggers a new Optuna optimization run
- [ ] Task 1.3.6: Track annotation-driven model improvements in MLflow (see Epic 4 Feature 4.2) with before/after AUC comparisons

### 🌿 Feature 1.4: New Disease Module Expansion

- [ ] Task 1.4.1: Create `configs/stroke.yaml` following the existing YAML schema, specifying the Framingham Stroke Risk Score features, model type (`xgboost`), and target column
- [ ] Task 1.4.2: Implement `features/stroke_features.py` with `StrokeFeatureEngineer` subclassing `BaseFeatureEngineer`, computing heuristic (Age × SystolicBP), clinical (Framingham score), and medical (atrial fibrillation risk index) paths
- [ ] Task 1.4.3: Train and export an XGBoost model for stroke risk using the Framingham Heart Study dataset; save as `models/stroke/omni_diag_xgb_stroke.pkl`
- [ ] Task 1.4.4: Add `StrokeInput` Pydantic schema to `backend/schemas.py` and register it in `DISEASE_SCHEMA_REGISTRY`
- [ ] Task 1.4.5: Create `configs/ckd.yaml` and `features/ckd_features.py` for Chronic Kidney Disease using the UCI CKD dataset (eGFR, creatinine, albumin features)
- [ ] Task 1.4.6: Add 2–3 mock patients per new disease to `frontend/src/mockPatients.js` with realistic clinical profiles

### 🌿 Feature 1.5: LLM-Powered Clinical Report Generation

- [ ] Task 1.5.1: Create `backend/report_generator.py` implementing a `ClinicalReportGenerator` class that formats SHAP values, prediction confidence, and counterfactuals into a structured prompt
- [ ] Task 1.5.2: Integrate the Anthropic Claude API (or OpenAI) to generate a 3–5 paragraph narrative clinical summary from the structured prompt, explaining the diagnosis in plain medical language
- [ ] Task 1.5.3: Add `POST /api/v4/{disease}/generate-report` endpoint that accepts patient data, runs predict + explain, then calls `ClinicalReportGenerator` and returns a `{ narrative: str, key_findings: list }` response
- [ ] Task 1.5.4: Add a "Generate AI Report" button in `ClinicalEmrMode.jsx` that calls the new endpoint and displays the narrative in a scrollable modal with copy-to-clipboard functionality
- [ ] Task 1.5.5: Store generated reports in the database linked to the patient and prediction record, with timestamps for audit purposes
- [ ] Task 1.5.6: Implement a hard token limit and content filter to prevent the LLM from generating diagnostic advice beyond the model's supported disease scope

### 🌿 Feature 1.6: Federated Learning Support

- [ ] Task 1.6.1: Add `flwr` (Flower) and `opacus` (differential privacy) to `requirements.txt`
- [ ] Task 1.6.2: Create `federated/server.py` implementing a Flower server with FedAvg aggregation strategy for XGBoost gradient aggregation
- [ ] Task 1.6.3: Create `federated/client.py` implementing a Flower client that performs local training on a hospital's private dataset without sharing raw data
- [ ] Task 1.6.4: Add Gaussian differential privacy noise injection (ε=1.0) to model updates before aggregation using `opacus`
- [ ] Task 1.6.5: Document the federated setup in `docs/federated_setup.md` with a step-by-step guide for hospital IT teams to onboard a new client node
- [ ] Task 1.6.6: Create a simulation script `federated/simulate.py` that runs 3 simulated hospital clients on data splits of the existing datasets to validate the aggregation pipeline

---

## 🌳 Epic 2: Backend, Database & Auth

### 🌿 Feature 2.1: PostgreSQL Database Layer

- [ ] Task 2.1.1: Add `sqlalchemy`, `asyncpg`, `alembic`, and `psycopg2-binary` to `requirements.txt`
- [ ] Task 2.1.2: Design the full ERD with tables: `users`, `roles`, `patients`, `predictions`, `explanations`, `review_queue`, `audit_logs` — document in `docs/erd.md`
- [ ] Task 2.1.3: Create `backend/database.py` with an async SQLAlchemy engine, `AsyncSessionLocal` factory, and `get_db()` FastAPI dependency
- [ ] Task 2.1.4: Implement SQLAlchemy ORM models in `backend/models/` (one file per table: `user.py`, `patient.py`, `prediction.py`, etc.)
- [ ] Task 2.1.5: Initialize Alembic (`alembic init alembic`) and create the initial migration for all tables; ensure migrations are idempotent
- [ ] Task 2.1.6: Add a `DATABASE_URL` environment variable to `.env.example` and update `docker-compose.yml` to include a PostgreSQL 15 service with a named volume
- [ ] Task 2.1.7: Create `scripts/seed_db.py` that populates the database with demo users, 3 CAD patients, and 2 DM patients matching the existing `mockPatients.js` data

### 🌿 Feature 2.2: JWT Authentication System

- [ ] Task 2.2.1: Add `python-jose[cryptography]` and `passlib[bcrypt]` to `requirements.txt`
- [ ] Task 2.2.2: Create `backend/auth/` package with `jwt.py` (token encode/decode), `hashing.py` (bcrypt utilities), and `dependencies.py` (`get_current_user` FastAPI dependency)
- [ ] Task 2.2.3: Implement `POST /auth/register` endpoint: validate unique email, hash password with bcrypt, insert user record, return success message
- [ ] Task 2.2.4: Implement `POST /auth/login` endpoint: verify credentials, issue access token (15 min expiry) and refresh token (7 day expiry) as HttpOnly cookies
- [ ] Task 2.2.5: Implement `POST /auth/refresh` endpoint that validates the refresh token cookie and issues a new access token
- [ ] Task 2.2.6: Implement `POST /auth/logout` endpoint that clears both token cookies
- [ ] Task 2.2.7: Add `Authorization: Bearer <token>` support as an alternative to cookies for programmatic API access

### 🌿 Feature 2.3: Role-Based Access Control (RBAC)

- [ ] Task 2.3.1: Define 4 roles in a `roles` table: `super_admin`, `doctor`, `nurse`, `viewer` — with a `user_roles` join table
- [ ] Task 2.3.2: Create `backend/auth/rbac.py` with a `require_role(*roles)` FastAPI dependency factory that raises HTTP 403 if the current user lacks the required role
- [ ] Task 2.3.3: Protect `POST /api/v4/{disease}/predict` and `explain` with `require_role("doctor", "nurse", "super_admin")`
- [ ] Task 2.3.4: Protect all `/admin/*` endpoints with `require_role("super_admin")`
- [ ] Task 2.3.5: Implement an audit logging middleware in `backend/middleware/audit.py` that writes every authenticated request (user_id, endpoint, timestamp, IP) to the `audit_logs` table
- [ ] Task 2.3.6: Add `GET /admin/audit-logs` endpoint with pagination and date-range filtering for compliance review

### 🌿 Feature 2.4: Patient Records & Prediction History API

- [ ] Task 2.4.1: Implement `POST /api/v4/patients` to create a new patient record (name, DOB, MRN, gender, contact)
- [ ] Task 2.4.2: Implement `GET /api/v4/patients/{id}` and `GET /api/v4/patients` (paginated list with search by MRN/name)
- [ ] Task 2.4.3: Modify the existing predict and explain endpoints to optionally accept a `patient_id` query parameter and persist the result to the `predictions` table
- [ ] Task 2.4.4: Implement `GET /api/v4/patients/{id}/predictions` returning paginated prediction history sorted by timestamp descending
- [ ] Task 2.4.5: Implement soft-delete (`DELETE /api/v4/patients/{id}`) that sets a `deleted_at` timestamp rather than physically removing the record, for GDPR compliance
- [ ] Task 2.4.6: Add `GET /api/v4/patients/{id}/export` endpoint that returns all patient data and predictions as a JSON bundle for data portability

### 🌿 Feature 2.5: Redis Caching Layer

- [ ] Task 2.5.1: Add `redis` and `fastapi-cache2` to `requirements.txt`; add a Redis 7 service to `docker-compose.yml`
- [ ] Task 2.5.2: Cache `GET /api/v4/{disease}/schema` responses in Redis with a 24-hour TTL since schemas change only on deployment
- [ ] Task 2.5.3: Cache `POST /api/v4/{disease}/predict` responses keyed by `hash(disease + sorted(patient_data.items()))` with a 5-minute TTL to avoid redundant model inference for identical inputs
- [ ] Task 2.5.4: Add a `Cache-Hit: true/false` response header so the frontend and monitoring tools can distinguish cached from computed responses
- [ ] Task 2.5.5: Implement a `POST /admin/cache/flush` endpoint (super_admin only) to invalidate all cached responses after a model update

### 🌿 Feature 2.6: API Gateway & Rate Limiting

- [ ] Task 2.6.1: Add `slowapi` to `requirements.txt` and configure a `Limiter` instance in `backend/main.py`
- [ ] Task 2.6.2: Apply rate limits per user: 30 req/min for `doctor`/`nurse`, 5 req/min for `viewer`, unlimited for `super_admin`
- [ ] Task 2.6.3: Implement API key authentication (`X-API-Key` header) as an alternative auth method for programmatic integrations; store hashed API keys in the `users` table
- [ ] Task 2.6.4: Add a `POST /admin/api-keys` endpoint for super admins to issue API keys with configurable scopes and expiry dates
- [ ] Task 2.6.5: Return standardized error envelopes `{ "error": str, "code": str, "request_id": uuid }` for all 4xx/5xx responses by adding a global exception handler in `main.py`

---

## 🌳 Epic 3: Frontend & UI/UX

### 🌿 Feature 3.1: Progressive Web App (PWA)

- [ ] Task 3.1.1: Install `vite-plugin-pwa` and configure it in `vite.config.js` with a `manifest.json` (app name, icons, theme color `#0f172a`)
- [ ] Task 3.1.2: Implement a service worker that pre-caches the app shell (HTML, CSS, JS bundles) and caches the last 5 API responses for offline viewing
- [ ] Task 3.1.3: Add an `OfflineBanner.jsx` component that displays a dismissible banner when the app detects no network connectivity
- [ ] Task 3.1.4: Redesign `ClinicalEmrMode.jsx` layout with a responsive 1-column stack on mobile (< 768px) and 2-column grid on tablet (768–1024px)
- [ ] Task 3.1.5: Add an "Install App" button in the header that triggers the PWA install prompt on supported browsers
- [ ] Task 3.1.6: Test the PWA on iOS Safari and Android Chrome; document any platform-specific limitations in `docs/pwa_notes.md`

### 🌿 Feature 3.2: PDF Clinical Report Export

- [ ] Task 3.2.1: Install `@react-pdf/renderer` and create a `PDFReport.jsx` component defining the PDF layout with OmniDiag header, patient info section, diagnosis summary, and footer
- [ ] Task 3.2.2: Convert the `ShapBarChart.jsx` Recharts component to an SVG snapshot using `html2canvas` and embed it in the PDF as an image
- [ ] Task 3.2.3: Render up to 3 counterfactual scenarios in the PDF as a formatted "Recommended Lifestyle Interventions" table with feasibility badges
- [ ] Task 3.2.4: Add doctor name, clinic name, date, and a signature placeholder line at the bottom of the PDF
- [ ] Task 3.2.5: Add an "Export PDF" button in `ClinicalEmrMode.jsx` that triggers client-side PDF generation using `@react-pdf/renderer`'s `PDFDownloadLink`; filename format: `OmniDiag_{patient_name}_{date}.pdf`
- [ ] Task 3.2.6: Add a "Print" button that opens a print-optimized CSS media query view of the report without needing the PDF library

### 🌿 Feature 3.3: Admin & Analytics Dashboard

- [ ] Task 3.3.1: Create `frontend/src/components/AdminDashboard.jsx` as a new top-level route accessible only to `super_admin` role
- [ ] Task 3.3.2: Add 4 KPI summary cards at the top: Total Predictions Today, Positive Rate (%), Average Confidence Score, Active Users
- [ ] Task 3.3.3: Implement a Recharts `BarChart` showing prediction volume per disease per day for the last 30 days, fetched from `GET /admin/stats/daily`
- [ ] Task 3.3.4: Implement a Recharts `LineChart` showing model accuracy trend over time (weekly aggregates), fetched from `GET /admin/stats/model-performance`
- [ ] Task 3.3.5: Add a User Management table with columns: Name, Email, Role, Last Login, Status — with actions to deactivate or change role
- [ ] Task 3.3.6: Add a heatmap (using a grid of colored `div`s) showing prediction request volume by hour-of-day vs day-of-week for capacity planning

### 🌿 Feature 3.4: Patient History Timeline

- [ ] Task 3.4.1: Create `frontend/src/components/PatientTimeline.jsx` rendering a vertical timeline of all prediction records for a selected patient
- [ ] Task 3.4.2: Implement a Recharts `LineChart` at the top of the timeline showing risk score (0–100%) across all visits with color-coded risk zone bands
- [ ] Task 3.4.3: For each visit node on the timeline, show: date, disease, prediction result badge, top 3 SHAP features, and a "View Details" expandable row
- [ ] Task 3.4.4: Add a "Trend Annotation" feature where doctors can attach free-text notes to specific timeline points (e.g., "Started metformin") stored in the database
- [ ] Task 3.4.5: Add a feature delta view: clicking between two visits shows which features changed and by how much, with arrows indicating direction of change
- [ ] Task 3.4.6: Integrate the timeline into `ClinicalEmrMode.jsx` as a collapsible "Patient History" drawer on the right side

### 🌿 Feature 3.5: Batch Prediction Interface

- [ ] Task 3.5.1: Create `backend/batch_processor.py` with a `BatchPredictor` class that accepts a list of patient dicts, validates each against the disease Pydantic schema, and returns a list of prediction results
- [ ] Task 3.5.2: Add `POST /api/v4/{disease}/batch-predict` endpoint with a request body of `{ "patients": [...] }` (max 500 records); run predictions concurrently using `asyncio.gather`
- [ ] Task 3.5.3: Create `frontend/src/components/BatchPrediction.jsx` with a drag-and-drop CSV upload area using the HTML5 File API
- [ ] Task 3.5.4: Parse the uploaded CSV on the frontend using `papaparse`, validate column headers against the disease schema, and show a preview table of the first 5 rows before submission
- [ ] Task 3.5.5: After batch prediction completes, display a results summary (total processed, positive count, negative count, error count) and a paginated results table
- [ ] Task 3.5.6: Add a "Download Results" button that exports the results table as a CSV with original patient columns plus appended `prediction`, `confidence`, and `top_risk_factor` columns

### 🌿 Feature 3.6: Before/After Comparison Mode

- [ ] Task 3.6.1: Add a "Compare Mode" toggle button in `EngineeringMode.jsx` that splits the layout into two equal panels (Scenario A / Scenario B)
- [ ] Task 3.6.2: Each panel contains an independent copy of `DynamicClinicalForm` with its own form state; initialize Panel B as a deep copy of Panel A's current values
- [ ] Task 3.6.3: Add a "Sync & Diverge" button that copies Panel A's values to Panel B, allowing the user to modify only the features they want to test
- [ ] Task 3.6.4: Display a delta summary below both panels: list changed features with their old → new values, and show risk score difference (e.g., "Risk decreased by 18.3%") with a color-coded arrow
- [ ] Task 3.6.5: Render a side-by-side SHAP bar chart comparison using two `ShapBarChart` instances with a shared Y-axis so feature names align for easy visual comparison

### 🌿 Feature 3.7: Dark Mode & WCAG Accessibility

- [ ] Task 3.7.1: Enable Tailwind's `darkMode: 'class'` strategy in `tailwind.config.js` and add a dark mode toggle button to the app header with state persisted in `localStorage`
- [ ] Task 3.7.2: Audit all components for WCAG AA color contrast (minimum 4.5:1 ratio) and update Tailwind color tokens where violations exist
- [ ] Task 3.7.3: Add `aria-label`, `role`, and `aria-describedby` attributes to all interactive elements (buttons, sliders, toggles, dropdowns) across the codebase
- [ ] Task 3.7.4: Implement full keyboard navigation: Tab order must follow logical form flow, all sliders must respond to arrow keys, and all modals must trap focus
- [ ] Task 3.7.5: Add a skip-to-content link (`<a href="#main-content">`) as the first focusable element for screen reader users
- [ ] Task 3.7.6: Run an automated accessibility audit using `axe-core` in the CI pipeline and fail the build if any critical violations are detected

---

## 🌳 Epic 4: DevOps & MLOps

### 🌿 Feature 4.1: Model Drift Monitoring

- [ ] Task 4.1.1: Add `evidently` to `requirements.txt` and create `mlops/drift_monitor.py` with a `DriftMonitor` class
- [ ] Task 4.1.2: Save a reference dataset snapshot (training data distribution statistics) for each disease as `mlops/reference/{disease}_reference.parquet` during the model training step
- [ ] Task 4.1.3: Implement a `DriftMonitor.check_drift(current_batch)` method that computes data drift (Jensen-Shannon divergence per feature) and prediction drift (PSI on output probabilities) against the reference
- [ ] Task 4.1.4: Add `GET /admin/drift/{disease}` endpoint that runs a drift check on the last 500 predictions stored in the database and returns a drift report with per-feature drift scores
- [ ] Task 4.1.5: Add a Drift Report card to the Admin Dashboard (`AdminDashboard.jsx`) showing a traffic-light indicator (green/yellow/red) per feature and a "Last Checked" timestamp
- [ ] Task 4.1.6: Configure a weekly cron job (GitHub Actions scheduled workflow) that runs `mlops/drift_monitor.py` and opens a GitHub Issue if drift score exceeds threshold 0.2

### 🌿 Feature 4.2: MLflow Experiment Tracking & Model Registry

- [ ] Task 4.2.1: Add `mlflow` to `requirements.txt`; add an MLflow Tracking Server service to `docker-compose.yml` with a PostgreSQL backend store and a local artifact store
- [ ] Task 4.2.2: Instrument existing training scripts in `experiment_files/` with `mlflow.start_run()`, logging all Optuna trial hyperparameters, CV scores, and final test metrics
- [ ] Task 4.2.3: Log trained model artifacts with `mlflow.xgboost.log_model()` and `mlflow.sklearn.log_model()` to the model registry after each training run
- [ ] Task 4.2.4: Tag the best-performing model version as `Production` in the MLflow registry; update `OmniDiagRouter` to optionally load models from the registry instead of local `.pkl` files
- [ ] Task 4.2.5: Create a `scripts/promote_model.py` CLI script that compares challenger vs champion model AUC on a held-out validation set and promotes the challenger only if improvement exceeds 0.5%
- [ ] Task 4.2.6: Add an MLflow UI link to the Admin Dashboard for experiment history browsing

### 🌿 Feature 4.3: Automated Retraining Pipeline

- [ ] Task 4.3.1: Create `scripts/retrain.py` as a self-contained training script that accepts `--disease`, `--data-path`, and `--n-trials` CLI arguments and outputs a new model artifact
- [ ] Task 4.3.2: Add a GitHub Actions workflow `.github/workflows/retrain.yml` that triggers manually (workflow_dispatch) or on a schedule (monthly), runs `retrain.py`, and pushes the new model to the MLflow registry
- [ ] Task 4.3.3: Integrate the drift detection check (Feature 4.1) as a gate in the retraining workflow: only retrain if drift score > 0.15 or if manually triggered
- [ ] Task 4.3.4: After retraining, automatically run the full test suite and evaluate the new model on the held-out test set; write results to a `retraining_report.md` artifact attached to the GitHub Actions run
- [ ] Task 4.3.5: Implement a Slack/email notification step at the end of the retraining workflow that posts the new model's performance metrics and a link to the MLflow run
- [ ] Task 4.3.6: Add a rollback mechanism: `scripts/rollback_model.py` that promotes the previous `Production` version back in the MLflow registry in case the new model underperforms

### 🌿 Feature 4.4: Docker Compose Full-Stack Orchestration

- [ ] Task 4.4.1: Create `docker-compose.yml` with 6 services: `backend` (FastAPI), `frontend` (Nginx serving Vite build), `postgres` (PostgreSQL 15), `redis` (Redis 7), `mlflow` (MLflow server), `nginx` (reverse proxy)
- [ ] Task 4.4.2: Configure Nginx as a reverse proxy routing `/api/*` to the backend service and `/*` to the frontend service; add SSL termination config for production
- [ ] Task 4.4.3: Add health checks for each service: `pg_isready` for PostgreSQL, `redis-cli ping` for Redis, and `curl /health` for the FastAPI backend
- [ ] Task 4.4.4: Create `docker-compose.override.yml` for local development with volume mounts for hot-reloading backend code and Vite dev server instead of Nginx
- [ ] Task 4.4.5: Create `.env.example` documenting all required environment variables: `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `MLFLOW_TRACKING_URI`, `HF_TOKEN`, `ANTHROPIC_API_KEY`
- [ ] Task 4.4.6: Write `docs/local_setup.md` with a single-command setup guide: `cp .env.example .env && docker-compose up --build`

### 🌿 Feature 4.5: Kubernetes Production Deployment

- [ ] Task 4.5.1: Create `k8s/` directory with Kubernetes manifests: `namespace.yaml`, `backend-deployment.yaml`, `frontend-deployment.yaml`, `postgres-statefulset.yaml`, `redis-deployment.yaml`
- [ ] Task 4.5.2: Configure a `HorizontalPodAutoscaler` for the backend deployment: min 2 pods, max 10 pods, scale up when CPU > 70% or memory > 80%
- [ ] Task 4.5.3: Create a `PersistentVolumeClaim` for PostgreSQL data storage and configure the StatefulSet to use it, ensuring data survives pod restarts
- [ ] Task 4.5.4: Create a Helm chart in `helm/omnidiag/` with a `values.yaml` file exposing configurable replicas, image tags, resource limits, and environment variables
- [ ] Task 4.5.5: Add `readinessProbe` (returns 200 on `GET /health`) and `livenessProbe` (returns 200 on `GET /`) to the backend container spec to enable zero-downtime rolling deployments
- [ ] Task 4.5.6: Create `k8s/ingress.yaml` using NGINX Ingress Controller with TLS termination via cert-manager (Let's Encrypt) for the production domain

### 🌿 Feature 4.6: Observability Stack

- [ ] Task 4.6.1: Add `prometheus-fastapi-instrumentator` to `requirements.txt` and register it in `backend/main.py` to expose `/metrics` with prediction latency histograms and request counters
- [ ] Task 4.6.2: Add Prometheus and Grafana services to `docker-compose.yml`; configure Prometheus to scrape the backend `/metrics` endpoint every 15 seconds
- [ ] Task 4.6.3: Create a Grafana dashboard JSON (`monitoring/grafana_dashboard.json`) with panels: prediction latency (p50/p95/p99), requests per second by disease, error rate, active users
- [ ] Task 4.6.4: Implement structured JSON logging throughout the backend using Python's `logging` module with a custom formatter outputting `{ timestamp, level, request_id, user_id, endpoint, duration_ms, message }`
- [ ] Task 4.6.5: Add OpenTelemetry tracing with `opentelemetry-instrumentation-fastapi`; export traces to a Jaeger instance in `docker-compose.yml` for end-to-end request tracing across predict → feature engineer → model inference
- [ ] Task 4.6.6: Configure Prometheus alerting rules for SLA violations: alert if p95 latency > 2s, error rate > 1%, or prediction throughput drops > 50% vs the 1-hour average

### 🌿 Feature 4.7: Security Hardening & HIPAA Compliance

- [ ] Task 4.7.1: Add `secure` middleware to FastAPI (`backend/middleware/security.py`) that injects OWASP security headers: `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`
- [ ] Task 4.7.2: Enable Dependabot in `.github/dependabot.yml` for both Python (`pip`) and JavaScript (`npm`) dependency vulnerability scanning with weekly checks
- [ ] Task 4.7.3: Run `bandit` (Python SAST) and `semgrep` as steps in the CI pipeline; fail the build on any HIGH severity findings
- [ ] Task 4.7.4: Implement at-rest encryption for the `predictions` and `patients` PostgreSQL tables using PostgreSQL's `pgcrypto` extension for PII fields (name, DOB, MRN)
- [ ] Task 4.7.5: Create `docs/hipaa_compliance_checklist.md` documenting how OmniDiag addresses each HIPAA Technical Safeguard: access controls (RBAC), audit controls (audit_logs table), integrity controls (predictions are immutable), transmission security (TLS)
- [ ] Task 4.7.6: Implement automated PII scrubbing in `backend/middleware/pii_scrubber.py` that redacts patient names and identifiers from all log outputs before they are written to disk

---

## 📊 Priority Matrix

| Feature | Impact | Effort | Recommended Phase |
|---|---|---|---|
| 2.1 PostgreSQL Database | ⭐⭐⭐⭐⭐ | High | Phase 1 (Foundation) |
| 2.2 JWT Auth | ⭐⭐⭐⭐⭐ | Medium | Phase 1 (Foundation) |
| 2.3 RBAC | ⭐⭐⭐⭐ | Medium | Phase 1 (Foundation) |
| 4.4 Docker Compose | ⭐⭐⭐⭐⭐ | Low | Phase 1 (Foundation) |
| 1.4 New Diseases (Stroke) | ⭐⭐⭐⭐⭐ | Medium | Phase 2 (Growth) |
| 3.2 PDF Export | ⭐⭐⭐⭐ | Low | Phase 2 (Growth) |
| 3.5 Batch Prediction | ⭐⭐⭐⭐ | Medium | Phase 2 (Growth) |
| 3.6 Before/After Compare | ⭐⭐⭐⭐ | Low | Phase 2 (Growth) |
| 4.1 Drift Monitoring | ⭐⭐⭐⭐⭐ | Medium | Phase 2 (Growth) |
| 4.2 MLflow Registry | ⭐⭐⭐⭐ | Medium | Phase 2 (Growth) |
| 1.1 NLP Notes Parser | ⭐⭐⭐⭐ | High | Phase 3 (Advanced) |
| 1.2 Time-Series LSTM | ⭐⭐⭐⭐ | High | Phase 3 (Advanced) |
| 1.5 LLM Reports | ⭐⭐⭐⭐ | Medium | Phase 3 (Advanced) |
| 3.1 PWA Mobile | ⭐⭐⭐ | Medium | Phase 3 (Advanced) |
| 4.5 Kubernetes | ⭐⭐⭐⭐ | High | Phase 4 (Enterprise) |
| 1.6 Federated Learning | ⭐⭐⭐ | Very High | Phase 4 (Enterprise) |
| 4.7 HIPAA Compliance | ⭐⭐⭐⭐⭐ | High | Phase 4 (Enterprise) |

---

*Generated for OmniDiag v2.0 planning. Total: 4 Epics · 27 Features · ~165 Tasks.*
