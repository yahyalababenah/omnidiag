# Claude Code Prompt — OmniDiag Test Suite Implementation

---

## Context

You are working on **OmniDiag**, a multi-disease clinical prediction API built with:
- **FastAPI** (async) + **SQLAlchemy 2.0 AsyncSession** + SQLite (test) / PostgreSQL (prod)
- **Auth**: JWT (access + refresh tokens), bcrypt, HttpOnly cookies
- **RBAC**: 4 roles — `super_admin`, `doctor`, `nurse`, `viewer`
- **ML**: XGBoost + LightGBM + Random Forest ensemble, SHAP explanations, DiCE counterfactuals
- **Active Learning**: entropy-based uncertainty sampling → `review_queue` table
- **Monitoring**: Evidently drift detection, MLflow tracking
- **LLM**: Anthropic Claude for clinical report generation (with rule-based fallback)
- **Caching**: Redis (prod) / InMemory (test) via `fastapi-cache`
- **Middleware**: AuditMiddleware (logs every request to `audit_logs` table), SecurityHeadersMiddleware, Rate limiting

The existing test infrastructure is already set up in `tests/conftest.py` with:
- SQLite in-memory database (no PostgreSQL needed)
- Mocked `OmniDiagRouter` (no ML models loaded)
- Module-scoped fixtures: `seeded_db`, `admin_token`, `doctor_token`, `viewer_token`
- Function-scoped fixtures: `client`, `db_session`
- pytest-asyncio with `asyncio_mode = auto`

**Read `tests/conftest.py` fully before writing any test** to understand available fixtures.

---

## Your Task

Implement the missing tests described in `plans/testing_checklist.md`. Create the files listed below. Do NOT modify any existing test files — only create new ones or append to `test_clinical.py` and `test_patients.py` where indicated.

---

## Files to Create

### 1. `tests/test_unit_auth.py`

Test `backend/auth/hashing.py` and `backend/auth/jwt.py` as pure unit tests — no HTTP client needed.

Cover all cases in the checklist section **"Unit Tests — test_unit_auth.py"** (U-1 through U-10).

Key points:
- Import and call `hash_password`, `verify_password`, `create_access_token`, `create_refresh_token`, `decode_token` directly.
- For expired token test: pass `expires_delta=timedelta(seconds=-1)` to `create_access_token`.
- For token type confusion test: create a refresh token then call `decode_token(token, expected_type="access")` and assert it raises `HTTPException` with status 401.
- `extract_token_from_request` requires a `starlette.requests.Request` mock — use `httpx.Request` or a simple `MagicMock`.

---

### 2. `tests/test_unit_active_learning.py`

Test `backend/active_learning/sampler.py` as pure unit tests.

Cover cases U-11 through U-18.

Key points:
- `prediction_entropy(0.5)` should return exactly `1.0` (use `pytest.approx`).
- `prediction_entropy(0.0)` and `prediction_entropy(1.0)` should return `0.0` (the function clamps to avoid log(0)).
- Use parametrize for `uncertainty_band` tests across the 4 bands.

---

### 3. `tests/test_unit_counterfactuals.py`

Test `backend/counterfactual_generator.py`.

Cover cases U-19 through U-23.

Key points:
- The generator needs a callable `model_fn` and a callable `engineer_fn`. Use simple lambdas that return `{"probability": 0.1}` (low risk = flipped prediction).
- To get counterfactuals that flip, set `model_fn` to always return `{"probability": 0.1}` so filtered results are non-empty.
- Verify IMMUTABLE_FEATURES don't change by comparing original input against each counterfactual's changed keys.
- Verify BMI bounds by iterating over returned counterfactuals.
- Import `IMMUTABLE_FEATURES`, `CLINICAL_BOUNDS`, `BINARY_FEATURES` from `backend.counterfactual_generator`.

---

### 4. `tests/test_audit_log.py`

Test that `AuditMiddleware` records requests correctly.

Cover cases I-7 through I-11. Uses `client`, `doctor_token`, `db_session` fixtures.

Key points:
- After an authenticated POST to `/api/v4/heart_disease/predict`, query `audit_logs` via `db_session` and assert a row exists with matching `endpoint` and `status_code`.
- For unauthenticated requests: query `audit_logs` and check `user_id IS NULL` (the middleware should still log the request but with no user).
- `duration_ms`: assert `> 0`.
- Import `AuditLog` from `backend.db_models.audit_log`.

---

### 5. `tests/test_security.py`

Test security properties: token attacks, rate limiting, headers.

Cover cases S-1 through S-8. Uses `client`, `doctor_token`, `seeded_db` fixtures.

Key points:
- **Expired token** (S-1): create one with `timedelta(seconds=-1)`, use it in a request, expect 401.
- **Refresh as access** (S-2): log in to get the `refresh_token` cookie, then use it as a Bearer token in the Authorization header for `/api/v4/heart_disease/predict`, expect 401.
- **Tampered JWT** (S-3): take `doctor_token`, split by `.`, replace payload with a base64-encoded `{"sub":"attacker","type":"access"}`, expect 401.
- **Rate limiting** (S-4): send 7 POST requests to `/auth/login` rapidly in a loop, assert at least one returns 429. Mark with `@pytest.mark.slow`.
- **Security headers** (S-5, S-6): do a GET `/api/v4/diseases`, check response headers contain `x-content-type-options`, `x-frame-options`.
- **SQL injection** (S-7): GET `/api/v4/patients/?search=' OR 1=1 --`, assert 200 (not 500), assert returns normal paginated structure.

---

### 6. `tests/test_database.py`

Test database-level constraints and behaviors.

Cover cases D-1 through D-5. Uses `db_session` fixture directly (no HTTP client).

Key points:
- **D-1 (unique MRN)**: add two `Patient` objects with the same `mrn` via `db_session`, call `await session.commit()`, expect `sqlalchemy.exc.IntegrityError`.
- **D-2 (soft delete)**: create a patient, set `patient.deleted_at = datetime.now(UTC)`, commit, then query — the row still exists in DB but `deleted_at` is not null.
- **D-3 (FK behavior)**: depends on whether you have ON DELETE CASCADE or SET NULL — test what's actually defined in the migration `alembic/versions/da87946a56a6_initial_schema.py`.
- **D-4 (JSON storage)**: create a `Prediction` with `input_features={"Age": 55, "Sex": "M"}`, commit, refresh, assert `prediction.input_features["Age"] == 55`.
- **D-5**: insert an `AuditLog` with `user_id=None`, commit, refresh — assert it persists.
- Import models from `backend.db_models.*`.

---

### 7. `tests/test_llm_report.py`

Test `backend/llm/report_generator.py` with a mocked Anthropic client.

Cover cases L-1 through L-4.

Key points:
- **L-1 (fallback)**: patch `backend.llm.report_generator._ANTHROPIC_API_KEY` to `""` and call the report generator — it should return a non-empty string without raising.
- **L-2 (fallback content)**: assert the returned string contains the disease name and probability.
- **L-3 (mocked API)**: use `unittest.mock.patch` on the Anthropic client class to return a mock response with `.content[0].text = "Test report"`, assert the generator returns that text.
- **L-4 (shap_summary)**: call the function that builds the SHAP summary string and assert it formats feature names and values correctly.
- Read `backend/llm/report_generator.py` first to understand the function signatures.

---

### 8. Additions to `tests/test_clinical.py`

Append a new class `TestDiabetes` and extend `TestPredict`:

```
class TestDiabetes:
    - test_diabetes_predict_returns_result: POST /api/v4/diabetes/predict with a valid diabetes payload
    - test_diabetes_explain_returns_chart_data
    - test_predict_missing_field_returns_422: omit a required field, expect 422
    - test_confidence_always_in_range: assert 0 <= data["confidence"] <= 1
    - test_prediction_is_binary: assert data["prediction"] in (0, 1)
```

The mock router already supports `diabetes` — check `conftest.py` `_make_mock_router()`.

For the diabetes payload, use fields from `omnidiag_diabetes_artifacts/configs/diabetes.yaml` or use a minimal dict — the router is mocked so any dict is accepted.

---

### 9. Additions to `tests/test_patients.py`

Append:

```
class TestSoftDelete:
    - test_delete_patient_sets_deleted_at: DELETE /api/v4/patients/{id} → 204, then GET → 404
    - test_deleted_patient_excluded_from_list: after delete, doesn't appear in list
    - test_viewer_cannot_create_patient: viewer_token → 403

class TestPredictionHistory:
    - test_patient_prediction_history: predict with patient_id, then GET /api/v4/patients/{id}/predictions → list contains the prediction
```

---

## Constraints & Rules

1. **Do not modify** `conftest.py`, `pytest.ini`, or existing test files — only append new classes if explicitly said above.
2. **All tests must be async** (`async def test_*`) since the app is async.
3. **Use only these imports** at the top: `pytest`, `pytest_asyncio` (if needed), `unittest.mock`, standard library. Everything else is already wired through fixtures.
4. **No ML models are loaded** — the mock router handles all predictions. Do not attempt to import or instantiate `OmniDiagRouter` in tests.
5. **Mark slow tests** with `@pytest.mark.slow` (rate limiting, actual model loading).
6. **Run `pytest tests/ -v` after each file** to confirm it passes before moving to the next.
7. **Coverage target**: after all files are done, run `pytest tests/ --cov=backend --cov-report=term-missing` and confirm overall coverage is ≥ 75%.

---

## Acceptance Criteria

- [ ] All 7 new test files exist and collect without errors
- [ ] `pytest tests/ -v -m "not slow"` passes with 0 failures
- [ ] No existing test is broken
- [ ] `pytest tests/ --cov=backend --cov-report=term-missing` shows ≥ 75% total coverage
- [ ] `backend/auth/hashing.py` coverage = 100%
- [ ] `backend/auth/jwt.py` coverage ≥ 90%
- [ ] `backend/active_learning/sampler.py` coverage = 100%
