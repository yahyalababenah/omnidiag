# Code Cleanup Report

**Branch:** `feature/omni-platform-final`  
**Date:** 2026-06-28  
**Scope:** Full codebase — frontend (React/JSX) + backend (Python/FastAPI)

---

## Summary

Two analysis agents scanned the frontend and backend in parallel. All confirmed findings were then applied. A total of **19 files** were modified across both layers.

---

## Frontend Changes

### 1. Extracted shared `ValueBadge` component and `SLIDER_CLASS` constant

**File:** `frontend/src/components/SchemaFieldFactory.jsx`

The live-value badge (blue pill showing the current slider value) and the range `<input type="range">` Tailwind class string were each copy-pasted verbatim between `SliderField` and `NumberField`.

**Fix:** Extracted a local `ValueBadge({ value })` component and a `SLIDER_CLASS` string constant at the top of the file. Both field types now reference them.

---

### 2. Merged duplicate patient-reset `useEffect`s

**File:** `frontend/src/components/ClinicalEmrMode.jsx`

Two `useEffect`s both fired on `selectedDisease` change:
- The first reset result/SHAP/error state and set `selectedPatient` from `patients.length`.
- The second also called `getPatientsForDisease` and set `selectedPatient`, silently overwriting the first.

**Fix:** Merged into one `useEffect` on `[selectedDisease]` that calls `getPatientsForDisease`, sets the patient, and clears result state.

---

### 3. Removed redundant schema cache from `DiseaseContext`

**File:** `frontend/src/context/DiseaseContext.jsx`

The context maintained its own `schemaCache` state, a `preFetchSchemas` function (called eagerly on app load for all diseases), and a `getSchemaForDisease` method. None of these were consumed anywhere outside the context itself — `useDiseaseSchema.js` already handles schema fetching and caching independently via its own module-level `Map`.

**Fix:** Removed `schemaCache` state, `preFetchSchemas`, and `getSchemaForDisease` (~40 lines). The context now only manages disease selection and the available diseases list.

---

### 4. Moved `@keyframes` animations out of JSX

**Files:** `frontend/src/components/FormSkeleton.jsx`, `frontend/src/components/VariableScalesModal.jsx`  
**Target:** `frontend/src/index.css`

Both components injected `<style>` tags with `@keyframes` blocks directly into the render output, re-injecting them into the document on every mount.

**Fix:** Moved `shimmer` and `slideInRight` animations to `index.css`. Components reference the class names only.

---

### 5. Removed dead exports

All of the following were exported but never imported anywhere in the codebase:

| File | Removed export |
|---|---|
| `components/ErrorBoundary.jsx` | `PredictionErrorBoundary` |
| `utils/schemaToZod.js` | `validateFormData` |
| `utils/medicalDictionary.js` | `isKnownMedicalTerm`, `getFullDictionary` |
| `mockPatients.js` | `getAvailablePatientDiseases` |
| `hooks/useDiseaseSchema.js` | `clearSchemaCache` |
| `components/SchemaFieldFactory.jsx` | `componentIcons` (defined, never referenced) |

---

## Backend Changes

### 6. Removed unused imports

| File | Removed |
|---|---|
| `backend/model_loader.py` | `import sys` |
| `backend/ensemble_loader.py` | `import sys`, `Tuple` and `Callable` from `typing` |
| `backend/ensemble_loader.py` | `import joblib as _joblib` inside a `for` loop body (top-level `joblib` already imported) |

---

### 7. Promoted deferred imports and loggers to module level

**File:** `backend/main.py`

Two route handlers (`explain_disease`, `counterfactuals_disease`) each executed `import traceback as _tb` and `_log = logging.getLogger(...)` on **every request**, deferring what should be module-level setup.

**Fix:**
- Added `import traceback` at the top of the file.
- Added `_explain_log` and `_cf_log` as module-level loggers alongside the existing `log`.
- Removed the per-request imports and local `_log` assignments from both handlers.

---

### 8. Extracted `_validate_patient_input` helper

**File:** `backend/main.py`

The schema-validation block appeared identically in all three clinical route handlers (`predict_disease`, `explain_disease`, `counterfactuals_disease`):

```python
schema = get_schema_for_disease(disease)
if schema:
    validated = schema(**patient)
    patient_data = validated.model_dump()
else:
    patient_data = patient
```

**Fix:** Extracted into `_validate_patient_input(disease, patient) -> dict` placed above the route definitions. All three handlers now call it with one line.

---

### 9. Extracted `extract_token_from_request` utility

**File:** `backend/auth/jwt.py` (new function)  
**Callers updated:** `backend/rate_limit.py`, `backend/middleware/audit.py`

The cookie-then-header token extraction pattern was duplicated in both files (and had diverged slightly — different variable names, one used `str | None`, the other `Optional[str]`):

```python
token = request.cookies.get("access_token")
if not token:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
```

**Fix:** Added `extract_token_from_request(request: Request) -> Optional[str]` to `auth/jwt.py`. Both callers import and use it.

---

### 10. Replaced `print()` with structured logging in `router.py`

**File:** `backend/router.py`

`_load_all_configs()` and `_create_loader()` used raw `print()` with emoji characters for status messages while the rest of the backend uses structured `logging`. This meant disease registration events were invisible to the configured log handler.

**Fix:** Added `log = logging.getLogger("omnidiag.router")` and replaced all 7 `print()` calls with `log.info()`, `log.warning()`, and `log.error()`.

---

### 11. Upgraded Pydantic v1 `class Config` to v2 `model_config`

**File:** `backend/schemas.py`

`HeartDiseaseInput` and `DiabetesInput` both used the Pydantic v1 inner-class style:

```python
class Config:
    json_schema_extra = { ... }
```

The rest of the backend (auth schemas, patient schemas) correctly uses Pydantic v2 style. Added `ConfigDict` to the import and replaced both with:

```python
model_config = ConfigDict(json_schema_extra={ ... })
```

---

## What Was Not Changed

The following duplication exists in `backend/model_loader.py` and `backend/ensemble_loader.py` and was identified but intentionally left alone:

| Duplication | Reason deferred |
|---|---|
| `_apply_preprocessors` — identical in both loaders | Extracting a `BaseModelLoader` carries regression risk without tests |
| `_get_feature_engineer` / `_engineer_features` — copy-pasted | Same as above |
| `preprocessors` property directory-scan loop | Same as above |
| XGBoost `base_score` monkey-patch | The two implementations have already diverged in detail; merging requires careful testing |

These should be addressed as a dedicated refactor with integration tests in place.
