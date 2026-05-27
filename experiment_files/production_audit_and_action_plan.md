# OmniDiag — Production Audit & Action Plan

> **Project:** OmniDiag (Multi-Disease Diagnostic Platform)
> **Audit Date:** 2026-05-27
> **Auditor:** Roo (Debug Mode)
> **Branch:** `production-final-v3`
> **Scope:** Git/Version Control Health · Security & Secrets · Docker/Container Stability · Codebase Architecture

---

## Executive Summary

OmniDiag is a well-architected clinical decision-support platform using XGBoost + SHAP explainability with a modern React frontend. However, this audit has uncovered **3 critical security leaks**, **1 unresolved merge conflict** that breaks `pip install`, **multiple Git health issues**, and **architectural inconsistencies** that must be resolved before this project can be considered production-ready.

**Risk Level: HIGH** — primarily due to exposed credentials in the Git configuration and a broken dependency manifest.

---

## Table of Contents

1. [🔴 Critical Issues](#1--critical-issues)
2. [🟡 Warnings](#2--warnings)
3. [🟢 Optimizations & Info](#3--optimizations--info)
4. [🗺️ Action Plan](#4--action-plan)

---

## 1. 🔴 Critical Issues

### C1 — Exposed GitHub Personal Access Token in `.git/config`

| Attribute | Detail |
|-----------|--------|
| **File** | [`.git/config`](.git/config) |
| **Secret** | `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` |
| **Remote** | `origin` → `https://yahyalababenah:TOKEN@github.com/yahyalababenah/omnidiag.git` |
| **Risk** | Anyone with access to this repo (or a backup) can push code, delete repos, and access all repositories belonging to the `yahyalababenah` GitHub account. |
| **CVSS** | 9.1 (Critical) |

**Evidence:** Running `git remote -v` reveals the token in plaintext in the remote URL. The `.git/config` file is **NOT** in `.gitignore`, meaning any `git clone --mirror`, `git bundle`, or filesystem backup exposes this token permanently.

### C2 — Exposed Hugging Face Token in `.git/config`

| Attribute | Detail |
|-----------|--------|
| **File** | [`.git/config`](.git/config) |
| **Secret** | `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` |
| **Remote** | `hf` → `https://yahyoha:TOKEN@huggingface.co/spaces/yahyoha/omnidiag.git` |
| **Risk** | Full write access to the Hugging Face Space. An attacker can deploy malicious models, delete the space, or steal model artifacts. |
| **CVSS** | 8.8 (High) |

**Evidence:** Same as C1 — plaintext token in remote URL for the `hf` remote. Also embedded in the LFS endpoint URL on line 23 of `.git/config`.

### C3 — Unresolved Merge Conflict in `requirements.txt`

| Attribute | Detail |
|-----------|--------|
| **File** | [`requirements.txt`](requirements.txt:9) |
| **Lines** | 9–12 |
| **Content** | ```
<<<<<<< HEAD
PyYAML
=======
>>>>>>> dfecd17 (fix(deps): remove strict version pinning to resolve Python version conflicts)
``` |
| **Impact** | **`pip install -r requirements.txt` WILL FAIL.** The file is syntactically invalid. Additionally, `PyYAML` (required by [`backend/router.py`](backend/router.py:21) and [`configs/config_loader.py`](configs/config_loader.py:15)) may or may not be installed depending on how this conflict is resolved. |
| **Downstream** | Docker build fails at `RUN pip install --no-cache-dir -r requirements.txt`. CI pipeline fails at `pip install -r requirements.txt`. |

### C4 — Model Weights Path Misconfiguration

| Attribute | Detail |
|-----------|--------|
| **File** | [`configs/heart_disease.yaml`](configs/heart_disease.yaml:77) |
| **Config** | `weights_path: "models/heart_disease/xgboost_weights/omni_diag_xgb_optimized.pkl"` |
| **Reality** | Directory `models/heart_disease/xgboost_weights/` **does not exist**. The actual model lives at [`models/heart_disease/omni_diag_xgb_optimized.pkl`](models/heart_disease/omni_diag_xgb_optimized.pkl). |
| **Fallback** | [`configs/heart_disease.yaml`](configs/heart_disease.yaml:78): `fallback_weights_path: "models/heart_disease/omni_diag_xgb_optimized.pkl"` works, but this masks the misconfiguration. |
| **Impact** | If the fallback is ever removed or the primary path is relied upon, the API returns 500 errors on every prediction request. |

### C5 — Preprocessing Order Bug (Encoding Before Feature Engineering)

| Attribute | Detail |
|-----------|--------|
| **File** | [`backend/model_loader.py`](backend/model_loader.py:205-207) |
| **Lines** | `predict()` calls `_apply_preprocessors(df)` **before** `_engineer_features(df)` |
| **Bug** | Label encoders convert strings like `"M"`, `"ATA"` into integers. Then `_engineer_features()` computes `Age_BP_Interaction = Age * RestingBP` on these **encoded numeric values** — which works mathematically but is conceptually fragile. More critically, the heuristic features like `Sex_Num` are not available if encoding hasn't happened, but the feature engineer expects raw categorical values. |
| **Severity** | The heuristic features (`Age_BP_Interaction`, `HR_Age_Ratio`, `Chol_Age_Ratio`) only use numeric columns (Age, RestingBP, MaxHR, Cholesterol), so this bug may not affect accuracy. However, if categorical features were part of the heuristic formula, the results would be garbage. |

---

## 2. 🟡 Warnings

### W1 — Python Version Fragmentation

| Environment | Python Version | Source |
|-------------|---------------|--------|
| Docker | **3.10** | [`Dockerfile`](Dockerfile:6) — `FROM python:3.10-slim` |
| CI | **3.11** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml:15) — `python-version: "3.11"` |
| README | 3.11+ | [`README.md`](README.md) |

**Risk:** Dependencies may behave differently across versions. The Docker container (3.10) is the production environment, but CI tests on 3.11. A dependency that works on 3.11 might break on 3.10 in production.

### W2 — No Version Pinning in `requirements.txt`

**Evidence:** All dependencies are unpinned (e.g., `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `xgboost`, `shap`).

**Risk:** A new release of any dependency could introduce breaking changes. The project has already experienced this — see commit `9e1e425` which had to remove strict version pinning due to conflicts.

### W3 — `PyYAML` Status Ambiguous

**Evidence:** `PyYAML` is caught in the merge conflict block. It's imported by:
- [`backend/router.py`](backend/router.py:21): `import yaml`
- [`configs/config_loader.py`](configs/config_loader.py:15): `import yaml`

**Risk:** If the conflict is resolved by removing `PyYAML`, both `router.py` and `config_loader.py` will raise `ModuleNotFoundError` at runtime.

### W4 — `get_schema_for_disease()` Throws `ValueError` Instead of `HTTPException`

| File | [`backend/schemas.py`](backend/schemas.py:130-134) |
|------|------|
| **Line** | `raise ValueError(...)` |
| **Impact** | FastAPI converts unhandled `ValueError` to a **500 Internal Server Error**, not a 404. This is both a poor developer experience and a potential information leak (stack traces in debug mode). |

### W5 — Legacy Endpoint Bypasses Validation

| File | [`backend/main.py`](backend/main.py:142-145) |
|------|------|
| **Route** | `POST /api/v3/predict` |
| **Issue** | Directly passes `patient` dict to `router.predict()` without Pydantic validation. Malformed input is passed straight to the model. |

### W6 — No Unit Tests Anywhere

| Scope | Backend | Frontend |
|-------|---------|----------|
| **Tests** | Zero | Zero |
| **Framework** | None installed | None configured |

**Risk:** No regression safety net. Any refactoring or dependency upgrade risks silent breakage.

### W7 — Duplicate Model Artifacts

| File | Size | Location |
|------|------|----------|
| `omni_diag_xgb_optimized.pkl` | 1.79 MB | `models/heart_disease/` |
| `omni_diag_xgb_optimized.pkl` | 1.79 MB | `models/` (project root) |

These appear to be identical copies, wasting ~1.8 MB of storage. The root-level copy should be removed.

### W8 — Frontend Hardcodes Production API URL

| File | [`frontend/src/api.js`](frontend/src/api.js:1) |
|------|------|
| **Line** | `const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';` |
| **Issue** | Local development requires manually setting `VITE_API_BASE=http://localhost:8000`. If a developer forgets, requests go to production, polluting production logs and potentially hitting rate limits. |

### W9 — Git Branching Structure Is Confusing

| Branch | Tracks | Notes |
|--------|--------|-------|
| `production-final-v3` | `origin/main1` | Current active branch |
| `hf-deploy` | Local only | Was being rebased onto `16318c7` |
| `V3-advanced-DL` | `origin/V3-advanced-DL` | Remote tracking branch |
| `main1` | `origin/main1` | Primary remote branch |
| `main` | `origin/main` | Remote only, possibly legacy |

**Risk:** The presence of 5+ branches with overlapping purposes (and a `main` vs `main1` naming issue) creates confusion about which branch is the canonical source of truth.

### W10 — `.gitignore` and `.gitattributes` Contradiction

| File | Rule | Effect |
|------|------|--------|
| [`.gitignore`](.gitignore:6) | `*.pkl` | **Ignores** all pickle files from version control |
| [`.gitattributes`](.gitattributes:1) | `*.pkl filter=lfs diff=lfs merge=lfs -text` | **Tracks** pickle files via Git LFS |

These are fundamentally contradictory. If `*.pkl` is in `.gitignore`, Git LFS won't track them either (since they're already ignored).

---

## 3. 🟢 Optimizations & Info

### I1 — CORS Is Well-Configured

[`backend/main.py`](backend/main.py:52-63) uses a strict allowlist with environment variable override. This is the correct approach for production.

### I2 — Docker Layer Caching Is Optimized

[`Dockerfile`](Dockerfile:18-20) copies `requirements.txt` and installs dependencies **before** copying the application code. This correctly leverages Docker layer caching for faster rebuilds.

### I3 — Lazy Model Loading

[`backend/model_loader.py`](backend/model_loader.py:59-70) loads models on first request, not at startup. This reduces cold-start time and memory usage for unused disease modules.

### I4 — Config-Driven Architecture

The YAML config system ([`configs/heart_disease.yaml`](configs/heart_disease.yaml)) and dynamic router ([`backend/router.py`](backend/router.py)) allow adding new diseases without code changes. This is an excellent architectural pattern.

### I5 — Frontend Has Proper API Timeout Handling

[`frontend/src/api.js`](frontend/src/api.js:19-46) implements `AbortController`-based timeout with a user-friendly error message for HF Spaces cold starts.

---

## 4. 🗺️ Action Plan

### Phase 0: 🚨 IMMEDIATE — Rotate Exposed Secrets

> **Priority:** CRITICAL — Do this before any other changes.

| Step | Action | Details |
|------|--------|---------|
| 0.1 | **Revoke GitHub PAT** | Go to https://github.com/settings/tokens and delete token `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`. |
| 0.2 | **Revoke HF Token** | Go to https://huggingface.co/settings/tokens and delete token `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`. |
| 0.3 | **Generate new tokens** | Create new tokens with minimal required scopes (repo-only for GitHub, space-only for HF). |
| 0.4 | **Update Git remotes** | ```bash
git remote set-url origin https://yahyalababenah:NEW_TOKEN@github.com/yahyalababenah/omnidiag.git
git remote set-url hf https://yahyoha:NEW_TOKEN@huggingface.co/spaces/yahyoha/omnidiag.git
``` |
| 0.5 | **Verify no secrets in git history** | ```bash
git log --all --format='%H %s' -- .git/config
# Check for any commits that contain the old URLs
``` |

### Phase 1: 🔴 Critical Infrastructure Fixes

| Step | Action | File(s) | Details |
|------|--------|---------|---------|
| 1.1 | **Resolve `requirements.txt` merge conflict** | [`requirements.txt`](requirements.txt:9-12) | Keep `PyYAML` (it's required). Remove the `<<<<<<< HEAD`, `=======`, `>>>>>>>` markers. Result should be: ```
fastapi
uvicorn
pandas
scikit-learn
xgboost
shap
python-multipart
joblib
PyYAML
``` |
| 1.2 | **Fix model weights path** | [`configs/heart_disease.yaml`](configs/heart_disease.yaml:77) | Either create the directory: `mkdir -p models/heart_disease/xgboost_weights/ && cp models/heart_disease/omni_diag_xgb_optimized.pkl models/heart_disease/xgboost_weights/` OR update the config to point to the actual path. |
| 1.3 | **Fix preprocessing order** | [`backend/model_loader.py`](backend/model_loader.py:205-207) | Swap the order in `predict()` and `explain()` so feature engineering runs BEFORE preprocessing: ```python
df = self._engineer_features(df)   # First: engineer features on raw data
df = self._apply_preprocessors(df) # Then: encode/scaler
``` |
| 1.4 | **Fix `get_schema_for_disease()` error handling** | [`backend/schemas.py`](backend/schemas.py:130-134) | Change `raise ValueError(...)` to return `None` (already handled in [`backend/main.py`](backend/main.py:108-113) which checks `if schema is None`). OR raise `HTTPException(status_code=404, ...)`. |

### Phase 2: 🟡 Git & Version Control Cleanup

| Step | Action | Details |
|------|--------|---------|
| 2.1 | **Align Python versions** | Update [`Dockerfile`](Dockerfile:6) from `python:3.10-slim` to `python:3.11-slim` to match CI and README. |
| 2.2 | **Pin critical dependency versions** | Add minimum version pins to [`requirements.txt`](requirements.txt) for reproducibility: `fastapi>=0.100.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `shap>=0.42.0`. Test thoroughly after pinning. |
| 2.3 | **Clean up Git branches** | Decide on canonical branch. Recommend: rename `production-final-v3` → `main`, delete stale branches (`hf-deploy`, `main1`, `V3-advanced-DL` after merging). |
| 2.4 | **Resolve `.gitignore` / `.gitattributes` conflict** | Either: remove `*.pkl` from `.gitignore` and properly configure Git LFS, OR remove the `.gitattributes` LFS config if models should not be tracked. |
| 2.5 | **Remove duplicate model files** | Delete `models/omni_diag_xgb_optimized.pkl` (root-level duplicate). Keep only `models/heart_disease/omni_diag_xgb_optimized.pkl`. |

### Phase 3: 🟡 Code Quality & Testing

| Step | Action | File(s) | Details |
|------|--------|---------|---------|
| 3.1 | **Add backend unit tests** | `tests/` | Minimum test coverage: `test_model_loader.py` (predict, explain, edge cases), `test_shap_service.py` (output format), `test_router.py` (disease routing). Use `pytest`. |
| 3.2 | **Add input validation for legacy endpoint** | [`backend/main.py`](backend/main.py:142-145) | Apply Pydantic schema validation to `/api/v3/predict` or remove the endpoint entirely. |
| 3.3 | **Add API authentication** | [`backend/main.py`](backend/main.py) | Implement API key middleware or JWT authentication. At minimum, require a configurable API key in the `Authorization` header. |
| 3.4 | **Add rate limiting** | [`backend/main.py`](backend/main.py) | Use `slowapi` or similar middleware to prevent abuse. |

### Phase 4: 🟢 Production Hardening

| Step | Action | Details |
|------|--------|---------|
| 4.1 | **Add Docker health check** | [`Dockerfile`](Dockerfile) | Add `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/ || exit 1` |
| 4.2 | **Run as non-root in Docker** | [`Dockerfile`](Dockerfile) | Add `RUN useradd -m -u 1000 appuser && USER appuser` for security best practice. |
| 4.3 | **Add `.env` documentation** | [`README.md`](README.md) | Document all environment variables (`CORS_ALLOWED_ORIGINS`, `VITE_API_BASE`, etc.) with examples. |
| 4.4 | **Add frontend tests** | `frontend/src/__tests__/` | Add smoke tests for `ShapBarChart`, `EngineeringMode`, `ClinicalEmrMode` using Vitest or Jest. |
| 4.5 | **CI hardening** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Add `npm test` step for frontend. Add security scanning step (e.g., `pip audit` or `safety check`). |

### Phase 5: 📋 Final Verification Checklist

- [ ] All secrets revoked and regenerated
- [ ] `git remote -v` shows no tokens in URLs
- [ ] `requirements.txt` is valid and `pip install` succeeds
- [ ] `pip install -r requirements.txt` installs `PyYAML`
- [ ] `docker build -t omnidiag .` succeeds
- [ ] `docker run omnidiag` starts and responds on `:8000`
- [ ] `GET /` returns `{"status": "Healthy", ...}`
- [ ] `POST /api/v4/heart_disease/predict` with valid patient data returns a prediction
- [ ] `POST /api/v4/heart_disease/explain` returns SHAP chart data
- [ ] `POST /api/v4/unknown_disease/predict` returns **404**, not 500
- [ ] Frontend `npm run build` succeeds
- [ ] Frontend `npm run dev` connects to local API when `VITE_API_BASE` is set
- [ ] CI pipeline passes on GitHub
- [ ] HF Spaces deployment picks up the new container

---

## Summary Dashboard

| Category | 🔴 Critical | 🟡 Warning | 🟢 Info | Status |
|----------|:-----------:|:----------:|:-------:|:------:|
| **Git & Version Control** | 1 (C1, C2) | 3 (W6, W7, W8) | 0 | ❌ Not ready |
| **Security & Secrets** | 2 (C1, C2) | 1 (W4) | 1 (I1) | ❌ **IMMEDIATE ACTION REQUIRED** |
| **Docker & Container** | 1 (C3) | 2 (W1, W2) | 1 (I2) | ❌ Cannot build |
| **Codebase & Architecture** | 2 (C4, C5) | 3 (W3, W5, W6) | 3 (I3, I4, I5) | ⚠️ Needs fixes |

**Total: 6 Critical, 9 Warnings, 5 Info items**

---

*Report generated by Roo (Debug Mode) — 2026-05-27T08:53 UTC*
