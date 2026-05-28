# Docker & requirements.txt — Compatibility Analysis

**Date:** 2026-05-28  
**Scope:** Verify that [`Dockerfile`](../Dockerfile) can successfully install all dependencies in [`requirements.txt`](../requirements.txt) without conflicts.

---

## 1. Environment

| Property | Value |
|----------|-------|
| Base image | `python:3.10-slim` (Debian-based, Python 3.10) |
| Architecture | Linux x86_64 (amd64) |
| System packages installed | `gcc` (C compiler), `libgomp1` (OpenMP for XGBoost) |
| Install method | `pip install --no-cache-dir -r requirements.txt` |

---

## 2. Library Compatibility Matrix

| Library | Min Version | Type | Has pre-built wheel for Linux x86_64 + Python 3.10? | Compatible? |
|---------|-------------|------|------------------------------------------------------|-------------|
| `fastapi` | ≥ 0.100.0 | Pure Python | ✅ N/A (no compilation needed) | ✔ |
| `uvicorn` | ≥ 0.20.0 | Pure Python | ✅ N/A (no compilation needed) | ✔ |
| `python-multipart` | ≥ 0.0.6 | Pure Python | ✅ N/A (no compilation needed) | ✔ |
| `pandas` | ≥ 2.0.0 | C-extensions | ✅ manylinux2014 wheel available | ✔ |
| `numpy` | ≥ 1.24.0 | C-extensions | ✅ manylinux2014 wheel available | ✔ |
| `scikit-learn` | ≥ 1.3.0 | C-extensions | ✅ manylinux2014 wheel available | ✔ |
| `xgboost` | ≥ 2.0.0 | C++/OpenMP | ✅ manylinux2014 wheel + `libgomp1` | ✔ |
| `joblib` | ≥ 1.3.0 | Pure Python | ✅ N/A (no compilation needed) | ✔ |
| `shap` | ≥ 0.42.0 | Python + numba | ✅ wheel available (numba + llvmlite) | ✔ |
| `pyyaml` | ≥ 6.0 | C-extensions | ✅ manylinux2014 wheel available | ✔ |
| `pydantic` | ≥ 2.0.0 | Rust (pydantic-core) | ✅ pre-built wheel available | ✔ |

> **Result:** All 11 libraries have pre-compiled wheels for the target platform. `pip` will NOT need to compile anything from source.

---

## 3. System Dependencies Explained

### `gcc` (GNU C Compiler) — 🟡 Optional

| | |
|---|---|
| **Why installed** | Safety net — if pip cannot find a pre-built wheel and falls back to source compilation |
| **Is it needed?** | No — all libraries have wheels on PyPI for `manylinux2014` + Python 3.10 |
| **Risk of removal** | Low. If a wheel is ever missing, pip would fail to compile from source |
| **Recommendation** | Keep it — minimal size cost, prevents rare failures |

### `libgomp1` (GNU OpenMP) — 🔴 Required

| | |
|---|---|
| **Why installed** | XGBoost uses OpenMP for parallel CPU processing. Without this library, XGBoost will either crash at import or run single-threaded |
| **Is it needed?** | **Yes** — XGBoost will fail without it |
| **Risk of removal** | High. XGBoost import error or silent performance degradation |
| **Recommendation** | Keep it — essential for XGBoost |

---

## 4. Potential Issues & Edge Cases

### 4.1 NumPy Version Conflict with SHAP

`shap>=0.42.0` depends on `numba`, which depends on `numpy`. If `numpy>=1.24.0` installs a version too new for `numba`, pip will resolve this automatically by finding a compatible version. No action needed.

### 4.2 Pydantic v2 Breaking Changes

`pydantic>=2.0.0` has a completely different API from v1. The project's [`schemas.py`](../backend/schemas.py) was written for v2 syntax (`BaseModel`, `Field`, `ConfigDict`). As long as the codebase uses v2 API (which it does), there is no issue.

### 4.3 Python-Multipart

FastAPI requires `python-multipart` to parse form data. The project uses JSON requests (`/api/v4/{disease}/predict` with `application/json`), so this dependency is not strictly needed at runtime. However, it is harmless to keep.

### 4.4 XGBoost OpenMP on Slim Images

`python:3.10-slim` does NOT include `libgomp1` by default. The explicit `apt-get install libgomp1` in the Dockerfile is correct and necessary.

---

## 5. Summary

| Aspect | Verdict |
|--------|---------|
| All libraries have wheels? | ✅ Yes |
| System deps sufficient? | ✅ Yes (`gcc` + `libgomp1` cover all needs) |
| Python version correct? | ✅ Python 3.10 supported by all libraries |
| Version conflicts? | ✅ None — pip resolves dependencies automatically |
| **Overall** | **✅ Fully compatible — no changes needed** |

---

## 6. Related Documents

- [`Dockerfile`](../Dockerfile) — Current Docker configuration
- [`requirements.txt`](../requirements.txt) — Current Python dependencies
- [`docker_fix_plan.md`](docker_fix_plan.md) — Docker fix plan (model weights, non-root user, HEALTHCHECK)
