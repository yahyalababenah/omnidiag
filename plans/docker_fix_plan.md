# Docker Fix Plan — OmniDiag

## Issues Found

| # | Severity | Issue | File |
|---|----------|-------|------|
| 1 | 🚨 Critical | Model weights + preprocessors `.pkl` files do not exist on disk | `models/heart_disease/` |
| 2 | 📦 Medium | `experiment_files/` copied into image unnecessarily | `.dockerignore` |
| 3 | 🛡️ Low | Container runs as root | `Dockerfile` |
| 4 | ❤️ Low | No HEALTHCHECK for container health monitoring | `Dockerfile` |

---

## Step 1: Generate Model Weights and Preprocessors

**Why:** The backend will crash on first request if `omni_diag_xgb_optimized.pkl` and preprocessors don't exist.

**Action:**
```bash
pip install -r requirements.txt
python experiment_files/models/train_best_xgb.py
```

**Expected output after running:**
```
models/heart_disease/
├── omni_diag_xgb_optimized.pkl      # ← New: trained XGBoost model
├── preprocessors/
│   ├── label_encoders.pkl            # ← New: label encoders
│   └── standard_scaler.pkl           # ← New: standard scaler
├── grid_best.json
├── metadata.json
└── metrics.json
```

---

## Step 2: Update `.dockerignore`

**Why:** `experiment_files/` contains training scripts, reports, and plans. These are not needed at runtime and add unnecessary size to the Docker image.

**File:** `.dockerignore`

**Change:** Add this line to the existing file:
```
experiment_files
```

**Verification:** After adding, `docker build` will skip the entire `experiment_files/` directory.

---

## Step 3: Update `Dockerfile` — Add Non-Root User (Optional but Recommended)

**Why:** Running as root is a security risk. If the container is compromised, the attacker has full system access.

**File:** `Dockerfile`

**Change:** Add these lines before the `CMD` line:
```dockerfile
# Create non-root user for security
RUN useradd -m -u 1000 omnidiag && chown -R omnidiag:omnidiag /app
USER omnidiag
```

**Placement:** After `COPY . .` and before `EXPOSE 8000`.

---

## Step 4: Update `Dockerfile` — Add HEALTHCHECK (Optional but Recommended)

**Why:** Hugging Face Spaces and orchestrators like Kubernetes need a way to know if the container is healthy. The API already has a `/` health endpoint.

**File:** `Dockerfile`

**Change:** Add this after `EXPOSE 8000`:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1
```

---

## Step 5: Rebuild and Test

```bash
# 1. Train the model first
python experiment_files/models/train_best_xgb.py

# 2. Build Docker image
docker build -t omnidiag:latest .

# 3. Run the container
docker run -d -p 8000:8000 --name omnidiag omnidiag:latest

# 4. Test health
curl http://localhost:8000/

# 5. Test prediction
curl -X POST http://localhost:8000/api/v4/heart_disease/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140, "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Flat"}'

# 6. Cleanup
docker stop omnidiag && docker rm omnidiag
```

---

## Final File States

### `.dockerignore` After Fix
```
.venv
__pycache__
.git
.env
*.log
*.md
mlruns.db
.vscode
.idea
.DS_Store
frontend/node_modules
frontend/dist
data/heart_disease/raw/*.xlsx
data/heart_disease/interim
data/interim
data/raw
data/processed
v2_archive
plans
deployment
scripts
reports/figures
*.png
*.csv
*.ipynb
experiment_files          ← NEW LINE
```

### `Dockerfile` After Fix
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 omnidiag && chown -R omnidiag:omnidiag /app
USER omnidiag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Summary

| Step | Action | Priority | Time |
|------|--------|----------|------|
| 1 | Train model + save preprocessors | 🚨 Required | ~2 min |
| 2 | Add `experiment_files` to `.dockerignore` | 📦 Recommended | 30 sec |
| 3 | Add non-root user to `Dockerfile` | 🛡️ Optional | 1 min |
| 4 | Add HEALTHCHECK to `Dockerfile` | ❤️ Optional | 1 min |
| 5 | Rebuild and test | ✅ Required | ~5 min |
