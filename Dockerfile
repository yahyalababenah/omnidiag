# =============================================================================
# OmniDiag — Multi-Disease Diagnostic Platform
# Production Dockerfile for Hugging Face Spaces deployment
# Build v4 — bakes model + preprocessors into image at build time
# =============================================================================

# Same interpreter as the validated local environment: the pinned numeric
# stack (numpy 2.4 / scikit-learn 1.9) requires Python >= 3.11.
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by scikit-learn / shap + curl for downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Download model weights + preprocessors at BUILD time so they're baked into
# the image. Zero download delay at startup — port 7860 responds instantly.
#
# heart_full_tuned.pkl is the previous heart model, kept in the image so that
# reverting to it stays a two-line config change (see below). It is a single
# self-contained bundle, so there is nothing else to fetch for it. The retired
# BRFSS diabetes weights are no longer downloaded (gate B7).
RUN mkdir -p models/heart_disease && \
    HF="https://huggingface.co/yahyoha/omnidiag-models/resolve/main" && \
    echo "=== heart_disease (revert path) ===" && \
    curl -fsSL "${HF}/heart_full_tuned.pkl"   -o models/heart_disease/heart_full_tuned.pkl && \
    echo "=== downloaded ==="

# NHANES dysglycaemia bundle (Phase 9). HF rejects any binary in a git push, so the
# .joblib cannot travel in the Space repository and is fetched here like the heart
# file above. The sha256 is checked: the bundle carries the conformal quantiles
# and the calibration, so a different file would silently change every decision.
# A local or GitHub checkout has this file already; the download overwrites it with
# the same bytes or fails the build.
RUN mkdir -p models/diabetes_nhanes && \
    curl -fsSL "https://huggingface.co/yahyoha/omnidiag-models/resolve/main/diabetes_nhanes/diabetes_nhanes_ebm.joblib" \
        -o models/diabetes_nhanes/diabetes_nhanes_ebm.joblib && \
    echo "fcceeb37b53295f115e5fe41b9a8618117263ef6ddb7898f9a90ad7b50b2c408  models/diabetes_nhanes/diabetes_nhanes_ebm.joblib" | sha256sum -c -

# Build the live heart model from the committed training CSV. No model binary
# is committed to the repository, so this step IS the shipping mechanism.
#
# The script refuses to produce a bundle unless (1) the training CSV hashes to
# the recorded value and (2) the decision and conformal set are identical to
# the recorded fingerprint for all 920 patients, with |delta p| <= 1e-6. A
# failure here fails the build on purpose: a heart model whose decisions differ
# from the measured ones must not ship silently.
#
# heart_full_tuned.pkl above stays in the image so that reverting to the
# previous XGBoost model is two lines in configs/heart_disease.yaml.
RUN python scripts/train_heart_glm.py --verify

# Rebuild the drift reference profiles and compare them against the committed
# ones. The heart profile is derived from the same training CSV as the model
# above and records its sha256, so a training file that changed without the
# profile being rebuilt fails the build here rather than showing up later as
# drift that came from nowhere.
RUN python scripts/build_drift_reference.py --verify

# Create non-root user for security
RUN useradd -m -u 1000 omnidiag && chown -R omnidiag:omnidiag /app
USER omnidiag

# Simple startup — model is already present, just launch uvicorn
RUN printf '#!/bin/bash\n\
echo "===== Application Startup at $(date -u +%%Y-%%m-%%d\\ %%H:%%M:%%S) ====="\n\
exec uvicorn backend.main:app --host 0.0.0.0 --port 7860\n\
' > /app/startup.sh && chmod +x /app/startup.sh

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Run startup script (downloads model if needed, then starts uvicorn)
CMD ["/app/startup.sh"]
