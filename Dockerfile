# =============================================================================
# OmniDiag — Multi-Disease Diagnostic Platform
# Production Dockerfile for Hugging Face Spaces deployment
# Build v4 — bakes model + preprocessors into image at build time
# =============================================================================

FROM python:3.10-slim

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
# Why preprocessors are separate:
#   *.pkl files are gitignored (too large for git history), so COPY . . doesn't
#   include them. label_encoders.pkl and standard_scaler.pkl are required by
#   ModelLoader._apply_preprocessors() to encode categorical fields (Sex, ChestPainType,
#   RestingECG, ExerciseAngina, ST_Slope) and scale numerical features before inference.
#   Without them, any predict/explain call raises a KeyError / ValueError → HTTP 500.
RUN mkdir -p models/heart_disease/preprocessors models/diabetes/preprocessors && \
    HF="https://huggingface.co/yahyoha/omnidiag-models/resolve/main" && \
    echo "=== heart_disease ===" && \
    curl -fsSL "${HF}/omni_diag_xgb_optimized.pkl"   -o models/heart_disease/omni_diag_xgb_optimized.pkl && \
    curl -fsSL "${HF}/label_encoders.pkl"             -o models/heart_disease/preprocessors/label_encoders.pkl && \
    curl -fsSL "${HF}/standard_scaler.pkl"            -o models/heart_disease/preprocessors/standard_scaler.pkl && \
    echo "=== diabetes ===" && \
    curl -fsSL "${HF}/diabetes/xgb_model.pkl"                     -o models/diabetes/xgb_model.pkl && \
    curl -fsSL "${HF}/diabetes/lgb_model.pkl"                     -o models/diabetes/lgb_model.pkl && \
    curl -fsSL "${HF}/diabetes/rf_model.pkl"                      -o models/diabetes/rf_model.pkl && \
    curl -fsSL "${HF}/diabetes/meta_learner.pkl"                  -o models/diabetes/meta_learner.pkl && \
    curl -fsSL "${HF}/diabetes/preprocessors/standard_scaler.pkl" -o models/diabetes/preprocessors/standard_scaler.pkl && \
    echo "=== all models downloaded ==="

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
