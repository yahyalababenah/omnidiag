# =============================================================================
# OmniDiag — Multi-Disease Diagnostic Platform
# Production Dockerfile for Hugging Face Spaces deployment
# Build v3 — includes engineer_medical fix + retrained 16-feature model
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

# Download model weights at BUILD time so they're baked into the image.
# This means zero download delay at startup — port 7860 responds instantly.
RUN mkdir -p models/heart_disease && \
    curl -fsSL "https://huggingface.co/yahyoha/omnidiag-models/resolve/main/omni_diag_xgb_optimized.pkl" \
         -o models/heart_disease/omni_diag_xgb_optimized.pkl && \
    echo "Model baked in: $(wc -c < models/heart_disease/omni_diag_xgb_optimized.pkl) bytes"

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
