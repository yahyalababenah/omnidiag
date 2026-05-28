# =============================================================================
# OmniDiag — Multi-Disease Diagnostic Platform
# Production Dockerfile for Hugging Face Spaces deployment
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
# Note: *.pkl files are in .gitignore, so model weights won't be in git.
# Preprocessors (standard_scaler.pkl, label_encoders.pkl) are already committed
# to the Space and will be copied here. The main model file is downloaded
# at startup from Hugging Face Hub (see startup.sh).
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 omnidiag && chown -R omnidiag:omnidiag /app
USER omnidiag

# Create startup script that downloads model weights if missing
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
# Download model weights from Hugging Face Hub if not present\n\
MODEL_FILE="models/heart_disease/omni_diag_xgb_optimized.pkl"\n\
MODEL_URL="https://huggingface.co/yahyoha/omnidiag-models/resolve/main/omni_diag_xgb_optimized.pkl"\n\
\n\
if [ ! -f "$MODEL_FILE" ]; then\n\
    echo "⬇️  Downloading model weights from Hugging Face Hub..."\n\
    mkdir -p "$(dirname "$MODEL_FILE")"\n\
    curl -sL "$MODEL_URL" -o "$MODEL_FILE"\n\
    echo "✅ Model weights downloaded ($(stat -c%s "$MODEL_FILE") bytes)"\n\
else\n\
    echo "✅ Model weights already present ($(stat -c%s "$MODEL_FILE") bytes)"\n\
fi\n\
\n\
# Start the FastAPI application\n\
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000\n\
' > /app/startup.sh && chmod +x /app/startup.sh

# Expose the FastAPI port
EXPOSE 8000

# Health check for container orchestration (Hugging Face Spaces, Kubernetes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run startup script (downloads model if needed, then starts uvicorn)
CMD ["/app/startup.sh"]
