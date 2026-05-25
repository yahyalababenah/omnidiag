# =============================================================================
# OmniDiag — Multi-Disease Diagnostic Platform
# Production Dockerfile for Hugging Face Spaces deployment
# =============================================================================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by scikit-learn / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
