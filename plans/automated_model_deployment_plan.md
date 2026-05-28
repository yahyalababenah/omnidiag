# Automated .pkl Model Deployment — FINAL

## What Was Done

### Problem
The Hugging Face Space's `.gitignore` blocks `*.pkl` files. The new model file (`omni_diag_xgb_optimized.pkl`, 1.7 MB) cannot be uploaded to the Space's git repo.

### Solution
- **Preprocessors** (`standard_scaler.pkl`, `label_encoders.pkl`): Already committed to the Space before `*.pkl` was gitignored → available at Docker build time via `COPY . .`
- **Model weights** (`omni_diag_xgb_optimized.pkl`): Uploaded to Hugging Face Model Hub at [`yahyoha/omnidiag-models`](https://huggingface.co/yahyoha/omnidiag-models) → downloaded at container startup via `startup.sh`

## Uploaded Files

| File | Location | Size |
|------|----------|------|
| `omni_diag_xgb_optimized.pkl` | Hugging Face Model Hub → yahyoha/omnidiag-models | 1.7 MB |
| `standard_scaler.pkl` (✅ CORRECT) | Hugging Face Model Hub + already on Space | 991 bytes |
| `label_encoders.pkl` | Hugging Face Model Hub + already on Space | 1.5 KB |

## Dockerfile Changes

[`Dockerfile`](Dockerfile) now:
1. Installs `curl` for model downloads
2. Adds `startup.sh` script that downloads model weights from Hub if missing
3. `CMD` points to `startup.sh` (which runs uvicorn after download)

## Remaining

- [ ] Push code changes to GitHub
- [ ] Hugging Face Space will rebuild Docker and download model at startup
- [ ] Old stale files deleted from Space: `models/omni_diag_xgb_optimized.pkl`, `models/final_model.pkl`, `models/heart_disease/final_model.pkl`
