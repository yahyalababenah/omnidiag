/**
 * Central source for the clinical decision thresholds shown in the UI.
 *
 * There is exactly ONE place in the frontend that knows a threshold number
 * (this file) and exactly one function (`getDisplayThreshold`) that resolves
 * which one to show. Never write a threshold literal anywhere else — read
 * it from here (or, better, from the API response) instead.
 *
 * ── Diabetes ────────────────────────────────────────────────────────────
 * The backend (`EnsembleModelLoader.predict()`, backend/ensemble_loader.py)
 * returns a live `inference_threshold` field on every /api/v4/diabetes/predict
 * response, sourced from configs/diabetes.yaml → model.inference_threshold
 * (currently 0.275, clinically tuned for a 2x false-negative cost). That
 * value is read straight from the API response below — nothing for
 * diabetes is hardcoded here.
 *
 * ── Heart disease ───────────────────────────────────────────────────────
 * `ModelLoader.predict()` (backend/model_loader.py) has no configurable
 * threshold at all — it just takes the model's own argmax (the sklearn
 * default 0.5 cutoff) and none of the API endpoints (/predict, /explain,
 * /schema, /api/v4/diseases) expose that number. Verified by reading all
 * four response shapes — see PR discussion. So it is defined here as a
 * fallback constant. If the backend is ever changed to return it, delete
 * this constant and read it from the API exactly like diabetes.
 */
export const HEART_DISEASE_DEFAULT_THRESHOLD = 0.5;

/**
 * Resolve the decision threshold to display alongside a prediction result.
 *
 * Prefers the live value from the API response (`result.inference_threshold`,
 * present for diabetes). Falls back to the heart-disease constant above
 * only when the API genuinely provides nothing — never invents a number
 * for a disease that isn't heart disease.
 *
 * @param {string} disease - disease key, e.g. 'diabetes' | 'heart_disease'
 * @param {object} result - the /predict (or /explain) response object
 * @returns {number|null} threshold in [0, 1], or null if unknown
 */
export function getDisplayThreshold(disease, result) {
  if (typeof result?.inference_threshold === 'number') {
    return result.inference_threshold;
  }
  if (disease === 'heart_disease') {
    return HEART_DISEASE_DEFAULT_THRESHOLD;
  }
  return null;
}
