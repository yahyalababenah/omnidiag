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
 * (tuned for a 2x false-negative cost). That value is read straight from
 * the API response below — nothing for diabetes is hardcoded here.
 *
 * Diabetes probabilities are prevalence-corrected on the backend: both
 * `result.confidence` and `result.inference_threshold` are stated on the
 * deployment prevalence (~14%), so comparing them here is like-for-like.
 * The raw model output is in `result.probability_raw` /
 * `result.inference_threshold_raw` for auditing — do not mix the two scales.
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

/**
 * ── Risk display bands (HIGH / MODERATE / LOW) ───────────────────────────
 *
 * These decide the colour of the risk badge; they are NOT the decision
 * threshold (that is `inference_threshold`). A patient can be Positive and
 * still sit in a lower band — the badge describes magnitude, the diagnosis
 * describes the decision.
 *
 * The backend returns `risk_bands` on every diabetes /predict response and
 * on `GET /api/v4/diseases` (`info.risk_bands`), already stated on the same
 * scale as the probability it returns — prevalence-corrected for diabetes,
 * raw for heart disease. Never compare a band to a probability from a
 * different scale.
 *
 * Heart disease configures no bands, so it falls back to the constants
 * below — the same numbers it has always used.
 */
export const DEFAULT_RISK_BANDS = { high: 0.7, moderate: 0.4 };

/**
 * Resolve the risk bands to use for a disease.
 *
 * @param {object|null} source - a /predict response, or a disease-info object
 *                               from GET /api/v4/diseases
 * @returns {{high: number, moderate: number}}
 */
export function getRiskBands(source) {
  const bands = source?.risk_bands;
  if (bands && typeof bands.high === 'number' && typeof bands.moderate === 'number') {
    return { high: bands.high, moderate: bands.moderate };
  }
  return DEFAULT_RISK_BANDS;
}

/**
 * Classify a probability into a display band.
 *
 * @param {number} probability - on the same scale the API returned
 * @param {object|null} source - /predict response or disease-info object
 * @returns {'HIGH'|'MODERATE'|'LOW'}
 */
export function classifyRisk(probability, source) {
  const { high, moderate } = getRiskBands(source);
  if (probability >= high) return 'HIGH';
  if (probability >= moderate) return 'MODERATE';
  return 'LOW';
}
