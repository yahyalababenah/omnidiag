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
 * ── Modules that decide without a threshold ─────────────────────────────
 * A module whose response carries `output_type: 'conformal_decision'` does
 * not compare a probability to a cut-point at all: it returns a decision
 * (`referral` / `no_referral` / `uncertain`) plus a Venn-Abers interval.
 * There is no threshold to display, and `getDisplayThreshold` returns null.
 *
 * Until Gate 8.4 this file exported a HEART_DISEASE_DEFAULT_THRESHOLD of 0.5
 * and showed it on three screens. The shipped heart model has no threshold of
 * any kind, so that number was invented here and displayed to a clinician as
 * if it came from the model. Never reintroduce a per-disease fallback: an
 * absent threshold is information, not a gap to fill.
 */

/**
 * Resolve the decision threshold to display alongside a prediction result.
 *
 * Reads the live value from the API response and invents nothing. Returns null
 * both when the module decides without a threshold (`conformal_decision`) and
 * when the response simply does not carry one — in either case the caller must
 * show no threshold rather than a placeholder.
 *
 * `disease` is kept in the signature for the call sites and is deliberately
 * unused: behaviour follows what the response declares, never the disease name.
 *
 * @param {string} _disease - unused; kept so existing call sites are unchanged
 * @param {object} result - the /predict (or /explain) response object
 * @returns {number|null} threshold in [0, 1], or null when there is none
 */
export function getDisplayThreshold(_disease, result) {
  if (typeof result?.inference_threshold === 'number') {
    return result.inference_threshold;
  }
  return null;
}

/**
 * True when this module reports a decision instead of thresholding a
 * probability. Screens use it to show the decision and its interval in place
 * of a threshold-and-band readout.
 *
 * @param {object|null} source - a /predict response or a disease-info object
 */
export function isConformalDecision(source) {
  return source?.output_type === 'conformal_decision';
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
 * A module that configures NO bands gets none: `getRiskBands` returns null and
 * `classifyRisk` returns null, and the caller shows the decision and interval
 * instead of a badge.
 *
 * Heart is that module, on purpose (D-32). Gate 6 measured that its
 * probability's meaning does not transport between hospitals, so a
 * HIGH/MODERATE/LOW badge on it claims a precision the model does not have.
 * Until Gate 8.4 this file substituted 0.7/0.4 for it anyway and the badge
 * appeared on screen — silently undoing the decision the config recorded.
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
  // An explicit null from a module that configures no bands is an answer, not a
  // missing value, and so is `output_type: 'conformal_decision'` on a /predict
  // response (which carries no risk_bands key at all). Only a source that says
  // nothing either way falls back.
  if (source && ('risk_bands' in source || isConformalDecision(source))) return null;
  return DEFAULT_RISK_BANDS;
}

/**
 * Classify a probability into a display band.
 *
 * @param {number} probability - on the same scale the API returned
 * @param {object|null} source - /predict response or disease-info object
 * @returns {'HIGH'|'MODERATE'|'LOW'|null} null when the module has no bands
 */
export function classifyRisk(probability, source) {
  const bands = getRiskBands(source);
  if (!bands) return null;
  if (probability >= bands.high) return 'HIGH';
  if (probability >= bands.moderate) return 'MODERATE';
  return 'LOW';
}
