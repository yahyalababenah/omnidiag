/**
 * Wording for the model's output. OmniDiag screens for risk; it does not
 * diagnose. The API still returns `diagnosis: "Positive" | "Negative"` (an
 * unchanged contract); this is the only place that turns it into text shown
 * to a clinician.
 */
export const SCREENING_LABEL = 'Screening result';
export const SCREENING_ELEVATED = 'Elevated risk — confirmatory testing recommended';
export const SCREENING_BELOW = 'Below threshold';

/** True when the model flagged the patient (prediction 1 / diagnosis "Positive"). */
export function isElevated(result) {
  if (!result) return false;
  if (result.prediction === 1 || result.prediction === 0) return result.prediction === 1;
  return result.diagnosis === 'Positive';
}

export function screeningText(result) {
  return isElevated(result) ? SCREENING_ELEVATED : SCREENING_BELOW;
}
