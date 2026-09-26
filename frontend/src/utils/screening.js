/**
 * Wording for the model's output. OmniDiag screens for risk; it does not
 * diagnose. The API still returns `diagnosis: "Positive" | "Negative"` (an
 * unchanged contract); this is the only place that turns it into text shown
 * to a clinician.
 *
 * Two vocabularies, chosen by what the module reports — never by disease name:
 *
 * * A THRESHOLD module says elevated or below its threshold.
 *
 * * A DECISION module (`output_type: 'conformal_decision'`) has three answers,
 *   and the third is the point of it. Until Gate 8.8 this file had only two
 *   strings, so an UNCERTAIN patient — one the model explicitly could not place
 *   — was shown "Elevated risk", and a non-referred one was shown "Below
 *   threshold" for a model that has no threshold. The model family exists to
 *   surface "I cannot tell"; collapsing it back to a binary on screen throws
 *   away the thing being demonstrated.
 */
export const SCREENING_LABEL = 'Screening result';
export const SCREENING_ELEVATED = 'Elevated risk — confirmatory testing recommended';
export const SCREENING_BELOW = 'Below threshold';
export const SCREENING_REFERRAL = 'Refer — confirmatory testing recommended';
export const SCREENING_UNCERTAIN = 'Uncertain — refer for further evaluation';
export const SCREENING_NO_REFERRAL = 'No referral indicated';

/** True when the model flagged the patient (prediction 1 / diagnosis "Positive"). */
export function isElevated(result) {
  if (!result) return false;
  if (result.prediction === 1 || result.prediction === 0) return result.prediction === 1;
  return result.diagnosis === 'Positive';
}

/**
 * True when this patient goes forward for evaluation.
 *
 * Reads the flag the API states outright rather than re-deriving it, so
 * "uncertain counts as a referral" is decided in one place on the server.
 */
export function goesForward(result) {
  if (result?.decision_is_referral != null) return result.decision_is_referral === true;
  return isElevated(result);
}

export function screeningText(result) {
  switch (result?.decision) {
    case 'referral':    return SCREENING_REFERRAL;
    case 'uncertain':   return SCREENING_UNCERTAIN;
    case 'no_referral': return SCREENING_NO_REFERRAL;
    default:            return isElevated(result) ? SCREENING_ELEVATED : SCREENING_BELOW;
  }
}
