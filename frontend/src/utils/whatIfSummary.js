/**
 * whatIfSummary — one description of the What-If result, for every surface.
 *
 * The screen and the exported PDF have to say the same thing about the same
 * patient. They did not: the PDF had no What-If section at all, so a patient
 * the screen told the clinician to refer ("Even with every modifiable factor
 * improved…") produced a printed report that mentioned neither the referral
 * nor the scenarios. A report that silently drops the recommendation is worse
 * than one that never had it, because it is the copy that leaves the building.
 *
 * Both WhatIfScenarioCard and PDFReport classify through this module, so the
 * wording is shared rather than duplicated and can only change in one place.
 *
 * Every number here comes from the /counterfactuals response. Nothing is
 * derived that the backend already computed — in particular relative risk
 * reduction, which is not the difference of two probabilities.
 */

import { normaliseScenario } from './counterfactuals';

export const SUBTITLE =
  "Scenarios show how the model's estimate responds to modifiable factors. " +
  'They are not a predicted treatment effect.';

export const NO_CROSSING_MESSAGE =
  'Even with every modifiable factor improved, the estimated risk remains above the threshold. ' +
  'The dominant factors are not modifiable. Referral is recommended.';

export const NO_IMPROVEMENT_MESSAGE =
  'No change to the modifiable factors lowers the estimated risk for this patient. ' +
  'Referral is recommended.';

export const NO_LEVERS_MESSAGE =
  'No modifiable factor is available to change for this patient (none of the ' +
  'modifiable inputs was supplied, or each is already at its target).';

export const LOW_RISK_HEADLINE = 'Low Clinical Risk';

export const LOW_RISK_MESSAGE =
  'Patient is currently at low clinical risk. No counterfactual interventions are necessary.';

/**
 * Section title for the scenarios table.
 *
 * It used to read "Recommended Lifestyle Interventions", which contradicts the
 * subtitle printed directly above it: a counterfactual is what the MODEL would
 * estimate under different inputs, not a course of action anyone is
 * recommending, and not a predicted treatment effect. The neutral name says
 * what the rows actually contain.
 */
export const SCENARIOS_TITLE = 'Modelled What-If Scenarios';

/** Post-intervention probability, on whichever scale the module returned. */
export function scenarioProbability(s) {
  if (typeof s.new_probability_corrected === 'number') return s.new_probability_corrected;
  if (typeof s.new_probability === 'number') return s.new_probability;
  if (typeof s.probability === 'number') return s.probability;
  return null;
}

/**
 * Relative risk reduction in percent — "removes N% of this patient's risk".
 *
 * The backend ships it as risk_reduction_relative_pct and it is used as-is. It
 * is NOT recomputed by subtracting probabilities: that gives percentage points,
 * a different and much smaller number (0.4795 -> 0.0395 is 92% relative but 44
 * points), and every label here says relative.
 */
export function scenarioReductionPct(s, baselineProbability) {
  if (typeof s.risk_reduction_relative_pct === 'number') {
    return Math.round(s.risk_reduction_relative_pct);
  }
  if (typeof s.risk_reduction === 'string') {
    const parsed = parseFloat(s.risk_reduction);
    if (!Number.isNaN(parsed)) return Math.round(parsed);
  }
  const prob = scenarioProbability(s);
  if (prob !== null && typeof baselineProbability === 'number' && baselineProbability > 0) {
    return Math.round(((baselineProbability - prob) / baselineProbability) * 100);
  }
  return null;
}

/**
 * Classify a What-If result.
 *
 * Returns { state, headline, message, detail, scenarios, best, referral },
 * where `state` is one of:
 *
 *   'unavailable'    — no response, or an error: say so, show no numbers
 *   'low_risk'       — below threshold, nothing to change
 *   'no_crossing'    — flagged, best achievable helps but does not cross
 *   'no_improvement' — flagged, nothing lowers the estimate at all
 *   'no_levers'      — flagged, no modifiable input to move
 *   'scenarios'      — the backend returned crossing scenarios
 *
 * `referral` is true for exactly the states where the screen recommends
 * referral, so a caller cannot render the outcome and drop the advice.
 */
export function summariseWhatIf({
  counterfactuals,
  prediction,
  baselineProbability,
  patientData = null,
  bestAchievable = null,
  error = null,
  message = null,
} = {}) {
  if (error || !Array.isArray(counterfactuals)) {
    return {
      state: 'unavailable',
      headline: 'What-If Scenarios',
      message: 'What-If scenarios could not be computed for this patient.',
      detail: null,
      scenarios: [],
      best: null,
      referral: false,
    };
  }

  const flaggedWithoutCrossing =
    counterfactuals.length === 0 && (bestAchievable || prediction === 1);

  if (flaggedWithoutCrossing) {
    const best = bestAchievable ? normaliseScenario(bestAchievable, 0, patientData) : null;
    const after = best ? scenarioProbability(best) : null;
    // A "best achievable" that does not lower the estimate is not an
    // improvement; showing "49.4% -> 50.4%" under that heading was wrong.
    const improves =
      typeof after === 'number' &&
      (typeof baselineProbability !== 'number' || after < baselineProbability);

    if (best && !improves) {
      return {
        state: 'no_improvement',
        headline: 'What-If Scenarios',
        message: NO_IMPROVEMENT_MESSAGE,
        detail: null,
        scenarios: [],
        best: null,
        referral: true,
      };
    }
    if (!best) {
      return {
        state: message ? 'no_crossing' : 'no_levers',
        headline: 'What-If Scenarios',
        message: message || NO_CROSSING_MESSAGE,
        detail: message ? null : NO_LEVERS_MESSAGE,
        scenarios: [],
        best: null,
        referral: true,
      };
    }
    return {
      state: 'no_crossing',
      headline: 'What-If Scenarios',
      message: NO_CROSSING_MESSAGE,
      detail: null,
      scenarios: [],
      best: {
        changes: best.changes ?? [],
        after,
        relativePct: typeof best.risk_reduction_relative_pct === 'number'
          ? best.risk_reduction_relative_pct : null,
        absolutePp: typeof best.risk_reduction_absolute_pp === 'number'
          ? best.risk_reduction_absolute_pp : null,
      },
      referral: true,
    };
  }

  if (counterfactuals.length === 0) {
    return {
      state: 'low_risk',
      headline: LOW_RISK_HEADLINE,
      message: LOW_RISK_MESSAGE,
      detail: null,
      scenarios: [],
      best: null,
      referral: false,
    };
  }

  const scenarios = counterfactuals.map((s, i) => {
    const cf = normaliseScenario(s, i, patientData);
    return {
      changes: cf.changes ?? [],
      after: scenarioProbability(cf),
      reductionPct: scenarioReductionPct(cf, baselineProbability),
    };
  });

  return {
    state: 'scenarios',
    headline: 'What-If Scenarios',
    message: null,
    detail: null,
    scenarios,
    best: null,
    referral: false,
  };
}
