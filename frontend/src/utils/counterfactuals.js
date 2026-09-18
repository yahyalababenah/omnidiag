/**
 * Normalise a counterfactual scenario to one shape, whichever module sent it.
 *
 * The two modules return different shapes from POST /counterfactuals:
 *
 *   heart:    { scenario_id, probability,
 *               changes: [{ feature, original_value, counterfactual_value, direction }] }
 *   diabetes: { scenario, new_probability, new_probability_corrected,
 *               risk_reduction_relative_pct, risk_reduction_absolute_pp, ...,
 *               changes: { feature: new_value, ... } }        ← an OBJECT
 *
 * Consumers used to assume the heart shape. For diabetes that meant the
 * What-If card fell into its mock branch ("If undefined drops from
 * undefined") and the PDF export threw on `changes.map`. Every consumer
 * now goes through this function and sees the heart-style array.
 *
 * Probability fields are passed through untouched — this function changes
 * the layout of `changes`, never a number's scale.
 *
 * @param {object} s            one scenario from the API
 * @param {number} idx          its position, used as scenario_id when absent
 * @param {object|null} patientData  the patient's original inputs, used to
 *                              fill `original_value` (diabetes omits it)
 * @returns {object} scenario with `scenario_id` and an array `changes`
 */
export function normaliseScenario(s, idx, patientData = null) {
  if (!s || typeof s !== 'object') return s;
  if (Array.isArray(s.changes)) {
    return s.scenario_id !== undefined ? s : { ...s, scenario_id: idx + 1 };
  }
  if (s.changes && typeof s.changes === 'object') {
    const changes = Object.entries(s.changes).map(([feature, value]) => {
      const original = patientData?.[feature];
      const shown = typeof value === 'number' && !Number.isInteger(value)
        ? Math.round(value * 10) / 10
        : value;
      let direction = 'change';
      if (typeof original === 'number' && typeof value === 'number') {
        direction = value < original ? 'decrease' : 'increase';
      }
      return {
        feature,
        original_value: original ?? '—',
        counterfactual_value: shown,
        direction,
      };
    });
    return { ...s, scenario_id: s.scenario_id ?? idx + 1, changes };
  }
  // Mock/placeholder scenarios have no `changes`; leave them alone.
  return s;
}
