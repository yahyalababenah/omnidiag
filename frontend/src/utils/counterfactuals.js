/**
 * Give every counterfactual scenario a `scenario_id` and an array `changes`.
 *
 * Both live modules (heart, NHANES) return the array shape from
 * POST /counterfactuals:
 *
 *   { scenario_id, probability,
 *     changes: [{ feature, original_value, counterfactual_value, direction }] }
 *
 * The object-shaped `changes` of the retired BRFSS diabetes module, and the
 * conversion it needed, went with that module (gate B6).
 *
 * @param {object} s            one scenario from the API
 * @param {number} idx          its position, used as scenario_id when absent
 * @param {object|null} _patientData  unused; kept for existing callers
 * @returns {object} scenario with `scenario_id` and an array `changes`
 */
// eslint-disable-next-line no-unused-vars
export function normaliseScenario(s, idx, _patientData = null) {
  if (!s || typeof s !== 'object') return s;
  if (Array.isArray(s.changes)) {
    return s.scenario_id !== undefined ? s : { ...s, scenario_id: idx + 1 };
  }
  // Mock/placeholder scenarios have no `changes`; leave them alone.
  return s;
}
