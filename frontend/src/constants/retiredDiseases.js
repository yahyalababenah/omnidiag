/**
 * Disease modules that have been retired.
 *
 * The backend's list (backend/retired_diseases.py) is the source of truth: a
 * retired module has no config there, so the API stops listing it and answers
 * 410 for anything that would score or write for it. This copy exists for one
 * reason: the frontend deploys before the backend does (Vercel on merge, the HF
 * Space on release), and against the older backend the retired module would
 * still be offered in the picker -- with none of the UI it needed left here.
 *
 * tests/test_frontend_retired_contract.py keeps this list equal to the
 * backend's. Remove it once every backend the frontend can reach has retired
 * the module itself (backlog: after the HF release).
 */
export const RETIRED_DISEASES = ['diabetes'];

/** The disease list from GET /api/v4/diseases, without retired modules. */
export function withoutRetired(diseases) {
  return (diseases || []).filter((d) => !RETIRED_DISEASES.includes(d?.name));
}
