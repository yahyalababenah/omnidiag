import fs from 'fs';
import { parseSchema } from './schemaFieldParser.mjs';
import { categorizeFields } from './featureCategorizer.mjs';
import { profileFor, randomPatient, randomValueFor } from './randomPatient.mjs';
import { normaliseScenario } from './counterfactuals.mjs';
import { summariseWhatIf, scenarioProbability, scenarioReductionPct } from './whatIfSummary.mjs';
import { lookupMedicalTerm } from './medicalDictionary.mjs';
import { FAVOURABLE_BINARY, NEUTRAL_FIELDS, HIGHER_IS_BETTER, RISK_CATEGORIES } from './clinicalDirection.mjs';
const SCHEMAS = process.argv[2];
const data = JSON.parse(fs.readFileSync(SCHEMAS));
let seed = 12345; Math.random = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
const out = {};
for (const d of ['heart_disease', 'diabetes_nhanes']) {
  const fields = parseSchema(data[d]);
  const names = fields.map(f => f.name);
  out[d] = {
    fields: names,
    categories: [...categorizeFields(fields).entries()].map(([c, fs]) => [c, fs.map(f => f.name)]),
    profiles: Object.fromEntries(names.map(n => [n, profileFor(d, n) ?? null])),
    random: Array.from({ length: 5 }, () => randomPatient(fields, d)),
    // Age/Sex are the two entries this gate rewrites on purpose (own commit).
    dictionary: Object.fromEntries(names.filter(n => !['Age', 'Sex'].includes(n)).map(n => [n, lookupMedicalTerm(n) ?? null])),
    dictionaryByLabel: Object.fromEntries(fields.filter(f => !['Age', 'Sex'].includes(f.name)).flatMap(f => [f.label, f.title, f.displayName].filter(x => typeof x === 'string').map(t => [t, lookupMedicalTerm(t) ?? null]))),
    direction: Object.fromEntries(names.map(n => [n, [FAVOURABLE_BINARY[n] ?? null, NEUTRAL_FIELDS?.has?.(n) ?? null, HIGHER_IS_BETTER?.has?.(n) ?? null, RISK_CATEGORIES?.[n] ?? null]])),
  };
}
out.cf = {};
for (const [k, v] of Object.entries(data._cf)) {
  const cf = v.cf, scen = (cf.counterfactuals || []).map((s, i) => normaliseScenario(s, i, v.patient));
  out.cf[k] = { scen, best: cf.best_achievable ? normaliseScenario(cf.best_achievable, 0, v.patient) : null,
    probs: scen.map(s => scenarioProbability(s)), red: scen.map(s => scenarioReductionPct ? scenarioReductionPct(s, cf.baseline_probability) : null),
    summary: summariseWhatIf({ counterfactuals: cf.counterfactuals, prediction: cf.status === 'not_applicable' ? 0 : 1, baselineProbability: cf.baseline_probability, patientData: v.patient, bestAchievable: cf.best_achievable ?? null, message: cf.message ?? null }) };
}
console.log(JSON.stringify(out, null, 1));
