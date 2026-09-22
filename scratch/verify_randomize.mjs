/**
 * Check the Randomize button's output: valid against the schema, plausible
 * against clinical bands, and accepted by the real /predict endpoint.
 *
 *   node scratch/verify_randomize.mjs [baseUrl]
 *
 * Randomize used to sample uniformly across each field's SCHEMA range, which
 * is a validation bound and not a clinical distribution -- a quarter of
 * randomised patients had a resting BP over 180. There is no JS test runner
 * in this project, so this script is the repeatable check.
 *
 * Exit code 0 only if every assertion holds.
 */

import { parseSchema } from '../frontend/src/utils/schemaFieldParser.js';
import { randomPatient, profileFor } from '../frontend/src/utils/randomPatient.js';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8000';
const N = 500;

let failures = 0;
function check(ok, message) {
  if (!ok) { console.log(`  FAIL  ${message}`); failures++; }
}

for (const disease of ['heart_disease', 'diabetes']) {
  console.log(`\n${disease}`);
  const schema = await (await fetch(`${BASE}/api/v4/${disease}/schema`)).json();
  const fields = parseSchema(schema);

  const stats = {};
  const patients = [];
  for (let i = 0; i < N; i++) {
    const p = randomPatient(fields, disease);
    patients.push(p);
    for (const f of fields) {
      const v = p[f.name];
      const { minimum, maximum, enum: enumValues } = f.validation || {};

      // 1. valid against the schema
      if (enumValues?.length && f.component === 'select') {
        check(enumValues.includes(v), `${f.name}: ${v} is not one of ${enumValues}`);
      } else if (typeof v === 'number') {
        if (minimum !== undefined) check(v >= minimum, `${f.name}: ${v} < minimum ${minimum}`);
        if (maximum !== undefined) check(v <= maximum, `${f.name}: ${v} > maximum ${maximum}`);
        check(Number.isFinite(v), `${f.name}: ${v} is not finite`);
        stats[f.name] ??= { min: Infinity, max: -Infinity, sum: 0, n: 0 };
        const s = stats[f.name];
        s.min = Math.min(s.min, v); s.max = Math.max(s.max, v); s.sum += v; s.n++;
      }
    }
  }

  // 2. plausible against the declared clinical band
  for (const [name, s] of Object.entries(stats)) {
    const profile = profileFor(disease, name);
    const mean = s.sum / s.n;
    if (profile) {
      check(s.min >= profile.min - 1e-9, `${name}: sampled ${s.min} below band min ${profile.min}`);
      check(s.max <= profile.max + 1e-9, `${name}: sampled ${s.max} above band max ${profile.max}`);
      check(mean > profile.min && mean < profile.max, `${name}: mean ${mean} outside band`);
    }
    console.log(`  ${name.padEnd(22)} min=${s.min.toFixed(1).padStart(6)}  mean=${mean.toFixed(1).padStart(6)}  max=${s.max.toFixed(1).padStart(6)}${profile ? `   band ${profile.min}-${profile.max}` : ''}`);
  }

  // 3. the server accepts them -- a 422 here means Randomize produces input
  //    the product itself rejects, which is what a judge would hit.
  // /predict is rate limited (30/min per IP), and a 429 says nothing about
  // whether the patient was valid -- only a 422 does. They are counted
  // separately so a throttled run is not reported as a validation failure.
  let accepted = 0, invalid = 0, throttled = 0;
  for (const p of patients.slice(0, 12)) {
    const res = await fetch(`${BASE}/api/v4/${disease}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    });
    if (res.ok) { accepted++; continue; }
    if (res.status === 429) { throttled++; continue; }
    invalid++;
    if (invalid === 1) console.log(`  FIRST REJECTION (${res.status}): ${(await res.text()).slice(0, 300)}`);
  }
  check(invalid === 0, `${invalid} randomised patients were rejected as invalid by /predict`);
  check(accepted > 0, 'every /predict call was throttled — nothing was actually verified');
  console.log(`  /predict: ${accepted} accepted, ${invalid} invalid, ${throttled} throttled`);
  // Stay under the per-minute limit before the next disease.
  await new Promise((r) => setTimeout(r, 3000));
}

console.log(failures === 0 ? '\nPASS — all assertions held' : `\nFAIL — ${failures} assertion(s)`);
process.exit(failures === 0 ? 0 : 1);
