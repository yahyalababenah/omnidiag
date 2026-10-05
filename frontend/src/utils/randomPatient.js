/**
 * randomPatient — plausible demo values for the Randomize button.
 *
 * Randomize used to sample each field uniformly across its whole SCHEMA
 * range. The schema range is a validation bound, not a clinical
 * distribution: RestingBP is allowed 80-220, so uniform sampling produced a
 * resting blood pressure over 180 about a quarter of the time, cholesterol
 * anywhere in 100-600, and a maximum heart rate picked independently of age.
 * The result was a patient no clinician would accept as real, shown to
 * judges as an example of what the product takes as input.
 *
 * Values here are sampled from a clinically plausible band inside the schema
 * range, centred on a typical value, and are then clamped and stepped back
 * onto the schema's own constraints — so a randomised patient is always
 * valid input AND always a believable person.
 *
 * These are demo inputs, not a population model. They are not used for
 * evaluation, calibration or any metric; nothing in evaluation_evidence/
 * depends on them.
 */

/**
 * Plausible band per numeric field: {min, typical, max}, all inside the
 * schema's own range. Sampling is triangular around `typical`, so values
 * near the middle are common and the extremes stay reachable but rare.
 *
 * Keyed BY DISEASE FIRST, because a bare field name is not enough to know
 * what a number means: the same name can carry a different unit in another
 * module. (The retired BRFSS module's `Age` was a 5-year band; a single
 * name-keyed table once gave it the heart band.)
 *
 * Sources are ordinary adult clinical reference ranges; the point is
 * believability, not precision.
 */
export const CLINICAL_PROFILES = {
  heart_disease: {
    Age: { min: 35, typical: 55, max: 78 },            // years
    RestingBP: { min: 100, typical: 130, max: 175 },   // mm Hg
    Cholesterol: { min: 150, typical: 220, max: 320 }, // mg/dL, total
    MaxHR: { min: 95, typical: 150, max: 190 },        // bpm
    Oldpeak: { min: 0, typical: 0.8, max: 4.0 },       // mm ST depression
  },
};

/** The band for one field of one disease, or undefined. */
export function profileFor(disease, fieldName) {
  return CLINICAL_PROFILES[disease]?.[fieldName];
}

/**
 * Probability that a binary field is 1, so a randomised patient does not
 * come out with every risk factor flipped on by a coin toss. Anything
 * unlisted falls back to 0.5.
 */
export const BINARY_PREVALENCE = {
  FastingBS: 0.2,
  Sex: 0.5,
};

/** Triangular sample on [min, max] with mode at `typical`. */
function triangular(min, typical, max, rnd = Math.random) {
  if (!(max > min)) return min;
  const mode = Math.min(Math.max(typical, min), max);
  const u = rnd();
  const c = (mode - min) / (max - min);
  return u < c
    ? min + Math.sqrt(u * (max - min) * (mode - min))
    : max - Math.sqrt((1 - u) * (max - min) * (max - mode));
}

/** Snap to the field's step and clamp to its schema bounds. */
function conform(value, field) {
  const { minimum, maximum, step } = field.validation || {};
  const s = step ?? (field.type === 'number' ? 0.1 : 1);
  let v = Math.round(value / s) * s;
  if (minimum !== undefined) v = Math.max(minimum, v);
  if (maximum !== undefined) v = Math.min(maximum, v);
  return s < 1 ? parseFloat(v.toFixed(10)) : Math.round(v);
}

/**
 * A plausible value for one field, guaranteed valid against its schema.
 *
 * `disease` selects which band table applies; `rnd` is injectable so checks
 * can drive the extremes deterministically.
 */
export function randomValueFor(field, disease, rnd = Math.random) {
  const { enum: enumValues, minimum, maximum } = field.validation || {};

  if (field.component === 'select' && enumValues?.length) {
    return enumValues[Math.floor(rnd() * enumValues.length)];
  }

  if (field.component === 'toggle' || field.component === 'segmented') {
    const p = BINARY_PREVALENCE[field.name] ?? 0.5;
    return rnd() < p ? 1 : 0;
  }

  if (field.type === 'number' || field.type === 'integer') {
    const schemaMin = minimum ?? 0;
    const schemaMax = maximum ?? 100;
    const profile = profileFor(disease, field.name);

    // Without a profile, stay in the middle half of the schema range rather
    // than reaching for its validation extremes.
    const band = profile ?? {
      min: schemaMin + (schemaMax - schemaMin) * 0.25,
      typical: (schemaMin + schemaMax) / 2,
      max: schemaMin + (schemaMax - schemaMin) * 0.75,
    };

    const lo = Math.max(band.min, schemaMin);
    const hi = Math.min(band.max, schemaMax);
    return conform(triangular(lo, band.typical, hi, rnd), field);
  }

  if (enumValues?.length) return enumValues[Math.floor(rnd() * enumValues.length)];
  return field.default ?? '';
}

/**
 * A whole plausible patient: {fieldName: value} for every field given.
 *
 * `disease` selects the band table — it is required for anything with a
 * profile, since the same field name means different things per module.
 */
export function randomPatient(fields, disease, rnd = Math.random) {
  const out = {};
  for (const field of fields) out[field.name] = randomValueFor(field, disease, rnd);
  return out;
}
