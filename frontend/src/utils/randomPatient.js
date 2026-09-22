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
 * what a number means. `Age` is years in the UCI heart cohort and a BRFSS
 * 5-year BAND (1 = 18-24 … 13 = 80+) in the diabetes module. A single
 * name-keyed table gave diabetes the heart band, which clamped to the
 * schema maximum and made every randomised diabetes patient Age = 13 —
 * constant, and the oldest bracket there is.
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
  diabetes: {
    Age: { min: 4, typical: 8, max: 12 },              // BRFSS band: 35-39 … 75-79
    BMI: { min: 19, typical: 28, max: 42 },
    GenHlth: { min: 1, typical: 3, max: 5 },
    MentHlth: { min: 0, typical: 2, max: 20 },
    PhysHlth: { min: 0, typical: 3, max: 20 },
    Education: { min: 3, typical: 5, max: 6 },
    Income: { min: 2, typical: 5, max: 8 },
  },
};

/** The band for one field of one disease, or undefined. */
export function profileFor(disease, fieldName) {
  return CLINICAL_PROFILES[disease]?.[fieldName];
}

/**
 * Probability that a binary field is 1, so a randomised patient does not
 * come out with every risk factor flipped on by a coin toss. Roughly
 * BRFSS-adult prevalences; anything unlisted falls back to 0.5.
 */
export const BINARY_PREVALENCE = {
  HighBP: 0.35,
  HighChol: 0.35,
  CholCheck: 0.95,
  Smoker: 0.4,
  Stroke: 0.05,
  HeartDiseaseorAttack: 0.08,
  PhysActivity: 0.7,
  Fruits: 0.6,
  Veggies: 0.8,
  HvyAlcoholConsump: 0.06,
  AnyHealthcare: 0.9,
  NoDocbcCost: 0.1,
  DiffWalk: 0.15,
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
