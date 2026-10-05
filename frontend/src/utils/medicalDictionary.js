/**
 * medicalDictionary.js
 * ======================
 * Maps medical feature names and common display labels to plain-English
 * clinical descriptions for interactive tooltips.
 *
 * The dictionary supports lookup by:
 *   - Raw field name (e.g., "FastingBS", "MaxHR")
 *   - Derived title (e.g., "Fasting Blood Sugar", "Maximum Heart Rate")
 *
 * Each entry now includes the value scale/range inline, so the existing
 * MedicalTooltip component shows clinicians both the definition AND the
 * valid input range on hover.
 *
 * Add new entries here when new disease features are introduced.
 */

const MEDICAL_DICTIONARY = {
  // ── Shared names ──

  // One entry serves heart's field name and NHANES's label ("Sex" for RIAGENDR),
  // and the two code it differently, so it states both.
  Sex:
    'Recorded sex — used as a demographic covariate in the risk model. '
    + 'Coding: M / F in the heart module; 1 = male, 0 = female in the NHANES module.',
  Age:
    'Patient age in years — a primary risk factor for coronary artery disease.',

  // ── Heart Disease features ──

  RPP:
    'Rate Pressure Product (RPP) = Heart Rate × Systolic Blood Pressure. '
    + 'Measures myocardial oxygen consumption.',
  FastingBS:
    'Fasting Blood Sugar — Glucose level after an overnight fast; key for metabolic assessment. '
    + 'Scale: 0 = Normal (<120 mg/dL), 1 = Elevated (≥120 mg/dL).',
  'Fasting Blood Sugar':
    'Fasting Blood Sugar — Glucose level after an overnight fast; key for metabolic assessment. '
    + 'Scale: 0 = Normal (<120 mg/dL), 1 = Elevated (≥120 mg/dL).',
  MaxHR:
    'Maximum Heart Rate — Highest heart rate achieved during clinical assessment. '
    + 'Range: 60–220 bpm.',
  'Max HR':
    'Maximum Heart Rate — Highest heart rate achieved during clinical assessment. '
    + 'Range: 60–220 bpm.',
  'Maximum Heart Rate':
    'Maximum Heart Rate — Highest heart rate achieved during clinical assessment. '
    + 'Range: 60–220 bpm.',
  ExerciseAngina:
    'Exercise-Induced Angina — Chest pain brought on by physical exertion. '
    + 'Scale: 0 = No angina, 1 = Exercise-induced angina.',
  'Exercise Angina':
    'Exercise-Induced Angina — Chest pain brought on by physical exertion. '
    + 'Scale: 0 = No angina, 1 = Exercise-induced angina.',
  'Exercise-Induced Angina':
    'Exercise-Induced Angina — Chest pain brought on by physical exertion. '
    + 'Scale: 0 = No angina, 1 = Exercise-induced angina.',
  RestingBP:
    'Resting Blood Pressure — Systolic blood pressure measured at rest. '
    + 'Range: 80–220 mmHg.',
  'Resting BP':
    'Resting Blood Pressure — Systolic blood pressure measured at rest. '
    + 'Range: 80–220 mmHg.',
  'Resting Blood Pressure':
    'Resting Blood Pressure — Systolic blood pressure measured at rest. '
    + 'Range: 80–220 mmHg.',
  Cholesterol:
    'Serum total cholesterol level — includes LDL, HDL, and other lipid components. '
    + 'Range: 100–600 mg/dL.',
  Oldpeak:
    'ST depression induced by exercise relative to rest — indicates exercise-induced ischemia. '
    + 'Range: 0.0–6.0.',
  ST_Slope:
    'Slope of the ST segment during peak exercise — a key ECG marker for coronary artery disease. '
    + 'Values: Up, Flat, Down.',
  'ST Slope':
    'Slope of the ST segment during peak exercise — a key ECG marker for coronary artery disease. '
    + 'Values: Up, Flat, Down.',
  RestingECG:
    'Resting Electrocardiogram results. '
    + 'Values: Normal, ST-T wave abnormality, Left ventricular hypertrophy.',
  'Resting ECG':
    'Resting Electrocardiogram results. '
    + 'Values: Normal, ST-T wave abnormality, Left ventricular hypertrophy.',
  'Resting Electrocardiogram':
    'Resting Electrocardiogram results. '
    + 'Values: Normal, ST-T wave abnormality, Left ventricular hypertrophy.',
  ChestPainType:
    'Type of chest pain experienced. '
    + 'Values: Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic.',
  'Chest Pain Type':
    'Type of chest pain experienced. '
    + 'Values: Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic.',
};

/**
 * Look up a clinical description for a given text string.
 * Performs case-insensitive matching against the dictionary.
 *
 * @param {string} text - The feature name or title to look up.
 * @returns {string|undefined} The clinical description, or undefined if not found.
 */
export function lookupMedicalTerm(text) {
  if (!text || typeof text !== 'string') return undefined;
  const trimmed = text.trim();
  // Direct lookup
  if (MEDICAL_DICTIONARY[trimmed]) return MEDICAL_DICTIONARY[trimmed];
  // Case-insensitive fallback
  const lower = trimmed.toLowerCase();
  for (const [key, value] of Object.entries(MEDICAL_DICTIONARY)) {
    if (key.toLowerCase() === lower) return value;
  }
  return undefined;
}

export default MEDICAL_DICTIONARY;
