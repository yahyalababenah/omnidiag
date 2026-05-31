/**
 * medicalDictionary.js
 * ======================
 * Maps medical feature names and common display labels to plain-English
 * clinical descriptions for interactive tooltips.
 *
 * The dictionary supports lookup by:
 *   - Raw field name (e.g., "HighBP", "BMI", "FastingBS")
 *   - Derived title (e.g., "High Blood Pressure", "Body Mass Index")
 *
 * Add new entries here when new disease features are introduced.
 */

const MEDICAL_DICTIONARY = {
  // ── Heart Disease features ──
  HighBP: 'High Blood Pressure (Hypertension) — Indicator of chronic cardiovascular strain.',
  'High blood pressure': 'High Blood Pressure (Hypertension) — Indicator of chronic cardiovascular strain.',
  HighChol: 'High Cholesterol — Elevated blood lipid levels, increasing plaque risk.',
  'High cholesterol': 'High Cholesterol — Elevated blood lipid levels, increasing plaque risk.',
  BMI: 'Body Mass Index (BMI) — A statistical measurement of body weight relative to height.',
  'Body Mass Index': 'Body Mass Index (BMI) — A statistical measurement of body weight relative to height.',
  RPP: 'Rate Pressure Product (RPP) = Heart Rate × Systolic Blood Pressure. Measures myocardial oxygen consumption.',
  FastingBS: 'Fasting Blood Sugar — Glucose level after an overnight fast; key for metabolic assessment.',
  'Fasting Blood Sugar': 'Fasting Blood Sugar — Glucose level after an overnight fast; key for metabolic assessment.',
  MaxHR: 'Maximum Heart Rate — Highest heart rate achieved during clinical assessment.',
  'Max HR': 'Maximum Heart Rate — Highest heart rate achieved during clinical assessment.',
  'Maximum Heart Rate': 'Maximum Heart Rate — Highest heart rate achieved during clinical assessment.',
  ExerciseAngina: 'Exercise-Induced Angina — Chest pain brought on by physical exertion.',
  'Exercise Angina': 'Exercise-Induced Angina — Chest pain brought on by physical exertion.',
  'Exercise-Induced Angina': 'Exercise-Induced Angina — Chest pain brought on by physical exertion.',

  // ── Diabetes / general health features ──
  GenHlth: 'General Health Rating — Self-reported or clinically assessed overall health scale.',
  'General Health': 'General Health Rating — Self-reported or clinically assessed overall health scale.',
  MentHlth: 'Mental Health Days — Number of days in the past 30 days with poor mental health or stress.',
  'Mental Health': 'Mental Health Days — Number of days in the past 30 days with poor mental health or stress.',
  PhysHlth: 'Physical Health Days — Number of days in the past 30 days with physical illness or injury.',
  'Physical Health': 'Physical Health Days — Number of days in the past 30 days with physical illness or injury.',
  DiffWalk: 'Difficulty Walking — Indicates limited physical mobility or difficulty climbing stairs.',
  'Difficulty Walking': 'Difficulty Walking — Indicates limited physical mobility or difficulty climbing stairs.',

  // ── Common vital / lab terms ──
  Age: 'Patient age at time of assessment — a primary risk factor for cardiovascular and metabolic disease.',
  Sex: 'Biological sex assigned at birth — used as a demographic covariate in risk models.',
  ChestPainType: 'Type of chest pain experienced — classified as Typical Angina, Atypical Angina, Non-Anginal Pain, or Asymptomatic.',
  'Chest Pain Type': 'Type of chest pain experienced — classified as Typical Angina, Atypical Angina, Non-Anginal Pain, or Asymptomatic.',
  RestingBP: 'Resting Blood Pressure — Systolic blood pressure measured at rest in mmHg.',
  'Resting BP': 'Resting Blood Pressure — Systolic blood pressure measured at rest in mmHg.',
  'Resting Blood Pressure': 'Resting Blood Pressure — Systolic blood pressure measured at rest in mmHg.',
  Cholesterol: 'Serum total cholesterol level in mg/dL — includes LDL, HDL, and other lipid components.',
  Oldpeak: 'ST depression induced by exercise relative to rest — indicates exercise-induced ischemia.',
  ST_Slope: 'Slope of the ST segment during peak exercise — a key ECG marker for coronary artery disease.',
  'ST Slope': 'Slope of the ST segment during peak exercise — a key ECG marker for coronary artery disease.',
  RestingECG: 'Resting Electrocardiogram results — classified as Normal, ST-T Wave Abnormality, or Left Ventricular Hypertrophy.',
  'Resting ECG': 'Resting Electrocardiogram results — classified as Normal, ST-T Wave Abnormality, or Left Ventricular Hypertrophy.',
  'Resting Electrocardiogram': 'Resting Electrocardiogram results — classified as Normal, ST-T Wave Abnormality, or Left Ventricular Hypertrophy.',
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

/**
 * Check if a text string matches a known medical term.
 *
 * @param {string} text
 * @returns {boolean}
 */
export function isKnownMedicalTerm(text) {
  return lookupMedicalTerm(text) !== undefined;
}

export default MEDICAL_DICTIONARY;
