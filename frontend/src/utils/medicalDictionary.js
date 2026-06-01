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
 * Each entry now includes the value scale/range inline, so the existing
 * MedicalTooltip component shows clinicians both the definition AND the
 * valid input range on hover.
 *
 * Add new entries here when new disease features are introduced.
 */

const MEDICAL_DICTIONARY = {
  // ── Diabetes / CDC BRFSS 2015 features ──

  HighBP:
    'High Blood Pressure (Hypertension) — Indicator of chronic cardiovascular strain. '
    + 'Scale: 0 = No high BP, 1 = High BP.',
  'High blood pressure':
    'High Blood Pressure (Hypertension) — Indicator of chronic cardiovascular strain. '
    + 'Scale: 0 = No high BP, 1 = High BP.',
  HighChol:
    'High Cholesterol — Elevated blood lipid levels, increasing plaque risk. '
    + 'Scale: 0 = No high cholesterol, 1 = High cholesterol.',
  'High cholesterol':
    'High Cholesterol — Elevated blood lipid levels, increasing plaque risk. '
    + 'Scale: 0 = No high cholesterol, 1 = High cholesterol.',
  BMI:
    'Body Mass Index (BMI) — A statistical measurement of body weight relative to height. '
    + 'Range: 10.0–100.0. WHO cut-offs: <18.5 underweight, 18.5–24.9 normal, 25–29.9 overweight, ≥30 obese.',
  'Body Mass Index':
    'Body Mass Index (BMI) — A statistical measurement of body weight relative to height. '
    + 'Range: 10.0–100.0. WHO cut-offs: <18.5 underweight, 18.5–24.9 normal, 25–29.9 overweight, ≥30 obese.',
  CholCheck:
    'Cholesterol Check — Has had a blood cholesterol test within the past 5 years. '
    + 'Scale: 0 = Not checked, 1 = Checked.',
  Smoker:
    'Smoker — Smoked at least 100 cigarettes over lifetime. '
    + 'Scale: 0 = Non-smoker, 1 = Smoker.',
  Stroke:
    'Stroke History — Ever told by a physician that they had a stroke. '
    + 'Scale: 0 = No stroke, 1 = Stroke history.',
  HeartDiseaseorAttack:
    'Heart Disease or MI — Diagnosed with coronary heart disease or myocardial infarction. '
    + 'Scale: 0 = No CHD/MI, 1 = History of CHD or MI.',
  PhysActivity:
    'Physical Activity — Performed any physical activity in the past 30 days (non-job). '
    + 'Scale: 0 = No activity, 1 = Any activity.',
  Fruits:
    'Fruit Consumption — Consumes fruit one or more times per day. '
    + 'Scale: 0 = Less than 1/day, 1 = 1+ servings/day.',
  Veggies:
    'Vegetable Consumption — Consumes vegetables one or more times per day. '
    + 'Scale: 0 = Less than 1/day, 1 = 1+ servings/day.',
  HvyAlcoholConsump:
    'Heavy Alcohol Consumption — >14 drinks/week (men) or >7 drinks/week (women). '
    + 'Scale: 0 = Not heavy drinker, 1 = Heavy drinker.',
  AnyHealthcare:
    'Healthcare Coverage — Has any form of health insurance. '
    + 'Scale: 0 = No coverage, 1 = Any coverage.',
  NoDocbcCost:
    'Could Not See Doctor Due to Cost — Needed care but could not afford it in past 12 months. '
    + 'Scale: 0 = Could see doctor, 1 = Could not due to cost.',
  GenHlth:
    'General Health Rating — Self-reported or clinically assessed overall health. '
    + 'Scale: 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor.',
  'General Health':
    'General Health Rating — Self-reported or clinically assessed overall health. '
    + 'Scale: 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor.',
  MentHlth:
    'Mental Health Days — Number of days in the past 30 days with poor mental health or stress. '
    + 'Range: 0–30 days.',
  'Mental Health':
    'Mental Health Days — Number of days in the past 30 days with poor mental health or stress. '
    + 'Range: 0–30 days.',
  PhysHlth:
    'Physical Health Days — Number of days in the past 30 days with physical illness or injury. '
    + 'Range: 0–30 days.',
  'Physical Health':
    'Physical Health Days — Number of days in the past 30 days with physical illness or injury. '
    + 'Range: 0–30 days.',
  DiffWalk:
    'Difficulty Walking — Self-reported serious difficulty walking or climbing stairs. '
    + 'Scale: 0 = No difficulty, 1 = Serious difficulty.',
  'Difficulty Walking':
    'Difficulty Walking — Self-reported serious difficulty walking or climbing stairs. '
    + 'Scale: 0 = No difficulty, 1 = Serious difficulty.',
  Sex:
    'Biological sex assigned at birth — used as a demographic covariate in risk models. '
    + 'Scale: 0 = Female, 1 = Male.',
  Age:
    'Patient age category — a primary risk factor for cardiovascular and metabolic disease. '
    + 'BRFSS categories: 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+.',
  Education:
    'Education Level — Highest level of educational attainment. '
    + 'Scale: 1=None/Kindergarten, 2=Grades 1–8, 3=Grades 9–11, 4=HS Grad/GED, 5=Some College, 6=College Grad.',
  Income:
    'Annual Household Income — Bracket scale. '
    + 'Scale: 1=<$10K, 2=$10–15K, 3=$15–20K, 4=$20–25K, 5=$25–35K, 6=$35–50K, 7=$50–75K, 8=≥$75K.',

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

  // ── Engineered Features (computed server-side, appear in SHAP charts) ──

  Diabetes_Clinical_Risk:
    'Diabetes Clinical Risk — Engineered logarithmic risk index. '
    + 'Formula: exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5). '
    + 'HighBP contributes 50% weight — the single strongest modifiable risk factor. '
    + 'BMI contributes 5% per unit; each unit increase raises risk exponentially. '
    + 'Range: ~2.7 (young/healthy) to ~90+ (elderly, obese, hypertensive). '
    + 'This feature almost always dominates the SHAP explanation for diabetes predictions.',
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

/**
 * Return the full structured dictionary for use in reference components
 * (e.g., VariableScalesModal). Keys are feature names; values are strings
 * with embedded definition + scale/range.
 *
 * @returns {Record<string, string>} The complete MEDICAL_DICTIONARY map.
 */
export function getFullDictionary() {
  return { ...MEDICAL_DICTIONARY };
}

export default MEDICAL_DICTIONARY;
