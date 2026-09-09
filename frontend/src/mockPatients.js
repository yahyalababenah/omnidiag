/**
 * Pre-defined mock patients for the Clinical EMR Mode.
 * Restructured for multi-disease support — keyed by disease name.
 *
 * Each patient has realistic vitals/data matching the disease's schema fields,
 * plus clinical meta-fields (history, medications, admittingComplaint).
 *
 * Diabetes profiles (4 total — the A/B/C/D demo set, verified live against
 * EnsembleModelLoader.predict()/.explain() so the printed probabilities
 * below are actual model output, not estimates):
 *   D-001 Dana Al-Amin (NEGATIVE, 23.6%)     — Case A: no risk factors, healthy baseline
 *   D-002 Bilal Hourani (POSITIVE, 44.0%)    — Case B: one isolated risk factor (HighBP), otherwise fit
 *   D-003 Fadi Boutros (POSITIVE, 90.0%)     — Case C: multiple compounding risk factors
 *   D-004 Rania Saad (POSITIVE, 29.3%)       — Case D: same healthy baseline as Dana + isolated HighBP —
 *                                               deliberately borderline, just above the 0.275 clinical threshold
 */
const mockPatients = {
  heart_disease: [
    {
      id: 'P-001',
      name: 'Ahmed Al-Rashid',
      age: 54,
      sex: 'M',
      avatar: 'AR',
      history: 'Hypertension (10 yrs), Type 2 Diabetes, Family history of CAD',
      medications: 'Lisinopril 10mg, Metformin 500mg, Atorvastatin 20mg',
      admittingComplaint: 'Chest tightness on exertion for 2 weeks',
      data: {
        Age: 54,
        Sex: 'M',
        ChestPainType: 'ATA',
        RestingBP: 140,
        Cholesterol: 289,
        FastingBS: 0,
        RestingECG: 'Normal',
        MaxHR: 122,
        ExerciseAngina: 'N',
        Oldpeak: 0.0,
        ST_Slope: 'Flat',
      },
    },
    {
      id: 'P-002',
      name: 'Fatima Hassan',
      age: 62,
      sex: 'F',
      avatar: 'FH',
      history: 'Dyslipidemia, Obesity (BMI 32), Post-menopausal',
      medications: 'Rosuvastatin 10mg, Aspirin 81mg',
      admittingComplaint: 'Shortness of breath and palpitations',
      data: {
        Age: 62,
        Sex: 'F',
        ChestPainType: 'ASY',
        RestingBP: 158,
        Cholesterol: 340,
        FastingBS: 1,
        RestingECG: 'LVH',
        MaxHR: 98,
        ExerciseAngina: 'Y',
        Oldpeak: 2.3,
        ST_Slope: 'Down',
      },
    },
    {
      id: 'P-003',
      name: 'Khalid Othman',
      age: 45,
      sex: 'M',
      avatar: 'KO',
      history: 'No significant history, Active smoker (20 pack-years)',
      medications: 'None',
      admittingComplaint: 'Routine check-up, occasional dizziness',
      data: {
        Age: 45,
        Sex: 'M',
        ChestPainType: 'NAP',
        RestingBP: 120,
        Cholesterol: 210,
        FastingBS: 0,
        RestingECG: 'Normal',
        MaxHR: 160,
        ExerciseAngina: 'N',
        Oldpeak: 0.5,
        ST_Slope: 'Up',
      },
    },
  ],

  diabetes: [
    // ── Case A — clear negative baseline (verified: 23.6%, Negative) ──
    // No risk factors present anywhere in the input: normal BMI, youngest
    // age band, excellent self-rated health, active, non-smoker.
    {
      id: 'D-001',
      name: 'Dana Al-Amin',
      age: 22,
      sex: 'F',
      avatar: 'DA',
      history: 'No significant medical history, non-smoker, physically active',
      medications: 'None',
      admittingComplaint: 'Routine annual check-up — no complaints',
      data: {
        HighBP: 0,
        HighChol: 0,
        CholCheck: 1,
        BMI: 21,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 1,
        Fruits: 1,
        Veggies: 1,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 1,
        MentHlth: 0,
        PhysHlth: 0,
        DiffWalk: 0,
        Sex: 0,
        Age: 1,
        Education: 5,
        Income: 6,
      },
    },
    // ── Case B — moderate positive (verified: 44.0%, Positive) ──
    // Exactly one real risk factor (HighBP, newly found) against an
    // otherwise fit, active, slightly-overweight profile — a "watch and
    // treat" case rather than an alarming one.
    {
      id: 'D-002',
      name: 'Bilal Hourani',
      age: 27,
      sex: 'M',
      avatar: 'BH',
      history: 'Hypertension diagnosed this year, otherwise healthy, physically active',
      medications: 'Amlodipine 5mg (started 2 months ago)',
      admittingComplaint: 'Follow-up visit for newly diagnosed high blood pressure',
      data: {
        HighBP: 1,
        HighChol: 0,
        CholCheck: 1,
        BMI: 27,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 1,
        Fruits: 1,
        Veggies: 1,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 2,
        MentHlth: 0,
        PhysHlth: 0,
        DiffWalk: 0,
        Sex: 1,
        Age: 2,
        Education: 4,
        Income: 5,
      },
    },
    // ── Case C — strong positive (verified: 90.0%, Positive) ──
    // Multiple compounding risk factors: obesity, hypertension,
    // hyperlipidemia, active smoking, prior MI, poor self-rated health,
    // and difficulty walking. Age band 11 = 70–74 (BRFSS coding).
    {
      id: 'D-003',
      name: 'Fadi Boutros',
      age: 72,
      sex: 'M',
      avatar: 'FB',
      history: 'Type 2 Diabetes risk profile: prior MI, hypertension, hyperlipidemia, ' +
        '40-yr smoking history, obesity (BMI 38), poor mobility',
      medications: 'Lisinopril 20mg, Atorvastatin 40mg, Aspirin 81mg',
      admittingComplaint: 'Chest discomfort on exertion, worsening fatigue, difficulty walking',
      data: {
        HighBP: 1,
        HighChol: 1,
        CholCheck: 1,
        BMI: 38,
        Smoker: 1,
        Stroke: 0,
        HeartDiseaseorAttack: 1,
        PhysActivity: 0,
        Fruits: 0,
        Veggies: 0,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 5,
        MentHlth: 5,
        PhysHlth: 15,
        DiffWalk: 1,
        Sex: 1,
        Age: 11,
        Education: 3,
        Income: 3,
      },
    },
    // ── Case D — deliberately borderline (verified: 29.3%, Positive) ──
    // Identical healthy baseline to Case A (Dana) with one isolated
    // HighBP added — crosses the 0.275 clinical threshold by a narrow
    // margin. Pairs with Case A to teach "one risk factor is sometimes
    // just enough". NOTE: entropy at this probability (~0.87) sits just
    // under the 0.88 active-learning review threshold, so this case does
    // NOT get auto-queued for review despite being probability-borderline
    // — a useful contrast to point out in a demo.
    {
      id: 'D-004',
      name: 'Rania Saad',
      age: 26,
      sex: 'F',
      avatar: 'RS',
      history: 'Hypertension found at a routine visit, otherwise healthy and active',
      medications: 'None yet — lifestyle modification advised',
      admittingComplaint: 'Routine check-up flagged elevated blood pressure',
      data: {
        HighBP: 1,
        HighChol: 0,
        CholCheck: 1,
        BMI: 21,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 1,
        Fruits: 1,
        Veggies: 1,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 1,
        MentHlth: 0,
        PhysHlth: 0,
        DiffWalk: 0,
        Sex: 0,
        Age: 2,
        Education: 5,
        Income: 6,
      },
    },
  ],
};

/**
 * Get all patients for a given disease.
 * Falls back to an empty array if none are defined.
 */
export function getPatientsForDisease(diseaseName) {
  return mockPatients[diseaseName] || [];
}

export default mockPatients;
