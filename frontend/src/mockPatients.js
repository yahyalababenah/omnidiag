/**
 * Pre-defined mock patients for the Clinical EMR Mode.
 * Restructured for multi-disease support — keyed by disease name.
 *
 * Each patient has realistic vitals/data matching the disease's schema fields,
 * plus clinical meta-fields (history, medications, admittingComplaint).
 *
 * The BRFSS diabetes demo set (D-001..D-004) was removed with that module
 * (retired; gate B6). backend/demo_patients.json mirrors this file.
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

  // ── Dysglycaemia screening (NHANES, Phase 9) ────────────────────────────
  // One patient per conformal decision. Every decision and probability below was
  // measured 2026-09-28 by calling DiabetesEbmConformalBackend.predict() locally
  // on exactly these inputs (configs/diabetes_nhanes.yaml, alpha 0.20). The
  // probability is Platt-calibrated on NHANES 2015-2016 and is NOT comparable
  // with the retired BRFSS module's.
  //
  //   N-001 Lina Haddad      28 F   p=0.036  no_referral   Low
  //   N-002 Rami Nasser      45 M   p=0.349  uncertain     Medium
  //   N-003 Waleed Khoury    58 M   p=0.646  referral      High
  //
  // All three were re-measured through DiabetesNhanesInput, so the values are the
  // ones the API accepts. Sex is coded 1 = male, 0 = female in this schema (not the
  // NHANES 1/2 code); an earlier N-001 measurement used 2 and skipped the schema.
  //
  // Do not read decisions off the probabilities across patients. The conformal
  // layer is conditioned on AGE BAND (D9-05), so the same probability can be a
  // referral in one band and a no_referral in another. That is by design, not an
  // inconsistency.
  //
  // N-001's no_referral is the case the module's safety_note exists for: at this
  // operating point 14.1% of genuinely dysglycaemic patients receive it.
  // N-002 is the one seeded into the review queue on startup (uncertain).
  //
  // ADIPOSITY_BAND is the clinician-facing level ("normal" | "increased" |
  // "high"), not an integer; the six mandatory fields are always present
  // because the module refuses a row with any of them blank (D9-06).
  diabetes_nhanes: [
    {
      id: 'N-001',
      name: 'Lina Haddad',
      age: 28,
      sex: 'F',
      avatar: 'LH',
      history: 'No known conditions, non-smoker, exercises regularly, no family history of diabetes',
      medications: 'None',
      admittingComplaint: 'Routine health check for a new employer',
      data: {
        RIDAGEYR: 28,
        RIAGENDR: 0,
        BMXBMI: 22,
        ADIPOSITY_BAND: 'normal',
        SBP: 108,
        DBP: 68,
        BPXPLS: 70,
        MCQ300C: 0,
        CVD_ANY: 0,
        PAQ650: 1,
        PAQ665: 1,
        LBDHDD: 66,
        LBXSCH: 185,
        LBXSTR: 70,
        LBXSATSI: 16,
        LBXSGTSI: 14,
        LBXSCR: 0.85,
        LBXSBU: 13,
        LBXSAL: 4.3,
        LBXSUA: 5.0,
      },
    },
    {
      id: 'N-002',
      name: 'Rami Nasser',
      age: 45,
      sex: 'M',
      avatar: 'RN',
      history: 'Office worker, weight gain over five years, no vigorous activity, no diagnosed conditions',
      medications: 'None',
      admittingComplaint: 'Occasional fatigue; wants a diabetes check',
      data: {
        RIDAGEYR: 45,
        RIAGENDR: 1,
        BMXBMI: 30,
        ADIPOSITY_BAND: 'high',
        SBP: 126,
        DBP: 80,
        BPXPLS: 72,
        MCQ300C: 0,
        CVD_ANY: 0,
        PAQ650: 0,
        PAQ665: 1,
        LBDHDD: 46,
        LBXSCH: 195,
        LBXSTR: 140,
        LBXSATSI: 26,
        LBXSGTSI: 30,
        LBXSCR: 0.9,
        LBXSBU: 14,
        LBXSAL: 4.3,
        LBXSUA: 5.8,
      },
    },
    {
      id: 'N-003',
      name: 'Waleed Khoury',
      age: 58,
      sex: 'M',
      avatar: 'WK',
      history: 'Obesity, sedentary, parent with type 2 diabetes, borderline blood pressure',
      medications: 'None',
      admittingComplaint: 'Increased thirst and tiredness over recent months',
      data: {
        RIDAGEYR: 58,
        RIAGENDR: 1,
        BMXBMI: 33,
        ADIPOSITY_BAND: 'high',
        SBP: 138,
        DBP: 84,
        BPXPLS: 70,
        MCQ300C: 1,
        CVD_ANY: 0,
        PAQ650: 0,
        PAQ665: 0,
        LBDHDD: 38,
        LBXSCH: 185,
        LBXSTR: 240,
        LBXSATSI: 36,
        LBXSGTSI: 52,
        LBXSCR: 0.85,
        LBXSBU: 13,
        LBXSAL: 4.3,
        LBXSUA: 6.8,
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
