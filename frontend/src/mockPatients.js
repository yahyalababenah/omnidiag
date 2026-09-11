/**
 * Pre-defined mock patients for the Clinical EMR Mode.
 * Restructured for multi-disease support — keyed by disease name.
 *
 * Each patient has realistic vitals/data matching the disease's schema fields,
 * plus clinical meta-fields (history, medications, admittingComplaint).
 *
 * Diabetes profiles (4 total — the A/B/C/D demo set, regenerated 2026-09-11
 * after the ensemble-model-mismatch fix; verified live against the deployed
 * Space's EnsembleModelLoader.predict(), so the printed probabilities below
 * are actual production output, not estimates):
 *   D-001 Noor Sabbagh (NEGATIVE, 9.6%)         — Case A: no real risk factors, healthy baseline
 *   D-002 Karim Yaghi (POSITIVE, 43.2%)         — Case B: one risk factor (HighBP) + fair self-rated health
 *   D-003 Samir Abu-Ghazaleh (POSITIVE, 75.8%)  — Case C: severe but fully mutable risk profile
 *                                                  (obesity, hypertension, hyperlipidemia, active
 *                                                  smoking, sedentary, fair GenHlth, DiffWalk)
 *   D-004 Hala Mansour (POSITIVE, 33.8%)        — Case D: hypertension only, otherwise clean —
 *                                                  deliberately borderline, comfortably above the 0.275
 *                                                  clinical threshold (not flush against it — LightGBM's
 *                                                  output varies slightly by inference environment, so a
 *                                                  case placed right at the edge could flip sides)
 *
 * Case C note: HeartDiseaseorAttack and Stroke are deliberately 0 (Negative).
 * Both are IMMUTABLE_FEATURES in counterfactual_generator.py — the What-If
 * engine never proposes changing past medical history — so a patient whose
 * severity depends on either one is structurally unreachable: no combination
 * of the remaining mutable features (BMI, HighBP, HighChol, Smoker,
 * PhysActivity, GenHlth, MentHlth, PhysHlth, DiffWalk) can flip the
 * prediction, and the endpoint returns zero valid scenarios no matter how
 * large n_samples is. Case C's severity comes entirely from mutable factors
 * instead, so What-If has real, actionable ground to work with.
 *
 * Case C stability caveat: the number of valid counterfactuals returned for
 * this patient has been observed to vary between 1 and 3 across different
 * container builds of the live Space, with the *baseline* confidence
 * (75.8%) identical bit-for-bit every time. The cause is inference jitter
 * (floating-point, likely multi-threaded XGBoost/LightGBM/RandomForest) that
 * is not controlled by CounterfactualGenerator's fixed random_state=42 — a
 * borderline candidate can land a hair on either side of the 0.275 decision
 * threshold depending on the container instance, even for identical input.
 * This patient was pushed as low-severity as practical to minimize that
 * risk, but it cannot be eliminated by patient-data tuning alone; a true
 * fix would require either ensembling/averaging repeated model calls inside
 * the generator or pinning single-threaded, deterministic inference.
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
    // ── Case A — clear negative baseline (verified live on Space: 9.6%, Negative) ──
    // No real risk factors: no hypertension, no high cholesterol, normal BMI,
    // good self-rated health, active, non-smoker. Regenerated 2026-09-11 after
    // the ensemble-model-mismatch fix (see mockPatients.js history).
    {
      id: 'D-001',
      name: 'Noor Sabbagh',
      age: 37,
      sex: 'F',
      avatar: 'NS',
      history: 'No significant medical history, non-smoker, physically active',
      medications: 'None',
      admittingComplaint: 'Routine annual check-up — no complaints',
      data: {
        HighBP: 0,
        HighChol: 0,
        CholCheck: 1,
        BMI: 24,
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
        Sex: 0,
        Age: 4,
        Education: 6,
        Income: 7,
      },
    },
    // ── Case B — moderate positive (verified live on Space: 43.2%, Positive) ──
    // One real risk factor (hypertension) plus only fair self-rated health and
    // a couple of recent unwell days — a "watch and treat" case rather than an
    // alarming one. No cholesterol issue and otherwise active.
    {
      id: 'D-002',
      name: 'Karim Yaghi',
      age: 48,
      sex: 'M',
      avatar: 'KY',
      history: 'Hypertension diagnosed two years ago, otherwise active, fair general health',
      medications: 'Losartan 50mg',
      admittingComplaint: 'Follow-up visit for blood pressure management',
      data: {
        HighBP: 1,
        HighChol: 0,
        CholCheck: 1,
        BMI: 29,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 1,
        Fruits: 1,
        Veggies: 1,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 3,
        MentHlth: 0,
        PhysHlth: 2,
        DiffWalk: 0,
        Sex: 1,
        Age: 6,
        Education: 5,
        Income: 6,
      },
    },
    // ── Case C — strong positive (verified live on Space: 75.8%, Positive) ──
    // Severity comes entirely from mutable risk factors: obesity (BMI 33),
    // hypertension, hyperlipidemia, active smoking, sedentary lifestyle,
    // fair self-rated general health, and difficulty walking. No prior
    // MI/stroke — see the "Case C note" at the top of this file for why
    // that matters for What-If.
    //
    // Tuning history: an earlier BMI-38 draft (86.5%) gave 0/3 valid
    // counterfactuals (too extreme to flip at n_samples=100); a BMI-34 draft
    // (81.1%) gave 3/3 on one deterministic live call but only 1/3 after
    // the next container rebuild, with the baseline confidence identical
    // bit-for-bit both times — see the "Case C stability caveat" at the top
    // of this file. This BMI-33 profile was pushed as low-severity as
    // practical to sit further from the 0.275 threshold, but per that
    // caveat, the same 1-to-3 variance cannot be ruled out here either —
    // it was observed to give 3/3 on its one live test.
    // Age band 9 = 60–64 (BRFSS coding).
    {
      id: 'D-003',
      name: 'Samir Abu-Ghazaleh',
      age: 62,
      sex: 'M',
      avatar: 'SA',
      history: 'Type 2 Diabetes risk profile: obesity (BMI 33), hypertension, ' +
        'hyperlipidemia, active smoker, sedentary lifestyle, fair general health, ' +
        'poor mobility',
      medications: 'Amlodipine 10mg, Rosuvastatin 20mg',
      admittingComplaint: 'Fatigue and difficulty walking, worsening over recent weeks',
      data: {
        HighBP: 1,
        HighChol: 1,
        CholCheck: 1,
        BMI: 33,
        Smoker: 1,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 0,
        Fruits: 0,
        Veggies: 0,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 4,
        MentHlth: 5,
        PhysHlth: 12,
        DiffWalk: 1,
        Sex: 1,
        Age: 9,
        Education: 4,
        Income: 4,
      },
    },
    // ── Case D — deliberately borderline (verified live on Space: 33.8%, Positive) ──
    // Hypertension only, everything else clean: no high cholesterol, active,
    // good diet, normal-to-mildly-elevated BMI. Lands comfortably above the
    // 0.275 clinical threshold (margin ≈ +0.06) rather than flush against it —
    // LightGBM's output shifts a few points between inference environments,
    // so a case placed right at the edge could flip Positive/Negative on
    // redeploy. Pairs with Case A to show "one risk factor is sometimes
    // just enough", without being fragile to reproduce.
    {
      id: 'D-004',
      name: 'Hala Mansour',
      age: 47,
      sex: 'F',
      avatar: 'HM',
      history: 'Hypertension found at a routine visit, otherwise healthy and active',
      medications: 'Amlodipine 5mg (started recently)',
      admittingComplaint: 'Routine check-up flagged elevated blood pressure',
      data: {
        HighBP: 1,
        HighChol: 0,
        CholCheck: 1,
        BMI: 26,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 1,
        Fruits: 1,
        Veggies: 1,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 3,
        MentHlth: 0,
        PhysHlth: 0,
        DiffWalk: 0,
        Sex: 0,
        Age: 6,
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
