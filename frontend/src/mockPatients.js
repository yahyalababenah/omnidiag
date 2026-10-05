/**
 * Pre-defined mock patients for the Clinical EMR Mode.
 * Restructured for multi-disease support — keyed by disease name.
 *
 * Each patient has realistic vitals/data matching the disease's schema fields,
 * plus clinical meta-fields (history, medications, admittingComplaint).
 *
 * Diabetes profiles (4 total — the A/B/C/D demo set). Raw probabilities were
 * measured 2026-09-18 by calling EnsembleModelLoader.predict() locally on
 * exactly these inputs; re-verified unchanged 2026-09-21 after the deployment
 * prevalence decision below (raw values and the model itself never changed —
 * only the correction target did). configs/diabetes.yaml as committed:
 * inference_threshold 0.108184 on the deployment prior (raw equivalent
 * 0.280854), prevalence_train 0.50 → prevalence_deploy 0.237 (Jordan's
 * diabetes prevalence, per archive/post_expo_2026-10/docs/OmniDiag_Proposal_Defense.md — was ~0.14,
 * a US/BRFSS placeholder, until 2026-09-21). If prevalence_deploy changes
 * again, every "shown" value below changes with it; the raw values and
 * every Positive/Negative decision do not — confirmed on the full 14,139-row
 * test set: 0 decisions changed by the prevalence move alone (same property
 * that already held for 0.14; see evaluation_evidence/diabetes/before_after.json).
 *
 * Two scales, both listed — the UI shows the CORRECTED one:
 *                                   raw     shown (corrected)  decision
 *   D-001 Noor Sabbagh            0.0791   2.6%               NEGATIVE — Case A: no real risk factors
 *   D-002 Karim Yaghi             0.4352  19.3%               POSITIVE — Case B: HighBP + fair GenHlth
 *   D-003 Samir Abu-Ghazaleh      0.8498  63.7%               POSITIVE — Case C: severe, fully mutable
 *                                                                        (obesity, hypertension, hyperlipidemia,
 *                                                                        smoking, sedentary, fair GenHlth, DiffWalk)
 *   D-004 Hala Mansour            0.3509  14.4%               POSITIVE — Case D: hypertension only —
 *                                                                        deliberately borderline
 *
 * A Positive at 14–19% is correct, not a bug: the corrected probabilities
 * sit on a ~23.7% base rate and the decision threshold is 10.82%, not 50%.
 *
 * The earlier figures in this file (9.6 / 43.2 / 75.8 / 33.8%) were raw-scale
 * values measured on the live Space on 2026-09-11 against the old 0.275
 * threshold. They are superseded on two counts: the UI no longer shows the
 * raw scale, and the raw values themselves moved (Case C: 75.8% → 85.0% raw).
 * These numbers were measured locally, not on the Space; LightGBM output can
 * shift slightly between inference environments.
 *
 * What-If update 2026-09-21 (fix/whatif-clinical-validity): HighBP, HighChol,
 * Smoker, GenHlth, MentHlth, PhysHlth and DiffWalk are now IMMUTABLE in the
 * What-If engine (BRFSS records them as history or outcomes, not levers). Only
 * BMI (down, floor 18.5), PhysActivity/Fruits/Veggies (-> 1) and
 * HvyAlcoholConsump (-> 0) may change. Case C therefore no longer crosses the
 * threshold: /counterfactuals returns best_achievable (BMI 18.5 + activity +
 * fruit + veg: 63.7% -> 42.3%) and the UI recommends referral. The notes
 * below describe the earlier policy and are kept for history.
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
 * identical bit-for-bit every time. The cause is inference jitter
 * (floating-point, likely multi-threaded XGBoost/LightGBM/RandomForest) that
 * is not controlled by CounterfactualGenerator's fixed random_state=42 — a
 * borderline candidate can land a hair on either side of the decision
 * threshold (0.108184 corrected / 0.280855 raw) depending on the container
 * instance, even for identical input. Re-checked locally 2026-09-18: two
 * uncached calls for Case C returned 2 and 3 valid scenarios.
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
    // ── Case A — clear negative baseline (raw 0.0791 · shown 2.6% · Negative) ──
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
    // ── Case B — moderate positive (raw 0.4352 · shown 19.3% · Positive, 1.78× threshold) ──
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
    // ── Case C — strong positive (raw 0.8498 · shown 63.7% · Positive, HIGH band) ──
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
    // practical to sit further from the decision threshold, but per that
    // caveat, the same 1-to-3 variance cannot be ruled out here either —
    // locally on 2026-09-18 it gave 2/3 and 3/3 on two uncached calls.
    // (The draft percentages above are historical raw-scale values.)
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
    // ── Case D — deliberately borderline (raw 0.3509 · shown 14.4% · Positive) ──
    // Hypertension only, everything else clean: no high cholesterol, active,
    // good diet, normal-to-mildly-elevated BMI. Lands above the decision
    // threshold with a margin of +0.070 raw (0.3509 vs 0.2809) = +0.0356
    // corrected (0.1438 vs 0.1082, i.e. 1.33× the threshold) rather than flush
    // against it —
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

  // ── Dysglycaemia screening (NHANES, Phase 9) ────────────────────────────
  // One patient per conformal decision. Every decision and probability below was
  // measured 2026-09-28 by calling DiabetesEbmConformalBackend.predict() locally
  // on exactly these inputs (configs/diabetes_nhanes.yaml, alpha 0.20). The
  // probability is Platt-calibrated on NHANES 2015-2016 and is NOT comparable
  // with the BRFSS module's.
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
