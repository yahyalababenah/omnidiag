/**
 * Pre-defined mock patients for the Clinical EMR Mode.
 * Restructured for multi-disease support — keyed by disease name.
 *
 * Each patient has realistic vitals/data matching the disease's schema fields,
 * plus clinical meta-fields (history, medications, admittingComplaint).
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
    {
      id: 'D-001',
      name: 'Layla Mansour',
      age: 58,
      sex: 'F',
      avatar: 'LM',
      history: 'Type 2 Diabetes (8 yrs), Hypertension, Hyperlipidemia',
      medications: 'Metformin 1000mg, Lisinopril 10mg, Atorvastatin 20mg',
      admittingComplaint: 'Follow-up visit — elevated HbA1c and fasting glucose',
      data: {
        HighBP: 1,
        HighChol: 1,
        CholCheck: 1,
        BMI: 32.4,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 0,
        Fruits: 0,
        Veggies: 0,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 3,
        MentHlth: 12,
        PhysHlth: 18,
        DiffWalk: 1,
        Sex: 0,
        Age: 10,
        Education: 3,
        Income: 4,
      },
    },
    {
      id: 'D-002',
      name: 'Mohammed Al-Sayed',
      age: 64,
      sex: 'M',
      avatar: 'MS',
      history: 'Type 2 Diabetes (15 yrs), CAD s/p PCI, CKD Stage 3',
      medications: 'Metformin 1000mg, Glipizide 10mg, Aspirin 81mg, Atorvastatin 40mg',
      admittingComplaint: 'Chest discomfort and shortness of breath × 3 days',
      data: {
        HighBP: 1,
        HighChol: 1,
        CholCheck: 1,
        BMI: 28.7,
        Smoker: 1,
        Stroke: 0,
        HeartDiseaseorAttack: 1,
        PhysActivity: 0,
        Fruits: 1,
        Veggies: 0,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 1,
        NoDocbcCost: 0,
        GenHlth: 4,
        MentHlth: 8,
        PhysHlth: 22,
        DiffWalk: 1,
        Sex: 1,
        Age: 11,
        Education: 2,
        Income: 3,
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

/**
 * Get all available disease keys that have mock patients.
 */
export function getAvailablePatientDiseases() {
  return Object.keys(mockPatients);
}

export default mockPatients;
