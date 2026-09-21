/**
 * Clinical direction per input field, for colouring patient data (not for
 * the model). A card is flagged only when its value is on the clinically
 * unfavourable side — never simply because a binary field equals 1.
 *
 * Where the What-If mutability policy defines a direction it is reused
 * verbatim (backend/counterfactual_generator.py DIABETES_POLICY and
 * backend/model_loader.py HEART_POLICY): PhysActivity/Fruits/Veggies -> 1,
 * HvyAlcoholConsump -> 0, FastingBS -> 0. The rest are the standard BRFSS /
 * UCI meanings of each field.
 */

// Binary (0/1) fields: the clinically favourable value.
export const FAVOURABLE_BINARY = {
  // from the What-If policy
  PhysActivity: 1,
  Fruits: 1,
  Veggies: 1,
  HvyAlcoholConsump: 0,
  FastingBS: 0,
  // BRFSS history / risk factors: 1 = present
  HighBP: 0,
  HighChol: 0,
  Smoker: 0,
  Stroke: 0,
  HeartDiseaseorAttack: 0,
  DiffWalk: 0,
  NoDocbcCost: 0,
  // access / screening: 1 = favourable
  AnyHealthcare: 1,
  CholCheck: 1,
};

// Demographics: never coloured as risk.
export const NEUTRAL_FIELDS = new Set(['Sex']);

// Numeric fields where a HIGHER value is favourable (the default assumes
// higher = worse, e.g. BMI, RestingBP, MentHlth).
export const HIGHER_IS_BETTER = new Set(['Income', 'Education', 'MaxHR']);

// Categorical fields: the values that are a risk finding. Fields listed here
// are coloured only by this list, not by their position in the enum.
export const RISK_CATEGORIES = {
  ExerciseAngina: ['Y'],
};
