/**
 * Clinical direction per input field, for colouring patient data (not for
 * the model). A card is flagged only when its value is on the clinically
 * unfavourable side — never simply because a binary field equals 1.
 *
 * Where the heart What-If policy defines a direction it is reused verbatim
 * (FastingBS -> 0). The rest are the standard UCI meanings of each field.
 * The BRFSS diabetes entries went with that module (retired; gate B6).
 */

// Binary (0/1) fields: the clinically favourable value.
export const FAVOURABLE_BINARY = {
  // from the heart What-If policy
  FastingBS: 0,
};

// Demographics: never coloured as risk.
export const NEUTRAL_FIELDS = new Set(['Sex']);

// Numeric fields where a HIGHER value is favourable (the default assumes
// higher = worse, e.g. RestingBP).
export const HIGHER_IS_BETTER = new Set(['MaxHR']);

// Categorical fields: the values that are a risk finding. Fields listed here
// are coloured only by this list, not by their position in the enum.
export const RISK_CATEGORIES = {
  ExerciseAngina: ['Y'],
};
