/**
 * featureCategorizer.js
 * =====================
 * Categorizes dynamically fetched fields into logical groups (Vitals,
 * Lifestyle, Demographics, Medical History, etc.) for visual hierarchy.
 *
 * Uses keyword matching on field names and descriptions. Categories
 * are returned as an ordered Map so they render in a consistent layout.
 *
 * Category definitions (the BRFSS column codes went with that module, gate B6):
 *   ┌─────────────────┬──────────────────────────────────────────────┐
 *   │ Category        │ Matched Keywords                             │
 *   ├─────────────────┼──────────────────────────────────────────────┤
 *   │ Vitals & Signs  │ Age, BP, HR, BMI, Cholesterol, Chest, ST,   │
 *   │                 │ Oldpeak, MaxHR, Pulse, Temp, Resp, O2        │
 *   │ Lifestyle       │ Smoke, Alc, Exercise, Diet, Activity        │
 *   │ Demographics    │ Sex, Age, Race, Region                       │
 *   │ Medical History │ Stroke, Heart, Diabetes, RestingECG,        │
 *   │                 │ FastingBS, prior conditions                  │
 *   │ Healthcare      │ Insurance, Doctor, Medication               │
 *   │ Mental Health   │ Depression, Anxiety, Mental                 │
 *   │ General         │ (fallback for uncategorized fields)          │
 *   └─────────────────┴──────────────────────────────────────────────┘
 */

/**
 * @typedef {import('./schemaFieldParser').FieldMetadata} FieldMetadata
 */

// ── Category definitions ──────────────────────────────────────────────
// ORDER MATTERS: categorizeField returns the FIRST category that matches, so
// Laboratory is declared first. Its keywords are NHANES column codes ONLY —
// never generic words like "cholesterol" — because a generic word would also
// match the heart module's `Cholesterol` field via its description and silently
// move it out of Vitals & Signs. Exact codes keep the blast radius at zero.
const CATEGORIES = {
  Laboratory: {
    icon: 'Pill',
    keywords: [
      'lbdhdd', 'lbxsch', 'lbxstr', 'lbxsatsi', 'lbxsgtsi',
      'lbxscr', 'lbxsbu', 'lbxsal', 'lbxsua',
    ],
  },
  // Declared before Vitals on purpose: the string "riagendr" contains the
  // substring "age", so Vitals' 'age' keyword would otherwise claim it.
  Demographics: {
    icon: 'User',
    keywords: [
      'sex', 'race', 'region',
      'geography', 'marital',
      'riagendr',
    ],
  },
  'Vitals & Signs': {
    icon: 'Activity',
    keywords: [
      'age', 'restingbp', 'maxhr', 'cholesterol', 'oldpeak',
      'chestpain', 'st_slope', 'fastingbs', 'bmi', 'heartrate',
      'pulse', 'temperature', 'respiratory', 'o2', 'bp', 'blood pressure',
      // NHANES codes, matched exactly (see the note above CATEGORIES)
      'ridageyr', 'bmxbmi', 'sbp', 'dbp', 'bpxpls', 'adiposity_band',
    ],
  },
  Lifestyle: {
    icon: 'Heart',
    keywords: [
      'smoker', 'exercise', 'diet', 'activity',
      'alcohol', 'smoking',
      'paq650', 'paq665',
    ],
  },
  'Medical History': {
    icon: 'Stethoscope',
    keywords: [
      'stroke', 'restingecg', 'diabetes', 'kidney', 'cancer',
      'condition', 'diagnosis',
      'mcq300c', 'cvd_any', 'cardiovascular',
    ],
  },
  'Healthcare Access': {
    icon: 'Pill',
    keywords: [
      'insurance', 'doctor', 'checkup', 'medication',
    ],
  },
  'Mental Health': {
    icon: 'Wind',
    keywords: [
      'depression', 'anxiety', 'mental', 'wellbeing',
    ],
  },
};

const CATEGORY_ORDER = [
  'Vitals & Signs',
  'Laboratory',
  'Medical History',
  'Lifestyle',
  'Demographics',
  'Mental Health',
  'Healthcare Access',
  'General',
];

// ── Categorization logic ──────────────────────────────────────────────

/**
 * Determine the category for a single field based on its name and description.
 */
function categorizeField(field) {
  const searchText = `${field.name} ${field.description}`.toLowerCase();

  for (const [category, def] of Object.entries(CATEGORIES)) {
    for (const keyword of def.keywords) {
      // Match whole field names (case-insensitive) or partial in description
      if (
        field.name.toLowerCase() === keyword ||
        searchText.includes(keyword)
      ) {
        return category;
      }
    }
  }

  return 'General';
}

/**
 * Categorize an array of FieldMetadata into an ordered Map.
 *
 * @param {FieldMetadata[]} fields - Parsed field metadata
 * @returns {Map<string, FieldMetadata[]>} Ordered map: category → fields[]
 */
export function categorizeFields(fields) {
  const grouped = new Map();

  // Initialize all categories in order
  for (const cat of CATEGORY_ORDER) {
    grouped.set(cat, []);
  }

  // Assign each field to a category
  for (const field of fields) {
    const category = categorizeField(field);
    field.category = category;

    if (!grouped.has(category)) {
      grouped.set(category, []);
    }
    grouped.get(category).push(field);
  }

  // Remove empty categories (preserving order)
  for (const [cat, fieldsInCat] of grouped) {
    if (fieldsInCat.length === 0) {
      grouped.delete(cat);
    }
  }

  return grouped;
}

/**
 * Get the icon name for a category.
 */
export function getCategoryIcon(category) {
  return CATEGORIES[category]?.icon || 'ClipboardList';
}
