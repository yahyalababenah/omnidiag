/**
 * featureCategorizer.js
 * =====================
 * Categorizes dynamically fetched fields into logical groups (Vitals,
 * Lifestyle, Demographics, Medical History, etc.) for visual hierarchy.
 *
 * Uses keyword matching on field names and descriptions. Categories
 * are returned as an ordered Map so they render in a consistent layout.
 *
 * Category definitions:
 *   ┌─────────────────┬──────────────────────────────────────────────┐
 *   │ Category        │ Matched Keywords                             │
 *   ├─────────────────┼──────────────────────────────────────────────┤
 *   │ Vitals & Signs  │ Age, BP, HR, BMI, Cholesterol, Chest, ST,   │
 *   │                 │ Oldpeak, MaxHR, Pulse, Temp, Resp, O2        │
 *   │ Lifestyle       │ Smoke, PhysActivity, Fruits, Veggies, Alc,  │
 *   │                 │ Exercise, Diet, Activity                     │
 *   │ Demographics    │ Sex, Age, Education, Income, Race, Region   │
 *   │ Medical History │ HighBP, HighChol, Stroke, Heart, Diabetes,  │
 *   │                 │ RestingECG, FastingBS, CholCheck, DiffWalk  │
 *   │ Healthcare      │ AnyHealthcare, NoDocbcCost, Insurance       │
 *   │ Mental Health   │ MentHlth, PhysHlth, GenHlth, DiffWalk       │
 *   │ General         │ (fallback for uncategorized fields)          │
 *   └─────────────────┴──────────────────────────────────────────────┘
 */

/**
 * @typedef {import('./schemaFieldParser').FieldMetadata} FieldMetadata
 */

// ── Category definitions ──────────────────────────────────────────────
const CATEGORIES = {
  'Vitals & Signs': {
    icon: 'Activity',
    keywords: [
      'age', 'restingbp', 'maxhr', 'cholesterol', 'oldpeak',
      'chestpain', 'st_slope', 'fastingbs', 'bmi', 'heartrate',
      'pulse', 'temperature', 'respiratory', 'o2', 'bp', 'blood pressure',
    ],
  },
  Lifestyle: {
    icon: 'Heart',
    keywords: [
      'smoker', 'physactivity', 'fruits', 'veggies',
      'hvyalcoholconsump', 'exercise', 'diet', 'activity',
      'alcohol', 'smoking',
    ],
  },
  Demographics: {
    icon: 'User',
    keywords: [
      'sex', 'education', 'income', 'race', 'region',
      'geography', 'marital',
    ],
  },
  'Medical History': {
    icon: 'Stethoscope',
    keywords: [
      'highbp', 'highchol', 'stroke', 'heartdiseaseorattack',
      'restingecg', 'cholcheck', 'diffwalk', 'diabetes',
      'anyhealthcare', 'nodocbccost', 'kidney', 'cancer',
      'condition', 'diagnosis',
    ],
  },
  'Healthcare Access': {
    icon: 'Pill',
    keywords: [
      'anyhealthcare', 'nodocbccost', 'insurance', 'doctor',
      'checkup', 'medication',
    ],
  },
  'Mental Health': {
    icon: 'Wind',
    keywords: [
      'menthlth', 'physhlth', 'genhlth', 'depression',
      'anxiety', 'mental', 'wellbeing',
    ],
  },
};

const CATEGORY_ORDER = [
  'Vitals & Signs',
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
