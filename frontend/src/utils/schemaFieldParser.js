/**
 * schemaFieldParser.js
 * =====================
 * Parses a JSON Schema (from GET /api/v4/{disease}/schema) into an array of
 * FieldMetadata objects that drive dynamic form rendering.
 *
 * Maps JSON Schema types to React component types:
 *   - integer + min=0 + max=1 + demographic  →  'segmented' (radio group, e.g. Sex)
 *   - integer + min=0 + max=1                →  'toggle'    (binary yes/no switch)
 *   - string + enum [...]                    →  'select'    (dropdown)
 *   - integer + range ≤ 12                   →  'slider'    (ordinal / small-range)
 *   - number (float)                         →  'number'    (constrained float input)
 *   - integer + range > 12                   →  'number'    (constrained int input)
 *   - string (no enum)                       →  'text'      (fallback text input)
 */

/**
 * @typedef {Object} FieldMetadata
 * @property {string}  name        - The field key (e.g., "Age", "HighBP")
 * @property {string}  title       - Human-readable label derived from description
 * @property {string}  type        - JSON Schema type: "string" | "integer" | "number"
 * @property {string}  component   - React component type: "toggle" | "select" | "number" | "slider" | "text"
 * @property {{ required: boolean, minimum?: number, maximum?: number, enum?: (string|number)[], step?: number }} validation
 * @property {string}  description - Full field description
 * @property {*}       default     - Default value from schema example
 * @property {string}  [category]  - Inferred grouping category (set later by featureCategorizer)
 */

/**
 * Extract a human-readable title from a field name + description.
 * Falls back to the field name with snake_case → Title Case conversion.
 */
function deriveTitle(fieldName, description) {
  if (description) {
    // Use the part before any colon or parentheses as the short title
    const clean = description.split(':')[0].split('(')[0].trim();
    if (clean && clean.length < 50) return clean;
  }
  // Snake_case / CamelCase → Title Case
  return fieldName
    .replace(/([A-Z])/g, ' $1')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

/**
 * Determine the React component type from schema field properties.
 */
/**
 * Fields that represent distinct demographic/clinical states (not on/off).
 * These get a segmented radio-group control instead of a toggle switch.
 */
const DEMOGRAPHIC_BINARY_FIELDS = new Set([
  'sex',
]);

function resolveComponentType(schemaField) {
  const { type, minimum, maximum, enum: enumValues } = schemaField;
  const name = (schemaField.name || '').toLowerCase();

  // Rule 1: integer with min=0, max=1, demographic → Segmented (radio group)
  if (type === 'integer' && minimum === 0 && maximum === 1 && DEMOGRAPHIC_BINARY_FIELDS.has(name)) {
    return 'segmented';
  }

  // Rule 2: integer with min=0, max=1 → Toggle (binary yes/no)
  if (type === 'integer' && minimum === 0 && maximum === 1) {
    return 'toggle';
  }

  // Rule 2: string with enum → Dropdown
  if (type === 'string' && Array.isArray(enumValues) && enumValues.length > 0) {
    return 'select';
  }

  // Rule 3: integer with small range (≤ 12 distinct values) → Slider
  if (type === 'integer' && minimum !== undefined && maximum !== undefined) {
    const range = maximum - minimum;
    if (range <= 12) return 'slider';
    return 'number';
  }

  // Rule 4: number (float) → Number input
  if (type === 'number') {
    return 'number';
  }

  // Rule 5: integer with wide range → Number input
  if (type === 'integer') {
    return 'number';
  }

  // Rule 6: string with no enum → Text input (fallback)
  if (type === 'string') {
    return 'text';
  }

  // Fallback
  return 'text';
}

/**
 * Extract default value for a field from the schema's example data.
 */
function extractDefault(fieldName, schema) {
  // Check in $defs first, then root-level example
  const example =
    schema?.$defs?.[Object.keys(schema?.$defs || {})[0]]?.example ||
    schema?.example ||
    schema?.json_schema_extra?.example ||
    {};

  if (fieldName in example) return example[fieldName];
  return undefined;
}

/**
 * Parse the full JSON Schema into an array of FieldMetadata.
 *
 * @param {object} schema - The JSON Schema object from api.getSchema(disease)
 * @returns {FieldMetadata[]} Ordered list of field definitions
 */
export function parseSchema(schema) {
  if (!schema || !schema.properties) {
    console.warn('[schemaFieldParser] Invalid schema — missing properties', schema);
    return [];
  }

  const requiredFields = new Set(schema.required || []);

  return Object.entries(schema.properties).map(([name, prop]) => {
    const fieldType = prop.type || 'string';
    const validation = {
      required: requiredFields.has(name),
      minimum: prop.minimum,
      maximum: prop.maximum,
      enum: prop.enum || undefined,
      step: prop.multipleOf || (fieldType === 'number' ? 0.1 : 1),
    };

    return {
      name,
      title: deriveTitle(name, prop.description || prop.title),
      type: fieldType,
      component: resolveComponentType({ ...prop, type: fieldType, name }),
      validation,
      description: prop.description || '',
      default: extractDefault(name, schema),
      category: null, // assigned later by featureCategorizer
    };
  });
}

/**
 * Parse default values from schema examples into a flat form values object.
 *
 * @param {FieldMetadata[]} fields
 * @param {object} schema - Original JSON Schema
 * @returns {object} Default form values keyed by field name
 */
export function extractDefaultValues(fields, schema) {
  const example =
    schema?.$defs?.[Object.keys(schema?.$defs || {})[0]]?.example ||
    schema?.example ||
    {};

  const defaults = {};
  for (const field of fields) {
    if (field.name in example) {
      defaults[field.name] = example[field.name];
    } else {
      // Fallback: use minimum, or 0 for numbers, '' for strings
      if (field.type === 'number' || field.type === 'integer') {
        defaults[field.name] = field.validation.minimum ?? 0;
      } else if (field.type === 'string') {
        defaults[field.name] = field.validation.enum?.[0] ?? '';
      } else {
        defaults[field.name] = '';
      }
    }
  }
  return defaults;
}
