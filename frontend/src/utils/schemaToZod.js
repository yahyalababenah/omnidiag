/**
 * schemaToZod.js
 * ==============
 * Converts JSON Schema field metadata into a Zod validation schema object.
 * Mirrors the Pydantic server-side validation (ge/le/type/required/enum)
 * so the client provides instant feedback without a round-trip.
 *
 * Dependencies: zod (must be installed in package.json)
 *
 * Usage:
 *   import { buildZodSchema } from '../utils/schemaToZod';
 *   import { z } from 'zod';
 *
 *   const fields = parseSchema(schemaJson);
 *   const zodSchema = buildZodSchema(fields);
 *   const result = zodSchema.safeParse(formData);
 */

import { z } from 'zod';

/**
 * Build a Zod object schema from parsed FieldMetadata array.
 *
 * @param {import('./schemaFieldParser').FieldMetadata[]} fields
 * @returns {z.ZodObject} A Zod schema that validates form data
 */
export function buildZodSchema(fields) {
  const shape = {};

  for (const field of fields) {
    let validator;

    switch (field.component) {
      // ── Toggle (binary 0/1 integer) ──
      case 'toggle': {
        validator = z
          .number({ invalid_type_error: `${field.title} is required` })
          .int()
          .min(0)
          .max(1);
        // Accept numbers and also 0/1 strings (coerce)
        validator = validator.or(z.literal(0)).or(z.literal(1));
        break;
      }

      // ── Segmented (binary 0/1 radio-group, e.g. Sex) ──
      case 'segmented': {
        validator = z
          .number({ invalid_type_error: `${field.title} is required` })
          .int()
          .min(0)
          .max(1);
        break;
      }

      // ── Dropdown (string enum) ──
      case 'select': {
        if (field.validation.enum && field.validation.enum.length > 0) {
          // Handle both string and number enums
          const enumValues = field.validation.enum;
          if (typeof enumValues[0] === 'string') {
            validator = z.enum(enumValues, {
              errorMap: () => ({ message: `Select a valid ${field.title}` }),
            });
          } else {
            validator = z.number().refine(
              (v) => enumValues.includes(v),
              { message: `Select a valid ${field.title}` }
            );
          }
        } else {
          validator = z.string().min(1, `${field.title} is required`);
        }
        break;
      }

      // ── Slider (small-range integer) ──
      case 'slider': {
        let numVal = z
          .number({ invalid_type_error: `${field.title} is required` })
          .int();
        if (field.validation.minimum !== undefined) {
          numVal = numVal.min(field.validation.minimum);
        }
        if (field.validation.maximum !== undefined) {
          numVal = numVal.max(field.validation.maximum);
        }
        validator = numVal;
        break;
      }

      // ── Number (float or wide-range integer) ──
      case 'number': {
        // Build the base numeric validator first (BEFORE .or() string-coercion)
        // so that .min()/.max() chain directly on ZodNumber, not on ZodUnion.
        let numVal;
        if (field.type === 'number') {
          numVal = z.number({
            invalid_type_error: `${field.title} must be a number`,
          });
        } else {
          numVal = z
            .number({ invalid_type_error: `${field.title} must be a number` })
            .int();
        }
        if (field.validation.minimum !== undefined) {
          numVal = numVal.min(field.validation.minimum, {
            message: `${field.title} must be ≥ ${field.validation.minimum}`,
          });
        }
        if (field.validation.maximum !== undefined) {
          numVal = numVal.max(field.validation.maximum, {
            message: `${field.title} must be ≤ ${field.validation.maximum}`,
          });
        }
        // Now chain the string-coercion AFTER min/max are set
        if (field.type === 'number') {
          numVal = numVal.or(z.string().transform((v) => parseFloat(v)));
        } else {
          numVal = numVal.or(z.string().transform((v) => parseInt(v, 10)));
        }
        validator = numVal;
        break;
      }

      // ── Text (fallback string) ──
      case 'text':
      default: {
        validator = z.string().min(1, `${field.title} is required`);
        break;
      }
    }

    // Make optional if not required
    if (!field.validation.required) {
      validator = validator.optional();
    }

    shape[field.name] = validator;
  }

  return z.object(shape);
}

