/**
 * useDiseaseForm.js
 * =================
 * Custom hook that combines schema fetching, Zod validation, and
 * React Hook Form into a single cohesive interface for the
 * DynamicClinicalForm component.
 *
 * Usage:
 *   const { form, fields, categorizedFields, schemaState, onSubmit } =
 *     useDiseaseForm('diabetes', { onPredict, onExplain });
 */

import { useMemo, useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useDiseaseSchema } from './useDiseaseSchema';
import { buildZodSchema } from '../utils/schemaToZod';
import { categorizeFields } from '../utils/featureCategorizer';

/**
 * @param {string} diseaseName - The disease identifier
 * @param {object} [options]
 * @param {function} [options.onSubmit] - Called with form data when submitted
 * @param {boolean} [options.skipCache=false] - Bypass schema cache
 */
export function useDiseaseForm(diseaseName, options = {}) {
  const { onSubmit, skipCache = false } = options;

  // ── Fetch schema ──
  const {
    fields,
    categorizedFields: initialCategorized,
    defaults,
    loading: schemaLoading,
    error: schemaError,
    refetch: refetchSchema,
  } = useDiseaseSchema(diseaseName, { skipCache });

  // ── Build Zod schema from fields (memoized) ──
  const zodSchema = useMemo(() => {
    if (!fields || fields.length === 0) return null;
    return buildZodSchema(fields);
  }, [fields]);

  // ── Initialize React Hook Form ──
  const form = useForm({
    defaultValues: defaults || {},
    resolver: zodSchema ? zodResolver(zodSchema) : undefined,
    mode: 'onBlur',
    reValidateMode: 'onChange',
  });

  // ── Reset form when disease changes ──
  const { reset } = form;
  useEffect(() => {
    if (defaults) {
      reset(defaults);
    }
  }, [defaults, reset, diseaseName]);

  // ── Submit handler ──
  const handleSubmit = useCallback(
    async (formData) => {
      if (onSubmit) {
        await onSubmit(formData);
      }
    },
    [onSubmit]
  );

  // ── Categorized fields (memoized from parsed fields) ──
  const categorizedFields = useMemo(() => {
    if (fields && fields.length > 0) {
      return categorizeFields(fields);
    }
    return initialCategorized || new Map();
  }, [fields, initialCategorized]);

  // ── Errors state (for convenience in parent) ──
  const formErrors = form.formState.errors;
  const hasErrors = Object.keys(formErrors).length > 0;

  return {
    // Form state
    form,
    fields: fields || [],
    categorizedFields,
    defaults,

    // Schema loading state
    schemaLoading,
    schemaError,
    refetchSchema,

    // Form helpers
    hasErrors,
    formErrors,

    // Submission
    submitForm: form.handleSubmit(handleSubmit),
    isSubmitting: form.formState.isSubmitting,

    // Manual reset
    resetForm: () => {
      if (defaults) reset(defaults);
    },
  };
}
