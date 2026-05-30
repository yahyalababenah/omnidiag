/**
 * useDiseaseSchema.js
 * ===================
 * Custom hook that fetches and parses the JSON Schema for a given disease.
 *
 * Caches the parsed FieldMetadata[] so switching diseases is instant
 * after the first load.
 *
 * Usage:
 *   const { fields, loading, error, refetch } = useDiseaseSchema('diabetes');
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api';
import { parseSchema, extractDefaultValues } from '../utils/schemaFieldParser';
import { categorizeFields } from '../utils/featureCategorizer';

const schemaCache = new Map();
const fieldsCache = new Map();
const categorizedCache = new Map();

/**
 * @param {string} diseaseName - The disease identifier (e.g., 'heart_disease')
 * @param {object} [options]
 * @param {boolean} [options.skipCache=false] - Bypass cache and re-fetch
 * @returns {{ fields: object[], categorizedFields: Map, defaults: object, loading: boolean, error: string|null, refetch: Function }}
 */
export function useDiseaseSchema(diseaseName, options = {}) {
  const { skipCache = false } = options;
  const [fields, setFields] = useState(() => {
    if (!skipCache && fieldsCache.has(diseaseName)) {
      return fieldsCache.get(diseaseName);
    }
    return null;
  });
  const [categorizedFields, setCategorizedFields] = useState(() => {
    if (!skipCache && categorizedCache.has(diseaseName)) {
      return categorizedCache.get(diseaseName);
    }
    return null;
  });
  const [defaults, setDefaults] = useState(null);
  const [loading, setLoading] = useState(!fields);
  const [error, setError] = useState(null);
  const mountedRef = useRef(true);

  const fetchSchema = useCallback(async (disease, bypassCache = false) => {
    if (!disease) {
      setFields(null);
      setCategorizedFields(null);
      setDefaults(null);
      setLoading(false);
      return;
    }

    // Check field-level cache
    if (!bypassCache && fieldsCache.has(disease)) {
      setFields(fieldsCache.get(disease));
      setCategorizedFields(categorizedCache.get(disease));
      setDefaults(null); // will be set below
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Fetch raw JSON Schema
      const schema = await api.getSchema(disease);

      // Parse into FieldMetadata[]
      const parsedFields = parseSchema(schema);

      // Categorize
      const categorized = categorizeFields(parsedFields);

      // Extract default values
      const defaultValues = extractDefaultValues(parsedFields, schema);

      // Cache
      fieldsCache.set(disease, parsedFields);
      categorizedCache.set(disease, categorized);
      schemaCache.set(disease, schema);

      if (mountedRef.current) {
        setFields(parsedFields);
        setCategorizedFields(categorized);
        setDefaults(defaultValues);
      }
    } catch (err) {
      console.error(`[useDiseaseSchema] Error fetching schema for "${disease}":`, err);
      if (mountedRef.current) {
        setError(err.message || 'Failed to load form schema.');
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  // Fetch when disease changes
  useEffect(() => {
    mountedRef.current = true;
    fetchSchema(diseaseName, skipCache);
    return () => { mountedRef.current = false; };
  }, [diseaseName, skipCache, fetchSchema]);

  const refetch = useCallback(() => {
    // Clear cache for this disease and re-fetch
    fieldsCache.delete(diseaseName);
    categorizedCache.delete(diseaseName);
    schemaCache.delete(diseaseName);
    fetchSchema(diseaseName, true);
  }, [diseaseName, fetchSchema]);

  return {
    fields,
    categorizedFields,
    defaults,
    loading,
    error,
    refetch,
  };
}

/**
 * Clear all cached schemas — useful for hard refresh.
 */
export function clearSchemaCache() {
  schemaCache.clear();
  fieldsCache.clear();
  categorizedCache.clear();
}
