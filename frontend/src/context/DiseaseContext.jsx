/**
 * DiseaseContext.jsx
 * ==================
 * Global React Context for multi-disease state management.
 *
 * Responsibilities:
 *   1. Fetch available diseases from GET /api/v4/diseases on mount
 *   2. Pre-fetch JSON schemas for each disease (cache for instant form rendering)
 *   3. Provide selectedDisease state + setter globally
 *   4. Persist selection in localStorage for session continuity
 */

import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api';

const DiseaseContext = createContext(null);

const STORAGE_KEY = 'omnidiag_selected_disease';

/**
 * DiseaseProvider — wraps the app and provides disease state.
 *
 * Usage:
 *   <DiseaseProvider>
 *     <App />
 *   </DiseaseProvider>
 */
export function DiseaseProvider({ children }) {
  const [availableDiseases, setAvailableDiseases] = useState([]);
  const [selectedDisease, setSelectedDisease] = useState(null);
  const [schemaCache, setSchemaCache] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const initialised = useRef(false);

  // ── Fetch available diseases on mount ──
  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    let cancelled = false;

    async function init() {
      try {
        setLoading(true);
        const res = await api.listDiseases();
        const diseases = res.diseases || [];

        if (cancelled) return;

        setAvailableDiseases(diseases);

        // Restore last-selected disease from localStorage, or default to first
        const stored = localStorage.getItem(STORAGE_KEY);
        const defaultDisease = stored && diseases.some((d) => d.name === stored)
          ? stored
          : diseases[0]?.name || null;

        setSelectedDisease(defaultDisease);

        // Pre-fetch schemas for all diseases
        await preFetchSchemas(diseases);
      } catch (err) {
        if (!cancelled) {
          console.error('[DiseaseContext] Failed to load diseases:', err);
          setError(err.message || 'Failed to connect to the diagnostic engine.');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    init();

    return () => { cancelled = true; };
  }, []);

  // ── Pre-fetch schemas for all diseases ──
  const preFetchSchemas = useCallback(async (diseases) => {
    const cache = {};
    const fetchPromises = diseases.map(async (d) => {
      try {
        const schema = await api.getSchema(d.name);
        cache[d.name] = schema;
      } catch (err) {
        console.warn(`[DiseaseContext] Failed to fetch schema for "${d.name}":`, err.message);
        cache[d.name] = null; // mark as failed, will retry on demand
      }
    });

    await Promise.allSettled(fetchPromises);
    setSchemaCache((prev) => ({ ...prev, ...cache }));
  }, []);

  // ── Select disease (persisted) ──
  const selectDisease = useCallback((diseaseName) => {
    setSelectedDisease(diseaseName);
    localStorage.setItem(STORAGE_KEY, diseaseName);
  }, []);

  // ── Get schema for the currently selected disease ──
  const getSchemaForDisease = useCallback(async (diseaseName) => {
    // Check cache first
    if (schemaCache[diseaseName]) {
      return schemaCache[diseaseName];
    }

    // Fetch on demand
    try {
      const schema = await api.getSchema(diseaseName);
      setSchemaCache((prev) => ({ ...prev, [diseaseName]: schema }));
      return schema;
    } catch (err) {
      console.error(`[DiseaseContext] Failed to fetch schema for "${diseaseName}":`, err);
      throw err;
    }
  }, [schemaCache]);

  // ── Retry fetching diseases (e.g., after error) ──
  const retry = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await api.listDiseases();
      const diseases = res.diseases || [];
      setAvailableDiseases(diseases);

      if (!selectedDisease && diseases.length > 0) {
        setSelectedDisease(diseases[0].name);
      }

      await preFetchSchemas(diseases);
    } catch (err) {
      setError(err.message || 'Retry failed.');
    } finally {
      setLoading(false);
    }
  }, [selectedDisease, preFetchSchemas]);

  // ── Get disease info for current selection ──
  const currentDiseaseInfo = availableDiseases.find(
    (d) => d.name === selectedDisease
  );

  const value = {
    // State
    availableDiseases,
    selectedDisease,
    schemaCache,
    loading,
    error,

    // Derived
    currentDiseaseInfo,

    // Actions
    selectDisease,
    getSchemaForDisease,
    retry,
  };

  return (
    <DiseaseContext.Provider value={value}>
      {children}
    </DiseaseContext.Provider>
  );
}

/**
 * Hook to access disease context. Must be used within DiseaseProvider.
 */
export function useDisease() {
  const ctx = useContext(DiseaseContext);
  if (!ctx) {
    throw new Error('useDisease() must be used within a <DiseaseProvider>');
  }
  return ctx;
}

export default DiseaseContext;
