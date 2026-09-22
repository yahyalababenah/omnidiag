/**
 * DiseaseContext.jsx
 * ==================
 * Global React Context for multi-disease state management.
 *
 * Responsibilities:
 *   1. Fetch available diseases from GET /api/v4/diseases on mount
 *   2. Provide selectedDisease state + setter globally
 *   3. Persist selection in localStorage for session continuity
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ── Fetch available diseases on mount ──
  //
  // No `initialised` ref guard here on purpose. Pairing one with the
  // `cancelled` flag deadlocks under StrictMode: the first run sets the ref,
  // its cleanup sets `cancelled`, and the second run returns early on the
  // ref — so nothing ever clears `loading` and the app sits on
  // "Loading diseases…" forever in `npm run dev` (X-4). The `cancelled` flag
  // alone is the correct guard: the second run is allowed to proceed and is
  // the one whose state updates land.
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setLoading(true);
        const res = await api.listDiseases();
        // Flatten { name, info: {...} } → { name, ...info } so consumers can
        // access currentDiseaseInfo.display_name / .supports_counterfactuals directly.
        const diseases = (res.diseases || []).map(d => ({ name: d.name, ...d.info }));

        if (cancelled) return;

        setAvailableDiseases(diseases);

        // Restore last-selected disease from localStorage, or default to first
        const stored = localStorage.getItem(STORAGE_KEY);
        const defaultDisease = stored && diseases.some((d) => d.name === stored)
          ? stored
          : diseases[0]?.name || null;

        setSelectedDisease(defaultDisease);
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

  // ── Select disease (persisted) ──
  const selectDisease = useCallback((diseaseName) => {
    setSelectedDisease(diseaseName);
    localStorage.setItem(STORAGE_KEY, diseaseName);
  }, []);

  // ── Retry fetching diseases (e.g., after error) ──
  const retry = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await api.listDiseases();
      const diseases = (res.diseases || []).map(d => ({ name: d.name, ...d.info }));
      setAvailableDiseases(diseases);

      if (!selectedDisease && diseases.length > 0) {
        setSelectedDisease(diseases[0].name);
      }
    } catch (err) {
      setError(err.message || 'Retry failed.');
    } finally {
      setLoading(false);
    }
  }, [selectedDisease]);

  // ── Get disease info for current selection ──
  const currentDiseaseInfo = availableDiseases.find(
    (d) => d.name === selectedDisease
  );

  const value = {
    availableDiseases,
    selectedDisease,
    loading,
    error,
    currentDiseaseInfo,
    selectDisease,
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
