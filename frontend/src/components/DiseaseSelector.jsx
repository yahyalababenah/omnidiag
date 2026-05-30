/**
 * DiseaseSelector.jsx
 * ====================
 * A polished dropdown component for switching between available diseases.
 * Placed in the sidebar header of App.jsx.
 *
 * Features:
 *   - Fetches disease list from DiseaseContext
 *   - Shows display_name, version badge, and description in dropdown
 *   - Persists selection via DiseaseContext (→ localStorage)
 *   - Keyboard accessible
 *   - Loading skeleton while diseases are being fetched
 *   - Error state with retry
 */

import { useState, useRef, useEffect } from 'react';
import {
  ChevronDown,
  AlertCircle,
  RefreshCw,
  Activity,
  Loader2,
} from 'lucide-react';
import { useDisease } from '../context/DiseaseContext';

export default function DiseaseSelector() {
  const {
    availableDiseases,
    selectedDisease,
    currentDiseaseInfo,
    selectDisease,
    loading,
    error,
    retry,
  } = useDisease();

  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // ── Loading state ──
  if (loading) {
    return (
      <div className="px-4 py-3 border-b border-clinical-border">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          Loading diseases...
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (error) {
    return (
      <div className="px-4 py-3 border-b border-clinical-border">
        <div className="flex items-start gap-2">
          <AlertCircle className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
          <div className="text-xs text-red-400">
            <p className="font-medium">Connection error</p>
            <button
              onClick={retry}
              className="mt-1 flex items-center gap-1 text-red-300 hover:text-red-200 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Empty state ──
  if (availableDiseases.length === 0) {
    return (
      <div className="px-4 py-3 border-b border-clinical-border">
        <p className="text-xs text-gray-400">No diseases registered.</p>
      </div>
    );
  }

  // ── Render ──
  return (
    <div className="border-b border-clinical-border" ref={dropdownRef}>
      {/* Selected disease trigger */}
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between gap-2 px-4 py-3 hover:bg-gray-50 transition-colors text-left"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        title={currentDiseaseInfo?.description || ''}
      >
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-primary-100 flex items-center justify-center shrink-0">
            <Activity className="w-3.5 h-3.5 text-primary-600" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">
              {currentDiseaseInfo?.display_name || selectedDisease || 'Select Disease'}
            </p>
            <p className="text-[10px] text-gray-500 truncate">
              v{currentDiseaseInfo?.version || '—'}
            </p>
          </div>
        </div>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 shrink-0 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          className="border-t border-clinical-border bg-white shadow-lg"
          role="listbox"
          aria-label="Select a disease"
        >
          {availableDiseases.map((d) => {
            const isActive = d.name === selectedDisease;
            return (
              <button
                key={d.name}
                onClick={() => {
                  selectDisease(d.name);
                  setIsOpen(false);
                }}
                role="option"
                aria-selected={isActive}
                className={`w-full flex items-start gap-3 px-4 py-3 text-left transition-colors ${
                  isActive
                    ? 'bg-primary-50 border-l-2 border-primary-500'
                    : 'hover:bg-gray-50 border-l-2 border-transparent'
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${
                    isActive ? 'bg-primary-500' : 'bg-gray-300'
                  }`}
                />
                <div className="min-w-0">
                  <p
                    className={`text-sm font-medium truncate ${
                      isActive ? 'text-primary-700' : 'text-gray-800'
                    }`}
                  >
                    {d.info?.display_name || d.name}
                  </p>
                  <p className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">
                    {d.info?.description || 'No description'}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] text-gray-400 font-mono">
                      v{d.info?.version || '—'}
                    </span>
                    <span className="text-[10px] text-gray-400">
                      {d.info?.model_type || '—'}
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
