/**
 * VariableScalesModal.jsx
 * ========================
 * A clean, minimalist reference modal that displays the Clinical Data Dictionary
 * for all variables in the current disease module.
 *
 * Features:
 *   - Slide-in drawer from the right (matches clinical aesthetic)
 *   - Categorized accordion groups matching the form layout
 *   - Columns: Feature Name · Type · Scale / Range · Description
 *   - Dismissible via close button, backdrop click, or Escape key
 *   - No external dependencies — pure Tailwind + React
 */

import { useState, useEffect, useCallback } from 'react';
import { X, ChevronDown, Search } from 'lucide-react';
import { lookupMedicalTerm } from '../utils/medicalDictionary';

/**
 * Format the validation constraints into a readable scale/range string.
 */
function formatScale(field) {
  const { minimum, maximum, enum: enumVals } = field.validation || {};

  // Enum fields — show comma-separated options
  if (Array.isArray(enumVals) && enumVals.length > 0) {
    return enumVals.map(String).join(', ');
  }

  // Toggle/binary fields — show as 0/1 labels
  if (minimum === 0 && maximum === 1) {
    return '0 / 1';
  }

  // Slider / number fields — show min–max range
  if (minimum !== undefined && maximum !== undefined) {
    const minStr = Number.isInteger(minimum) ? minimum.toString() : minimum.toFixed(1);
    const maxStr = Number.isInteger(maximum) ? maximum.toString() : maximum.toFixed(1);
    return `${minStr} – ${maxStr}`;
  }

  if (minimum !== undefined) {
    const minStr = Number.isInteger(minimum) ? minimum.toString() : minimum.toFixed(1);
    return `≥ ${minStr}`;
  }

  if (maximum !== undefined) {
    const maxStr = Number.isInteger(maximum) ? maximum.toString() : maximum.toFixed(1);
    return `≤ ${maxStr}`;
  }

  return '—';
}

/**
 * Format the value type for display.
 */
function formatType(field) {
  const t = field.type || 'string';
  if (t === 'integer') return 'Integer';
  if (t === 'number') return 'Float';
  if (t === 'string') return 'Categorical';
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/**
 * Get a short label hint for the scale (e.g., "Binary", "Ordinal 1-5").
 */
function getScaleHint(field) {
  const { minimum, maximum, enum: enumVals } = field.validation || {};

  if (Array.isArray(enumVals) && enumVals.length > 0) {
    return 'Categorical';
  }
  if (minimum === 0 && maximum === 1) {
    return 'Binary';
  }
  if (field.type === 'integer' && minimum !== undefined && maximum !== undefined) {
    const range = maximum - minimum;
    if (range <= 5) return `Ordinal ${minimum}–${maximum}`;
    if (range <= 12) return `Scale ${minimum}–${maximum}`;
    return 'Numeric range';
  }
  if (field.type === 'number') {
    return 'Continuous';
  }
  return '';
}

/**
 * VariableScalesModal — slide-in drawer from the right.
 *
 * @param {{
 *   isOpen: boolean,
 *   onClose: () => void,
 *   fields: Array<import('../utils/schemaFieldParser').FieldMetadata>,
 *   diseaseName: string,
 *   categorizedFields?: Map<string, Array<import('../utils/schemaFieldParser').FieldMetadata>>,
 * }} props
 */
export default function VariableScalesModal({
  isOpen,
  onClose,
  fields = [],
  diseaseName = '',
  categorizedFields,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState(new Set());

  // Expand all categories on open
  useEffect(() => {
    if (isOpen && categorizedFields) {
      setExpandedCategories(new Set(categorizedFields.keys()));
    }
  }, [isOpen, categorizedFields]);

  // Close on Escape key
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleKeyDown]);

  // Toggle category accordion
  const toggleCategory = (cat) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  // Filter fields by search query
  const filteredFields = searchQuery.trim()
    ? fields.filter((f) => {
        const q = searchQuery.toLowerCase();
        return (
          f.name.toLowerCase().includes(q) ||
          f.title.toLowerCase().includes(q) ||
          (f.description && f.description.toLowerCase().includes(q)) ||
          (lookupMedicalTerm(f.title) || '').toLowerCase().includes(q)
        );
      })
    : fields;

  // Re-group filtered fields into categories
  const filteredGroups = new Map();
  if (categorizedFields && !searchQuery.trim()) {
    // Use the original categorized structure if no search
    for (const [cat, catFields] of categorizedFields) {
      filteredGroups.set(cat, catFields);
    }
  } else {
    // Assign filtered fields back to their categories
    for (const field of filteredFields) {
      const cat = field.category || 'General';
      if (!filteredGroups.has(cat)) filteredGroups.set(cat, []);
      filteredGroups.get(cat).push(field);
    }
  }

  if (!isOpen) return null;

  const fieldCount = fields.length;

  return (
    <div className="fixed inset-0 z-[60] flex justify-end">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <div
        className="relative w-full max-w-2xl bg-white shadow-2xl border-l border-clinical-border
                   flex flex-col animate-slide-in-right"
        role="dialog"
        aria-modal="true"
        aria-label={`Variable scales for ${diseaseName}`}
      >
        {/* ── Header ── */}
        <div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-clinical-border">
          <div>
            <h2 className="text-base font-bold text-gray-900">Variable Reference</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {diseaseName} · {fieldCount} variable{fieldCount !== 1 ? 's' : ''}
              {' — '}Clinical Data Dictionary
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-400 hover:text-gray-600"
            aria-label="Close reference"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* ── Search ── */}
        <div className="shrink-0 px-6 py-3 border-b border-clinical-border">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search variables by name, description, or scale…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm border border-clinical-border rounded-lg
                         focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500
                         placeholder:text-gray-400"
            />
          </div>
        </div>

        {/* ── Body — scrollable variable list ── */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {filteredGroups.size === 0 && (
            <div className="text-center py-12">
              <p className="text-sm text-gray-400">
                No variables match &ldquo;{searchQuery}&rdquo;
              </p>
            </div>
          )}

          {Array.from(filteredGroups.entries()).map(([category, catFields]) => (
            <div key={category} className="card overflow-hidden border border-clinical-border">
              {/* Category header — clickable accordion */}
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between px-4 py-3 bg-gray-50
                           hover:bg-gray-100 transition-colors cursor-pointer"
              >
                <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  {category}
                  <span className="font-normal text-gray-400 ml-2">
                    ({catFields.length})
                  </span>
                </span>
                <ChevronDown
                  className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
                    expandedCategories.has(category) ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {/* Table — shown when expanded */}
              {expandedCategories.has(category) && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-clinical-border bg-white">
                        <th className="text-left px-4 py-2.5 font-semibold text-gray-600 w-[140px]">
                          Feature
                        </th>
                        <th className="text-left px-4 py-2.5 font-semibold text-gray-600 w-[72px]">
                          Type
                        </th>
                        <th className="text-left px-4 py-2.5 font-semibold text-gray-600 w-[120px]">
                          Scale / Range
                        </th>
                        <th className="text-left px-4 py-2.5 font-semibold text-gray-600">
                          Clinical Description
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {catFields.map((field, idx) => {
                        const dictDesc = lookupMedicalTerm(field.title);
                        const displayDesc = dictDesc || field.description || '—';
                        const scale = formatScale(field);
                        const typeLabel = formatType(field);
                        const scaleHint = getScaleHint(field);

                        return (
                          <tr
                            key={field.name}
                            className={`border-b border-clinical-border/50 last:border-b-0
                                        ${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}
                                        hover:bg-blue-50/40 transition-colors`}
                          >
                            <td className="px-4 py-2.5">
                              <code className="text-[11px] font-mono font-semibold text-primary-700 bg-primary-50 px-1.5 py-0.5 rounded">
                                {field.name}
                              </code>
                            </td>
                            <td className="px-4 py-2.5">
                              <span className="text-gray-600">{typeLabel}</span>
                            </td>
                            <td className="px-4 py-2.5">
                              <div className="space-y-0.5">
                                <span className="text-gray-800 font-medium">{scale}</span>
                                {scaleHint && (
                                  <span className="block text-[10px] text-gray-400 uppercase tracking-wider">
                                    {scaleHint}
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-4 py-2.5 text-gray-600 leading-relaxed">
                              {displayDesc}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ── Footer ── */}
        <div className="shrink-0 px-6 py-3 border-t border-clinical-border bg-gray-50">
          <p className="text-[10px] text-gray-400 text-center">
            Data source: CDC BRFSS 2015 Health Indicators · Scales reflect validated schema constraints
          </p>
        </div>
      </div>

    </div>
  );
}
