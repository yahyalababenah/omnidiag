/**
 * MedicalTooltip.jsx
 * ===================
 * A lightweight, accessible hover tooltip for medical terms and abbreviations.
 *
 * Uses pure Tailwind CSS with no external dependencies.
 * Features:
 *   - Safe 400ms hover delay before showing (reduces false triggers)
 *   - z-50 stacking for proper overlay above charts/modals
 *   - Arrow indicator pointing to the anchor element
 *   - Smooth fade-in transition
 *   - Fallback: if no dictionary entry found, renders children unchanged
 */

import { lookupMedicalTerm } from '../utils/medicalDictionary';

/**
 * MedicalTooltip — wraps children with an interactive tooltip.
 *
 * @param {{
 *   term: string,          // The term/label to look up in the dictionary
 *   children: ReactNode,   // The rendered element that triggers the tooltip
 *   className?: string,    // Additional classes for the wrapper
 * }} props
 */
export default function MedicalTooltip({ term, children, className = '' }) {
  const description = lookupMedicalTerm(term);

  // No dictionary entry — render children as-is (no tooltip wrapper)
  if (!description) {
    return <>{children}</>;
  }

  return (
    <span className={`group/tooltip relative inline-flex ${className}`}>
      {children}
      <span
        role="tooltip"
        className="
          invisible group-hover/tooltip:visible
          opacity-0 group-hover/tooltip:opacity-100
          transition-all duration-200 delay-[400ms]
          absolute bottom-full left-1/2 -translate-x-1/2 mb-2
          z-50
          max-w-[280px] w-max
          px-3 py-2
          rounded-lg
          bg-gray-900 text-white
          text-xs leading-relaxed
          shadow-lg
          pointer-events-none
        "
      >
        {description}
        {/* Arrow */}
        <span
          className="
            absolute top-full left-1/2 -translate-x-1/2
            w-0 h-0
            border-4 border-transparent border-t-gray-900
          "
        />
      </span>
    </span>
  );
}
