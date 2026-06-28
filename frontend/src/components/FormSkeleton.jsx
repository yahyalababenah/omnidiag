/**
 * FormSkeleton.jsx
 * ================
 * A shimmer-animated loading skeleton that mimics the form grid layout
 * while the schema is being fetched from the backend.
 *
 * Renders placeholder divs sized like toggle switches, number inputs,
 * and dropdowns to give a realistic "almost there" feel.
 */

import { Loader2 } from 'lucide-react';

/**
 * @param {{ fieldCount?: number, diseaseName?: string }} props
 */
export default function FormSkeleton({ fieldCount = 8, diseaseName = '' }) {
  // Generate placeholder rows
  const placeholders = Array.from({ length: fieldCount }, (_, i) => {
    // Alternate between field shapes for visual variety
    const type = i % 3;
    return (
      <div key={i} className="space-y-2">
        <div className="h-3 w-24 bg-gray-200 rounded shimmer" />
        <div
          className={`bg-gray-200 rounded shimmer ${
            type === 0 ? 'h-9 w-full'       // select / number input
              : type === 1 ? 'h-9 w-3/4'     // shorter input
                : 'h-6 w-full'               // toggle row
          }`}
        />
      </div>
    );
  });

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />
          <h2 className="text-lg font-semibold text-gray-400">
            Loading Patient Parameters
          </h2>
        </div>
      </div>
      <div className="card-body">
        <p className="text-xs text-gray-400 mb-4">
          Fetching schema for <span className="font-mono">{diseaseName || '...'}</span>
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {placeholders}
        </div>
      </div>

    </div>
  );
}
