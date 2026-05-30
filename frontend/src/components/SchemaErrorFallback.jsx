/**
 * SchemaErrorFallback.jsx
 * =======================
 * Error UI shown when the JSON Schema fails to load for a disease.
 *
 * Features:
 *   - Clear error message explaining what went wrong
 *   - Retry button
 *   - Suggestion to switch disease
 *   - Clean, non-alarming design
 */

import { AlertCircle, RefreshCw, Activity } from 'lucide-react';

/**
 * @param {{
 *   error: Error | null,
 *   onRetry: () => void,
 *   disease?: string,
 * }} props
 */
export default function SchemaErrorFallback({ error, onRetry, disease }) {
  const isConnectionError =
    error?.message?.toLowerCase().includes('network') ||
    error?.message?.toLowerCase().includes('fetch') ||
    error?.message?.toLowerCase().includes('abort') ||
    !error;

  return (
    <div className="card border-amber-200">
      <div className="card-body">
        <div className="flex flex-col items-center text-center py-6">
          {/* Icon */}
          <div className="w-12 h-12 rounded-full bg-amber-100 flex items-center justify-center mb-4">
            <AlertCircle className="w-6 h-6 text-amber-600" />
          </div>

          {/* Title */}
          <h3 className="text-base font-semibold text-gray-800 mb-2">
            {isConnectionError
              ? 'Unable to Load Form'
              : 'Schema Load Error'}
          </h3>

          {/* Description */}
          <p className="text-sm text-gray-600 max-w-md mb-1">
            {isConnectionError
              ? `The diagnostic engine could not be reached. This is normal during cold starts (first scan of the day).`
              : `Failed to load the form configuration for "${disease || 'this disease'}".`}
          </p>

          {/* Error detail (if available) */}
          {error?.message && !isConnectionError && (
            <p className="text-xs text-red-500 font-mono mt-2 mb-3 px-4 py-2 bg-red-50 rounded-lg max-w-md break-all">
              {error.message}
            </p>
          )}

          {isConnectionError && (
            <p className="text-xs text-amber-600 mt-2 mb-3">
              The backend may be waking up — this can take 30-60 seconds on the first request.
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 mt-2">
            <button
              onClick={onRetry}
              className="btn-primary text-sm"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>

            {disease && (
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <Activity className="w-3.5 h-3.5" />
                <span>{disease}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
