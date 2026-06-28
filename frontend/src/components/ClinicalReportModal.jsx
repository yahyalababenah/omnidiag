import { useState } from 'react';
import { X, FileText, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';

/**
 * Modal that generates and displays an AI clinical report for a single prediction.
 * Opens when the user clicks "Generate AI Report" after a prediction + SHAP explain.
 *
 * Props:
 *   disease        — disease key (e.g. 'heart_disease')
 *   probability    — prediction probability (0–1)
 *   label          — prediction label string (e.g. 'Positive')
 *   confidenceBand — string (e.g. 'HIGH', 'MODERATE', 'LOW')
 *   shapValues     — array of { feature, shap_value }
 *   features       — original patient feature dict
 *   onClose        — () => void
 */
export default function ClinicalReportModal({
  disease,
  probability,
  label,
  confidenceBand,
  shapValues = [],
  features = {},
  onClose,
}) {
  const { token } = useAuth();
  const [report, setReport] = useState('');
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleGenerate() {
    setLoading(true);
    setError('');
    setReport('');
    try {
      const res = await api.generateReport(
        {
          disease,
          probability,
          label,
          confidence_band: confidenceBand,
          shap_values: shapValues,
          features,
        },
        token
      );
      setReport(res.report);
      setSource(res.source);
    } catch (err) {
      setError(err.message || 'Failed to generate report');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="relative w-full max-w-2xl rounded-xl bg-white dark:bg-gray-900 shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              AI Clinical Report
            </h2>
            {source && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                source === 'llm'
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                  : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
              }`}>
                {source === 'llm' ? 'AI Generated' : 'Rule-Based'}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            aria-label="Close report"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Prediction summary */}
          <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-4 text-sm grid grid-cols-3 gap-3">
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-xs mb-0.5">Disease</p>
              <p className="font-medium text-gray-900 dark:text-white capitalize">
                {disease.replace(/_/g, ' ')}
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-xs mb-0.5">Probability</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {(probability * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-xs mb-0.5">Risk Band</p>
              <p className={`font-semibold ${
                confidenceBand === 'HIGH' ? 'text-red-600' :
                confidenceBand === 'MODERATE' ? 'text-amber-600' : 'text-green-600'
              }`}>
                {confidenceBand}
              </p>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          {/* Report output */}
          {report && (
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
              <div className="flex items-center gap-1.5 mb-3 text-xs text-gray-500 dark:text-gray-400">
                <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                Report generated successfully
              </div>
              <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap text-gray-800 dark:text-gray-200 font-mono text-xs leading-relaxed">
                {report}
              </div>
            </div>
          )}

          {!report && !loading && !error && (
            <p className="text-center text-sm text-gray-500 dark:text-gray-400 py-6">
              Click "Generate Report" to create a structured clinical narrative from this prediction.
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="btn-secondary text-sm"
          >
            Close
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="btn-primary text-sm flex items-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating…
              </>
            ) : (
              <>
                <FileText className="w-4 h-4" />
                Generate Report
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
