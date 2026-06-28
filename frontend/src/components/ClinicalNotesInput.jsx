import { useState } from 'react';
import { FileText, Loader2, Sparkles, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

/**
 * Textarea that parses free-text clinical notes and pre-fills the parent form.
 *
 * Props:
 *   onExtracted — (mappedFields: object) => void
 *                 Called with disease-mapped feature names so the parent
 *                 can merge them directly into patient.data.
 *   disease     — string (e.g. "heart_disease", "diabetes") passed to backend
 *                 so it returns schema-keyed mapped_features.
 */
export default function ClinicalNotesInput({ onExtracted, disease }) {
  const { token } = useAuth();
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  async function handleParse() {
    if (!note.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/v4/parse-notes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ note, disease: disease || null, use_bert: false }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || res.statusText);
      }
      const data = await res.json();
      setResult(data);
      if (onExtracted) {
        const fields = Object.keys(data.mapped_features || {}).length > 0
          ? data.mapped_features
          : data.extracted_features;
        onExtracted(fields);
      }
    } catch (err) {
      setError(err.message || 'Failed to parse note');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <FileText className="w-4 h-4 text-purple-600" />
        <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
          Clinical Notes Parser
          <span className="ml-2 text-xs font-normal text-purple-600 bg-purple-50 dark:bg-purple-900/30 px-1.5 py-0.5 rounded">
            NLP
          </span>
        </h3>
      </div>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder={`Paste free-text clinical note here…\n\nExample:\n"65-year-old male with hypertension and diabetes. BP 145/90. Cholesterol 240 mg/dl. BMI 29.5. Smoker."`}
        rows={6}
        className="input-field w-full resize-none text-sm font-mono"
      />

      <button
        onClick={handleParse}
        disabled={loading || !note.trim()}
        className="btn-primary text-sm flex items-center gap-2"
      >
        {loading ? (
          <><Loader2 className="w-4 h-4 animate-spin" /> Parsing…</>
        ) : (
          <><Sparkles className="w-4 h-4" /> Extract Features</>
        )}
      </button>

      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result && (
        <div className="rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 p-3">
          <p className="text-xs font-semibold text-purple-800 dark:text-purple-300 mb-2">
            {result.field_count} field{result.field_count !== 1 ? 's' : ''} extracted — patient data updated
          </p>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(
              Object.keys(result.mapped_features || {}).length > 0
                ? result.mapped_features
                : result.extracted_features
            ).map(([k, v]) => (
              <span
                key={k}
                className="inline-flex items-center gap-1 text-xs bg-white dark:bg-gray-800 border border-purple-200 dark:border-purple-700 rounded px-2 py-0.5 text-purple-700 dark:text-purple-300"
              >
                <span className="font-medium">{k}</span>
                <span className="text-purple-400">→</span>
                <span>{String(v)}</span>
              </span>
            ))}
          </div>
          {result.field_count === 0 && (
            <p className="text-xs text-purple-600 dark:text-purple-400">
              No structured fields detected. Try adding explicit values (e.g., "BP 140/90", "age 55").
            </p>
          )}
        </div>
      )}
    </div>
  );
}
