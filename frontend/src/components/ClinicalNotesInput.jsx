import { useState } from 'react';
import { FileText, Loader2, Sparkles, AlertCircle, Check, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

/**
 * Textarea that parses free-text clinical notes into schema fields.
 *
 * Nothing is applied automatically. The extracted fields are shown as a
 * checklist, the clinician unticks anything wrong, and only "Apply" merges
 * the ticked fields into the patient — the prediction reruns after that.
 * Applying on parse let a misread note ("no stroke" → Stroke = 1) change a
 * screening result with no human check.
 *
 * Only disease-mapped fields (schema names) are ever applied; the parser's
 * raw keys (e.g. `bp_diastolic`) are not model inputs.
 *
 * The parser is English-only. When the note contains Arabic, the server says
 * so (`language.supported === false`) and that is shown prominently instead
 * of the generic "no fields recognised" note — which told a clinician their
 * note was empty rather than that the tool cannot read their language.
 *
 * Props:
 *   onExtracted — (fields: object) => void, called with the confirmed fields
 *   disease     — string (e.g. "heart_disease", "diabetes")
 */
export default function ClinicalNotesInput({ onExtracted, disease }) {
  const { token } = useAuth();
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState({});
  const [applied, setApplied] = useState(null);
  const [error, setError] = useState('');

  const fields = result?.mapped_features || {};
  const fieldNames = Object.keys(fields);
  const selectedNames = fieldNames.filter((k) => selected[k]);

  async function handleParse() {
    if (!note.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    setApplied(null);
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
        throw new Error(body?.detail?.error || body?.error || res.statusText);
      }
      const data = await res.json();
      setResult(data);
      setSelected(Object.fromEntries(Object.keys(data.mapped_features || {}).map((k) => [k, true])));
    } catch (err) {
      setError(err.message || 'Failed to parse note');
    } finally {
      setLoading(false);
    }
  }

  function handleApply() {
    const confirmed = Object.fromEntries(selectedNames.map((k) => [k, fields[k]]));
    if (onExtracted && selectedNames.length > 0) onExtracted(confirmed);
    setApplied(selectedNames.length);
    setResult(null);
  }

  function handleDiscard() {
    setResult(null);
    setApplied(0);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <FileText className="w-4 h-4 text-purple-600" />
        <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
          Clinical Notes Parser
          <span className="ml-2 text-xs font-normal text-purple-600 bg-purple-50 dark:bg-purple-900/30 px-1.5 py-0.5 rounded">
            Rule-based · English only
          </span>
        </h3>
      </div>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder={`Paste an English clinical note here…\n\nExample:\n"65-year-old male with hypertension. BP 145/90. Total cholesterol 240 mg/dl. BMI 29.5. Non-smoker."`}
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
          <><Sparkles className="w-4 h-4" /> Extract Fields</>
        )}
      </button>

      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Language first: an Arabic note that extracted nothing is not the
          same event as an English note that extracted nothing. */}
      {result?.language && result.language.supported === false && (
        <div
          dir="auto"
          className="flex items-start gap-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-800 p-3 text-sm text-amber-900 dark:text-amber-200"
        >
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">
              {result.language.script === 'arabic'
                ? 'Arabic notes are not supported'
                : 'Mixed-language note — only the English parts were read'}
            </p>
            <p className="mt-0.5">{result.language.message}</p>
            <p className="mt-1 text-xs" lang="ar" dir="rtl">
              هذا المحلّل يقرأ الملاحظات الإنجليزية فقط؛ النص العربي لم يُقرأ.
            </p>
          </div>
        </div>
      )}

      {result && fieldNames.length === 0 && result.language?.supported !== false && (
        <p className="text-xs text-purple-700 dark:text-purple-300 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-3">
          No fields recognised — nothing was changed. The parser reads English notes with explicit
          values (e.g. &quot;BP 140/90&quot;, &quot;age 55&quot;, &quot;non-smoker&quot;).
        </p>
      )}

      {result && fieldNames.length > 0 && (
        <div className="rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 p-3 space-y-2">
          <p className="text-xs font-semibold text-purple-800 dark:text-purple-300">
            {fieldNames.length} field{fieldNames.length !== 1 ? 's' : ''} found — review before applying.
            Untick anything that is wrong.
          </p>
          <ul className="space-y-1">
            {fieldNames.map((k) => (
              <li key={k}>
                <label className="flex items-center gap-2 text-xs text-purple-900 dark:text-purple-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={Boolean(selected[k])}
                    onChange={(e) => setSelected((s) => ({ ...s, [k]: e.target.checked }))}
                  />
                  <span className="font-medium">{k}</span>
                  <span className="text-purple-400">→</span>
                  <span className="font-mono">{String(fields[k])}</span>
                </label>
              </li>
            ))}
          </ul>
          <div className="flex gap-2 pt-1">
            <button
              onClick={handleApply}
              disabled={selectedNames.length === 0}
              className="btn-primary text-xs flex items-center gap-1.5"
            >
              <Check className="w-3.5 h-3.5" /> Apply {selectedNames.length} field{selectedNames.length !== 1 ? 's' : ''}
            </button>
            <button onClick={handleDiscard} className="btn-secondary text-xs flex items-center gap-1.5">
              <X className="w-3.5 h-3.5" /> Discard
            </button>
          </div>
        </div>
      )}

      {applied !== null && !result && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {applied > 0
            ? `${applied} field${applied !== 1 ? 's' : ''} applied — screening re-run with the confirmed values.`
            : 'Nothing applied.'}
        </p>
      )}
    </div>
  );
}
