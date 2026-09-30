import { useState } from 'react';
import { CheckCircle, XCircle, Loader2, Brain } from 'lucide-react';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';

const LABEL_ROLES = ['doctor', 'nurse', 'super_admin', 'admin'];

/**
 * Human-in-the-loop control for the patient view. Rendered only when the
 * screening was queued for review (the predict response carries `review_id`)
 * and the signed-in user holds a clinical role. The clinician's label is the
 * ground truth the next retraining cycle reads.
 */
export default function ReviewLabelBox({ reviewId }) {
  const { user } = useAuth();
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(null);
  const [error, setError] = useState('');
  const [notes, setNotes] = useState('');

  const roles = (user?.roles ?? []).map((r) => (typeof r === 'string' ? r : r?.name));
  if (!reviewId || !roles.some((r) => LABEL_ROLES.includes(r))) return null;

  async function submit(label) {
    setBusy(true);
    setError('');
    try {
      await api.annotateReview(reviewId, label, notes.trim() || null);
      setDone(label);
    } catch (e) {
      setError(e.message || 'Could not save the label.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-amber-600" />
        <p className="text-sm font-semibold text-amber-900">Model is unsure about this patient</p>
      </div>
      {done !== null ? (
        <p className="text-sm text-green-700">
          Recorded as {done === 1 ? 'Positive' : 'Negative'}. Thank you — this label feeds the next retraining cycle.
        </p>
      ) : (
        <>
          <p className="text-xs text-amber-800">
            If you have confirmed the diagnosis, label this case. Leave it unlabeled if you are not sure.
          </p>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Reasoning (optional)"
            rows={2}
            className="w-full text-xs rounded-lg border border-amber-200 p-2 bg-white"
          />
          <div className="flex gap-2">
            <button onClick={() => submit(1)} disabled={busy}
              className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-red-100 hover:bg-red-200 text-red-700 text-xs font-medium py-2 disabled:opacity-50">
              {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle className="w-3.5 h-3.5" />} Positive
            </button>
            <button onClick={() => submit(0)} disabled={busy}
              className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-green-100 hover:bg-green-200 text-green-700 text-xs font-medium py-2 disabled:opacity-50">
              {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <XCircle className="w-3.5 h-3.5" />} Negative
            </button>
          </div>
          {error && <p className="text-xs text-red-600">{error}</p>}
        </>
      )}
    </div>
  );
}
