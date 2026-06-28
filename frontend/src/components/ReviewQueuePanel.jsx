import { useState, useEffect, useCallback } from 'react';
import { CheckCircle, XCircle, SkipForward, Loader2, AlertCircle, Brain, RefreshCw } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

function UncertaintyBar({ score }) {
  const pct = Math.round((score ?? 0) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
        <div
          className="bg-amber-500 h-1.5 rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-gray-600 dark:text-gray-400">{pct}%</span>
    </div>
  );
}

function ReviewCard({ item, onAnnotate, onSkip }) {
  const [busy, setBusy] = useState(false);

  async function handleAction(action) {
    setBusy(true);
    if (action === 'skip') {
      await onSkip(item.id);
    } else {
      await onAnnotate(item.id, action === 'positive' ? 1 : 0);
    }
    setBusy(false);
  }

  return (
    <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/10 p-4 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">{item.id.slice(0, 8)}…</p>
          <p className="font-semibold text-gray-900 dark:text-white capitalize text-sm">
            {(item.disease || 'unknown').replace(/_/g, ' ')}
          </p>
        </div>
        <div className="text-right text-xs">
          <p className="text-gray-500 dark:text-gray-400">Model said</p>
          <p className={`font-bold ${item.model_prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
            {item.model_prediction === 1 ? 'Positive' : 'Negative'}
          </p>
          <p className="text-gray-500 dark:text-gray-400 font-mono">{((item.confidence ?? 0) * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Uncertainty */}
      <div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Uncertainty (entropy)</p>
        <UncertaintyBar score={item.uncertainty_score} />
      </div>

      {/* Feature preview */}
      {item.features && (
        <details className="text-xs">
          <summary className="cursor-pointer text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
            Patient features ({Object.keys(item.features).length})
          </summary>
          <div className="mt-2 grid grid-cols-2 gap-1 max-h-24 overflow-y-auto">
            {Object.entries(item.features).slice(0, 12).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-1">
                <span className="text-gray-500 dark:text-gray-400 truncate">{k}</span>
                <span className="font-mono text-gray-800 dark:text-gray-200 shrink-0">{String(v)}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={() => handleAction('positive')}
          disabled={busy}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-400 text-xs font-medium py-1.5 transition-colors disabled:opacity-50"
        >
          <CheckCircle className="w-3.5 h-3.5" />
          Positive
        </button>
        <button
          onClick={() => handleAction('negative')}
          disabled={busy}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-green-100 hover:bg-green-200 dark:bg-green-900/30 dark:hover:bg-green-900/50 text-green-700 dark:text-green-400 text-xs font-medium py-1.5 transition-colors disabled:opacity-50"
        >
          <XCircle className="w-3.5 h-3.5" />
          Negative
        </button>
        <button
          onClick={() => handleAction('skip')}
          disabled={busy}
          className="px-3 flex items-center gap-1 rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs font-medium py-1.5 transition-colors disabled:opacity-50"
        >
          {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <SkipForward className="w-3.5 h-3.5" />}
          Skip
        </button>
      </div>
    </div>
  );
}

/**
 * Active Learning review queue panel.
 * Shows uncertain predictions awaiting expert annotation.
 */
export default function ReviewQueuePanel() {
  const { token } = useAuth();
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  const load = useCallback(async (p = 1) => {
    setLoading(true);
    setError('');
    try {
      const [queueRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/api/v4/review/queue?page=${p}&limit=10`, { headers }),
        fetch(`${API_BASE}/api/v4/review/stats`, { headers }),
      ]);
      if (!queueRes.ok) throw new Error((await queueRes.json().catch(() => ({}))).detail || queueRes.statusText);
      const queueData = await queueRes.json();
      const statsData = statsRes.ok ? await statsRes.json() : null;
      setItems(queueData.items || []);
      setTotalPages(queueData.pages || 1);
      setStats(statsData);
      setPage(p);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { load(1); }, [load]);

  async function handleAnnotate(id, label) {
    await fetch(`${API_BASE}/api/v4/review/${id}/annotate`, {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    load(page);
  }

  async function handleSkip(id) {
    await fetch(`${API_BASE}/api/v4/review/${id}/skip`, {
      method: 'POST',
      headers,
    });
    load(page);
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-amber-600" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Review Queue</h3>
          <span className="text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 px-2 py-0.5 rounded-full font-medium">
            Active Learning
          </span>
        </div>
        <button onClick={() => load(1)} className="btn-secondary text-xs flex items-center gap-1">
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* Stats strip */}
      {stats && (
        <div className="grid grid-cols-3 gap-3 text-center">
          {[
            { label: 'Pending', value: stats.pending, color: 'text-amber-600' },
            { label: 'Reviewed', value: stats.reviewed, color: 'text-green-600' },
            { label: 'Skipped', value: stats.skipped, color: 'text-gray-500' },
          ].map(({ label, value, color }) => (
            <div key={label} className="rounded-lg bg-gray-50 dark:bg-gray-800 p-2">
              <p className={`text-lg font-bold ${color}`}>{value}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Items */}
      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-amber-500" />
        </div>
      ) : items.length === 0 ? (
        <p className="text-center text-sm text-gray-400 py-8">
          No pending items. Uncertain predictions will appear here automatically.
        </p>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <ReviewCard key={item.id} item={item} onAnnotate={handleAnnotate} onSkip={handleSkip} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center gap-2 pt-2">
          <button disabled={page === 1} onClick={() => load(page - 1)} className="btn-secondary text-xs px-3 py-1 disabled:opacity-40">←</button>
          <span className="text-xs text-gray-500 dark:text-gray-400 self-center">
            {page} / {totalPages}
          </span>
          <button disabled={page === totalPages} onClick={() => load(page + 1)} className="btn-secondary text-xs px-3 py-1 disabled:opacity-40">→</button>
        </div>
      )}
    </div>
  );
}
