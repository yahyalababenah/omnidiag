import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, AlertCircle, Loader2, Activity } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import { useAuth } from '../context/AuthContext';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

function RiskBadge({ score }) {
  if (score >= 0.7) return <span className="badge-positive text-xs">HIGH</span>;
  if (score >= 0.4) return <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">MOD</span>;
  return <span className="badge-negative text-xs">LOW</span>;
}

function TrendIcon({ visits }) {
  if (visits.length < 2) return <Minus className="w-4 h-4 text-gray-400" />;
  const delta = visits[visits.length - 1].risk_score - visits[0].risk_score;
  if (delta > 0.05) return <TrendingUp className="w-4 h-4 text-red-500" />;
  if (delta < -0.05) return <TrendingDown className="w-4 h-4 text-green-500" />;
  return <Minus className="w-4 h-4 text-gray-400" />;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow p-3 text-xs">
      <p className="font-semibold text-gray-700 dark:text-gray-300 mb-1">{label}</p>
      <p className="text-gray-600 dark:text-gray-400">
        Risk: <span className="font-mono font-bold text-gray-900 dark:text-white">{(v * 100).toFixed(1)}%</span>
      </p>
    </div>
  );
}

/**
 * Displays a patient's longitudinal risk trajectory as a line chart + visit list.
 *
 * Props:
 *   patientId  — patient UUID
 *   disease    — optional disease filter string
 */
export default function PatientRiskTimeline({ patientId, disease }) {
  const { token } = useAuth();
  const [visits, setVisits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedDisease, setSelectedDisease] = useState(disease || '');

  useEffect(() => {
    if (!patientId) return;
    loadVisits();
  }, [patientId, selectedDisease]);

  async function loadVisits() {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ limit: '100' });
      if (selectedDisease) params.set('disease', selectedDisease);
      const res = await fetch(`${API_BASE}/api/v4/patients/${patientId}/visits?${params}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const data = await res.json();
      setVisits(data.visits || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const chartData = visits.map((v) => ({
    date: new Date(v.visit_date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
    risk: v.risk_score,
    rawDate: v.visit_date,
  }));

  const diseases = [...new Set(visits.map((v) => v.disease))];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Risk Timeline</h3>
          {visits.length > 0 && <TrendIcon visits={visits} />}
        </div>
        <select
          value={selectedDisease}
          onChange={(e) => setSelectedDisease(e.target.value)}
          className="select-field text-xs py-1 px-2"
        >
          <option value="">All diseases</option>
          {diseases.map((d) => (
            <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="flex justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {!loading && !error && visits.length === 0 && (
        <p className="text-center text-sm text-gray-400 py-6">
          No visits recorded yet. Risk data will appear here after predictions are saved as visits.
        </p>
      )}

      {chartData.length > 0 && (
        <>
          {/* Line Chart */}
          <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-4">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: '50%', fontSize: 10, fill: '#f59e0b' }} />
                <ReferenceLine y={0.7} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '70%', fontSize: 10, fill: '#ef4444' }} />
                <Line
                  type="monotone"
                  dataKey="risk"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 4, fill: '#3b82f6' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Visit Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 dark:text-gray-400">Date</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 dark:text-gray-400">Disease</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 dark:text-gray-400">Risk Score</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 dark:text-gray-400">Band</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 dark:text-gray-400">Notes</th>
                </tr>
              </thead>
              <tbody>
                {[...visits].reverse().map((v) => (
                  <tr key={v.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <td className="py-2 px-3 font-mono text-xs text-gray-600 dark:text-gray-400">
                      {new Date(v.visit_date).toLocaleDateString()}
                    </td>
                    <td className="py-2 px-3 text-xs text-gray-700 dark:text-gray-300 capitalize">
                      {v.disease.replace(/_/g, ' ')}
                    </td>
                    <td className="py-2 px-3 font-mono text-xs font-semibold text-gray-900 dark:text-white">
                      {(v.risk_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3">
                      <RiskBadge score={v.risk_score} />
                    </td>
                    <td className="py-2 px-3 text-xs text-gray-500 dark:text-gray-400 max-w-[200px] truncate">
                      {v.notes || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
