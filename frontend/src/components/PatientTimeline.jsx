/**
 * PatientTimeline — Feature 3.4
 *
 * Shows a patient's complete prediction history as a vertical timeline,
 * with a risk-trend sparkline at the top. Requires the user to be logged in
 * (token from AuthContext) since patient APIs require clinical roles.
 *
 * Props:
 *   patientId  — UUID of the patient (required)
 *   patientName — display name (optional, for the header)
 *   onClose    — callback to close/dismiss the panel (optional)
 */

import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import {
  Activity, ChevronDown, ChevronUp, ChevronLeft, ChevronRight,
  AlertCircle, TrendingUp, TrendingDown, Clock, X,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const BASE = '/api/v4'

async function apiFetch(path, token) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error || body?.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Risk trend sparkline tooltip ──────────────────────────────────────────────

function SparkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const p = payload[0]
  return (
    <div className="bg-white border border-clinical-border rounded-lg shadow-lg px-3 py-2 text-xs">
      <p className="text-gray-500 mb-0.5">{label}</p>
      <p className={`font-semibold ${p.value >= 0.5 ? 'text-red-600' : 'text-green-600'}`}>
        Confidence: {Math.round(p.value * 100)}%
      </p>
    </div>
  )
}

// ── Prediction entry card ─────────────────────────────────────────────────────

function PredictionCard({ prediction, index }) {
  const [expanded, setExpanded] = useState(false)

  const isPositive = prediction.prediction === 1
  const conf = Math.round((prediction.confidence ?? 0) * 100)
  const dateStr = new Date(prediction.created_at).toLocaleDateString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
  })
  const timeStr = new Date(prediction.created_at).toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit',
  })

  const features = prediction.input_features || {}
  const featureKeys = Object.keys(features)

  return (
    <div className="flex gap-4">
      {/* Timeline dot + line */}
      <div className="flex flex-col items-center flex-shrink-0">
        <div className={`w-4 h-4 rounded-full border-2 flex-shrink-0 mt-1 ${
          isPositive
            ? 'bg-red-100 border-red-500'
            : 'bg-green-100 border-green-500'
        }`} />
        <div className="w-0.5 bg-clinical-border flex-1 mt-1" />
      </div>

      {/* Card */}
      <div className="card flex-1 mb-4">
        {/* Header */}
        <button
          className="card-header flex items-center justify-between w-full text-left"
          onClick={() => setExpanded(e => !e)}
        >
          <div className="flex items-center gap-3 min-w-0">
            <span className={`px-2.5 py-1 rounded-full text-xs font-semibold flex-shrink-0 ${
              isPositive ? 'badge-positive' : 'badge-negative'
            }`}>
              {prediction.diagnosis ?? (isPositive ? 'Positive' : 'Negative')}
            </span>
            <div className="min-w-0">
              <p className="text-xs font-medium text-gray-900 truncate capitalize">
                {prediction.disease?.replace(/_/g, ' ')}
              </p>
              <p className="text-[10px] text-gray-500 flex items-center gap-1">
                <Clock className="w-3 h-3" /> {dateStr} · {timeStr}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0 ml-2">
            {/* Confidence bar */}
            <div className="hidden sm:block">
              <p className="text-[10px] text-gray-500 text-right mb-0.5">Confidence {conf}%</p>
              <div className="w-20 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${isPositive ? 'bg-red-500' : 'bg-green-500'}`}
                  style={{ width: `${conf}%` }}
                />
              </div>
            </div>
            {expanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
          </div>
        </button>

        {/* Expanded details */}
        {expanded && (
          <div className="card-body pt-0">
            <div className="border-t border-clinical-border pt-4">
              <p className="text-xs font-medium text-gray-700 mb-3">Input Features</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-2">
                {featureKeys.map(k => (
                  <div key={k}>
                    <p className="text-[10px] text-gray-400 uppercase tracking-wide">{k}</p>
                    <p className="text-xs font-medium text-gray-900">{String(features[k])}</p>
                  </div>
                ))}
                {featureKeys.length === 0 && (
                  <p className="text-xs text-gray-400 col-span-full">No feature data recorded</p>
                )}
              </div>

              {prediction.shap_chart_data && (
                <div className="mt-4">
                  <p className="text-xs font-medium text-gray-700 mb-2">Top SHAP Features</p>
                  <div className="space-y-1.5">
                    {prediction.shap_chart_data.slice(0, 5).map((f, i) => {
                      const isNeg = f.shap_value < 0
                      const barWidth = Math.min(100, Math.abs(f.shap_value) * 100)
                      return (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="w-24 text-gray-600 truncate text-right text-[11px]">{f.feature}</span>
                          <div className="flex-1 flex items-center gap-1">
                            {isNeg && (
                              <div className="flex-1 flex justify-end">
                                <div className="h-3 rounded-sm bg-green-500 opacity-75" style={{ width: `${barWidth}%` }} />
                              </div>
                            )}
                            {!isNeg && (
                              <div className="flex-1">
                                <div className="h-3 rounded-sm bg-red-500 opacity-75" style={{ width: `${barWidth}%` }} />
                              </div>
                            )}
                          </div>
                          <span className={`w-12 text-right font-mono text-[10px] ${isNeg ? 'text-green-600' : 'text-red-600'}`}>
                            {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Risk trend summary ────────────────────────────────────────────────────────

function RiskTrendCard({ predictions }) {
  if (predictions.length < 2) return null

  const chartData = [...predictions]
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .map(p => ({
      date: new Date(p.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
      confidence: p.prediction === 1 ? p.confidence : (1 - p.confidence),
      raw: p.confidence,
      positive: p.prediction === 1,
    }))

  const latest = chartData[chartData.length - 1]
  const prev = chartData[chartData.length - 2]
  const trend = latest.confidence - prev.confidence
  const improving = trend < 0

  return (
    <div className="card mb-6">
      <div className="card-header flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary-600" /> Risk Trend
        </h3>
        <div className={`flex items-center gap-1 text-xs font-medium ${improving ? 'text-green-600' : 'text-red-600'}`}>
          {improving ? <TrendingDown className="w-4 h-4" /> : <TrendingUp className="w-4 h-4" />}
          {improving ? 'Improving' : 'Worsening'}
          <span className="text-gray-400 font-normal ml-1">vs prior visit</span>
        </div>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fontSize: 10 }} />
            <Tooltip content={<SparkTooltip />} />
            <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 2" label={{ value: '50%', fontSize: 9, fill: '#94a3b8' }} />
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#2563eb"
              strokeWidth={2}
              dot={{ r: 4, fill: '#2563eb' }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-gray-400 mt-2 text-center">
          Risk score = model confidence toward positive prediction
        </p>
      </div>
    </div>
  )
}

// ── Main PatientTimeline ──────────────────────────────────────────────────────

export default function PatientTimeline({ patientId, patientName, onClose }) {
  const { token } = useAuth()
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [page, setPage]       = useState(1)
  const [disease, setDisease] = useState('')

  const load = useCallback(async () => {
    if (!patientId || !token) return
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({ page, limit: 10, ...(disease ? { disease } : {}) })
      const d = await apiFetch(`/patients/${patientId}/predictions?${qs}`, token)
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [patientId, token, page, disease])

  useEffect(() => { load() }, [load])

  if (!token) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <AlertCircle className="w-10 h-10 text-gray-300 mb-3" />
        <p className="text-sm text-gray-500">Sign in to view patient history.</p>
      </div>
    )
  }

  const diseases = data?.items
    ? [...new Set(data.items.map(p => p.disease))].filter(Boolean)
    : []

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary-600" />
            Patient History
          </h2>
          {patientName && (
            <p className="text-sm text-gray-500 mt-0.5">{patientName}</p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Disease filter */}
          {diseases.length > 0 && (
            <select
              className="select-field text-xs py-1.5 w-36"
              value={disease}
              onChange={e => { setDisease(e.target.value); setPage(1) }}
            >
              <option value="">All diseases</option>
              {diseases.map(d => (
                <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>
              ))}
            </select>
          )}
          {onClose && (
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors">
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="card-header h-14 bg-gray-50" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <>
          {/* Summary */}
          <div className="flex items-center gap-4 mb-6 text-sm">
            <span className="text-gray-500">
              <span className="font-semibold text-gray-900">{data.total}</span> prediction{data.total !== 1 ? 's' : ''}
            </span>
            {data.items.length > 0 && (
              <>
                <span className="text-gray-300">·</span>
                <span className="text-gray-500">
                  <span className="font-semibold text-red-600">
                    {data.items.filter(p => p.prediction === 1).length}
                  </span> positive
                </span>
                <span className="text-gray-300">·</span>
                <span className="text-gray-500">
                  <span className="font-semibold text-green-600">
                    {data.items.filter(p => p.prediction === 0).length}
                  </span> negative
                </span>
              </>
            )}
          </div>

          {/* Risk trend */}
          <RiskTrendCard predictions={data.items} />

          {/* Timeline */}
          {data.items.length === 0 ? (
            <div className="text-center py-16">
              <Clock className="w-10 h-10 text-gray-200 mx-auto mb-3" />
              <p className="text-sm text-gray-400">No predictions recorded yet.</p>
            </div>
          ) : (
            <div>
              {data.items.map((pred, i) => (
                <PredictionCard key={pred.id} prediction={pred} index={i} />
              ))}
            </div>
          )}

          {/* Pagination */}
          {data.pages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <span className="text-xs text-gray-500">Page {data.page} of {data.pages}</span>
              <div className="flex gap-2">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage(p => p - 1)}
                  className="btn-secondary py-1.5 px-3 text-xs disabled:opacity-40 flex items-center gap-1"
                >
                  <ChevronLeft className="w-3.5 h-3.5" /> Prev
                </button>
                <button
                  disabled={page >= data.pages}
                  onClick={() => setPage(p => p + 1)}
                  className="btn-secondary py-1.5 px-3 text-xs disabled:opacity-40 flex items-center gap-1"
                >
                  Next <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
