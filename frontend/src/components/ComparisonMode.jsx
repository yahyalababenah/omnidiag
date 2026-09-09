/**
 * ComparisonMode — Feature 3.6
 *
 * Side-by-side comparison of two prediction scenarios (Before/After).
 * Users pick two mock patients (or two sets of parameters) and the component
 * runs both predictions and highlights the differences.
 *
 * Works with Engineering Mode style (no auth required) so it uses the
 * public /predict endpoint via the existing api utility.
 */

import { useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, ResponsiveContainer,
} from 'recharts'
import {
  ArrowLeftRight, TrendingUp, TrendingDown, Minus, Loader2, AlertCircle,
  ChevronDown, CheckCircle2, XCircle,
} from 'lucide-react'
import { api } from '../api'
import { useDisease } from '../context/DiseaseContext'
import { useDiseaseSchema } from '../hooks/useDiseaseSchema'
import { getPatientsForDisease } from '../mockPatients'

// ── Helpers ───────────────────────────────────────────────────────────────────

function riskScore(result) {
  if (!result) return null
  return result.prediction === 1 ? result.confidence : 1 - result.confidence
}

function DeltaBadge({ before, after }) {
  if (before == null || after == null) return null
  const delta = after - before
  const pct = Math.round(Math.abs(delta) * 100)
  if (pct === 0) return <span className="text-xs text-gray-400 flex items-center gap-0.5"><Minus className="w-3 h-3" /> No change</span>
  const improving = delta < 0
  return (
    <span className={`text-xs font-semibold flex items-center gap-0.5 ${improving ? 'text-green-600' : 'text-red-600'}`}>
      {improving ? <TrendingDown className="w-3.5 h-3.5" /> : <TrendingUp className="w-3.5 h-3.5" />}
      {improving ? '↓' : '↑'}{pct}% risk
    </span>
  )
}

// ── Patient picker ────────────────────────────────────────────────────────────

function PatientPicker({ label, patients, selected, onSelect, color }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <p className={`text-xs font-semibold uppercase tracking-wide mb-1 ${color}`}>{label}</p>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between gap-2 p-3 bg-gray-50 border border-clinical-border rounded-lg hover:bg-gray-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold flex-shrink-0">
            {selected?.avatar}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">{selected?.name ?? 'Select patient'}</p>
            {selected && <p className="text-[10px] text-gray-500">{selected.age}yrs · {selected.sex}</p>}
          </div>
        </div>
        <ChevronDown className={`w-4 h-4 text-gray-400 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-clinical-border rounded-lg shadow-lg z-20 overflow-hidden">
            {patients.map(p => (
              <button
                key={p.id}
                onClick={() => { onSelect(p); setOpen(false) }}
                className={`w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-gray-50 transition-colors ${selected?.id === p.id ? 'bg-primary-50' : ''}`}
              >
                <div className="w-7 h-7 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center text-xs font-bold flex-shrink-0">
                  {p.avatar}
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">{p.name}</p>
                  <p className="text-[10px] text-gray-500">{p.age}yrs · {p.sex}</p>
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ── Result column ─────────────────────────────────────────────────────────────

function ResultColumn({ label, result, loading, error, color, colorLight }) {
  if (loading) return (
    <div className="flex-1 flex items-center justify-center py-12">
      <Loader2 className={`w-6 h-6 animate-spin ${color}`} />
    </div>
  )
  if (error) return (
    <div className="flex-1 flex items-center gap-2 py-8 px-4">
      <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
      <p className="text-xs text-red-600">{error}</p>
    </div>
  )
  if (!result) return (
    <div className="flex-1 flex items-center justify-center py-12 text-gray-300 text-sm">
      No result yet
    </div>
  )

  const isPositive = result.prediction === 1
  const conf = Math.round((result.confidence ?? 0) * 100)

  return (
    <div className="flex-1 space-y-4 p-4">
      {/* Badge */}
      <div className={`rounded-xl p-4 text-center ${isPositive ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
        <div className="flex items-center justify-center gap-2 mb-2">
          {isPositive
            ? <XCircle className="w-5 h-5 text-red-600" />
            : <CheckCircle2 className="w-5 h-5 text-green-600" />}
          <span className={`text-base font-bold ${isPositive ? 'text-red-700' : 'text-green-700'}`}>
            {result.diagnosis ?? (isPositive ? 'Positive' : 'Negative')}
          </span>
        </div>
        <p className="text-xs text-gray-500">Risk Probability</p>
        <p className={`text-2xl font-black ${isPositive ? 'text-red-600' : 'text-green-600'}`}>{conf}%</p>
        <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${isPositive ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ width: `${conf}%` }}
          />
        </div>
      </div>

      {/* Risk score */}
      <div className="text-center">
        <p className="text-xs text-gray-500 uppercase tracking-wide">Risk Score</p>
        <p className={`text-3xl font-black mt-1 ${colorLight}`}>
          {Math.round(riskScore(result) * 100)}%
        </p>
      </div>
    </div>
  )
}

// ── Feature diff table ────────────────────────────────────────────────────────

function FeatureDiffTable({ beforePatient, afterPatient, fields }) {
  if (!beforePatient || !afterPatient) return null

  const b = beforePatient.data || {}
  const a = afterPatient.data || {}

  const diffs = fields
    .filter(f => String(b[f.name]) !== String(a[f.name]))
    .map(f => ({ name: f.name, label: f.label || f.name, before: b[f.name], after: a[f.name] }))

  if (diffs.length === 0) {
    return (
      <div className="text-center py-6 text-sm text-gray-400">
        Both patients have identical feature values.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-clinical-border">
      <table className="w-full text-xs">
        <thead className="bg-slate-50">
          <tr>
            <th className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">Feature</th>
            <th className="text-left px-4 py-2.5 text-blue-600 font-medium uppercase tracking-wide text-[10px]">Before</th>
            <th className="text-left px-4 py-2.5 text-violet-600 font-medium uppercase tracking-wide text-[10px]">After</th>
            <th className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">Change</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-clinical-border">
          {diffs.map(d => {
            const numBefore = parseFloat(d.before)
            const numAfter = parseFloat(d.after)
            const isNumeric = !isNaN(numBefore) && !isNaN(numAfter)
            const delta = isNumeric ? numAfter - numBefore : null
            return (
              <tr key={d.name} className="hover:bg-slate-50 transition-colors">
                <td className="px-4 py-2.5 font-medium text-gray-800">{d.label}</td>
                <td className="px-4 py-2.5 text-blue-700 font-mono">{String(d.before ?? '—')}</td>
                <td className="px-4 py-2.5 text-violet-700 font-mono">{String(d.after ?? '—')}</td>
                <td className="px-4 py-2.5">
                  {delta != null ? (
                    <span className={`font-semibold ${delta > 0 ? 'text-red-500' : delta < 0 ? 'text-green-600' : 'text-gray-400'}`}>
                      {delta > 0 ? `+${delta.toFixed(1)}` : delta < 0 ? delta.toFixed(1) : '='}
                    </span>
                  ) : (
                    <span className="text-gray-400">{d.before} → {d.after}</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Side-by-side confidence chart ─────────────────────────────────────────────

function ComparisonChart({ beforeResult, afterResult, beforeName, afterName }) {
  if (!beforeResult || !afterResult) return null

  const data = [
    {
      name: 'Risk Score',
      Before: Math.round(riskScore(beforeResult) * 100),
      After:  Math.round(riskScore(afterResult) * 100),
    },
    {
      name: 'Risk Probability',
      Before: Math.round((beforeResult.confidence ?? 0) * 100),
      After:  Math.round((afterResult.confidence ?? 0) * 100),
    },
  ]

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-sm font-semibold text-gray-900">Side-by-Side Metrics</h3>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 10 }} />
            <Tooltip formatter={(v) => [`${v}%`]} />
            <Legend />
            <Bar name={beforeName} dataKey="Before" fill="#2563eb" radius={[4, 4, 0, 0]} />
            <Bar name={afterName}  dataKey="After"  fill="#7c3aed" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ── Main ComparisonMode ───────────────────────────────────────────────────────

export default function ComparisonMode() {
  const { selectedDisease } = useDisease()
  const { fields } = useDiseaseSchema(selectedDisease)
  const patients = getPatientsForDisease(selectedDisease)

  const [beforePatient, setBeforePatient] = useState(patients[0] ?? null)
  const [afterPatient,  setAfterPatient]  = useState(patients[1] ?? patients[0] ?? null)
  const [beforeResult,  setBeforeResult]  = useState(null)
  const [afterResult,   setAfterResult]   = useState(null)
  const [beforeLoading, setBeforeLoading] = useState(false)
  const [afterLoading,  setAfterLoading]  = useState(false)
  const [beforeError,   setBeforeError]   = useState(null)
  const [afterError,    setAfterError]    = useState(null)
  const [ran, setRan] = useState(false)

  const runComparison = useCallback(async () => {
    if (!beforePatient || !afterPatient || !selectedDisease) return
    setRan(true)
    setBeforeResult(null)
    setAfterResult(null)
    setBeforeError(null)
    setAfterError(null)

    setBeforeLoading(true)
    setAfterLoading(true)

    const [bRes, aRes] = await Promise.allSettled([
      api.predict(selectedDisease, beforePatient.data),
      api.predict(selectedDisease, afterPatient.data),
    ])

    setBeforeLoading(false)
    setAfterLoading(false)

    if (bRes.status === 'fulfilled') setBeforeResult(bRes.value)
    else setBeforeError(bRes.reason?.message ?? 'Prediction failed')

    if (aRes.status === 'fulfilled') setAfterResult(aRes.value)
    else setAfterError(aRes.reason?.message ?? 'Prediction failed')
  }, [beforePatient, afterPatient, selectedDisease])

  const riskBefore = riskScore(beforeResult)
  const riskAfter  = riskScore(afterResult)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <ArrowLeftRight className="w-5 h-5 text-primary-600" /> Before / After Comparison
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Compare two patient profiles side by side to evaluate risk changes
          </p>
        </div>
      </div>

      {/* Patient selectors + compare button */}
      <div className="card">
        <div className="card-body">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-end">
            <PatientPicker
              label="Before"
              patients={patients}
              selected={beforePatient}
              onSelect={p => { setBeforePatient(p); setRan(false) }}
              color="text-blue-600"
            />
            <PatientPicker
              label="After"
              patients={patients}
              selected={afterPatient}
              onSelect={p => { setAfterPatient(p); setRan(false) }}
              color="text-violet-600"
            />
          </div>
          <div className="mt-4 flex justify-center">
            <button
              onClick={runComparison}
              disabled={!beforePatient || !afterPatient || beforeLoading || afterLoading}
              className="btn-primary text-sm flex items-center gap-2"
            >
              {(beforeLoading || afterLoading)
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <ArrowLeftRight className="w-4 h-4" />}
              {(beforeLoading || afterLoading) ? 'Running…' : 'Compare'}
            </button>
          </div>
        </div>
      </div>

      {/* Results panel */}
      {ran && (
        <>
          {/* Side-by-side result columns */}
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-900">Prediction Results</h3>
              {riskBefore != null && riskAfter != null && (
                <DeltaBadge before={riskBefore} after={riskAfter} />
              )}
            </div>
            <div className="flex divide-x divide-clinical-border">
              <div className="flex-1">
                <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide text-center pt-4 px-4">
                  Before — {beforePatient?.name}
                </p>
                <ResultColumn
                  label="Before"
                  result={beforeResult}
                  loading={beforeLoading}
                  error={beforeError}
                  color="text-blue-600"
                  colorLight="text-blue-700"
                />
              </div>
              <div className="flex-1">
                <p className="text-xs font-semibold text-violet-600 uppercase tracking-wide text-center pt-4 px-4">
                  After — {afterPatient?.name}
                </p>
                <ResultColumn
                  label="After"
                  result={afterResult}
                  loading={afterLoading}
                  error={afterError}
                  color="text-violet-600"
                  colorLight="text-violet-700"
                />
              </div>
            </div>
          </div>

          {/* Chart comparison */}
          <ComparisonChart
            beforeResult={beforeResult}
            afterResult={afterResult}
            beforeName={beforePatient?.name ?? 'Before'}
            afterName={afterPatient?.name ?? 'After'}
          />

          {/* Feature diff table */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-sm font-semibold text-gray-900">Changed Features</h3>
            </div>
            <div className="card-body">
              <FeatureDiffTable
                beforePatient={beforePatient}
                afterPatient={afterPatient}
                fields={fields ?? []}
              />
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!ran && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <ArrowLeftRight className="w-12 h-12 text-gray-200 mb-4" />
          <p className="text-gray-500 text-sm">Select two patients and click Compare to see the results.</p>
        </div>
      )}
    </div>
  )
}
