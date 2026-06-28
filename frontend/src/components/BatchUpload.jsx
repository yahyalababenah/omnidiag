/**
 * BatchUpload — Feature 3.5
 *
 * CSV upload interface for batch predictions. Displays per-row results
 * in a color-coded table with a summary bar chart and CSV download.
 *
 * Requires: auth token (AuthContext) + clinical role on backend.
 */

import { useState, useRef, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'
import {
  Upload, FileText, Download, AlertCircle, CheckCircle2, XCircle,
  Loader2, RefreshCw, Table2,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useDisease } from '../context/DiseaseContext'

const BASE = '/api/v4'

function downloadCsv(rows, disease) {
  const headers = ['row', 'status', 'prediction', 'confidence', 'diagnosis', 'error']
  const lines = [headers.join(',')]
  for (const r of rows) {
    lines.push([
      r.row,
      r.status,
      r.prediction ?? '',
      r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '',
      r.diagnosis ?? '',
      r.error ? `"${r.error.replace(/"/g, '""')}"` : '',
    ].join(','))
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `omnidiag_batch_${disease}_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// ── Drop zone ─────────────────────────────────────────────────────────────────

function DropZone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) onFile(f)
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`
        border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
        ${dragOver ? 'border-primary-400 bg-primary-50' : 'border-clinical-border hover:border-primary-300 hover:bg-slate-50'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <Upload className="w-10 h-10 text-gray-300 mx-auto mb-3" />
      <p className="text-sm font-medium text-gray-700">Drop a CSV file here, or click to browse</p>
      <p className="text-xs text-gray-400 mt-1">
        Max 500 rows · UTF-8 encoding · Header row required
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        disabled={disabled}
        onChange={e => e.target.files[0] && onFile(e.target.files[0])}
      />
    </div>
  )
}

// ── Results table ─────────────────────────────────────────────────────────────

function ResultsTable({ results }) {
  const [filter, setFilter] = useState('all')

  const visible = results.filter(r =>
    filter === 'all' ? true : filter === 'ok' ? r.status === 'ok' : r.status === 'error'
  )

  return (
    <div>
      {/* Filter tabs */}
      <div className="flex items-center gap-2 mb-3">
        {[
          { id: 'all', label: 'All' },
          { id: 'ok', label: 'Succeeded' },
          { id: 'error', label: 'Failed' },
        ].map(f => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors ${
              filter === f.id
                ? 'bg-primary-100 text-primary-700'
                : 'text-gray-500 hover:bg-gray-100'
            }`}
          >
            {f.label} ({f.id === 'all' ? results.length : results.filter(r => r.status === (f.id === 'ok' ? 'ok' : 'error')).length})
          </button>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-clinical-border">
        <table className="w-full text-xs">
          <thead className="bg-slate-50">
            <tr>
              {['Row', 'Status', 'Prediction', 'Confidence', 'Diagnosis', 'Error'].map(h => (
                <th key={h} className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-clinical-border">
            {visible.slice(0, 200).map(r => (
              <tr key={r.row} className={`${r.status === 'error' ? 'bg-red-50' : 'hover:bg-slate-50'} transition-colors`}>
                <td className="px-4 py-2 text-gray-500 font-mono">{r.row}</td>
                <td className="px-4 py-2">
                  {r.status === 'ok'
                    ? <span className="flex items-center gap-1 text-green-700"><CheckCircle2 className="w-3 h-3" /> ok</span>
                    : <span className="flex items-center gap-1 text-red-600"><XCircle className="w-3 h-3" /> error</span>}
                </td>
                <td className="px-4 py-2">
                  {r.prediction != null ? (
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${r.prediction === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                      {r.prediction === 1 ? 'Positive' : 'Negative'}
                    </span>
                  ) : '—'}
                </td>
                <td className="px-4 py-2 text-gray-700">
                  {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}
                </td>
                <td className="px-4 py-2 text-gray-700">{r.diagnosis ?? '—'}</td>
                <td className="px-4 py-2 text-red-600 max-w-[200px] truncate" title={r.error ?? ''}>
                  {r.error ?? '—'}
                </td>
              </tr>
            ))}
            {visible.length > 200 && (
              <tr>
                <td colSpan={6} className="px-4 py-3 text-center text-xs text-gray-400">
                  Showing first 200 of {visible.length} rows. Download CSV for full results.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Summary bar chart ─────────────────────────────────────────────────────────

function SummaryChart({ data }) {
  const chartData = [
    { name: 'Positive', count: data.results.filter(r => r.prediction === 1).length, fill: '#dc2626' },
    { name: 'Negative', count: data.results.filter(r => r.prediction === 0).length, fill: '#16a34a' },
    { name: 'Error',    count: data.failed, fill: '#94a3b8' },
  ].filter(d => d.count > 0)

  return (
    <div className="card mb-6">
      <div className="card-header">
        <h3 className="text-sm font-semibold text-gray-900">Result Distribution</h3>
      </div>
      <div className="card-body flex items-center gap-8">
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 10 }} allowDecimals={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={60} />
            <Tooltip formatter={(v) => [v, 'Patients']} />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="grid grid-cols-1 gap-3 flex-shrink-0">
          <Stat label="Total" value={data.total} color="text-gray-900" />
          <Stat label="Succeeded" value={data.succeeded} color="text-green-600" />
          <Stat label="Failed" value={data.failed} color="text-red-500" />
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div className="text-center">
      <p className={`text-xl font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-gray-400 uppercase tracking-wide">{label}</p>
    </div>
  )
}

// ── Main BatchUpload ──────────────────────────────────────────────────────────

export default function BatchUpload() {
  const { token } = useAuth()
  const { selectedDisease } = useDisease()
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const handleFile = useCallback((f) => {
    setFile(f)
    setData(null)
    setError(null)
  }, [])

  async function runBatch() {
    if (!file || !selectedDisease) return
    if (!token) { setError('Please sign in to use batch predictions.'); return }

    setLoading(true)
    setError(null)
    setData(null)

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${BASE}/${selectedDisease}/batch`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json?.error || json?.detail || `HTTP ${res.status}`)
      setData(json)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setFile(null)
    setData(null)
    setError(null)
  }

  const diseaseName = (selectedDisease || 'disease').replace(/_/g, ' ')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Table2 className="w-5 h-5 text-primary-600" /> Batch Prediction
          </h2>
          <p className="text-sm text-gray-500 mt-0.5 capitalize">
            Upload a CSV to predict {diseaseName} risk for multiple patients
          </p>
        </div>
        {data && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => downloadCsv(data.results, selectedDisease)}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <Download className="w-4 h-4" /> Download CSV
            </button>
            <button onClick={reset} className="btn-secondary text-sm flex items-center gap-2">
              <RefreshCw className="w-4 h-4" /> New Batch
            </button>
          </div>
        )}
      </div>

      {!token && (
        <div className="flex items-center gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          Sign in with a clinical account to run batch predictions.
        </div>
      )}

      {/* Template hint */}
      {!data && (
        <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-xl text-sm text-blue-800">
          <FileText className="w-5 h-5 flex-shrink-0 mt-0.5 text-blue-500" />
          <div>
            <p className="font-medium">CSV Format</p>
            <p className="text-xs mt-1 text-blue-600">
              The first row must be a header matching the {diseaseName} schema field names.
              Each subsequent row is one patient. Numeric and categorical values are accepted as-is.
            </p>
          </div>
        </div>
      )}

      {/* Upload area */}
      {!data && (
        <>
          <DropZone onFile={handleFile} disabled={loading || !token} />

          {file && (
            <div className="flex items-center justify-between p-4 bg-slate-50 border border-clinical-border rounded-xl">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-primary-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">{file.name}</p>
                  <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <button
                onClick={runBatch}
                disabled={loading || !token}
                className="btn-primary text-sm flex items-center gap-2"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                {loading ? 'Running…' : 'Run Batch'}
              </button>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Results */}
      {data && (
        <>
          <SummaryChart data={data} />
          <ResultsTable results={data.results} />
        </>
      )}
    </div>
  )
}
