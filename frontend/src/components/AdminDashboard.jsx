import { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts'
import {
  Users, Activity, Database, Zap, TrendingUp, AlertCircle,
  RefreshCw, Trash2, Key, ChevronLeft, ChevronRight, Search, Shield,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const BASE = '/api/v4'

const DISEASE_COLORS = [
  '#2563eb', '#16a34a', '#dc2626', '#d97706', '#7c3aed', '#0891b2',
]

// ── Fetch helpers ─────────────────────────────────────────────────────────────

async function apiFetch(path, token, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...(opts.headers || {}) },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error || body?.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = 'text-primary-600' }) {
  return (
    <div className="card">
      <div className="card-body flex items-start gap-4">
        <div className={`p-3 rounded-xl bg-slate-100 ${color}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-medium">{label}</p>
          <p className="text-2xl font-bold text-gray-900 mt-0.5">{value}</p>
          {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
        </div>
      </div>
    </div>
  )
}

// ── Users table ───────────────────────────────────────────────────────────────

function UsersTable({ token, onIssueKey, onRevokeKey }) {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({ page, limit: 10, ...(query ? { search: query } : {}) })
      const d = await apiFetch(`/admin/users?${qs}`, token)
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [page, query, token])

  useEffect(() => { load() }, [load])

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
          <Users className="w-4 h-4 text-primary-600" /> Users
        </h3>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
            <input
              className="input-field pl-8 text-xs py-1.5"
              placeholder="Search email or name…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { setQuery(search); setPage(1) } }}
            />
          </div>
          <button
            onClick={() => { setQuery(search); setPage(1) }}
            className="btn-secondary text-xs py-1.5 px-3"
          >
            Go
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        {error && <p className="text-xs text-red-600 p-4">{error}</p>}
        {loading && !data && <p className="text-xs text-gray-400 p-4">Loading…</p>}
        {data && (
          <table className="w-full text-xs">
            <thead className="bg-slate-50">
              <tr>
                {['Name', 'Email', 'Roles', 'Status', 'Joined', 'API Key'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">{h}</th>
                ))}
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-clinical-border">
              {data.items.map(u => (
                <tr key={u.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-2.5 font-medium text-gray-900">{u.full_name}</td>
                  <td className="px-4 py-2.5 text-gray-600">{u.email}</td>
                  <td className="px-4 py-2.5">
                    <div className="flex flex-wrap gap-1">
                      {u.roles.map(r => (
                        <span key={r} className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-100 text-blue-700">{r}</span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${u.is_active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                      {u.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-gray-500">{new Date(u.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-2.5">
                    {u.has_api_key
                      ? <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-100 text-amber-700">Active</span>
                      : <span className="text-gray-400">—</span>}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-1.5">
                      <button
                        title="Issue API key"
                        onClick={() => onIssueKey(u.id)}
                        className="p-1 rounded hover:bg-blue-50 text-blue-600 transition-colors"
                      >
                        <Key className="w-3.5 h-3.5" />
                      </button>
                      {u.has_api_key && (
                        <button
                          title="Revoke API key"
                          onClick={() => onRevokeKey(u.id)}
                          className="p-1 rounded hover:bg-red-50 text-red-500 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {data && data.pages > 1 && (
        <div className="px-4 py-3 border-t border-clinical-border flex items-center justify-between">
          <span className="text-xs text-gray-500">{data.total} users · page {data.page}/{data.pages}</span>
          <div className="flex gap-1">
            <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="btn-secondary py-1 px-2 disabled:opacity-40">
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button disabled={page >= data.pages} onClick={() => setPage(p => p + 1)} className="btn-secondary py-1 px-2 disabled:opacity-40">
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Audit log table ───────────────────────────────────────────────────────────

function AuditLogTable({ token }) {
  const [page, setPage] = useState(1)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({ page, limit: 15 })
      const d = await apiFetch(`/admin/audit-logs?${qs}`, token)
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [page, token])

  useEffect(() => { load() }, [load])

  const statusColor = (code) => {
    if (!code) return 'text-gray-400'
    if (code < 300) return 'text-green-600'
    if (code < 400) return 'text-blue-600'
    if (code < 500) return 'text-amber-600'
    return 'text-red-600'
  }

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary-600" /> Audit Log
        </h3>
        <button onClick={load} disabled={loading} className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>
      <div className="overflow-x-auto">
        {error && <p className="text-xs text-red-600 p-4">{error}</p>}
        {data && (
          <table className="w-full text-xs">
            <thead className="bg-slate-50">
              <tr>
                {['Time', 'Method', 'Endpoint', 'Status', 'Latency', 'IP'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-clinical-border">
              {data.items.map(log => (
                <tr key={log.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-2 text-gray-500 whitespace-nowrap">
                    {new Date(log.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      log.method === 'GET' ? 'bg-blue-100 text-blue-700'
                      : log.method === 'POST' ? 'bg-green-100 text-green-700'
                      : log.method === 'DELETE' ? 'bg-red-100 text-red-700'
                      : 'bg-gray-100 text-gray-600'
                    }`}>{log.method}</span>
                  </td>
                  <td className="px-4 py-2 text-gray-700 font-mono max-w-[200px] truncate">{log.endpoint}</td>
                  <td className={`px-4 py-2 font-medium ${statusColor(log.status_code)}`}>{log.status_code ?? '—'}</td>
                  <td className="px-4 py-2 text-gray-500">{log.duration_ms != null ? `${Math.round(log.duration_ms)} ms` : '—'}</td>
                  <td className="px-4 py-2 text-gray-500 font-mono">{log.ip_address ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {data && data.pages > 1 && (
        <div className="px-4 py-3 border-t border-clinical-border flex items-center justify-between">
          <span className="text-xs text-gray-500">{data.total} entries · page {data.page}/{data.pages}</span>
          <div className="flex gap-1">
            <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="btn-secondary py-1 px-2 disabled:opacity-40">
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button disabled={page >= data.pages} onClick={() => setPage(p => p + 1)} className="btn-secondary py-1 px-2 disabled:opacity-40">
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Issue API Key modal ───────────────────────────────────────────────────────

function IssueKeyModal({ userId, token, onClose }) {
  const [expires, setExpires] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function submit() {
    setLoading(true)
    setError(null)
    try {
      const body = { user_id: userId, ...(expires ? { expires_at: new Date(expires).toISOString() } : {}) }
      const d = await apiFetch('/admin/api-keys', token, { method: 'POST', body: JSON.stringify(body) })
      setResult(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Key className="w-5 h-5 text-primary-600" /> Issue API Key
        </h3>
        {!result ? (
          <>
            <div className="mb-4">
              <label className="block text-xs font-medium text-gray-700 mb-1">Expiry (optional)</label>
              <input
                type="datetime-local"
                className="input-field text-sm"
                value={expires}
                onChange={e => setExpires(e.target.value)}
              />
            </div>
            {error && <p className="text-xs text-red-600 mb-3">{error}</p>}
            <div className="flex gap-2 justify-end">
              <button onClick={onClose} className="btn-secondary text-sm">Cancel</button>
              <button onClick={submit} disabled={loading} className="btn-primary text-sm">
                {loading ? 'Issuing…' : 'Issue Key'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4">
              <p className="text-xs font-medium text-amber-800 mb-2">Copy this key now — it will not be shown again.</p>
              <code className="text-xs font-mono text-amber-900 break-all">{result.api_key}</code>
            </div>
            <button onClick={onClose} className="btn-primary w-full text-sm">Done</button>
          </>
        )}
      </div>
    </div>
  )
}

// ── Main AdminDashboard ───────────────────────────────────────────────────────

export default function AdminDashboard() {
  const { token } = useAuth()
  const [stats, setStats] = useState(null)
  const [statsError, setStatsError] = useState(null)
  const [statsLoading, setStatsLoading] = useState(false)
  const [cacheMsg, setCacheMsg] = useState(null)
  const [cacheLoading, setCacheLoading] = useState(false)
  const [issueKeyUserId, setIssueKeyUserId] = useState(null)
  const [usersRefresh, setUsersRefresh] = useState(0)

  const loadStats = useCallback(async () => {
    setStatsLoading(true)
    setStatsError(null)
    try {
      const d = await apiFetch('/admin/stats', token)
      setStats(d)
    } catch (e) {
      setStatsError(e.message)
    } finally {
      setStatsLoading(false)
    }
  }, [token])

  useEffect(() => { loadStats() }, [loadStats])

  async function flushCache() {
    setCacheLoading(true)
    setCacheMsg(null)
    try {
      const d = await apiFetch('/admin/cache/flush', token, { method: 'POST' })
      setCacheMsg(`✓ ${d.message} (${d.keys_deleted} keys)`)
    } catch (e) {
      setCacheMsg(`Error: ${e.message}`)
    } finally {
      setCacheLoading(false)
    }
  }

  async function revokeKey(userId) {
    if (!confirm('Revoke this user\'s API key?')) return
    try {
      await apiFetch(`/admin/api-keys/${userId}`, token, { method: 'DELETE' })
      setUsersRefresh(r => r + 1)
    } catch (e) {
      alert(`Error: ${e.message}`)
    }
  }

  const pct = (v) => `${Math.round(v * 100)}%`

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary-600" /> Admin Dashboard
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">Platform analytics and user management</p>
        </div>
        <div className="flex items-center gap-3">
          {cacheMsg && (
            <span className={`text-xs px-3 py-1.5 rounded-lg ${cacheMsg.startsWith('Error') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
              {cacheMsg}
            </span>
          )}
          <button
            onClick={flushCache}
            disabled={cacheLoading}
            className="btn-secondary text-sm flex items-center gap-2"
          >
            <Trash2 className={`w-4 h-4 ${cacheLoading ? 'opacity-50' : ''}`} />
            {cacheLoading ? 'Flushing…' : 'Flush Cache'}
          </button>
          <button onClick={loadStats} disabled={statsLoading} className="btn-primary text-sm flex items-center gap-2">
            <RefreshCw className={`w-4 h-4 ${statsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {statsError && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {statsError}
        </div>
      )}

      {/* Stat cards */}
      {stats && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
            <StatCard
              icon={Activity}
              label="Total Predictions"
              value={stats.total_predictions.toLocaleString()}
              color="text-blue-600"
            />
            <StatCard
              icon={Users}
              label="Total Users"
              value={stats.total_users.toLocaleString()}
              color="text-violet-600"
            />
            <StatCard
              icon={Database}
              label="Total Patients"
              value={stats.total_patients.toLocaleString()}
              color="text-teal-600"
            />
            <StatCard
              icon={TrendingUp}
              label="Avg Confidence"
              value={pct(stats.avg_confidence)}
              color="text-green-600"
            />
            <StatCard
              icon={AlertCircle}
              label="Positive Rate"
              value={pct(stats.positive_rate)}
              color="text-red-600"
            />
            <StatCard
              icon={Zap}
              label="Avg Latency"
              value={`${stats.avg_latency_ms} ms`}
              color="text-amber-600"
            />
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Predictions last 7 days */}
            <div className="card">
              <div className="card-header">
                <h3 className="text-sm font-semibold text-gray-900">Predictions — Last 7 Days</h3>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={stats.predictions_last_7_days} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10 }}
                      tickFormatter={d => d.slice(5)}
                    />
                    <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                    <Tooltip
                      labelFormatter={l => `Date: ${l}`}
                      formatter={v => [v, 'Predictions']}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* By disease pie */}
            <div className="card">
              <div className="card-header">
                <h3 className="text-sm font-semibold text-gray-900">Predictions by Disease</h3>
              </div>
              <div className="card-body">
                {stats.predictions_by_disease.length > 0 ? (
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={stats.predictions_by_disease}
                        dataKey="total"
                        nameKey="disease"
                        cx="50%"
                        cy="50%"
                        outerRadius={75}
                        label={({ name, percent }) => `${name} ${Math.round(percent * 100)}%`}
                        labelLine={false}
                      >
                        {stats.predictions_by_disease.map((_, i) => (
                          <Cell key={i} fill={DISEASE_COLORS[i % DISEASE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(v, name) => [v, name]} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-sm text-gray-400 text-center py-12">No predictions yet</p>
                )}
              </div>
            </div>
          </div>

          {/* Top endpoints */}
          {stats.top_endpoints.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-primary-600" /> Top Endpoints
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-slate-50">
                    <tr>
                      {['Method', 'Endpoint', 'Calls', 'Avg Latency'].map(h => (
                        <th key={h} className="text-left px-4 py-2.5 text-gray-500 font-medium uppercase tracking-wide text-[10px]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-clinical-border">
                    {stats.top_endpoints.map((ep, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="px-4 py-2">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            ep.method === 'GET' ? 'bg-blue-100 text-blue-700'
                            : ep.method === 'POST' ? 'bg-green-100 text-green-700'
                            : 'bg-gray-100 text-gray-600'
                          }`}>{ep.method}</span>
                        </td>
                        <td className="px-4 py-2 font-mono text-gray-700">{ep.endpoint}</td>
                        <td className="px-4 py-2 text-gray-900 font-medium">{ep.count.toLocaleString()}</td>
                        <td className="px-4 py-2 text-gray-500">{ep.avg_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Users table */}
      <UsersTable
        key={usersRefresh}
        token={token}
        onIssueKey={userId => setIssueKeyUserId(userId)}
        onRevokeKey={revokeKey}
      />

      {/* Audit log */}
      <AuditLogTable token={token} />

      {/* Issue key modal */}
      {issueKeyUserId && (
        <IssueKeyModal
          userId={issueKeyUserId}
          token={token}
          onClose={() => { setIssueKeyUserId(null); setUsersRefresh(r => r + 1) }}
        />
      )}
    </div>
  )
}
