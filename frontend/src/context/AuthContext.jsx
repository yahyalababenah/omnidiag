import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { api } from '../api'

const AuthContext = createContext(null)

const TOKEN_KEY = 'omnidiag_token'
const USER_KEY  = 'omnidiag_user'

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => {
    const stored = localStorage.getItem(TOKEN_KEY)
    if (stored) api.setToken(stored)   // sync api client on page reload
    return stored
  })
  const [user, setUser]   = useState(() => {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') }
    catch { return null }
  })
  const [loginError, setLoginError] = useState(null)
  const [loginLoading, setLoginLoading] = useState(false)

  // Verify stored token is still valid on mount
  useEffect(() => {
    if (!token) return
    fetch('/api/v4/auth/me', { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(u => { setUser(u); localStorage.setItem(USER_KEY, JSON.stringify(u)) })
      .catch(() => { setToken(null); setUser(null); localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY) })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const login = useCallback(async (email, password) => {
    setLoginError(null)
    setLoginLoading(true)
    try {
      const res = await fetch('/api/v4/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || data?.detail || 'Login failed')

      const accessToken = data.access_token
      setToken(accessToken)
      api.setToken(accessToken)           // wire token into api client
      localStorage.setItem(TOKEN_KEY, accessToken)

      // Fetch user profile
      const meRes = await fetch('/api/v4/auth/me', { headers: { Authorization: `Bearer ${accessToken}` } })
      const me = await meRes.json()
      setUser(me)
      localStorage.setItem(USER_KEY, JSON.stringify(me))
      return true
    } catch (e) {
      setLoginError(e.message)
      return false
    } finally {
      setLoginLoading(false)
    }
  }, [])

  const logout = useCallback(() => {
    fetch('/api/v4/auth/logout', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    }).catch(() => {})
    setToken(null)
    setUser(null)
    api.setToken(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }, [token])

  const isAdmin = user?.roles?.some(r => ['admin', 'super_admin'].includes(r)) ?? false

  return (
    <AuthContext.Provider value={{ token, user, isAdmin, login, logout, loginError, loginLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
