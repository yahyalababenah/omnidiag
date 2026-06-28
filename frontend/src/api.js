const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

/** Default timeout for API requests (ms). HF Spaces cold starts can take 60-120s. */
const REQUEST_TIMEOUT_MS = 120_000;

/**
 * OmniDiag API client.
 * Call api.setToken(token) after login so all subsequent requests are authenticated.
 */
class OmniDiagApi {
  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this._token = null;
  }

  /** Set the JWT token — called by AuthContext after login */
  setToken(token) {
    this._token = token;
  }

  /** Build auth header if a token is available */
  _authHeader() {
    return this._token ? { Authorization: `Bearer ${this._token}` } : {};
  }

  async _fetch(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const url = `${this.baseUrl}${path}`;
    try {
      const res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...this._authHeader(),
          ...options.headers,
        },
        signal: controller.signal,
        ...options,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const msg = body?.detail?.error || body?.detail || res.statusText;
        throw new Error(msg);
      }
      return res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(
          'The diagnostic engine is waking up — this can take a minute or two on the first scan of the day. Please try again.'
        );
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /** GET / — health check */
  health() {
    return this._fetch('/');
  }

  /** GET /api/v4/diseases — list registered diseases */
  listDiseases() {
    return this._fetch('/api/v4/diseases');
  }

  /** GET /api/v4/{disease}/schema */
  getSchema(disease) {
    return this._fetch(`/api/v4/${disease}/schema`);
  }

  /** POST /api/v4/{disease}/predict */
  predict(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/predict`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/{disease}/explain */
  explain(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/explain`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/{disease}/counterfactuals */
  counterfactuals(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/counterfactuals`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/generate-report */
  generateReport(payload) {
    return this._fetch('/api/v4/generate-report', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /** POST /api/v4/parse-notes */
  parseNotes(note, useBert = false) {
    return this._fetch('/api/v4/parse-notes', {
      method: 'POST',
      body: JSON.stringify({ note, use_bert: useBert }),
    });
  }
}

export const api = new OmniDiagApi();
export default OmniDiagApi;
