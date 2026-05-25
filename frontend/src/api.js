const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

/**
 * OmniDiag API client.
 * All endpoints return parsed JSON or throw on error.
 */
class OmniDiagApi {
  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async _fetch(path, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const msg = body?.detail?.error || body?.detail || res.statusText;
      throw new Error(msg);
    }
    return res.json();
  }

  /** GET / — health check + available diseases */
  health() {
    return this._fetch('/');
  }

  /** GET /api/v4/diseases — list all registered diseases */
  listDiseases() {
    return this._fetch('/api/v4/diseases');
  }

  /** POST /api/v4/{disease}/predict — run inference */
  predict(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/predict`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/{disease}/explain — SHAP explanation */
  explain(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/explain`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }
}

export const api = new OmniDiagApi();
export default OmniDiagApi;
