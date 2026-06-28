const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

/** Default timeout for API requests (ms). HF Spaces cold starts can take 60-120s. */
const REQUEST_TIMEOUT_MS = 120_000; // 2 minutes

/**
 * OmniDiag API client.
 * All endpoints return parsed JSON or throw on error.
 */
class OmniDiagApi {
  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  /**
   * Fetch with AbortController timeout.
   * Throws a descriptive error if the request times out (e.g., HF Spaces cold start).
   */
  async _fetch(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const url = `${this.baseUrl}${path}`;
    try {
      const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
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

  /** GET / — health check + available diseases */
  health() {
    return this._fetch('/');
  }

  /** GET /api/v4/diseases — list all registered diseases */
  listDiseases() {
    return this._fetch('/api/v4/diseases');
  }

  /** GET /api/v4/{disease}/schema — get JSON Schema for a disease's patient input fields */
  getSchema(disease) {
    return this._fetch(`/api/v4/${disease}/schema`);
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

  /** POST /api/v4/{disease}/counterfactuals — DiCE what-if scenarios */
  counterfactuals(disease, patientData) {
    return this._fetch(`/api/v4/${disease}/counterfactuals`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/generate-report — LLM clinical narrative report */
  generateReport(payload, token) {
    return this._fetch('/api/v4/generate-report', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    });
  }
}

export const api = new OmniDiagApi();
export default OmniDiagApi;
