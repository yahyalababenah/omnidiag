export const API_BASE = import.meta.env.VITE_API_BASE || 'https://yahyoha-omnidiag.hf.space';

/** Default timeout for API requests (ms). HF Spaces cold starts can take 60-120s. */
const REQUEST_TIMEOUT_MS = 120_000;

/**
 * A 502/503/504 comes from the host's proxy, not from the model: on the free
 * Space tier roughly one request in four is rejected there before it reaches
 * the app (the app's own log shows only 200s for the same calls). Retrying
 * with a short back-off almost always lands on a healthy path.
 */
const RETRY_STATUSES = [502, 503, 504];
const RETRY_DELAYS_MS = [400, 1000, 2000];

/**
 * Turn an error body into readable text. `detail` can be a string, an
 * object with `error`, or a pydantic list of {loc, msg}; passing the list
 * straight to `new Error()` rendered as "[object Object],[object Object]".
 */
function errorMessage(body, res) {
  const pick = body?.detail?.error ?? body?.error ?? body?.detail;
  if (typeof pick === 'string') return pick;
  if (Array.isArray(pick)) {
    return pick
      .map((e) => (typeof e === 'string' ? e : `${(e?.loc || []).join('.')}: ${e?.msg ?? ''}`))
      .join('; ');
  }
  return res.statusText || `HTTP ${res.status}`;
}

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

  /** One request, retried on the proxy-level failures listed in RETRY_STATUSES. */
  async _fetch(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await this._fetchOnce(path, options, timeoutMs);
      } catch (err) {
        if (!RETRY_STATUSES.includes(err.status) || attempt >= RETRY_DELAYS_MS.length) throw err;
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
      }
    }
  }

  async _fetchOnce(path, options, timeoutMs) {
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
        const failure = new Error(errorMessage(body, res));
        failure.status = res.status;
        throw failure;
      }
      return res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(
          'The screening service is waking up — this can take a minute or two on the first scan of the day. Please try again.'
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

  /**
   * POST /api/v4/{disease}/predict
   *
   * `patientId` links the screening to a patient record, which is what makes
   * it appear in that patient's History and what a clinical note can later
   * be attached to. It is a query parameter: the response is identical with
   * or without it.
   */
  predict(disease, patientData, patientId = null) {
    return this._fetch(`/api/v4/${disease}/predict${this._patientQuery(patientId)}`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** POST /api/v4/{disease}/explain */
  explain(disease, patientData, patientId = null) {
    return this._fetch(`/api/v4/${disease}/explain${this._patientQuery(patientId)}`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  }

  /** `?patient_id=…`, or '' when no patient is linked. */
  _patientQuery(patientId) {
    return patientId ? `?patient_id=${encodeURIComponent(patientId)}` : '';
  }

  /**
   * POST /api/v4/patients/{id}/notes — store the clinician's free text
   * against their latest screening for this disease.
   *
   * The note is recorded and shown back; it is never a model input and never
   * reaches the report LLM.
   */
  saveClinicalNote(patientId, disease, notes) {
    return this._fetch(`/api/v4/patients/${encodeURIComponent(patientId)}/notes`, {
      method: 'POST',
      body: JSON.stringify({ disease, notes }),
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
