/**
 * The reason a non-OK response gives, as one line for the screen.
 *
 * Every backend error body carries `error` (backend/main.py wraps each
 * HTTPException in the same envelope); a refusal for a retired module also
 * carries `reason` (410 DISEASE_RETIRED). Anything else falls back to the
 * HTTP status text, so a failure is never shown as nothing at all.
 */
export async function responseError(res) {
  let body = {};
  try {
    body = await res.json();
  } catch {
    // not JSON
  }
  const main = body?.error
    || (typeof body?.detail === 'string' ? body.detail : null)
    || res.statusText
    || `HTTP ${res.status}`;
  return body?.reason ? `${main} ${body.reason}` : main;
}
