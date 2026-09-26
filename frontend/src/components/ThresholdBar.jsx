/**
 * ThresholdBar — horizontal indicator of a risk probability, shown against
 * whatever the module actually decides by.
 *
 * Two shapes, chosen by what the caller passes, never by the disease name:
 *
 * * A THRESHOLD module (diabetes) passes `threshold`. It is drawn as a vertical
 *   tick, and the fill is red at/above it and green below.
 *
 * * A DECISION module (heart, `output_type: 'conformal_decision'`) passes
 *   `decision` and `interval` instead. There is no threshold to draw, so the
 *   fill follows the decision itself and the Venn-Abers interval is drawn as a
 *   lighter band around the estimate.
 *
 * Why the decision has to be passed: before Gate 8.4 the fill was
 * `isAboveThreshold === false ? green : red`, so with no threshold every
 * patient's bar came out RED — including the ones the model had decided not to
 * refer. The colour was reporting the absence of a threshold as danger.
 */
export default function ThresholdBar({ probability, threshold, decision = null, interval = null }) {
  const clamp = (v) => Math.max(0, Math.min(100, v * 100));
  const pct = clamp(probability ?? 0);
  const thresholdPct = typeof threshold === 'number' ? clamp(threshold) : null;

  // Green only when something actually says this patient is below the line:
  // a probability under the threshold, or an explicit no-referral decision.
  // Never green merely because nothing was passed.
  const isBelow =
    thresholdPct != null ? probability < threshold
      : decision === 'no_referral' ? true
      : decision === 'referral' || decision === 'uncertain' ? false
      : null;

  const fill =
    isBelow === true ? 'bg-green-500'
      : decision === 'uncertain' ? 'bg-amber-500'
      : 'bg-red-500';

  const lowerPct = interval && typeof interval.lower === 'number' ? clamp(interval.lower) : null;
  const upperPct = interval && typeof interval.upper === 'number' ? clamp(interval.upper) : null;
  const hasInterval = lowerPct != null && upperPct != null && upperPct > lowerPct;

  return (
    <div className="relative h-2 rounded-full bg-gray-200 overflow-visible">
      <div
        className={`h-full rounded-full transition-all ${fill}`}
        style={{ width: `${pct}%` }}
      />
      {hasInterval && (
        <div
          className="absolute top-1/2 -translate-y-1/2 h-2 rounded-full bg-gray-800/25"
          style={{ left: `${lowerPct}%`, width: `${upperPct - lowerPct}%` }}
          title={`Venn-Abers interval: ${lowerPct.toFixed(1)}%–${upperPct.toFixed(1)}%`}
        />
      )}
      {thresholdPct != null && (
        <div
          className="absolute top-1/2 -translate-y-1/2 w-0.5 h-3.5 bg-gray-800 rounded-full"
          style={{ left: `${thresholdPct}%` }}
          title={`Clinical threshold: ${thresholdPct.toFixed(1)}%`}
        />
      )}
    </div>
  );
}
