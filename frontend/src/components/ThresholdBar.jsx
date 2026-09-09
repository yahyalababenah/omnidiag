/**
 * ThresholdBar — horizontal indicator of a risk probability relative to
 * the disease's clinical decision threshold.
 *
 * Fill color reflects which side of the threshold the probability falls
 * on (red = at/above threshold, green = below). The threshold itself is
 * drawn as a vertical tick — pass `threshold={null}` to omit it (e.g. a
 * disease with no known threshold) and the bar still renders the plain
 * probability fill.
 */
export default function ThresholdBar({ probability, threshold }) {
  const pct = Math.max(0, Math.min(100, (probability ?? 0) * 100));
  const thresholdPct =
    typeof threshold === 'number' ? Math.max(0, Math.min(100, threshold * 100)) : null;
  const isAboveThreshold = thresholdPct != null ? probability >= threshold : null;

  return (
    <div className="relative h-2 rounded-full bg-gray-200 overflow-visible">
      <div
        className={`h-full rounded-full transition-all ${
          isAboveThreshold === false ? 'bg-green-500' : 'bg-red-500'
        }`}
        style={{ width: `${pct}%` }}
      />
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
