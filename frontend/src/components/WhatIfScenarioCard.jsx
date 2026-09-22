import { Zap, Lightbulb, TrendingDown, CheckCircle2, Loader2, AlertTriangle } from 'lucide-react';
import { normaliseScenario } from '../utils/counterfactuals';
import {
  SUBTITLE, NO_CROSSING_MESSAGE, NO_IMPROVEMENT_MESSAGE, NO_LEVERS_MESSAGE,
  LOW_RISK_HEADLINE, LOW_RISK_MESSAGE,
} from '../utils/whatIfSummary';
import MedicalTooltip from './MedicalTooltip';

/**
 * WhatIfScenarioCard — counterfactual scenarios from the backend
 *
 * Rendering states, in order:
 * 1. loading                         → spinner
 * 2. error, or no response at all    → "could not be computed" — no numbers
 * 3. empty list + flagged patient    → no allowed change crosses the threshold:
 *    a. best_achievable lowers the estimate → show it, recommend referral
 *    b. nothing lowers the estimate          → say so, recommend referral
 * 4. empty list + not flagged        → below threshold, nothing to do
 * 5. non-empty list                  → the backend's scenarios
 *
 * Every number shown comes from the /counterfactuals response. There is no
 * placeholder or illustrative data: when the backend has no answer the card
 * says so instead of inventing one.
 */

function Header({ icon: Icon, iconClass }) {
  return (
    <div className="card-header">
      <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <Icon className={`w-5 h-5 ${iconClass}`} />
        What-If Scenarios
      </h2>
    </div>
  );
}

export default function WhatIfScenarioCard({
  counterfactuals,
  loading,
  prediction,
  baselineProbability,
  patientData = null,
  bestAchievable = null,
  error = null,
  message = null,
}) {
  /* ════════════════════════════════════════
     State 1 — Loading spinner
     ════════════════════════════════════════ */
  if (loading) {
    return (
      <div className="card border border-blue-100 bg-blue-50/30">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-500" />
            What-If Scenarios
          </h2>
        </div>
        <div className="card-body">
          <div className="flex items-center justify-center py-8 text-gray-400 text-sm">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Computing counterfactual scenarios...
          </div>
        </div>
      </div>
    );
  }

  if (error || !Array.isArray(counterfactuals)) {
    return (
      <div className="card border border-gray-200">
        <Header icon={AlertTriangle} iconClass="text-gray-400" />
        <div className="card-body">
          <p className="text-sm text-gray-600">What-If scenarios could not be computed for this patient.</p>
          {error && <p className="text-xs text-gray-400 mt-1">{error}</p>}
        </div>
      </div>
    );
  }

  /* ════════════════════════════════════════
     State 1b — Flagged patient, but no allowed change crosses the threshold
     (backend returned an empty list, with or without best_achievable)
     ════════════════════════════════════════ */
  const noCrossing =
    Array.isArray(counterfactuals) &&
    counterfactuals.length === 0 &&
    (bestAchievable || prediction === 1);
  if (noCrossing) {
    const best = bestAchievable ? normaliseScenario(bestAchievable, 0, patientData) : null;
    const after = best
      ? (typeof best.new_probability_corrected === 'number' ? best.new_probability_corrected
        : typeof best.new_probability === 'number' ? best.new_probability
        : best.probability)
      : null;
    // A "best achievable" that does not lower the estimate is not an
    // improvement; showing "49.4% → 50.4%" under "best achievable" was wrong.
    const improves =
      typeof after === 'number' &&
      (typeof baselineProbability !== 'number' || after < baselineProbability);
    const relative = best && typeof best.risk_reduction_relative_pct === 'number'
      ? best.risk_reduction_relative_pct
      : null;
    const absolute = best && typeof best.risk_reduction_absolute_pp === 'number'
      ? best.risk_reduction_absolute_pp
      : null;
    return (
      <div className="card border border-amber-200 bg-amber-50/40">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            What-If Scenarios
          </h2>
        </div>
        <div className="card-body space-y-4">
          <p className="text-xs text-gray-500 leading-relaxed">{SUBTITLE}</p>
          <div className="p-4 rounded-lg border border-amber-200 bg-white">
            <p className="text-sm font-medium text-amber-900">
              {best && !improves
                ? NO_IMPROVEMENT_MESSAGE
                : !best && message
                  ? message
                  : NO_CROSSING_MESSAGE}
            </p>
            {best && improves ? (
              <div className="mt-3">
                <p className="text-xs text-gray-600">
                  Best achievable with every modifiable factor improved:
                </p>
                <ul className="mt-1 text-sm text-gray-700 space-y-0.5">
                  {best.changes.map((c) => (
                    <li key={c.feature}>
                      <MedicalTooltip term={c.feature}>
                        <strong className="text-gray-900">{c.feature}</strong>
                      </MedicalTooltip>{' '}
                      {String(c.original_value)} {' → '} {String(c.counterfactual_value)}
                    </li>
                  ))}
                </ul>
                <p className="text-xs text-gray-600 mt-2">
                  {typeof baselineProbability === 'number' && (
                    <>Estimated risk {(baselineProbability * 100).toFixed(1)}%{' → '}</>
                  )}
                  {typeof after === 'number' && <strong>{(after * 100).toFixed(1)}%</strong>}
                  {relative !== null && (
                    <> — relative reduction {Math.round(relative)}%</>
                  )}
                  {absolute !== null && <> ({absolute.toFixed(1)} pts absolute)</>}
                  {' '}— still above the threshold.
                </p>
              </div>
            ) : !best && !message ? (
              <p className="text-xs text-gray-600 mt-2">{NO_LEVERS_MESSAGE}</p>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  /* ════════════════════════════════════════
     State 4 — below threshold, empty list: nothing to change
     ════════════════════════════════════════ */
  if (counterfactuals.length === 0) {
    return (
      <div className="card border border-green-200 bg-green-50/40">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-green-600" />
            What-If Scenarios
          </h2>
        </div>
        <div className="card-body">
          <div className="flex flex-col items-center text-center py-6 px-4">
            <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center mb-3">
              <CheckCircle2 className="w-6 h-6 text-green-600" />
            </div>
            <p className="text-sm font-medium text-green-800 mb-1">{LOW_RISK_HEADLINE}</p>
            <p className="text-sm text-green-700 leading-relaxed max-w-md">{LOW_RISK_MESSAGE}</p>
          </div>
        </div>
      </div>
    );
  }

  /* ════════════════════════════════════════
     State 5 — the backend's scenarios, normalised to one shape
     (diabetes sends `changes` as an object and no scenario_id)
     ════════════════════════════════════════ */
  const scenarios = counterfactuals.map((s, i) => normaliseScenario(s, i, patientData));

  /**
   * Post-intervention probability, on the scale the API returned it.
   *
   * The diabetes endpoint names this field `new_probability_corrected`
   * (with `new_probability` kept as its original alias); the heart endpoint
   * names it `probability`. Reading only `probability` used to make every
   * diabetes scenario render as "-0%".
   */
  const getScenarioProbability = (s) => {
    if (typeof s.new_probability_corrected === 'number') return s.new_probability_corrected;
    if (typeof s.new_probability === 'number') return s.new_probability;
    if (typeof s.probability === 'number') return s.probability;
    return null;
  };

  /**
   * Risk reduction, relative — "this intervention removes N% of the patient's
   * risk", not "N percentage points".
   *
   * The backend already computes this from the two corrected probabilities
   * and ships it as `risk_reduction_relative_pct`, so it is used as-is. It is
   * NOT recomputed locally: subtracting two corrected probabilities gives
   * percentage points, a different and much smaller number (0.4795 -> 0.0395
   * is 92% relative but 44 points), and the card is labelled as the former.
   *
   * - Backend (preferred): s.risk_reduction_relative_pct
   * - Backend (legacy string): s.risk_reduction, e.g. "92%"
   * - Heart: derived, since that endpoint sends no precomputed reduction
   */
  const getReductionPct = (s) => {
    if (typeof s.risk_reduction_relative_pct === 'number') {
      return Math.round(s.risk_reduction_relative_pct);
    }
    if (typeof s.risk_reduction === 'string') {
      const parsed = parseFloat(s.risk_reduction);
      if (!Number.isNaN(parsed)) return Math.round(parsed);
    }
    const prob = getScenarioProbability(s);
    if (prob !== null && typeof baselineProbability === 'number' && baselineProbability > 0) {
      return Math.round(((baselineProbability - prob) / baselineProbability) * 100);
    }
    return 0;
  };

  // Scenarios are alternatives, not a stack — the headline is the best one
  // available, not their sum. Summing three ~90% reductions used to saturate
  // at the 100% clamp and read as "risk eliminated".
  const totalReduction = scenarios.length
    ? Math.min(Math.max(...scenarios.map(getReductionPct)), 100)
    : 0;

  return (
    <div className="card border border-blue-100 bg-blue-50/30">
      <div className="card-header">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          What-If Scenarios
        </h2>
      </div>

      <div className="card-body space-y-4">
        <p className="text-xs text-gray-500 leading-relaxed">
          {SUBTITLE}
        </p>

        {scenarios.map((s, idx) => {
          const reductionPct = getReductionPct(s);
          const isHighImpact = reductionPct >= 30;

          /* ── Backend format: multi-feature counterfactual scenario ──
             s.changes is an array of { feature, original_value, counterfactual_value, direction } */
          const changes = s.changes || [];
          const featureNames = changes.map((c) => c.feature).join(', ');

          return (
            <div
              key={`cf-${idx}`}
              className={`p-4 rounded-lg border transition-colors ${
                isHighImpact
                  ? 'border-green-200 bg-green-50'
                  : 'border-blue-100 bg-white'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <TrendingDown className="w-4 h-4 text-green-500 shrink-0" />
                    <MedicalTooltip term={changes[0]?.feature}>
                      <span className="text-sm font-semibold text-gray-800">
                        {featureNames}
                      </span>
                    </MedicalTooltip>
                    {isHighImpact && (
                      <span className="text-[10px] font-medium text-green-600 bg-green-100 px-1.5 py-0.5 rounded">
                        High Impact
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-700 mt-1">
                    {changes.map((c, i) => (
                      <span key={c.feature ?? i}>
                        {i > 0 && ', '}
                        <strong className="text-gray-900">{c.feature}</strong>{' '}
                        <span className="text-red-600 font-semibold">
                          {String(c.original_value)}
                        </span>
                        {' → '}
                        <span className="text-green-600 font-semibold">
                          {String(c.counterfactual_value)}
                        </span>
                      </span>
                    ))}
                  </p>
                  {getScenarioProbability(s) !== null && (
                    <p className="text-xs text-gray-500 mt-1">
                      Post-intervention probability:{' '}
                      <strong>{(getScenarioProbability(s) * 100).toFixed(1)}%</strong>
                      {s.probability_scale === 'corrected' && (
                        <span className="text-gray-400">
                          {' '}(calibrated to real-world prevalence)
                        </span>
                      )}
                    </p>
                  )}
                </div>
                <div className="shrink-0 text-right">
                  <div className="text-lg font-bold text-green-600">
                    -{reductionPct}%
                  </div>
                  <div className="text-[10px] text-gray-400">
                    Relative Risk Reduction
                  </div>
                  {typeof s.risk_reduction_absolute_pp === 'number' && (
                    <div className="text-[10px] text-gray-400">
                      ({s.risk_reduction_absolute_pp.toFixed(1)} pts absolute)
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {/* ── Cumulative risk-reduction progress bar ── */}
        <div className="pt-2">
          <div className="flex items-center gap-2 mb-1.5">
            <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
            <span className="text-xs font-medium text-gray-600">
              Best Single-Scenario Risk Reduction
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-gradient-to-r from-green-400 to-green-600 h-2.5 rounded-full"
              style={{ width: `${totalReduction}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-400 mt-1">
            {scenarios.length === 1
              ? 'The scenario above reduces relative risk by '
              : `The most effective of the ${scenarios.length} scenarios above reduces relative risk by up to `}
            <strong>{totalReduction}%</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
