import { Zap, Lightbulb, TrendingDown, CheckCircle2, Loader2, AlertTriangle } from 'lucide-react';
import { normaliseScenario } from '../utils/counterfactuals';
import MedicalTooltip from './MedicalTooltip';

/**
 * WhatIfScenarioCard — DiCE Counterfactuals Viewer
 *
 * Four-state rendering logic:
 * 1. loading=true           → Loading spinner
 * 2. counterfactuals === null         → Mock data with "Coming Soon" badge
 *    (API error or unsupported disease)
 * 3. prediction === 0 && counterfactuals !== null && counterfactuals.length === 0
 *    → Green "Low Risk" message (truly negative patient — no interventions needed)
 * 4. prediction === 1 && (counterfactuals === null || counterfactuals.length === 0)
 *    → Mock data with "Coming Soon" (positive patient, counterfactuals pending/failed)
 * 5. counterfactuals.length > 0       → Real scenarios from backend DiCE engine
 *
 * The `prediction` prop (binary 0|1|undefined) decouples logic from presentation:
 * - If the patient is truly negative (prediction=0), counterfactuals are irrelevant → show green card.
 * - If the patient is positive (prediction=1), ALWAYS show actionable scenarios (real or mock).
 * - Never let a missing/empty counterfactuals array suppress the What-If UI for a positive case.
 *
 * Real backend scenario shape (POST /api/v4/{disease}/counterfactuals):
 *   { scenario_id, probability, changes: [{ feature, original_value, counterfactual_value, direction }] }
 * `baselineProbability` (the request's `baseline_probability`) is required to turn a scenario's
 * `probability` into a risk-reduction percentage; pass it whenever available.
 *
 * `bestAchievable` (the response's `best_achievable`) is set when NO allowed change crosses the
 * threshold: every modifiable factor improved at once, flagged `crosses_threshold: false`. That
 * case renders its own explicit state — never an empty card or an unexplained "0%".
 */
const SUBTITLE =
  "Scenarios show how the model's estimate responds to modifiable factors. " +
  'They are not a predicted treatment effect.';
const NO_CROSSING_MESSAGE =
  'Even with every modifiable factor improved, the estimated risk remains above the threshold. ' +
  'The dominant factors are not modifiable. Referral is recommended.';

export default function WhatIfScenarioCard({
  counterfactuals,
  loading,
  prediction,
  baselineProbability,
  patientData = null,
  bestAchievable = null,
}) {
  /* ── Mock placeholder data (fallback when API unavailable) ── */
  const mockScenarios = [
    {
      feature: 'BMI',
      current: 32.4,
      proposed: 27.0,
      riskReduction: 44,
      description:
        'Reducing BMI into the overweight range significantly lowers cardiovascular strain.',
    },
    {
      feature: 'RestingBP',
      current: 158,
      proposed: 130,
      riskReduction: 28,
      description:
        'Controlling systolic BP under 130 mmHg reduces hypertensive stress on the heart.',
    },
    {
      feature: 'Cholesterol',
      current: 340,
      proposed: 200,
      riskReduction: 31,
      description:
        'Lowering total cholesterol to normal range reduces plaque formation risk.',
    },
  ];

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
            <p className="text-sm font-medium text-amber-900">{NO_CROSSING_MESSAGE}</p>
            {best ? (
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
            ) : (
              <p className="text-xs text-gray-600 mt-2">
                No modifiable factor is available to change for this patient (none of the
                modifiable inputs was supplied, or each is already at its target).
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  /* ════════════════════════════════════════
     State 2 — Truly negative patient (prediction=0)
     (prediction !== 1 && counterfactuals !== null && length === 0)

     If prediction is 1 (positive), we NEVER show "Low Clinical Risk" —
     the patient IS positive and deserves actionable scenarios even if
     counterfactuals are empty (backend returned not_applicable).
     Instead, fall through to mock data with "Coming Soon" badge.
     ════════════════════════════════════════ */
  if (prediction !== 1 && counterfactuals !== null && counterfactuals.length === 0) {
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
            <p className="text-sm font-medium text-green-800 mb-1">
              Low Clinical Risk
            </p>
            <p className="text-sm text-green-700 leading-relaxed max-w-md">
              Patient is currently at low clinical risk. No counterfactual
              interventions are necessary.
            </p>
          </div>
        </div>
      </div>
    );
  }

  /* ════════════════════════════════════════
     State 3 / 4 — Mock or real scenarios
     counterfactuals === null  → mock + Coming Soon badge (API error)
     counterfactuals.length > 0 → real backend data
     ════════════════════════════════════════ */
  const isMock = counterfactuals === null;
  // Real scenarios are normalised to one shape: diabetes sends `changes` as
  // an object and no scenario_id, which used to route it into the mock
  // branch below and render "If undefined drops from undefined".
  const scenarios = isMock
    ? mockScenarios
    : counterfactuals.map((s, i) => normaliseScenario(s, i, patientData));

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
   * - Mock: s.riskReduction (number)
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
    if (typeof s.riskReduction === 'number') return s.riskReduction;
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
      <div className="card-header flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          What-If Scenarios
          {isMock && (
            <span className="text-[10px] font-medium text-amber-600 bg-amber-100 px-2 py-0.5 rounded-full ml-1">
              Coming Soon
            </span>
          )}
        </h2>
        <span className="text-[10px] text-gray-400">Powered by DiCE</span>
      </div>

      <div className="card-body space-y-4">
        <p className="text-xs text-gray-500 leading-relaxed">
          {SUBTITLE}
          {isMock &&
            ' Below are illustrative examples — backend DiCE engine integration pending.'}
        </p>

        {scenarios.map((s, idx) => {
          const isBackendFormat = s.scenario_id !== undefined;
          const reductionPct = getReductionPct(s);
          const isHighImpact = reductionPct >= 30;

          /* ── Mock format: single-feature scenario ── */
          if (!isBackendFormat) {
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
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingDown className="w-4 h-4 text-green-500 shrink-0" />
                      <MedicalTooltip term={s.feature}>
                        <span className="text-sm font-semibold text-gray-800">
                          {s.feature}
                        </span>
                      </MedicalTooltip>
                      {isHighImpact && (
                        <span className="text-[10px] font-medium text-green-600 bg-green-100 px-1.5 py-0.5 rounded">
                          High Impact
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-700">
                      Scenario: If{' '}
                      <strong className="text-gray-900">{s.feature}</strong>{' '}
                      drops from{' '}
                      <span className="text-red-600 font-semibold">
                        {s.current}
                      </span>
                      {' → '}
                      <span className="text-green-600 font-semibold">
                        {s.proposed}
                      </span>
                      {' → '}Risk reduces by{' '}
                      <strong className="text-green-600">
                        {s.riskReduction}%
                      </strong>
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {s.description}
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="text-lg font-bold text-green-600">
                      -{reductionPct}%
                    </div>
                    <div className="text-[10px] text-gray-400">
                      Risk Reduction
                    </div>
                  </div>
                </div>
              </div>
            );
          }

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
