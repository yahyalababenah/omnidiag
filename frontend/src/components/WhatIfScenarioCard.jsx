import { Zap, Lightbulb, TrendingDown, CheckCircle2, Loader2 } from 'lucide-react';
import MedicalTooltip from './MedicalTooltip';

/**
 * WhatIfScenarioCard — DiCE Counterfactuals Viewer
 *
 * Three-state rendering logic:
 * 1. loading=true      → Loading spinner
 * 2. counterfactuals === null        → Mock data with "Coming Soon" badge
 *    (API error or unsupported disease)
 * 3. counterfactuals !== null && counterfactuals.length === 0
 *    → Green "Low Risk" message (healthy patient — no interventions needed)
 * 4. counterfactuals.length > 0      → Real scenarios from backend DiCE engine
 */
export default function WhatIfScenarioCard({ counterfactuals, loading }) {
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
     State 2 — Healthy patient
     (counterfactuals !== null && length === 0)
     ════════════════════════════════════════ */
  if (counterfactuals !== null && counterfactuals.length === 0) {
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
  const scenarios = isMock ? mockScenarios : counterfactuals;

  /**
   * Extract the risk-reduction percentage from both data formats:
   * - Mock:  s.riskReduction  (number, e.g. 44)
   * - Backend: s.risk_reduction (string, e.g. "44%")
   */
  const getReductionPct = (s) => {
    if (typeof s.riskReduction === 'number') return s.riskReduction;
    if (typeof s.risk_reduction === 'string')
      return parseInt(s.risk_reduction, 10) || 0;
    return 0;
  };

  const totalReduction = Math.min(
    scenarios.reduce((acc, s) => acc + getReductionPct(s), 0),
    100,
  );

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
          Explore how modifying key risk factors could alter the predicted
          outcome.
          {isMock &&
            ' Below are illustrative examples — backend DiCE engine integration pending.'}
        </p>

        {scenarios.map((s, idx) => {
          const isBackendFormat = s.scenario !== undefined;
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

          /* ── Backend format: multi-feature counterfactual scenario ── */
          const featureNames = Object.keys(s.changes || {}).join(', ');
          const feasibility = s.feasibility || null;
          const feasibilityColor =
            feasibility === 'high'
              ? 'text-green-600 bg-green-100'
              : feasibility === 'medium'
                ? 'text-amber-600 bg-amber-100'
                : 'text-red-600 bg-red-100';

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
                    <MedicalTooltip term={featureNames.split(', ')[0]}>
                      <span className="text-sm font-semibold text-gray-800">
                        {featureNames}
                      </span>
                    </MedicalTooltip>
                    {isHighImpact && (
                      <span className="text-[10px] font-medium text-green-600 bg-green-100 px-1.5 py-0.5 rounded">
                        High Impact
                      </span>
                    )}
                    {feasibility && (
                      <span
                        className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${feasibilityColor}`}
                      >
                        {feasibility.charAt(0).toUpperCase() +
                          feasibility.slice(1)}{' '}
                        Feasibility
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-700 mt-1">{s.scenario}</p>
                  {s.new_probability !== undefined && (
                    <p className="text-xs text-gray-500 mt-1">
                      Post-intervention probability:{' '}
                      <strong>
                        {(s.new_probability * 100).toFixed(1)}%
                      </strong>
                    </p>
                  )}
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
        })}

        {/* ── Cumulative risk-reduction progress bar ── */}
        <div className="pt-2">
          <div className="flex items-center gap-2 mb-1.5">
            <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
            <span className="text-xs font-medium text-gray-600">
              Cumulative Risk Reduction Potential
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-gradient-to-r from-green-400 to-green-600 h-2.5 rounded-full"
              style={{ width: `${totalReduction}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-400 mt-1">
            Addressing all suggested factors could reduce overall risk by up to{' '}
            <strong>{totalReduction}%</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
