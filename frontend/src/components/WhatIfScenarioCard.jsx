import { Zap, Lightbulb, TrendingDown } from 'lucide-react';

/**
 * WhatIfScenarioCard — DiCE Counterfactuals Placeholder
 *
 * Displays mock "what-if" scenarios showing how changing a patient's feature
 * (e.g., BMI, RestingBP) would reduce their risk. This is a placeholder UI
 * for the Phase 2 DiCE (Diverse Counterfactual Explanations) engine integration.
 *
 * Props (future):
 *   counterfactuals — Array of { feature, current, proposed, riskReduction }
 *   loading         — Boolean indicating backend computation in progress
 */
export default function WhatIfScenarioCard({ counterfactuals, loading }) {
  // ── Mock placeholder data ──
  const mockScenarios = [
    {
      feature: 'BMI',
      current: 32.4,
      proposed: 27.0,
      riskReduction: 44,
      description: 'Reducing BMI into the overweight range significantly lowers cardiovascular strain.',
    },
    {
      feature: 'RestingBP',
      current: 158,
      proposed: 130,
      riskReduction: 28,
      description: 'Controlling systolic BP under 130 mmHg reduces hypertensive stress on the heart.',
    },
    {
      feature: 'Cholesterol',
      current: 340,
      proposed: 200,
      riskReduction: 31,
      description: 'Lowering total cholesterol to normal range reduces plaque formation risk.',
    },
  ];

  // Use real counterfactuals when backend provides them, otherwise use mock
  const scenarios = counterfactuals?.length > 0 ? counterfactuals : mockScenarios;

  return (
    <div className="card border border-blue-100 bg-blue-50/30">
      <div className="card-header flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          What-If Scenarios
          <span className="text-[10px] font-medium text-amber-600 bg-amber-100 px-2 py-0.5 rounded-full ml-1">
            Coming Soon
          </span>
        </h2>
        <span className="text-[10px] text-gray-400">Powered by DiCE</span>
      </div>
      <div className="card-body space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-6 text-gray-400 text-sm">
            <Zap className="w-4 h-4 animate-pulse mr-2" />
            Computing counterfactual scenarios...
          </div>
        ) : (
          <>
            <p className="text-xs text-gray-500 leading-relaxed">
              Explore how modifying key risk factors could alter the predicted outcome.
              {!counterfactuals && ' Below are illustrative examples — backend DiCE engine integration pending.'}
            </p>

            {scenarios.map((s, idx) => {
              const reduction = s.riskReduction;
              const isHighImpact = reduction >= 30;

              return (
                <div
                  key={`${s.feature}-${idx}`}
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
                        <span className="text-sm font-semibold text-gray-800">
                          {s.feature}
                        </span>
                        {isHighImpact && (
                          <span className="text-[10px] font-medium text-green-600 bg-green-100 px-1.5 py-0.5 rounded">
                            High Impact
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700">
                        Scenario: If{' '}
                        <strong className="text-gray-900">{s.feature}</strong> drops from{' '}
                        <span className="text-red-600 font-semibold">{s.current}</span>
                        {' → '}
                        <span className="text-green-600 font-semibold">{s.proposed}</span>
                        {' → '}
                        Risk reduces by{' '}
                        <strong className="text-green-600">{s.riskReduction}%</strong>
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{s.description}</p>
                    </div>
                    <div className="shrink-0 text-right">
                      <div className="text-lg font-bold text-green-600">-{s.riskReduction}%</div>
                      <div className="text-[10px] text-gray-400">Risk Reduction</div>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Visual progress bar showing cumulative improvement */}
            <div className="pt-2">
              <div className="flex items-center gap-2 mb-1.5">
                <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-medium text-gray-600">Cumulative Risk Reduction Potential</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-gradient-to-r from-green-400 to-green-600 h-2.5 rounded-full"
                  style={{ width: `${Math.min(scenarios.reduce((acc, s) => acc + s.riskReduction, 0), 100)}%` }}
                />
              </div>
              <p className="text-[10px] text-gray-400 mt-1">
                Addressing all suggested factors could reduce overall risk by up to{' '}
                <strong>{Math.min(scenarios.reduce((acc, s) => acc + s.riskReduction, 0), 100)}%</strong>
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
