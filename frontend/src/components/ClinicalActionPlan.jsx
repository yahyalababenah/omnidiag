import { ClipboardList, AlertTriangle } from 'lucide-react';

/**
 * Clinical action plan — the next step implied by the conformal decision.
 *
 * Renders the `clinical_action_plan` object a dysglycaemia /predict response
 * carries (backend/schemas_clinical_action.py). It renders exactly what the
 * server sent: no tier, wording or urgency is derived here, because the
 * wording is reviewed as policy on the backend. Returns null when the response
 * has no plan (the heart module), so the heart result panel is unchanged.
 *
 * The safety note is always shown, never collapsed: on the no_referral tier it
 * is what stops "no test indicated" being read as a clearance.
 */
const TIER_STYLES = {
  High: {
    box: 'border-red-300 bg-red-50 text-red-900',
    badge: 'bg-red-600 text-white',
  },
  Medium: {
    box: 'border-amber-300 bg-amber-50 text-amber-900',
    badge: 'bg-amber-500 text-white',
  },
  Low: {
    box: 'border-slate-300 bg-slate-50 text-slate-900',
    badge: 'bg-slate-600 text-white',
  },
};

const DECISION_LABEL = {
  referral: 'Referral',
  uncertain: 'Uncertain',
  no_referral: 'No referral',
};

export default function ClinicalActionPlan({ plan }) {
  if (!plan || !plan.urgency || !plan.recommended_test) return null;

  // An unrecognised urgency falls back to the neutral style rather than to a
  // colour that would imply a tier the server did not send.
  const style = TIER_STYLES[plan.urgency] || TIER_STYLES.Low;

  return (
    <section
      className={`rounded-lg border p-4 ${style.box}`}
      aria-label="Clinical action plan"
      data-testid="clinical-action-plan"
      data-decision={plan.decision}
    >
      <header className="flex flex-wrap items-center gap-2 mb-2">
        <ClipboardList className="w-4 h-4" aria-hidden="true" />
        <h3 className="text-sm font-semibold">Next clinical step</h3>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${style.badge}`}>
          {plan.urgency} urgency
        </span>
        {plan.decision && (
          <span className="text-xs opacity-80">
            from decision: {DECISION_LABEL[plan.decision] || plan.decision}
          </span>
        )}
      </header>

      <p className="text-sm font-medium" data-testid="action-plan-test">
        {plan.recommended_test}
      </p>
      {plan.rationale && <p className="text-sm mt-1">{plan.rationale}</p>}

      {plan.safety_note && (
        <p
          className="mt-3 flex gap-2 text-xs leading-relaxed border-t border-current/20 pt-2"
          role="note"
          data-testid="action-plan-safety-note"
        >
          <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{plan.safety_note}</span>
        </p>
      )}
    </section>
  );
}
