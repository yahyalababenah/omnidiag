"""
OmniDiag — LLM Clinical Report Generator
==========================================
Generates structured clinical narrative reports from prediction results
using the Anthropic API (Claude). Falls back to a rule-based template
when the API key is unavailable (e.g. in offline/demo environments).
"""

import os
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("omnidiag.llm")

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

_SYSTEM_PROMPT = """You are a senior clinical decision support AI embedded in OmniDiag,
a multi-disease risk assessment platform used by healthcare professionals.

Your task is to produce a concise, evidence-based clinical report for a single patient
assessment. The report must be:
- Written for a qualified clinician (not the patient)
- Factual and grounded only in the provided data
- Free of speculative diagnoses not supported by the input
- Structured with clear sections
- Under 350 words

Do NOT add disclaimers about seeking medical advice (the audience is medical professionals).
Do NOT hallucinate lab values or history not provided."""

_USER_PROMPT_TEMPLATE = """Generate a clinical assessment report for the following patient.

Disease Module: {disease_display}
Risk Probability: {probability:.1%}
Risk Label: {label}
Confidence Band: {confidence_band}

Top Risk Factors (SHAP-ranked):
{shap_summary}

Patient Features:
{features_summary}

Structure the report as:
1. Clinical Summary (2-3 sentences)
2. Key Risk Drivers (bullet list from SHAP)
3. Recommended Actions (evidence-based, 3-5 bullets)
4. Risk Stratification Note (1 sentence)
"""


def _format_shap(shap_values: List[Dict[str, Any]], top_n: int = 5) -> str:
    sorted_shap = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:top_n]
    lines = []
    for item in sorted_shap:
        direction = "↑ increases risk" if item["shap_value"] > 0 else "↓ decreases risk"
        lines.append(f"  - {item['feature']}: SHAP={item['shap_value']:+.3f} ({direction})")
    return "\n".join(lines) or "  - No SHAP data available"


def _format_features(features: Dict[str, Any]) -> str:
    lines = [f"  {k}: {v}" for k, v in features.items()]
    return "\n".join(lines[:20])  # cap to avoid prompt bloat


def _rule_based_report(
    disease_display: str,
    probability: float,
    label: str,
    shap_values: List[Dict[str, Any]],
    features: Dict[str, Any],
) -> str:
    top = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    top_names = [s["feature"] for s in top]
    actions = {
        "HIGH": "- Urgent specialist referral recommended\n- Review medication adherence\n- Order confirmatory investigations",
        "MODERATE": "- Schedule follow-up within 4 weeks\n- Lifestyle modification counselling\n- Monitor key biomarkers",
        "LOW": "- Routine follow-up\n- Reinforce preventive measures\n- Rescreen in 12 months",
    }
    band = "HIGH" if probability >= 0.7 else ("MODERATE" if probability >= 0.4 else "LOW")
    report = (
        f"**Clinical Summary**\n"
        f"Patient assessed for {disease_display} risk. "
        f"Model probability: {probability:.1%} ({label}). "
        f"Top contributing factors: {', '.join(top_names)}.\n\n"
        f"**Key Risk Drivers**\n"
        + "\n".join(f"- {s['feature']} (SHAP {s['shap_value']:+.3f})" for s in top)
        + f"\n\n**Recommended Actions**\n{actions.get(band, actions['MODERATE'])}\n\n"
        f"**Risk Stratification Note**\n"
        f"This assessment is {band.lower()} priority based on the {probability:.1%} probability estimate."
    )
    return report


async def generate_report(
    disease_display: str,
    probability: float,
    label: str,
    confidence_band: str,
    shap_values: List[Dict[str, Any]],
    features: Dict[str, Any],
    model: str = "claude-haiku-4-5-20251001",
) -> Dict[str, str]:
    """
    Generate a structured clinical report.

    Returns a dict with keys:
        - report: The generated markdown report text
        - source: 'llm' | 'rule_based'
    """
    if not _ANTHROPIC_API_KEY:
        log.info("ANTHROPIC_API_KEY not set — using rule-based report fallback")
        return {
            "report": _rule_based_report(disease_display, probability, label, shap_values, features),
            "source": "rule_based",
        }

    try:
        import anthropic  # lazy import — only needed when API key present

        client = anthropic.AsyncAnthropic(api_key=_ANTHROPIC_API_KEY)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            disease_display=disease_display,
            probability=probability,
            label=label,
            confidence_band=confidence_band,
            shap_summary=_format_shap(shap_values),
            features_summary=_format_features(features),
        )

        message = await client.messages.create(
            model=model,
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        report_text = message.content[0].text
        return {"report": report_text, "source": "llm"}

    except Exception as exc:
        log.warning(f"LLM report generation failed ({exc!r}), falling back to rule-based")
        return {
            "report": _rule_based_report(disease_display, probability, label, shap_values, features),
            "source": "rule_based",
        }
