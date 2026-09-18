"""
OmniDiag — LLM Clinical Report Generator
==========================================
Generates structured clinical narrative reports from prediction results
using the DeepSeek API (OpenAI-compatible). Falls back to a rule-based
template when the API key is unavailable (e.g. in offline/demo environments).

── Probability scale ──────────────────────────────────────────────────────
`probability_corrected` is on the deployment scale — the same scale as the
diagnosis label and as `risk_bands`, both taken from the /predict response.
For diabetes that means a Positive patient can read 11%, because the decision
threshold there is 5.98%, not 50%.

Bands therefore come from `risk_bands` and never from a literal. Reading
HIGH/MODERATE off hardcoded 0.70/0.40 put every corrected-scale patient in
LOW: on the 14,139-row test split that was 6,576 Positive patients being told
"routine follow-up, rescreen in 12 months", and HIGH became unreachable.

The band shown in the report, the band used to pick the recommended actions
and the band sent to the LLM are all the same value, computed once here.
"""

import os
import logging
from typing import Any, Dict, List, Mapping, Optional

from backend.probability_scale import classify_band

log = logging.getLogger("omnidiag.llm")

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Fallback display bands for a disease that configures none. These are the
# same numbers as DEFAULT_RISK_BANDS in frontend/src/constants/thresholds.js
# and are correct for heart_disease, whose probabilities are on the model's
# own scale with an argmax 0.5 cut-point. A disease WITH configured bands
# (diabetes) always passes them in; this constant is never its band source.
DEFAULT_RISK_BANDS: Dict[str, float] = {"high": 0.7, "moderate": 0.4}


def _get_api_key() -> str:
    """Read key lazily so HF Space secrets (injected after startup) are picked up."""
    return os.getenv("DEEPSEEK_API_KEY", "")

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
Risk Probability: {probability:.1%}  (calibrated to real-world prevalence)
Decision Threshold: {threshold_note}
Risk Label: {label}
Risk Band: {confidence_band}

Note: this probability is stated on the deployment population's prevalence, so
it is NOT comparable to a 50% cut-off. Judge it against the decision threshold
and the risk band above, never against 50%.

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
    probability_corrected: float,
    label: str,
    shap_values: Optional[List[Dict[str, Any]]] = None,
    features: Optional[Dict[str, Any]] = None,
    risk_bands: Optional[Mapping[str, float]] = None,
) -> str:
    shap_values = shap_values or []
    features = features or {}
    top = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    top_names = [s["feature"] for s in top]
    actions = {
        "HIGH": "- Urgent specialist referral recommended\n- Review medication adherence\n- Order confirmatory investigations",
        "MODERATE": "- Schedule follow-up within 4 weeks\n- Lifestyle modification counselling\n- Monitor key biomarkers",
        "LOW": "- Routine follow-up\n- Reinforce preventive measures\n- Rescreen in 12 months",
    }
    band = classify_band(probability_corrected, risk_bands or DEFAULT_RISK_BANDS)
    report = (
        f"**Clinical Summary**\n"
        f"Patient assessed for {disease_display} risk. "
        f"Model probability: {probability_corrected:.1%} ({label}). "
        f"Top contributing factors: {', '.join(top_names)}.\n\n"
        f"**Key Risk Drivers**\n"
        + "\n".join(f"- {s['feature']} (SHAP {s['shap_value']:+.3f})" for s in top)
        + f"\n\n**Recommended Actions**\n{actions.get(band, actions['MODERATE'])}\n\n"
        f"**Risk Stratification Note**\n"
        f"This assessment is {band.lower()} priority: a {probability_corrected:.1%} "
        f"probability on this population's prevalence, classified against the "
        f"module's own risk bands."
    )
    return report


async def generate_report(
    disease_display: str,
    probability_corrected: float,
    label: str,
    confidence_band: Optional[str] = None,
    shap_values: Optional[List[Dict[str, Any]]] = None,
    features: Optional[Dict[str, Any]] = None,
    model: str = "deepseek-chat",
    risk_bands: Optional[Mapping[str, float]] = None,
    decision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a structured clinical report.

    `probability_corrected` and `risk_bands` must be on the same scale — both
    come straight from the /predict response for this disease.

    `confidence_band` is accepted for backwards compatibility but is NOT
    trusted: the band is recomputed here from the probability and the bands,
    so the narrative, the recommended actions and the label the LLM is given
    can never disagree with one another. A caller-supplied band that differs
    is logged and discarded.

    Returns a dict with keys:
        - report:        The generated markdown report text
        - source:        'llm' | 'rule_based'
        - risk_band:     The band actually used
        - llm_model:     Model name used (only when source='llm')
        - latency_ms:    Round-trip time in milliseconds (only when source='llm')
        - fallback_reason: Why rule-based was used (only when source='rule_based')
    """
    import time

    probability_corrected = float(probability_corrected)
    shap_values = shap_values or []
    features = features or {}
    bands = risk_bands or DEFAULT_RISK_BANDS
    band = classify_band(probability_corrected, bands)
    if confidence_band and confidence_band != band:
        log.info(
            "Discarding caller-supplied confidence_band=%r; %.4f against bands %r is %r",
            confidence_band, probability_corrected, dict(bands), band,
        )
    threshold_note = (
        f"{decision_threshold:.4f} — at or above this the patient is classified Positive"
        if decision_threshold is not None
        else "not exposed by this module (argmax)"
    )

    api_key = _get_api_key()

    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — falling back to rule-based report")
        return {
            "report": _rule_based_report(
                disease_display, probability_corrected, label, shap_values, features, bands
            ),
            "source": "rule_based",
            "risk_band": band,
            "fallback_reason": "DEEPSEEK_API_KEY environment variable is not set",
        }

    try:
        from openai import AsyncOpenAI  # lazy import — only needed when API key present

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=_DEEPSEEK_BASE_URL,
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            disease_display=disease_display,
            probability=probability_corrected,
            threshold_note=threshold_note,
            label=label,
            confidence_band=band,
            shap_summary=_format_shap(shap_values),
            features_summary=_format_features(features),
        )

        log.info(f"Calling DeepSeek API (model={model}) for {disease_display} report...")
        t0 = time.perf_counter()

        response = await client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        report_text = response.choices[0].message.content
        log.info(f"DeepSeek report generated in {latency_ms}ms ({len(report_text)} chars)")

        return {
            "report": report_text,
            "source": "llm",
            "risk_band": band,
            "llm_model": model,
            "latency_ms": latency_ms,
        }

    except Exception as exc:
        log.error(f"DeepSeek API call failed: {exc!r}")
        return {
            "report": _rule_based_report(
                disease_display, probability_corrected, label, shap_values, features, bands
            ),
            "source": "rule_based",
            "risk_band": band,
            "fallback_reason": str(exc),
        }
