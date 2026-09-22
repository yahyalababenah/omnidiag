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
# same numbers as DEFAULT_RISK_BANDS in frontend/src/constants/thresholds.js.
# heart_disease is the only disease that falls back to this constant (it
# configures no risk_bands in configs/heart_disease.yaml) — but its actual
# decision threshold is 0.3695 (models/heart_disease/heart_full_tuned.pkl,
# read at runtime in backend/model_loader.py), NOT the 0.5 argmax cut-point
# these numbers used to assume. That leaves a real gap: a patient with
# probability in [0.3695, 0.4) is classified Positive by predict() but still
# displays as the LOW band here, since 0.4 is the "moderate" cut-point.
# A disease WITH configured bands (diabetes) always passes them in; this
# constant is never its band source.
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
Do NOT hallucinate lab values or history not provided.

You are given the model's outputs only — a probability, a decision threshold,
a label, a risk band and the SHAP-ranked feature names with their signed
contributions. You are NOT given the patient's measured values, their age or
their sex. Never state, guess or imply a specific measurement, age or
demographic: write "cholesterol is the largest upward contributor", never
"cholesterol of 340" and never "this 62-year-old woman". Refer to a factor by
the exact name given in the glossary.

This is a SCREENING estimate, not a diagnosis. Hard rules:
- Never state or imply that the patient has, or is diagnosed with, any disease.
  Do not write "diagnosis", "diagnosed", "confirms", "consistent with <disease>",
  or interpret a value as proof of a condition (e.g. "indicates ischemia").
  Describe only the estimated risk, the decision threshold and the risk drivers.
- Never name a medication, drug class or dose, and never recommend starting,
  stopping or changing any drug therapy. Recommended actions are limited to
  confirmatory testing, referral, follow-up timing and lifestyle counselling.
- Never name a laboratory analyte that is not in the glossary you are given.
  In particular do not write LDL, HDL, triglycerides or HbA1c: this platform
  measures serum TOTAL cholesterol and a fasting-blood-sugar flag, and naming
  a fraction it does not measure is a factual error about the patient.
- The word "diagnosis" and its forms are banned outright, including in
  headings and in phrases like "the diagnosis of X is not established". Write
  "screening estimate" or "risk assessment" instead."""

# Output check. A report that breaks either rule above is replaced by the
# deterministic report — the prompt asks, this enforces.
import re as _re

_FORBIDDEN_PATTERNS = [
    # diagnosis wording
    (r"\bdiagnos(?:ed|is\s+of|es\s+(?:of|the))\b", "diagnosis wording"),
    (r"\b(?:has|have|with)\s+(?:established\s+|confirmed\s+|known\s+)?"
     r"(?:type\s*[12]\s+)?(?:diabetes|diabetes\s+mellitus|coronary\s+artery\s+disease|CAD|"
     r"heart\s+disease|ischemi\w*|ischaemi\w*)\b", "states the disease is present"),
    (r"\bconsistent\s+with\s+(?:a\s+)?(?:diabetes|coronary|CAD|ischemi\w*|heart\s+disease)", "diagnosis wording"),
    (r"\bconfirm(?:s|ed|ing)?\s+(?:the\s+)?(?:presence|diabetes|coronary|CAD|ischemi\w*|heart\s+disease)", "diagnosis wording"),
    (r"\bindicat\w*\s+(?:inducible\s+|myocardial\s+)?ischemi\w*", "diagnosis wording"),
    # medication / dose advice
    (r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|units?|iu)\b(?!\s*/\s*dl)", "dose"),
    # hallucinated lab analytes: the platform measures serum TOTAL
    # cholesterol and a fasting-blood-sugar flag. Reports were describing
    # that as "LDL burden" — a different lipid fraction, never measured here.
    # Naming the wrong analyte is a factual error about the patient.
    (r"\b(?:LDL|HDL|triglycerides?|HbA1c|A1c|haemoglobin\s+a1c|hemoglobin\s+a1c)\b",
     "names a lab value this platform does not measure"),
    (r"\b(?:statins?|atorvastatin|rosuvastatin|simvastatin|metformin|insulin|aspirin|"
     r"clopidogrel|antiplatelet\w*|anticoagula\w*|beta[\s-]?blockers?|ace[\s-]?inhibitors?|"
     r"arbs?|angiotensin|diuretics?|nitrates?|nitroglycerin|glp-?1|sglt-?2|sulfonylureas?|"
     r"lisinopril|losartan|amlodipine|antihypertensive\w*|anti-?ischemic|pharmacotherap\w*|"
     r"medications?|drugs?|prescri\w+)\b", "medication advice"),
]


def forbidden_content(text: str) -> list:
    """Every rule the text breaks, as short labels. Empty list = acceptable."""
    found = []
    for pattern, label in _FORBIDDEN_PATTERNS:
        m = _re.search(pattern, text or "", flags=_re.IGNORECASE)
        if m:
            found.append(f"{label}: '{m.group(0)}'")
    return found

_USER_PROMPT_TEMPLATE = """Generate a clinical assessment report for the following patient.

Disease Module: {disease_display}
Risk Probability: {probability:.1%}  (calibrated to real-world prevalence)
Decision Threshold: {threshold_note}
Risk Label: {label}
Risk Band: {confidence_band}

Note: this probability is stated on the deployment population's prevalence, so
it is NOT comparable to a 50% cut-off. Judge it against the decision threshold
and the risk band above, never against 50%.

The label and the band answer different questions and do not contradict each
other: the LABEL says which side of the decision threshold this patient falls
(whether to act at all), the BAND says how urgently among those flagged. A
Positive patient in the MODERATE band is above the threshold and warrants
follow-up; do not describe such a patient as low risk, and do not describe a
Negative patient as flagged.

Top Risk Factors (SHAP-ranked):
{shap_summary}

What each factor is:
{glossary_summary}

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


# What each model input actually measures, in the words a clinician would
# use. Sent instead of the patient's values so the narrative can name a
# factor precisely without being told the number.
#
# `Cholesterol` is the entry that mattered: it is SERUM TOTAL cholesterol in
# the UCI heart data, and reports were describing it as "LDL burden" — a
# different lipid fraction, and one this platform never measures. Naming the
# wrong analyte in a clinical report is a factual error about the patient,
# not a wording preference.
_FEATURE_GLOSSARY: Dict[str, str] = {
    # heart_disease
    "Age": "age in years",
    "Sex": "recorded sex",
    "ChestPainType": "chest pain character (TA typical angina / ATA atypical / NAP non-anginal / ASY asymptomatic)",
    "RestingBP": "resting systolic blood pressure (mm Hg)",
    "Cholesterol": "serum TOTAL cholesterol (mg/dL) — not LDL, not HDL, and no fractions are measured",
    "FastingBS": "fasting blood sugar above 120 mg/dL (yes/no flag, not a glucose value)",
    "RestingECG": "resting electrocardiogram category",
    "MaxHR": "maximum heart rate achieved during exercise testing",
    "ExerciseAngina": "exercise-induced angina (yes/no)",
    "Oldpeak": "ST depression induced by exercise relative to rest",
    "ST_Slope": "slope of the peak exercise ST segment",
    # diabetes (BRFSS self-report)
    "HighBP": "self-reported history of high blood pressure (yes/no)",
    "HighChol": "self-reported history of high cholesterol (yes/no)",
    "CholCheck": "cholesterol checked in the last 5 years (yes/no)",
    "BMI": "body mass index",
    "Smoker": "smoked at least 100 cigarettes in their lifetime (yes/no)",
    "Stroke": "self-reported history of stroke (yes/no)",
    "HeartDiseaseorAttack": "self-reported coronary heart disease or myocardial infarction (yes/no)",
    "PhysActivity": "any physical activity in the past 30 days (yes/no)",
    "Fruits": "eats fruit at least once a day (yes/no)",
    "Veggies": "eats vegetables at least once a day (yes/no)",
    "HvyAlcoholConsump": "heavy alcohol consumption (yes/no)",
    "AnyHealthcare": "has any health coverage (yes/no)",
    "NoDocbcCost": "could not see a doctor because of cost in the past year (yes/no)",
    "GenHlth": "self-rated general health, 1 (excellent) to 5 (poor)",
    "MentHlth": "days of poor mental health in the past 30",
    "PhysHlth": "days of poor physical health in the past 30",
    "DiffWalk": "serious difficulty walking or climbing stairs (yes/no)",
    "Education": "education level band",
    "Income": "income band",
}


def _format_glossary(shap_values: List[Dict[str, Any]], top_n: int = 5) -> str:
    """
    Describe the factors the report will discuss — names only, no values.

    Only the features that actually appear in the SHAP list are described, so
    the prompt stays short and the model is never handed a factor it was not
    asked to write about.
    """
    ranked = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:top_n]
    lines = []
    for item in ranked:
        name = item.get("feature", "")
        meaning = _FEATURE_GLOSSARY.get(name)
        if meaning:
            lines.append(f"  - {name}: {meaning}")
        else:
            # An engineered or unknown feature: say so rather than let the
            # model invent a clinical meaning for it.
            lines.append(f"  - {name}: a model-internal feature; describe it by name only")
    return "\n".join(lines) or "  - No factors available"


def band_for_report(
    probability_corrected: float,
    risk_bands: Mapping[str, float],
    decision_threshold: Optional[float] = None,
) -> str:
    """
    Display band, floored so it cannot contradict the screening decision.

    classify_band() alone produced "Positive" beside "LOW" — D-004 sits at
    14.4% with a 10.8% decision threshold and a 17.2% moderate cut-point, so
    the patient was flagged for follow-up and simultaneously told "routine
    follow-up, rescreen in 12 months". The two numbers answer different
    questions (which side of the threshold vs how urgent among the flagged),
    but LOW is not an available answer to the second question for a patient
    who is above the threshold: being flagged IS the floor.

    A patient at or above the decision threshold is therefore never LOW. The
    bands are otherwise untouched, and a Negative patient is unaffected.
    """
    band = classify_band(probability_corrected, risk_bands)
    if (
        decision_threshold is not None
        and probability_corrected >= decision_threshold
        and band == "LOW"
    ):
        return "MODERATE"
    return band


def _rule_based_report(
    disease_display: str,
    probability_corrected: float,
    label: str,
    shap_values: Optional[List[Dict[str, Any]]] = None,
    features: Optional[Dict[str, Any]] = None,
    risk_bands: Optional[Mapping[str, float]] = None,
    decision_threshold: Optional[float] = None,
) -> str:
    shap_values = shap_values or []
    features = features or {}
    top = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    top_names = [s["feature"] for s in top]
    actions = {
        "HIGH": "- Urgent specialist referral recommended\n- Order confirmatory investigations\n- Review the current care plan with the treating clinician",
        "MODERATE": "- Schedule follow-up within 4 weeks\n- Lifestyle modification counselling\n- Monitor key biomarkers",
        "LOW": "- Routine follow-up\n- Reinforce preventive measures\n- Rescreen in 12 months",
    }
    band = band_for_report(
        probability_corrected, risk_bands or DEFAULT_RISK_BANDS, decision_threshold
    )
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

    `features` is accepted and DELIBERATELY NOT SENT to the LLM. The clients
    still post it and the parameter is kept so they keep working, but the
    prompt now carries only the model's outputs — probability, threshold,
    label, band, and the SHAP-ranked feature NAMES with a glossary of what
    each one measures. Previously the first 20 raw patient values went to a
    third-party API on every report, which is how the narrative came to
    contain "62-year-old female" and "cholesterol 340". Those are health data
    about an individual; nothing in the report requires them, because SHAP
    already says which factors drove the estimate and in which direction.

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
    band = band_for_report(probability_corrected, bands, decision_threshold)
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
                disease_display, probability_corrected, label, shap_values,
                features, bands, decision_threshold,
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
            glossary_summary=_format_glossary(shap_values),
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

        violations = forbidden_content(report_text)
        if violations:
            log.warning("LLM report rejected (%s) — using rule-based report", "; ".join(violations))
            return {
                "report": _rule_based_report(
                    disease_display, probability_corrected, label, shap_values,
                    features, bands, decision_threshold,
                ),
                "source": "rule_based",
                "risk_band": band,
                "fallback_reason": "LLM output contained " + "; ".join(violations),
            }

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
                disease_display, probability_corrected, label, shap_values,
                features, bands, decision_threshold,
            ),
            "source": "rule_based",
            "risk_band": band,
            "fallback_reason": str(exc),
        }
