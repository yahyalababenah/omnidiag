"""
OmniDiag — LLM Clinical Report Generator
==========================================
Generates structured clinical narrative reports from prediction results
using the DeepSeek API (OpenAI-compatible). Falls back to a rule-based
template when the API key is unavailable (e.g. in offline/demo environments).

── Decisions, not bands ───────────────────────────────────────────────────
Every live module (heart, NHANES) reports a conformal decision: there is no
risk band and no decision threshold, and the report says so instead of
inventing either. The threshold-and-band wording that the retired BRFSS
diabetes module needed was removed in gate B5, together with its bands.

A report rendered from a stored row of a RETIRED module (backend/
retired_diseases.py) never reaches the LLM: `archived_report()` restates the
stored result under an "archived module" banner, rule-based, with no band, no
threshold and no recommended actions.
"""

import os
import logging
from typing import Any, Dict, List, Mapping, Optional

from backend.retired_diseases import RETIRED_DISEASES

log = logging.getLogger("omnidiag.llm")

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

#: Recommended-action blocks keyed on a CONFORMAL DECISION rather than a band.
#: A conformal module has no band to key on, and "uncertain" is a real third
#: answer -- not a middle amount of risk -- so it gets its own block.
_CONFORMAL_ACTIONS: Dict[str, str] = {
    "referral": (
        "- Specialist referral recommended\n"
        "- Order confirmatory investigations\n"
        "- Review the current care plan with the treating clinician"
    ),
    "uncertain": (
        "- Refer for further evaluation: the model could not place this patient "
        "confidently in either group\n"
        "- Treat the estimate as provisional and weigh the clinical picture\n"
        "- Additional information about this patient is likely to change the answer"
    ),
    "no_referral": (
        "- Routine follow-up\n"
        "- Reinforce preventive measures\n"
        "- Rescreen per local guidance"
    ),
}


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

# How the module decides, stated for the LLM in terms that are TRUE of the
# module that produced the numbers. Before Gate 8.4 there was one block, written
# for a threshold model on a prevalence-corrected probability, and heart was
# handed it unchanged: it was told heart's probability was "calibrated to
# real-world prevalence" (it is calibrated to the training hospitals' mix), that
# the label said "which side of the decision threshold" the patient fell on
# (heart has no threshold), and it was given a HIGH/MODERATE/LOW band heart
# configures none of. Five false premises, before the model wrote a word -- in
# the text a reviewer reads.
#: How a module that is NOT conformal decided. No live module takes this path;
#: it exists so one never gets a band it does not have.
_PLAIN_DECISION_BLOCK = """Estimated Probability: {probability:.1%}
Decision Threshold: {threshold_note}
Label: {label}

This module configures no risk band. Do not invent one, and do not describe the
probability as high, moderate or low risk."""

_CONFORMAL_DECISION_BLOCK = """Estimated Probability: {probability:.1%}{interval_note}
Decision: {decision_text}

This module does NOT compare a probability against a cut-off, and there is no
decision threshold and no HIGH/MODERATE/LOW risk band to report. Do not invent
either one, and do not describe the probability as high, moderate or low risk.
The decision above is the model's answer; the probability is context for it.

{scale_note}

{uncertain_note}"""


def _decision_block(
    probability: float, label: str, band: Optional[str], threshold_note: str,
    output_type: Optional[str], probability_scale: Optional[str],
    decision: Optional[str], lower: Optional[float], upper: Optional[float],
) -> str:
    """The prompt's description of how this module decided, per module."""
    if output_type != "conformal_decision":
        return _PLAIN_DECISION_BLOCK.format(
            probability=probability, threshold_note=threshold_note, label=label,
        )

    # The interval's NAME follows the module's calibration layer. Calling a Platt
    # bootstrap interval "Venn-Abers" in a clinical report is a factual error about
    # how the number was produced (Gate 9.3).
    interval_name = (
        "calibration interval"
        if probability_scale == "platt_calibrated_nhanes_2015_2016"
        else "Venn-Abers interval"
    )
    interval = (
        f"  ({interval_name} {lower:.1%} to {upper:.1%})"
        if lower is not None and upper is not None else ""
    )
    decision_text = {
        "referral": "REFER for further evaluation",
        "no_referral": "DO NOT refer; no further evaluation indicated on this estimate",
        "uncertain": (
            "UNCERTAIN — refer for further evaluation. The model could not place this "
            "patient confidently in either group"
        ),
    }.get(decision or "", label)

    scale_note = (
        "This probability is calibrated to the mix of the four teaching hospitals the "
        "model was trained on, NOT to the population of the hospital reading it. Do not "
        "present it as this patient's population risk."
        if probability_scale == "ivap_calibrated_training_mix"
        else (
            "This probability is calibrated to a US national survey sample of adults "
            "who had NOT been diagnosed with diabetes and were NOT on treatment. It is "
            "not calibrated to this clinic's population and must not be presented as "
            "this patient's absolute risk. The estimate targets dysglycaemia (HbA1c at "
            "or above 5.7%), which is not a diabetes diagnosis."
        )
        if probability_scale == "platt_calibrated_nhanes_2015_2016"
        else "State no assumption about what this probability is calibrated to."
    )
    uncertain_note = (
        "An UNCERTAIN result is not a middle amount of risk; it means the evidence did "
        "not separate the two groups for this patient. Say that plainly rather than "
        "describing it as moderate risk."
        if decision == "uncertain" else ""
    )
    return _CONFORMAL_DECISION_BLOCK.format(
        probability=probability, interval_note=interval, decision_text=decision_text,
        scale_note=scale_note, uncertain_note=uncertain_note,
    ).strip()


_USER_PROMPT_TEMPLATE = """Generate a clinical assessment report for the following patient.

Disease Module: {disease_display}
{decision_block}

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
    # diabetes_nhanes (NHANES, measured — not self-reported). Added in Gate 9.3.
    # These are clinical measurements and lab analytes, so naming the wrong one
    # is a factual error about the patient, exactly as with `Cholesterol` above.
    "RIDAGEYR": "age in years",
    "RIAGENDR": "recorded sex",
    "BMXBMI": "body mass index (kg/m^2), from measured height and weight",
    "ADIPOSITY_BAND": "central adiposity assessed by the clinician (normal / increased / high), standing in for waist-to-height ratio",
    "SBP": "systolic blood pressure (mm Hg), mean of the seated readings",
    "DBP": "diastolic blood pressure (mm Hg), mean of the seated readings",
    "BPXPLS": "resting pulse (beats per minute)",
    "MCQ300C": "close blood relative with diabetes (yes/no)",
    "CVD_ANY": "any prior cardiovascular disease — heart failure, coronary disease, myocardial infarction or stroke (yes/no)",
    "PAQ650": "vigorous recreational physical activity in a typical week (yes/no)",
    "PAQ665": "moderate recreational physical activity in a typical week (yes/no)",
    "LBDHDD": "serum HDL cholesterol (mg/dL) — in this module a marker of insulin resistance, not a lipid target on its own",
    "LBXSCH": "serum TOTAL cholesterol (mg/dL) — not LDL, and no fractions are measured",
    "LBXSTR": "serum triglycerides (mg/dL)",
    "LBXSATSI": "alanine aminotransferase, ALT (U/L) — a liver enzyme, raised in hepatic steatosis",
    "LBXSGTSI": "gamma-glutamyl transferase, GGT (U/L) — a liver enzyme",
    "LBXSCR": "serum creatinine (mg/dL) — renal function",
    "LBXSBU": "blood urea nitrogen (mg/dL) — renal function",
    "LBXSAL": "serum albumin (g/dL)",
    "LBXSUA": "serum uric acid (mg/dL)",
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


#: The banner every report on a retired module's row starts with.
ARCHIVED_BANNER = "ARCHIVED MODULE — replaced by {replacement}"


def archived_report(
    disease: str,
    probability: float,
    label: str,
    shap_values: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    The report for a stored row of a retired module: what was recorded, under a
    banner that says the module is archived and what replaced it.

    Rule-based on purpose -- an LLM is not asked to interpret the output of a
    model that is no longer maintained. No band, no threshold and no
    recommended actions: those belonged to the retired module and are not
    maintained either. Patient feature values are not an input at all.
    """
    info = RETIRED_DISEASES[disease]
    top = sorted(shap_values or [], key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    names = ", ".join(s["feature"] for s in top if "feature" in s) or "not recorded"
    return (
        f"**{ARCHIVED_BANNER.format(replacement=info['replaced_by_display'])}**\n"
        f"{info['display_name']} was retired on {info['retired_on']} and replaced by "
        f"the {info['replaced_by_display']} ({info['replaced_by']}). This report restates "
        f"the result exactly as it was stored. It is not a current assessment, and the "
        f"retired module's thresholds and bands are no longer maintained.\n\n"
        f"**Stored Result**\n"
        f"Estimate as recorded: {float(probability):.1%} ({label}).\n"
        f"Factors recorded as most influential: {names}.\n\n"
        f"**For a Current Assessment**\n"
        f"Screen the patient again with the {info['replaced_by_display']}."
    )


def _rule_based_report(
    disease_display: str,
    probability_corrected: float,
    label: str,
    shap_values: Optional[List[Dict[str, Any]]] = None,
    features: Optional[Dict[str, Any]] = None,
    risk_bands: Optional[Mapping[str, float]] = None,
    decision_threshold: Optional[float] = None,
    output_type: Optional[str] = None,
    probability_scale: Optional[str] = None,
    decision: Optional[str] = None,
    probability_lower: Optional[float] = None,
    probability_upper: Optional[float] = None,
) -> str:
    """
    The report written without the LLM -- which is what runs with no API key,
    and therefore what a demo usually shows.

    For a conformal module this must not talk about bands or population
    prevalence. Until Gate 8.4 it told a heart patient their result was
    "<n>% probability on this population's prevalence, classified against the
    module's own risk bands": the probability is calibrated to the training
    hospitals' mix, and heart configures no risk bands at all.
    """
    shap_values = shap_values or []
    features = features or {}
    top = sorted(shap_values, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:3]
    top_names = [s["feature"] for s in top]
    drivers = "\n".join(f"- {s['feature']} (SHAP {s['shap_value']:+.3f})" for s in top)

    if output_type == "conformal_decision":
        interval = (
            f" (interval {probability_lower:.1%} to {probability_upper:.1%})"
            if probability_lower is not None and probability_upper is not None else ""
        )
        scale_sentence = (
            "calibrated to the mix of the four teaching hospitals this model was "
            "trained on, not to the population of the hospital reading it"
            if probability_scale == "ivap_calibrated_training_mix"
            else "on the scale this module reports"
        )
        stratification = (
            # Phrased without the words "risk band" on purpose: the test that
            # guards this text checks for them bluntly, and a blunt check that
            # cannot be argued with is worth more than a sentence that mentions
            # what it is denying.
            "This module reports a decision rather than a severity rating: the estimate is "
            f"{probability_corrected:.1%}{interval}, {scale_sentence}. "
            + (
                "An uncertain result means the evidence did not separate the two "
                "groups for this patient, which is why the referral stands."
                if decision == "uncertain"
                else "There is no decision threshold to compare it against."
            )
        )
        return (
            f"**Clinical Summary**\n"
            f"Patient assessed for {disease_display} risk. "
            f"Model estimate: {probability_corrected:.1%}{interval} — {label}. "
            f"Top contributing factors: {', '.join(top_names)}.\n\n"
            f"**Key Risk Drivers**\n{drivers}\n\n"
            f"**Recommended Actions**\n"
            f"{_CONFORMAL_ACTIONS.get(decision or '', _CONFORMAL_ACTIONS['uncertain'])}\n\n"
            f"**Decision Note**\n{stratification}"
        )

    # Not conformal (no live module): the probability and the label, and no band.
    decision_note = (
        f"The module's decision threshold is {decision_threshold:.4f}; this result is {label}."
        if decision_threshold is not None
        else "This module configures no decision threshold and no risk band, and none is invented here."
    )
    return (
        f"**Clinical Summary**\n"
        f"Patient assessed for {disease_display} risk. "
        f"Model probability: {probability_corrected:.1%} ({label}). "
        f"Top contributing factors: {', '.join(top_names)}.\n\n"
        f"**Key Risk Drivers**\n{drivers}\n\n"
        f"**Decision Note**\n{decision_note}"
    )


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
    output_type: Optional[str] = None,
    probability_scale: Optional[str] = None,
    decision: Optional[str] = None,
    probability_lower: Optional[float] = None,
    probability_upper: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a structured clinical report.

    `output_type` decides how the report talks about the decision, and it comes
    from the disease config -- never from the disease name. 'conformal_decision'
    means there is no threshold and no band: the report states the decision and
    the Venn-Abers interval, and says so. Anything else gets the probability,
    the label and the threshold if it has one -- never a band.

    `risk_bands` is accepted for existing callers and ignored: no module
    configures bands since gate B5.

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
    # A conformal module has NO band -- not a default one and not a floored one.
    # It configures none because its probability's meaning does not transport
    # between hospitals (D-32), so no band is computed here at all: computing one
    # and then dropping it would still log a "discarding caller band" line that
    # implies a correct band exists.
    # No module configures risk bands any more (gate B5; a config declaring
    # them is refused at load), so no band is computed and a caller's is
    # discarded. `risk_band` stays in the result, always None, for clients.
    bands = None
    band = None
    if confidence_band:
        log.info(
            "Discarding caller-supplied confidence_band=%r: %s reports no band",
            confidence_band, disease_display,
        )
    threshold_note = (
        f"{decision_threshold:.4f} — at or above this the patient is classified Positive"
        if decision_threshold is not None
        else "this module does not decide by a threshold"
    )

    api_key = _get_api_key()

    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — falling back to rule-based report")
        return {
            "report": _rule_based_report(
                disease_display, probability_corrected, label, shap_values,
                features, bands, decision_threshold, output_type,
                probability_scale, decision, probability_lower, probability_upper,
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
            decision_block=_decision_block(
                probability_corrected, label, band, threshold_note,
                output_type, probability_scale, decision,
                probability_lower, probability_upper,
            ),
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
                    features, bands, decision_threshold, output_type,
                    probability_scale, decision, probability_lower, probability_upper,
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
                features, bands, decision_threshold, output_type,
                probability_scale, decision, probability_lower, probability_upper,
            ),
            "source": "rule_based",
            "risk_band": band,
            "fallback_reason": str(exc),
        }
