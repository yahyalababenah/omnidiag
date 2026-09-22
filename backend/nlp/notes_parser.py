"""
OmniDiag — NLP Clinical Notes Parser
======================================
Extracts structured clinical features from free-text clinical notes using
a two-tier approach:

  1. Primary: BioBERT / ClinicalBERT NER via HuggingFace Transformers
     (loaded lazily — first call initialises the pipeline).
  2. Fallback: Rule-based regex patterns (runs offline, zero dependencies).

The output is a dict suitable for passing directly to the prediction API,
pre-filled with whatever values could be extracted from the note.
Missing values are omitted so the frontend can prompt the user to fill them in.

Supported diseases: heart_disease, diabetes
"""

import re
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("omnidiag.nlp")

# ---------------------------------------------------------------------------
# Regex patterns for the rule-based fallback
# ---------------------------------------------------------------------------
#
# Numeric patterns capture the value in group 1. Condition patterns mark the
# condition word itself with the named group `k`; negation is judged from
# the text between the start of the clause and `k` (see _negated). A negated
# condition is extracted as an explicit 0 — "non-smoker" is information, not
# an absence of it. Before negation handling, "No stroke, no heart disease,
# non-smoker" was extracted as stroke = heart disease = smoker = 1.

_NUMERIC: Dict[str, list] = {
    "age": [
        r"\b(\d{1,3})[- ]?(?:year[s]?[- ]?old|y/?o|yr[s]?)\b",
        r"\bage[:\s]+(\d{1,3})\b",
        r"\b(\d{2,3})\s*[-,]?\s*(?:year[s]?[-\s]?old\s+)?(?:male|female|man|woman)\b",
    ],
    "bp_systolic": [
        r"\b(?:bp|blood pressure)[:\s]*(\d{2,3})\s*/\s*\d{2,3}",
        r"\b(?:systolic|sbp)[:\s]*(\d{2,3})",
        r"\b(?:bp|blood pressure)[:\s]+(\d{2,3})\b",
    ],
    "bp_diastolic": [
        r"\b(?:bp|blood pressure)[:\s]*\d{2,3}\s*/\s*(\d{2,3})",
        r"\b(?:diastolic|dbp)[:\s]*(\d{2,3})",
    ],
    # Total cholesterol only: "LDL 160" is not the Cholesterol field.
    "cholesterol": [
        r"\b(?:total\s+cholesterol|total\s+chol|cholesterol)\b(?:\s+checked[^:\d]*)?[:\s]*(\d{2,3})",
    ],
    "fasting_glucose": [
        r"\b(?:fasting\s+(?:blood\s+)?(?:sugar|glucose)|fbs|fpg)[:\s]*(\d{2,3})",
    ],
    "bmi": [
        r"\b(?:bmi|body mass index)[:\s]*(\d{1,2}(?:\.\d)?)",
    ],
    # Maximum heart rate only. A resting "HR 72" or "pulse 88" is not MaxHR.
    "max_heart_rate": [
        r"\b(?:max(?:imum|imal)?|peak)\s+(?:heart\s+rate|hr)\s*(?:achieved|reached)?[:\s]*(?:of\s+)?(\d{2,3})",
        r"\b(?:heart\s+rate|hr)\s+max(?:imum)?[:\s]*(\d{2,3})",
    ],
    "oldpeak": [
        r"\b(?:st\s+depression|oldpeak)[:\s]*(?:of\s+)?(\d+(?:\.\d+)?)",
    ],
}

_CONDITIONS: Dict[str, list] = {
    "smoking": [
        r"\b(?P<k>smok(?:er|ing|es|ed)?|cigarettes?|tobacco)\b",
    ],
    "hypertension": [
        r"\b(?P<k>hypertension|htn|high\s+blood\s+pressure)\b",
    ],
    "diabetes": [
        r"\b(?P<k>diabetes|diabetic|t2dm|t1dm|dm)\b",
    ],
    "heart_disease_history": [
        r"\b(?P<k>heart\s+disease|cad|coronary\s+artery\s+disease|mi|myocardial\s+infarction|chd|heart\s+attack)\b",
    ],
    "stroke_history": [
        r"\b(?P<k>stroke|cva|tia|cerebrovascular)\b",
    ],
    "exercise_angina": [
        r"\b(?P<k>exercise.?induced\s+angina|angina\s+on\s+exertion|exertional\s+angina)\b",
        r"\b(?:exercise|treadmill|stress|tolerance)\b[^.;]*?\b(?P<k>angina)\b",
    ],
}

# Chest pain type, most specific first ("atypical angina" contains "typical angina").
_CHEST_PAIN = [
    ("ATA", r"\batypical\s+(?:chest\s+pain|angina)\b"),
    ("NAP", r"\bnon[-\s]?anginal\s+(?:chest\s+)?pain\b"),
    ("TA", r"(?<!a)\btypical\s+angina\b|\btypical\s+chest\s+pain\b"),
    ("ASY", r"\b(?:denies|denied|no)\s+(?:any\s+)?chest\s+pain\b|\basymptomatic\b"),
]

_RESTING_ECG = [
    ("LVH", r"\blvh\b|left\s+ventricular\s+hypertrophy"),
    ("ST", r"\bst[-\s]?t\s+(?:wave\s+)?(?:abnormalit\w*|changes)"),
    ("Normal", r"\bnormal\s+(?:resting\s+)?(?:ecg|ekg)\b|\b(?:ecg|ekg)[:\s]+normal\b"),
]

_NEGATION_CUES = re.compile(
    r"\b(?:no|not|denies|denied|deny|without|negative\s+for|never|free\s+of|absence\s+of|absent)\b",
    re.IGNORECASE,
)


def _negated(text: str, start: int) -> bool:
    """
    True when the term starting at `start` is negated: it is prefixed by
    "non-"/"non " (non-smoker), or a negation cue appears earlier in the
    same clause (clauses end at . ; , : newline or " but ").
    """
    if re.search(r"\bnon[-\s]?$", text[max(0, start - 4):start], re.IGNORECASE):
        return True
    clause_start = max(
        text.rfind(ch, 0, start) for ch in (".", ";", ",", ":", "\n")
    ) + 1
    but = text.lower().rfind(" but ", clause_start, start)
    if but >= 0:
        clause_start = but + 5
    return bool(_NEGATION_CUES.search(text[clause_start:start]))


def _condition(text: str, patterns: list) -> Optional[int]:
    """1 if any affirmed mention, 0 if every mention is negated, None if absent."""
    seen = False
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            seen = True
            start = m.start("k") if "k" in m.re.groupindex else m.start()
            if not _negated(text, start):
                return 1
    return 0 if seen else None


def _regex_extract(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    extracted: Dict[str, Any] = {}

    for key, patterns in _NUMERIC.items():
        for pat in patterns:
            m = re.search(pat, text_lower, re.IGNORECASE)
            if m:
                try:
                    extracted[key] = float(m.group(1))
                    break
                except (IndexError, ValueError):
                    pass

    # Sex: explicit words first; pronouns only when no explicit word exists.
    if re.search(r"\b(?:female|woman|lady|girl|mrs|ms)\b", text_lower):
        extracted["sex"] = "Female"
    elif re.search(r"\b(?:male|man|gentleman|boy|mr)\b", text_lower):
        extracted["sex"] = "Male"
    elif re.search(r"\b(?:she|her)\b", text_lower):
        extracted["sex"] = "Female"
    elif re.search(r"\b(?:he|his|him)\b", text_lower):
        extracted["sex"] = "Male"

    flags = {
        "hypertension": "hypertension",
        "diabetes": "diabetes_flag",
        "heart_disease_history": "heart_disease_flag",
        "stroke_history": "stroke_flag",
        "smoking": "smoking_flag",
        "exercise_angina": "exercise_angina",
    }
    for cond, key in flags.items():
        value = _condition(text_lower, _CONDITIONS[cond])
        if value is not None:
            extracted[key] = value

    for code, pat in _CHEST_PAIN:
        if re.search(pat, text_lower, re.IGNORECASE):
            extracted["chest_pain_type"] = code
            break

    for code, pat in _RESTING_ECG:
        if re.search(pat, text_lower, re.IGNORECASE):
            extracted["resting_ecg"] = code
            break

    return extracted


# ---------------------------------------------------------------------------
# BioBERT / ClinicalBERT NER (lazy-loaded)
# ---------------------------------------------------------------------------

_ner_pipeline = None
_NER_MODEL = os.getenv("CLINICAL_NER_MODEL", "d4data/biomedical-ner-all")


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore
            log.info(f"Loading clinical NER model: {_NER_MODEL}")
            _ner_pipeline = pipeline(
                "ner",
                model=_NER_MODEL,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
            log.info("Clinical NER pipeline ready")
        except Exception as exc:
            log.warning(f"Failed to load NER model ({exc!r}). Falling back to regex.")
            _ner_pipeline = "unavailable"
    return _ner_pipeline if _ner_pipeline != "unavailable" else None


def _bert_extract(text: str) -> Dict[str, Any]:
    pipe = _get_ner_pipeline()
    if pipe is None:
        return {}
    try:
        entities = pipe(text)
        extracted: Dict[str, Any] = {}
        for ent in entities:
            label = ent.get("entity_group", "").upper()
            word = ent.get("word", "").strip()
            score = ent.get("score", 0.0)
            if score < 0.7:
                continue
            if label in ("AGE",):
                m = re.search(r"\d+", word)
                if m:
                    extracted["age"] = float(m.group())
            elif label in ("DISEASE", "CONDITION", "PROBLEM"):
                word_lower = word.lower()
                if any(x in word_lower for x in ("hypertension", "htn")):
                    extracted["hypertension"] = 1
                if any(x in word_lower for x in ("diabetes", "dm")):
                    extracted["diabetes_flag"] = 1
                if any(x in word_lower for x in ("stroke", "cva")):
                    extracted["stroke_flag"] = 1
                if any(x in word_lower for x in ("heart disease", "cad")):
                    extracted["heart_disease_flag"] = 1
                if any(x in word_lower for x in ("anemia", "anaemia")):
                    extracted["anemia_flag"] = 1
        return extracted
    except Exception as exc:
        log.warning(f"NER extraction error: {exc!r}")
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Disease-specific field mappers
# Maps generic extracted keys → schema field names for each disease
# ---------------------------------------------------------------------------

def _brfss_age_bucket(age: float) -> int:
    """BRFSS _AGEG5YR: 1 = 18-24, 2 = 25-29, ... 12 = 75-79, 13 = 80+."""
    age = int(age)
    if age < 25:
        return 1
    return min(13, (age - 25) // 5 + 2)


_HEART_DISEASE_MAP = {
    "age":             ("Age",            lambda v: int(v)),
    "sex":             ("Sex",            lambda v: "M" if str(v).lower().startswith("m") else "F"),
    "bp_systolic":     ("RestingBP",      lambda v: int(v)),
    "cholesterol":     ("Cholesterol",    lambda v: int(v)),
    "max_heart_rate":  ("MaxHR",          lambda v: int(v)),
    "oldpeak":         ("Oldpeak",        lambda v: float(v)),
    "fasting_glucose": ("FastingBS",      lambda v: 1 if float(v) > 120 else 0),
    "chest_pain_type": ("ChestPainType",  lambda v: v),
    "resting_ecg":     ("RestingECG",     lambda v: v),
    "exercise_angina": ("ExerciseAngina", lambda v: "Y" if int(v) else "N"),
}

# Order matters: a later key overwrites an earlier one mapped to the same
# field, so a stated hypertension history wins over a single BP reading.
_DIABETES_MAP = {
    "age":                   ("Age",                 _brfss_age_bucket),
    "bmi":                   ("BMI",                 lambda v: float(v)),
    "bp_systolic":           ("HighBP",              lambda v: 1 if int(v) >= 130 else 0),
    "cholesterol":           ("HighChol",            lambda v: 1 if int(v) >= 200 else 0),
    "sex":                   ("Sex",                 lambda v: 1 if str(v).lower().startswith("m") else 0),
    "hypertension":          ("HighBP",              lambda v: int(v)),
    "heart_disease_flag":    ("HeartDiseaseorAttack",lambda v: int(v)),
    "stroke_flag":           ("Stroke",              lambda v: int(v)),
    "smoking_flag":          ("Smoker",              lambda v: int(v)),
}


def map_to_disease_schema(extracted: Dict[str, Any], disease: str) -> Dict[str, Any]:
    """Map generic NLP-extracted fields to disease-specific schema field names."""
    mapping = {"heart_disease": _HEART_DISEASE_MAP, "diabetes": _DIABETES_MAP}.get(disease, {})
    result: Dict[str, Any] = {}
    for generic_key, (schema_key, transform) in mapping.items():
        if generic_key in extracted:
            try:
                result[schema_key] = transform(extracted[generic_key])
            except Exception:
                pass
    return result


def bert_status() -> Dict[str, Any]:
    """
    Whether the BERT NER path can run in this environment, WITHOUT loading
    the model: `transformers` and a backend (`torch`) must both import. The
    model itself is downloaded on first use.
    """
    import importlib.util
    missing = [m for m in ("transformers", "torch") if importlib.util.find_spec(m) is None]
    return {
        "available": not missing,
        "model": _NER_MODEL,
        "reason": None if not missing else f"not installed: {', '.join(missing)}",
    }


def parse_clinical_note(note: str, use_bert: bool = True) -> Dict[str, Any]:
    """
    Parse a free-text clinical note and extract structured features.

    Args:
        note:      The clinical note text.
        use_bert:  Whether to attempt BioBERT NER (falls back to regex on failure).

    Returns:
        Dict of extracted feature_name → value. Only present for detected values.
        Numeric values are Python floats; categorical values are strings.
    """
    if not note or not note.strip():
        return {}

    # Regex baseline (always runs)
    result = _regex_extract(note)

    # Merge BERT results (BERT takes precedence for overlapping keys)
    if use_bert:
        bert_result = _bert_extract(note)
        result.update(bert_result)

    return result


# ── Language support ─────────────────────────────────────────────────────────
#
# Every pattern in this module is an English regex. An Arabic note therefore
# extracts nothing, and the UI used to report that as "0 fields extracted —
# patient data updated", which reads as "the note contained nothing useful"
# rather than "this tool cannot read your language". A clinician at a Jordan
# demo typing an Arabic note deserves to be told which it is.
#
# Detection is by script, not by language model: the Arabic block is
# unambiguous and this only needs to answer "can the English patterns
# possibly work here".

_ARABIC_BLOCK = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")

SUPPORTED_LANGUAGE = "en"

UNSUPPORTED_LANGUAGE_MESSAGE = (
    "This parser reads English clinical notes only. Arabic text is not "
    "supported and no fields were extracted from it — enter the values "
    "directly, or paste an English note."
)

MIXED_LANGUAGE_MESSAGE = (
    "This parser reads English clinical notes only. The Arabic parts of this "
    "note were not read; only English terms and abbreviations were extracted. "
    "Check the fields below before applying them."
)


def detect_script(note: str) -> str:
    """
    'arabic', 'mixed', or 'latin' — which scripts the note is written in.

    'mixed' matters on its own: an Arabic note sprinkled with English
    abbreviations (BP, HTN, DM, BMI) yields a handful of fields, which looks
    like a successful parse while most of the note was silently skipped.
    """
    if not note:
        return "latin"
    has_arabic = bool(_ARABIC_BLOCK.search(note))
    has_latin = bool(_LATIN_LETTER.search(note))
    if has_arabic and has_latin:
        return "mixed"
    if has_arabic:
        return "arabic"
    return "latin"


def language_support(note: str) -> Dict[str, Any]:
    """
    Whether this note is in a language the parser can actually read.

    Returns {script, supported, message}. `supported` is False for anything
    containing Arabic, including mixed notes: a partial read of a clinical
    note is not a supported read, and saying so is the whole point.
    """
    script = detect_script(note)
    if script == "arabic":
        return {"script": script, "supported": False, "message": UNSUPPORTED_LANGUAGE_MESSAGE}
    if script == "mixed":
        return {"script": script, "supported": False, "message": MIXED_LANGUAGE_MESSAGE}
    return {"script": script, "supported": True, "message": None}
