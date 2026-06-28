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

_PATTERNS: Dict[str, list] = {
    "age": [
        r"\b(\d{1,3})[- ]?(?:year[s]?[- ]?old|y/?o|yr[s]?)\b",
        r"\bage[:\s]+(\d{1,3})\b",
    ],
    "sex_male": [r"\b(male|man|he|his|gentleman|boy)\b"],
    "sex_female": [r"\b(female|woman|she|her|lady|girl)\b"],
    "bp_systolic": [
        r"(?:bp|blood pressure)[:\s]*(\d{2,3})\s*/\s*\d{2,3}",
        r"(?:systolic|sbp)[:\s]*(\d{2,3})",
    ],
    "bp_diastolic": [
        r"(?:bp|blood pressure)[:\s]*\d{2,3}\s*/\s*(\d{2,3})",
        r"(?:diastolic|dbp)[:\s]*(\d{2,3})",
    ],
    "cholesterol": [
        r"(?:cholesterol|ldl|hdl|total chol)[:\s]*(\d{2,3})\s*(?:mg/dl|mg)?",
    ],
    "glucose": [
        r"(?:glucose|blood sugar|fbs|rbs|bgr)[:\s]*(\d{2,3})\s*(?:mg/dl|mg)?",
    ],
    "bmi": [
        r"(?:bmi|body mass index)[:\s]*(\d{1,2}(?:\.\d)?)",
    ],
    "heart_rate": [
        r"(?:hr|heart rate|pulse)[:\s]*(\d{2,3})\s*(?:bpm)?",
    ],
    "creatinine": [
        r"(?:creatinine|cr|scr)[:\s]*(\d+(?:\.\d+)?)\s*(?:mg/dl|mg)?",
    ],
    "hemoglobin": [
        r"(?:hemoglobin|hgb|hb)[:\s]*(\d+(?:\.\d+)?)\s*(?:g/dl|gms?)?",
    ],
    "smoking": [
        r"\b(smok(?:er|ing|ed)|smoker|cigarette|tobacco)\b",
    ],
    "hypertension": [
        r"\b(hypertension|htn|high blood pressure)\b",
    ],
    "diabetes": [
        r"\b(diabetes|diabetic|dm|t2dm|t1dm)\b",
    ],
    "heart_disease_history": [
        r"\b(heart disease|cad|coronary artery disease|mi|myocardial infarction|chd)\b",
    ],
    "stroke_history": [
        r"\b(stroke|cva|tia|cerebrovascular)\b",
    ],
    "chest_pain": [
        r"\b(chest pain|angina|ata|typical angina|atypical angina)\b",
    ],
    "exercise_angina": [
        r"\b(exercise.?induced angina|angina on exertion|exertional angina)\b",
    ],
    "oldpeak": [
        r"(?:st depression|oldpeak|st.?segment)[:\s]*(\d+(?:\.\d+)?)",
    ],
    "marriage": [r"\b(married|spouse|husband|wife)\b"],
    "edema": [r"\b(edema|oedema|swelling|pedal edema)\b"],
    "appetite": [r"\b(poor appetite|anorexia|not eating|reduced appetite)\b"],
    "anemia": [r"\b(anemia|anaemia|low haemoglobin|iron deficiency)\b"],
}


def _regex_extract(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    extracted: Dict[str, Any] = {}

    def first_match(patterns):
        for pat in patterns:
            m = re.search(pat, text_lower, re.IGNORECASE)
            if m:
                return m
        return None

    # Numeric extractions
    for key in ("age", "bp_systolic", "bp_diastolic", "cholesterol", "glucose",
                "bmi", "heart_rate", "creatinine", "hemoglobin", "oldpeak"):
        m = first_match(_PATTERNS[key])
        if m:
            try:
                extracted[key] = float(m.group(1))
            except (IndexError, ValueError):
                pass

    # Boolean / categorical extractions
    if first_match(_PATTERNS["sex_male"]):
        extracted["sex"] = "Male"
    elif first_match(_PATTERNS["sex_female"]):
        extracted["sex"] = "Female"

    extracted["hypertension"] = 1 if first_match(_PATTERNS["hypertension"]) else None
    extracted["diabetes_flag"] = 1 if first_match(_PATTERNS["diabetes"]) else None
    extracted["heart_disease_flag"] = 1 if first_match(_PATTERNS["heart_disease_history"]) else None
    extracted["stroke_flag"] = 1 if first_match(_PATTERNS["stroke_history"]) else None
    extracted["smoking_flag"] = 1 if first_match(_PATTERNS["smoking"]) else None
    extracted["chest_pain_flag"] = 1 if first_match(_PATTERNS["chest_pain"]) else None
    extracted["exercise_angina"] = "Y" if first_match(_PATTERNS["exercise_angina"]) else None
    extracted["ever_married"] = "Yes" if first_match(_PATTERNS["marriage"]) else None
    extracted["edema_flag"] = 1 if first_match(_PATTERNS["edema"]) else None
    extracted["poor_appetite"] = 1 if first_match(_PATTERNS["appetite"]) else None
    extracted["anemia_flag"] = 1 if first_match(_PATTERNS["anemia"]) else None

    # Remove None values
    return {k: v for k, v in extracted.items() if v is not None}


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

_HEART_DISEASE_MAP = {
    "age":             ("Age",            lambda v: int(v)),
    "sex":             ("Sex",            lambda v: "M" if str(v).lower().startswith("m") else "F"),
    "bp_systolic":     ("RestingBP",      lambda v: int(v)),
    "cholesterol":     ("Cholesterol",    lambda v: int(v)),
    "heart_rate":      ("MaxHR",          lambda v: int(v)),
    "oldpeak":         ("Oldpeak",        lambda v: float(v)),
    "hypertension":    ("FastingBS",      lambda v: 1),
    "chest_pain_flag": ("ChestPainType",  lambda v: "ASY"),
    "exercise_angina": ("ExerciseAngina", lambda v: "Y"),
}

_DIABETES_MAP = {
    "age":                   ("Age",                 lambda v: max(1, min(13, round(int(v) / 7)))),
    "bmi":                   ("BMI",                 lambda v: float(v)),
    "bp_systolic":           ("HighBP",              lambda v: 1 if int(v) >= 130 else 0),
    "cholesterol":           ("HighChol",            lambda v: 1 if int(v) >= 200 else 0),
    "sex":                   ("Sex",                 lambda v: 1 if str(v).lower().startswith("m") else 0),
    "hypertension":          ("HighBP",              lambda v: 1),
    "heart_disease_flag":    ("HeartDiseaseorAttack",lambda v: 1),
    "stroke_flag":           ("Stroke",              lambda v: 1),
    "smoking_flag":          ("Smoker",              lambda v: 1),
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
