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

import logging
import os
import re
from typing import Any, Dict, Optional

try:
    import spacy
    
except ImportError:  # spaCy is optional: without it the regex baseline still runs
    spacy = None

log = logging.getLogger("omnidiag.nlp")

# ---------------------------------------------------------------------------
# spaCy NLP Logic (Lazy-loaded)
# ---------------------------------------------------------------------------
_nlp_spacy = None


def _get_spacy_pipeline():
    global _nlp_spacy
    if _nlp_spacy is None:
        if spacy is None:
            _nlp_spacy = "unavailable"
            return None
        try:
            log.info("Loading spaCy en_core_web_sm model...")
            _nlp_spacy = spacy.load("en_core_web_sm")
            log.info("spaCy pipeline ready")
        except Exception as exc:
            log.warning(f"spaCy model not found or failed to load: {exc!r}")
            _nlp_spacy = "unavailable"

    return _nlp_spacy if _nlp_spacy != "unavailable" else None


# A number is assigned to the clinical concept nearest to it in the sentence's
# dependency tree, not to a fixed "keyword then digits" template. So
# "his total cholesterol came back at 240" and "she has a BMI of 31" resolve
# the same way whatever words sit in between.
_CONCEPT_WORDS = {
    "age": {"age", "aged"},
    "bp": {"pressure", "bp"},
    "cholesterol": {"cholesterol", "chol"},
    "bmi": {"bmi"},
    "glucose": {"glucose", "sugar", "fbs", "fpg"},
    "hr": {"rate", "hr"},
    "st": {"oldpeak", "depression"},
}
# concept -> (feature, min, max)
_CONCEPT_RANGE = {
    "age": ("age", 0, 120),
    "cholesterol": ("cholesterol", 50, 700),
    "bmi": ("bmi", 10, 80),
    "glucose": ("fasting_glucose", 30, 700),
    "hr": ("max_heart_rate", 30, 250),
    "st": ("oldpeak", -3, 10),
}
_WORD_TO_CONCEPT = {w: c for c, ws in _CONCEPT_WORDS.items() for w in ws}
_MAX_TREE_DIST = 4
_NON_CLINICAL_UNITS = {"month", "months", "week", "weeks", "day", "days", "hours", "hour", "kg", "kgs", "lb", "lbs", "cm", "m", "pounds", "kilograms", "feet", "ft", "inches"}
_PERSON_WORDS = {"he", "she", "who", "patient", "pt", "man", "woman", "gentleman", "lady"}


def _tree_dist(a, b) -> int:
    pa = [a] + list(a.ancestors)
    pb = [b] + list(b.ancestors)
    for i, x in enumerate(pa):
        if x in pb:
            return i + pb.index(x)
    return 99


def _prev_words(doc, i, n=3):
    return {t.lower_ for t in doc[max(0, i - n):i]}


def _concept_of(doc, tok):
    """Concept a token names, or None. Resolves ambiguous words from context."""
    c = _WORD_TO_CONCEPT.get(tok.lower_)
    if c is None:
        return None
    prev = _prev_words(doc, tok.i)
    if c == "hr":
        # Only MAXIMUM heart rate is a feature; resting "HR 88" is not MaxHR.
        near = prev | {t.lower_ for t in doc[tok.i + 1:tok.i + 3]}
        if not near & {"max", "maximum", "maximal", "peak"}:
            return None
        if tok.lower_ == "rate" and "heart" not in prev:
            return None
    if c == "st" and tok.lower_ == "depression" and "st" not in prev:
        return None
    if c == "glucose" and tok.lower_ in {"glucose", "sugar"} and not (
        prev | {t.lower_ for t in doc[tok.i + 1:tok.i + 2]}
    ) & {"fasting", "fbs", "fpg"}:
        return None
    return c


def _number_mentions(doc):
    """(value(s), first_token, last_token) for every numeric mention."""
    out, i, n = [], 0, len(doc)
    while i < n:
        t = doc[i]
        m = re.match(r"(\d{2,3})/(\d{2,3})\W*$", t.text)
        if m:
            out.append(((float(m.group(1)), float(m.group(2))), t, t))
            i += 1
            continue
        if not t.like_num:
            i += 1
            continue
        j = i
        # join spelled-out numbers: "sixty five"
        while not doc[i].text.replace(".", "").isdigit():
            if j + 1 < n and doc[j + 1].like_num:
                j += 1
            elif j + 2 < n and doc[j + 1].text == "-" and doc[j + 2].like_num:
                j += 2  # "sixty-two"
            else:
                break
        digits = _words_to_digits(" ".join(x.lower_ for x in doc[i:j + 1]))
        try:
            v = float(digits.replace(",", ""))
        except ValueError:
            i = j + 1
            continue
        # "150 over 95" / "150 / 95"
        if j + 2 < n and doc[j + 1].lower_ in {"over", "/"} and doc[j + 2].text.isdigit():
            out.append(((v, float(doc[j + 2].text)), t, doc[j + 2]))
            i = j + 3
            continue
        out.append(((v,), t, doc[j]))
        i = j + 1
    return out


def _spacy_extract(text: str) -> dict:
    nlp = _get_spacy_pipeline()
    if nlp is None:
        return {}

    doc = nlp(text)
    extracted: dict = {}
    concepts = [(t, _concept_of(doc, t)) for t in doc]
    concepts = [(t, c) for t, c in concepts if c]

    mentions = _number_mentions(doc)
    pending, mi_count, fallback = [], {}, []
    for mi, (vals, first, last) in enumerate(mentions):
        fallback.append(mi)
        nxt = doc[last.i + 1] if last.i + 1 < len(doc) else None
        nxt2 = doc[last.i + 2] if last.i + 2 < len(doc) else None
        # "50 years old", "50-year-old", "50 yo": the phrase itself says age.
        if nxt is not None and (
            nxt.lower_ in {"yo", "y/o", "yrs", "yr"}
            or (nxt.lower_ in {"year", "years"} and nxt2 is not None and nxt2.lower_ in {"old", "-"})
            or nxt.lower_ in {"year-old", "years-old"}
        ):
            if len(vals) == 1 and 0 < vals[0] <= 120:
                extracted.setdefault("age", vals[0])
            continue
        if nxt is not None and nxt.lower_ in _NON_CLINICAL_UNITS:
            continue
        cands = [
            ((0 if 0 < first.i - c_tok.i <= 1 else min(_tree_dist(first, c_tok), _MAX_TREE_DIST), abs(c_tok.i - first.i)), c, mi)
            for c_tok, c in concepts
            # Broken parses (telegraphic or mixed-language notes) fall back to
            # plain adjacency: "BP 132/84".
            if _tree_dist(first, c_tok) <= _MAX_TREE_DIST or 0 < first.i - c_tok.i <= 2
        ]
        pending.extend((k, c, mi) for k, c, _ in cands)
        mi_count[mi] = len(cands)
    # Each number goes to one concept and each concept takes one number: the
    # closest pair first, so "BMI of 31" keeps 31 even beside "sixty five".
    used_n, used_c = set(), set()
    for _, c, mi in sorted(pending):
        if mi in used_n or c in used_c:
            continue
        used_n.add(mi)
        used_c.add(c)
        vals = mentions[mi][0]
        if c == "bp":
            if len(vals) == 2 and 50 <= vals[0] <= 260 and 30 <= vals[1] <= 160:
                extracted.setdefault("bp_systolic", vals[0])
                extracted.setdefault("bp_diastolic", vals[1])
            elif len(vals) == 1 and 50 <= vals[0] <= 260:
                extracted.setdefault("bp_systolic", vals[0])
            continue
        feat, lo, hi = _CONCEPT_RANGE[c]
        if lo <= vals[0] <= hi:
            extracted.setdefault(feat, vals[0])

    # "who is now 58": a bare number predicated of the patient is the age.
    if "age" not in extracted:
        for mi in fallback:
            if mi in used_n:
                continue
            first = mentions[mi][1]
            head = first.head
            subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if head.lemma_ == "be" and subj and subj[0].lower_ in _PERSON_WORDS:
                v = mentions[mi][0]
                if len(v) == 1 and 0 < v[0] <= 120:
                    extracted["age"] = v[0]
                    break

    return extracted


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

# Words that may sit between a field name and its value: "age is 50",
# "cholesterol level was around 240", "BP of 150 over 95", "HR: 172 bpm".
_L = r"(?:\s*[:=\-]\s*|\s+(?:(?:is|was|of|at|about|around|approximately|approx|reading|level|levels|measured|measures|came\s+(?:back\s+)?(?:at|to)|to|the)\b\s*)*)"
_SLASH = r"\s*(?:/|over)\s*"

_NUMERIC: Dict[str, list] = {
    "age": [
        r"\b(\d{1,3})[- ]?(?:year[s]?[- ]?old|y/?o|yr[s]?)\b",
        r"\bage[d]?" + _L + r"(\d{1,3})\b(?!\s*(?:months?|weeks?|days?))",
        r"\bage[d]?\s+(\d{1,3})\b(?!\s*(?:months?|weeks?|days?))",
        r"\b(\d{2,3})\s*[-,]?\s*(?:year[s]?[-\s]?old\s+)?(?:male|female|man|woman)\b",
        r"\b(\d{2,3})\s?[mf]\b",
        r"\b(?:patient|pt|he|she)\s+is\s+(?:a\s+)?(\d{2,3})\b(?!\s*(?:kg|cm|mg|mmhg|bpm|%))",
        r"\bturned\s+(\d{2,3})\b",
    ],
    "bp_systolic": [
        r"\b(?:bp|blood\s+pressure)(?:" + _L + r")?(\d{2,3})" + _SLASH + r"\d{2,3}",
        r"\b(?:systolic|sbp)(?:\s+(?:bp|blood\s+pressure))?" + _L + r"(\d{2,3})",
        r"\b(?:bp|blood\s+pressure)" + _L + r"(\d{2,3})\b",
    ],
    "bp_diastolic": [
        r"\b(?:bp|blood\s+pressure)(?:" + _L + r")?\d{2,3}" + _SLASH + r"(\d{2,3})",
        r"\b(?:diastolic|dbp)(?:\s+(?:bp|blood\s+pressure))?" + _L + r"(\d{2,3})",
    ],
    # Total cholesterol only: "LDL 160" is not the Cholesterol field.
    "cholesterol": [
        r"\b(?:total\s+cholesterol|total\s+chol|cholesterol)\b(?:\s+checked[^:\d]*)?" + _L + r"(\d{2,3})\b",
    ],
    "fasting_glucose": [
        r"\b(?:fasting\s+(?:blood\s+)?(?:sugar|glucose)|fbs|fpg)" + _L + r"(\d{2,3})",
    ],
    "bmi": [
        r"\b(?:bmi|body\s+mass\s+index)" + _L + r"(\d{1,2}(?:\.\d+)?)",
    ],
    # Maximum heart rate only. A resting "HR 72" or "pulse 88" is not MaxHR.
    "max_heart_rate": [
        r"\b(?:max(?:imum|imal)?|peak)\s+(?:heart\s+rate|hr)\s*(?:achieved|reached)?" + _L + r"(\d{2,3})",
        r"\b(?:heart\s+rate|hr)\s+max(?:imum)?" + _L + r"(\d{2,3})",
    ],
    "oldpeak": [
        r"\b(?:st\s+depression|oldpeak)" + _L + r"(\d+(?:\.\d+)?)",
    ],
}

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[-\s]+(" + "|".join(k for k in _ONES if 0 < _ONES[k] < 10) + r"))?\b"
    r"|\b(" + "|".join(_ONES) + r")\b"
)


def _words_to_digits(text: str) -> str:
    """'fifty five' -> '55', so the numeric patterns see one form."""
    def sub(m):
        if m.group(1):
            return str(_TENS[m.group(1)] + (_ONES[m.group(2)] if m.group(2) else 0))
        return str(_ONES[m.group(3)])
    return _NUMWORD.sub(sub, text)


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
    if re.search(r"\bnon[-\s]?$", text[max(0, start - 4) : start], re.IGNORECASE):
        return True
    clause_start = (
        max(text.rfind(ch, 0, start) for ch in (".", ";", ",", ":", "\n")) + 1
    )
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
    text_lower = _words_to_digits(text.lower())
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
    if re.search(r"\b(?:female|woman|lady|girl|mrs|ms)\b|\b\d{2,3}\s?f\b", text_lower):
        extracted["sex"] = "Female"
    elif re.search(r"\b(?:male|man|gentleman|boy|mr)\b|\b\d{2,3}\s?m\b", text_lower):
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


_HEART_DISEASE_MAP = {
    "age": ("Age", lambda v: int(v)),
    "sex": ("Sex", lambda v: "M" if str(v).lower().startswith("m") else "F"),
    "bp_systolic": ("RestingBP", lambda v: int(v)),
    "cholesterol": ("Cholesterol", lambda v: int(v)),
    "max_heart_rate": ("MaxHR", lambda v: int(v)),
    "oldpeak": ("Oldpeak", lambda v: float(v)),
    "fasting_glucose": ("FastingBS", lambda v: 1 if float(v) > 120 else 0),
    "chest_pain_type": ("ChestPainType", lambda v: v),
    "resting_ecg": ("RestingECG", lambda v: v),
    "exercise_angina": ("ExerciseAngina", lambda v: "Y" if int(v) else "N"),
}

def map_to_disease_schema(extracted: Dict[str, Any], disease: str) -> Dict[str, Any]:
    """Map generic NLP-extracted fields to disease-specific schema field names."""
    # Heart only. The BRFSS diabetes map and its age-band feature went with that
    # module (gate B5); the NHANES module has no notes map.
    mapping = {"heart_disease": _HEART_DISEASE_MAP}.get(disease, {})
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

    missing = [
        m for m in ("transformers", "torch") if importlib.util.find_spec(m) is None
    ]
    return {
        "available": not missing,
        "model": _NER_MODEL,
        "reason": None if not missing else f"not installed: {', '.join(missing)}",
    }


def parse_clinical_note(note: str, use_bert: bool = True) -> Dict[str, Any]:
    """
    Parse a free-text clinical note and extract structured features using a hybrid NLP pipeline.

    Args:
        note:      The clinical note text.
        use_bert:  Whether to attempt BioBERT NER. The extraction pipeline merges
                results from baseline Regex, spaCy pattern matching, and BioBERT.

    Returns:
        Dict of extracted feature_name → value. Only present for detected values.
        Numeric values are Python floats; categorical values are strings.
    """
    if not note or not note.strip():
        return {}

    # 1. Regex baseline (always runs)
    result = _regex_extract(note)

    # 2. spaCy Extraction (يطغى على Regex في حالة إيجاد العمر بصياغة معقدة)
    # The dependency-based reading wins over the regex baseline, which stays
    # as the fallback when spaCy is not installed.
    result.update(_spacy_extract(note))

    # 3. Merge BERT results (BERT takes precedence for overlapping keys)
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
        return {
            "script": script,
            "supported": False,
            "message": UNSUPPORTED_LANGUAGE_MESSAGE,
        }
    if script == "mixed":
        return {"script": script, "supported": False, "message": MIXED_LANGUAGE_MESSAGE}
    return {"script": script, "supported": True, "message": None}
