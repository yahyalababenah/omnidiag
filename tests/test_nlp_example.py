"""Example tests for the clinical-note parser (regex/spaCy path, no BERT)."""

from backend.nlp.notes_parser import detect_script, parse_clinical_note


def _parse(note):
    return parse_clinical_note(note, use_bert=False)


def test_extracts_vitals_and_demographics():
    r = _parse("57 year old male, BP 138/88, cholesterol 240, max heart rate 141")
    assert r["age"] == 57
    assert r["sex"] == "Male"
    assert r["bp_systolic"] == 138
    assert r["bp_diastolic"] == 88
    assert r["cholesterol"] == 240
    assert r["max_heart_rate"] == 141


def test_negation_gives_zero_flag():
    r = _parse("Patient is not diabetic, denies smoking")
    assert r["diabetes_flag"] == 0
    assert r["smoking_flag"] == 0


def test_positive_condition_gives_one_flag():
    assert _parse("Known diabetic, 60 year old woman")["diabetes_flag"] == 1


def test_empty_note_returns_empty_dict():
    assert _parse("") == {}
    assert _parse("   ") == {}


def test_arabic_note_is_detected_as_non_english():
    assert detect_script("مريض عمره 57 سنة") != "latin"
