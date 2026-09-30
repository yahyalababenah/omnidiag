"""spaCy dependency-based numeric extraction: context, not keyword templates."""
import pytest

from backend.nlp.notes_parser import _get_spacy_pipeline, parse_clinical_note

# The regex baseline reads most of these; "who is now 58" needs the dependency
# parse. Without spaCy installed that case cannot pass, so it is skipped rather
# than reported as a parser bug.
NEEDS_SPACY = "who is now 58"

CASES = [
    ("the age is 50", {"age": 50}),
    ("patient aged 58", {"age": 58}),
    ("She is sixty five with a BMI of 31", {"age": 65, "bmi": 31}),
    ("The patient's age is sixty-two", {"age": 62}),
    ("Pt 71F c/o SOB, HTN, HR 88", {"age": 71}),
    ("His total cholesterol came back at 240 and his blood pressure was 150 over 95",
     {"cholesterol": 240, "bp_systolic": 150, "bp_diastolic": 95}),
    ("The patient, who is now 58, has a BP reading of 145/92 and an LDL of 160",
     {"age": 58, "bp_systolic": 145, "bp_diastolic": 92}),
    ("Weighs 80 kg, age 40, resting HR 72", {"age": 40}),
    ("Resting heart rate 72; max heart rate achieved 148", {"max_heart_rate": 148}),
    ("He smoked for 20 years, currently 49 years old", {"age": 49}),
    ("Glucose 95 (random), fasting glucose 130", {"fasting_glucose": 130}),
]
ABSENT = [("Age 5 months", "age"), ("Her LDL is 160 and HDL 45", "cholesterol"),
          ("weight 70 kg", "age"), ("diabetic for 10 years", "age")]


@pytest.mark.parametrize("note,expected", CASES)
def test_values(note, expected):
    if NEEDS_SPACY in note and _get_spacy_pipeline() is None:
        pytest.skip("spaCy / en_core_web_sm not installed in this environment")
    got = parse_clinical_note(note, use_bert=False)
    for k, v in expected.items():
        assert got.get(k) == v, (note, k, got)


@pytest.mark.parametrize("note,field", ABSENT)
def test_no_wrong_value(note, field):
    assert field not in parse_clinical_note(note, use_bert=False)
