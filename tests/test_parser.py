"""Age extraction: the spaCy matcher must not mistake a duration for an age."""
import pytest

from backend.nlp.notes_parser import parse_clinical_note


@pytest.mark.parametrize(
    "note, age",
    [
        ("The patient's age is 50. BP: 130/85.", 50),
        ("Age: 62. Fasting glucose 110.", 62),
        ("Patient is a 45 years old male.", 45),
        ("Smoked for 20 years. Age: 45.", 45),
        ("Diabetic for 10 years, 58 years old male.", 58),
    ],
)
def test_age_extracted(note, age):
    assert parse_clinical_note(note, use_bert=False)["age"] == age


@pytest.mark.parametrize("note", ["Age unknown. BP 140/90.", "Smoked for 20 years."])
def test_no_age_invented(note):
    assert "age" not in parse_clinical_note(note, use_bert=False)
