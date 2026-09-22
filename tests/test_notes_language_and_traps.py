"""
15 — the 12-note evaluation set, and Arabic told the truth.

The parser is English regex end to end, so an Arabic note extracts nothing.
The UI reported that as "0 fields extracted — patient data updated", which
tells a clinician their note was empty rather than that the tool cannot read
their language. /parse-notes now returns a `language` block and the UI shows
it.

The note set lives in tests/fixtures/clinical_notes.json. It was
reconstructed on 2026-09-23 from the case descriptions in
docs/FEATURE_VERIFICATION.md section 3 -- the original notes.json was in a
session scratchpad that no longer exists -- so each note reproduces the
documented trap rather than the exact original wording.

The traps below are the four systematic errors the audit recorded, pinned so
they cannot come back.
"""

import json
import os

import pytest

from backend.nlp.notes_parser import (
    detect_script,
    language_support,
    map_to_disease_schema,
    parse_clinical_note,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "clinical_notes.json")

with open(FIXTURE, encoding="utf-8") as _f:
    NOTES = {n["id"]: n for n in json.load(_f)["notes"]}


def mapped(note_id):
    note = NOTES[note_id]
    return map_to_disease_schema(
        parse_clinical_note(note["text"], use_bert=False), note["disease"]
    )


class TestScriptDetection:
    @pytest.mark.parametrize("note_id", ["H-AR-1", "H-AR-3", "D-AR-1", "D-AR-3"])
    def test_pure_arabic_is_detected(self, note_id):
        assert detect_script(NOTES[note_id]["text"]) == "arabic"

    @pytest.mark.parametrize("note_id", ["H-AR-2", "D-AR-2"])
    def test_mixed_script_is_detected(self, note_id):
        assert detect_script(NOTES[note_id]["text"]) == "mixed"

    @pytest.mark.parametrize("note_id", ["H-EN-1", "H-EN-2", "H-EN-3",
                                         "D-EN-1", "D-EN-2", "D-EN-3"])
    def test_english_is_detected(self, note_id):
        assert detect_script(NOTES[note_id]["text"]) == "latin"

    def test_an_empty_note_does_not_crash(self):
        assert detect_script("") == "latin"
        assert language_support("")["supported"] is True


class TestLanguageIsReportedHonestly:
    @pytest.mark.parametrize("note_id", ["H-AR-1", "H-AR-3", "D-AR-1", "D-AR-3"])
    def test_arabic_is_reported_unsupported_and_extracts_nothing(self, note_id):
        support = language_support(NOTES[note_id]["text"])
        assert support["supported"] is False
        assert "English" in support["message"]
        assert mapped(note_id) == {}, (
            "an Arabic note produced fields — the message would then be wrong"
        )

    @pytest.mark.parametrize("note_id", ["H-AR-2", "D-AR-2"])
    def test_a_mixed_note_is_also_unsupported_even_though_it_yields_fields(self, note_id):
        """
        A partial read of a clinical note is not a supported read. The mixed
        notes do return a field or two from their English abbreviations, and
        reporting that as success is exactly the failure mode: it looks like
        the note parsed while most of it was skipped.
        """
        support = language_support(NOTES[note_id]["text"])
        assert support["supported"] is False
        assert support["script"] == "mixed"
        assert mapped(note_id), "this note should still yield its English tokens"

    @pytest.mark.parametrize("note_id", ["H-EN-1", "D-EN-2"])
    def test_english_notes_are_supported(self, note_id):
        assert language_support(NOTES[note_id]["text"])["supported"] is True


class TestTheFourSystematicErrors:
    """The traps the audit recorded, one test each."""

    def test_typical_angina_is_not_flattened_to_asymptomatic(self):
        """H-EN-1: any mention of chest pain used to yield ASY."""
        assert mapped("H-EN-1")["ChestPainType"] == "TA"

    def test_atypical_angina_is_distinguished(self):
        assert mapped("H-EN-2")["ChestPainType"] == "ATA"

    def test_denied_chest_pain_is_asymptomatic(self):
        """H-EN-3: ASY is the correct answer here, and now for the right reason."""
        assert mapped("H-EN-3")["ChestPainType"] == "ASY"

    def test_ldl_is_not_read_as_total_cholesterol(self):
        """H-EN-2 states LDL 160 and total 230; 230 is the model's input."""
        assert mapped("H-EN-2")["Cholesterol"] == 230

    @pytest.mark.parametrize("note_id", ["H-EN-2", "H-EN-3"])
    def test_a_resting_pulse_is_not_read_as_max_heart_rate(self, note_id):
        """'Pulse 88 at rest' / 'HR 72 at rest' are not MaxHR."""
        assert "MaxHR" not in mapped(note_id)

    def test_a_real_max_heart_rate_is_still_read(self):
        """The fix must not have simply stopped reading MaxHR."""
        assert mapped("H-EN-1")["MaxHR"] == 138

    def test_hypertension_does_not_become_fasting_blood_sugar(self):
        """H-EN-2 mentions hypertension; FastingBS must not be inferred from it."""
        result = mapped("H-EN-2")
        assert result.get("FastingBS") != 1 or "FastingBS" not in result

    def test_negation_yields_zero_not_one(self):
        """
        D-EN-2: "non-smoker. No history of hypertension. Denies stroke, no
        heart disease." Every one of these used to extract as 1 and flipped
        the screening result.
        """
        result = mapped("D-EN-2")
        assert result["Smoker"] == 0
        assert result["HighBP"] == 0
        assert result["Stroke"] == 0
        assert result["HeartDiseaseorAttack"] == 0

    def test_affirmative_history_still_yields_one(self):
        """The negation fix must not have zeroed everything."""
        result = mapped("D-EN-1")
        assert result["HighBP"] == 1
        assert result["Smoker"] == 1
        assert result["HeartDiseaseorAttack"] == 1


class TestBrfssAgeBuckets:
    """BRFSS codes 5-year bands from 2 (25-29); the note states a real age."""

    @pytest.mark.parametrize("note_id,age,bucket", [
        ("D-EN-1", 52, 7),   # 50-54
        ("D-EN-2", 47, 6),   # 45-49
        ("D-EN-3", 42, 5),   # 40-44
    ])
    def test_stated_age_maps_to_the_right_band(self, note_id, age, bucket):
        assert mapped(note_id)["Age"] == bucket


class TestTheWholeSetStillRuns:
    def test_every_note_parses_without_raising(self):
        for note_id in NOTES:
            mapped(note_id)

    def test_no_english_note_comes_back_empty(self):
        for note_id in ("H-EN-1", "H-EN-2", "H-EN-3", "D-EN-1", "D-EN-2", "D-EN-3"):
            assert mapped(note_id), f"{note_id} extracted nothing"
