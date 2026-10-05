"""
Gate 9.8 — what the frontend actually receives.

The form is schema-driven: `useDiseaseSchema` fetches the JSON Schema from the API,
`schemaFieldParser` picks a widget from it, and `featureCategorizer` groups the
fields. So the FORM is a consequence of the schema, and these tests pin the parts
of the schema the form depends on. A change here is a change to what a clinician
sees.
"""

import json
import re

import pytest
import yaml

from backend.schemas_diabetes_nhanes import (
    ADIPOSITY_BAND_LEVELS,
    MANDATORY_FIELDS,
    DiabetesNhanesInput,
)

CATEGORIZER = "frontend/src/utils/featureCategorizer.js"


@pytest.fixture(scope="module")
def schema():
    return DiabetesNhanesInput.model_json_schema()


@pytest.fixture(scope="module")
def config():
    with open("configs/diabetes_nhanes.yaml") as handle:
        return yaml.safe_load(handle)


def test_the_form_gets_all_twenty_fields(schema):
    assert len(schema["properties"]) == 20


def test_adiposity_renders_as_a_labelled_dropdown(schema):
    """schemaFieldParser: `type === 'string'` with an enum gives a select. As an
    integer 0-2 it fell to Rule 3 and rendered an UNLABELLED 0-2 slider, which
    asks a clinician to express a visual judgement by dragging a number."""
    field = schema["properties"]["ADIPOSITY_BAND"]
    assert field["type"] == "string"
    assert field["enum"] == ["normal", "increased", "high"]


def test_the_levels_are_ordered_by_risk(schema):
    """The dropdown order is the risk order, so the control reads the way the
    model does."""
    assert list(ADIPOSITY_BAND_LEVELS) == schema["properties"]["ADIPOSITY_BAND"]["enum"]
    assert list(ADIPOSITY_BAND_LEVELS.values()) == [0, 1, 2]


def test_levels_and_integers_score_identically(backend_or_none):
    if backend_or_none is None:
        pytest.skip("bundle not available")
    backend = backend_or_none
    patient = {
        "RIDAGEYR": 58, "RIAGENDR": 1.0, "BMXBMI": 31.2, "SBP": 138, "DBP": 84,
        "BPXPLS": 78, "MCQ300C": 1.0, "CVD_ANY": 0.0, "PAQ650": 0.0, "PAQ665": 0.0,
        "LBDHDD": 41, "LBXSCH": 205, "LBXSTR": 190, "LBXSATSI": 28, "LBXSGTSI": 34,
        "LBXSCR": 0.95, "LBXSBU": 15, "LBXSAL": 4.2, "LBXSUA": 6.4,
    }
    for level, ordinal in ADIPOSITY_BAND_LEVELS.items():
        a = backend.predict({**patient, "ADIPOSITY_BAND": level})["confidence"]
        b = backend.predict({**patient, "ADIPOSITY_BAND": ordinal})["confidence"]
        assert abs(a - b) < 1e-12


def test_raising_the_band_raises_the_risk(backend_or_none):
    if backend_or_none is None:
        pytest.skip("bundle not available")
    patient = {
        "RIDAGEYR": 58, "RIAGENDR": 1.0, "BMXBMI": 31.2, "SBP": 138, "DBP": 84,
        "BPXPLS": 78, "MCQ300C": 1.0, "CVD_ANY": 0.0, "PAQ650": 0.0, "PAQ665": 0.0,
        "LBDHDD": 41, "LBXSCH": 205, "LBXSTR": 190, "LBXSATSI": 28, "LBXSGTSI": 34,
        "LBXSCR": 0.95, "LBXSBU": 15, "LBXSAL": 4.2, "LBXSUA": 6.4,
    }
    probabilities = [
        backend_or_none.predict({**patient, "ADIPOSITY_BAND": level})["confidence"]
        for level in ("normal", "increased", "high")
    ]
    assert probabilities == sorted(probabilities)


def test_there_is_no_unknown_level(schema):
    """F9-25: blanking this field LOWERS predicted risk, so an 'unknown' option
    would be a false reassurance the clinician did not intend."""
    enum = schema["properties"]["ADIPOSITY_BAND"]["enum"]
    for forbidden in ("unknown", "not assessed", "n/a", "unsure", ""):
        assert forbidden not in enum


def test_mandatory_fields_are_required_in_the_schema(schema):
    required = set(schema.get("required", []))
    for field in MANDATORY_FIELDS:
        assert field in required, f"{field} must be required, D9-06"


def test_mandatory_list_matches_the_config(config):
    assert sorted(MANDATORY_FIELDS) == sorted(config["features"]["mandatory"])


def test_every_field_carries_a_description(schema):
    """The categorizer matches on name AND description, and the report glossary
    needs one too. A field with no description lands in 'General' unexplained."""
    for name, prop in schema["properties"].items():
        text = prop.get("description", "")
        assert len(text) > 15, f"{name} has no usable description"


# ── the categorizer, read from the actual file ───────────────────────────

def _categories():
    source = open(CATEGORIZER).read()
    block = re.search(r"const CATEGORIES = \{(.*?)\n\};", source, re.S).group(1)
    pairs = re.findall(r"([\w' &]+):\s*\{[^}]*?keywords:\s*\[(.*?)\]", block, re.S)
    return {
        name.strip().strip("'"): re.findall(r"'([^']+)'", keywords)
        for name, keywords in pairs
    }


def _categorize(name, description, categories):
    text = f"{name} {description}".lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if name.lower() == keyword or keyword in text:
                return category
    return "General"


def test_no_nhanes_field_falls_into_general(schema):
    """A field in 'General' tells the clinician nothing about what kind of input
    it is. All twenty should land somewhere meaningful."""
    categories = _categories()
    uncategorized = [
        name for name, prop in schema["properties"].items()
        if _categorize(name, prop.get("description", ""), categories) == "General"
    ]
    assert not uncategorized, f"uncategorized: {uncategorized}"


def test_the_lab_panel_groups_together(schema):
    categories = _categories()
    labs = {
        name for name, prop in schema["properties"].items()
        if _categorize(name, prop.get("description", ""), categories) == "Laboratory"
    }
    assert labs == {
        "LBDHDD", "LBXSCH", "LBXSTR", "LBXSATSI", "LBXSGTSI",
        "LBXSCR", "LBXSBU", "LBXSAL", "LBXSUA",
    }


def test_laboratory_keywords_are_nhanes_codes_only():
    """Generic words like 'cholesterol' would also match the heart module's
    `Cholesterol` field through its description and silently move it out of
    Vitals & Signs. Exact codes keep the blast radius at zero."""
    for keyword in _categories()["Laboratory"]:
        assert re.fullmatch(r"lb[dx]s?[a-z0-9]+", keyword), keyword


@pytest.mark.parametrize("name,description,expected", [
    ("Age", "age in years", "Vitals & Signs"),
    ("Sex", "recorded sex", "Demographics"),
    ("Cholesterol", "serum TOTAL cholesterol (mg/dL) — not LDL", "Vitals & Signs"),
    ("RestingBP", "resting systolic blood pressure (mm Hg)", "Vitals & Signs"),
])
def test_existing_modules_keep_their_categories(name, description, expected):
    """The heart form must lay out exactly as it did before (BRFSS retired, gate B6)."""
    assert _categorize(name, description, _categories()) == expected


@pytest.fixture(scope="module")
def backend_or_none(config):
    try:
        from backend.model_backends import get_backend

        return get_backend("ebm_platt_conformal")(config).load()
    except Exception:
        return None
