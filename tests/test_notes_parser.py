"""
Tests — clinical notes parser (regex path)
==========================================
The 12 notes are the ones used in docs/FEATURE_VERIFICATION.md §3. Before
negation handling and the mapping fixes, 10 extracted values were WRONG
(e.g. "No stroke, no heart disease. Non-smoker." became Stroke = Heart
disease = Smoker = 1 and flipped the patient to Positive). The rule tested
here: the parser may MISS a field, but it must never extract a wrong one.

The six BRFSS diabetes notes (D-*) left with that module's field map in gate
B5; their negation cases are covered on the shared extraction below and in
tests/test_notes_language_and_traps.py.
"""

import pytest

from backend.nlp.notes_parser import (
    bert_status,
    map_to_disease_schema,
    parse_clinical_note,
)

NOTES = [
    ("H-EN-1", "heart_disease",
     "58-year-old male presents with typical angina on exertion for 3 weeks. BP 150/95. Total cholesterol 265 mg/dl. Fasting blood sugar 135 mg/dl. Resting ECG shows LVH. Treadmill test: max heart rate 128 bpm, exercise-induced angina, ST depression 2.1 mm with downsloping ST segment.",
     {"Age": 58, "Sex": "M", "ChestPainType": "TA", "RestingBP": 150, "Cholesterol": 265, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 128, "ExerciseAngina": "Y", "Oldpeak": 2.1}),
    ("H-EN-2", "heart_disease",
     "Mrs. K, 67 y/o woman, history of hypertension and type 2 diabetes, complains of atypical chest pain at rest. Blood pressure 132/84, pulse 88. LDL 160, total cholesterol 230. ECG: ST-T wave abnormality. No exercise test performed. Her husband smokes.",
     {"Age": 67, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 132, "Cholesterol": 230, "RestingECG": "ST", "MaxHR": None, "ExerciseAngina": None, "Oldpeak": None}),
    ("H-EN-3", "heart_disease",
     "41 yo male, non-smoker, referred for pre-employment screening. Denies chest pain. BP 118/76. Cholesterol 185. HR 72 at rest. Normal ECG. Exercise tolerance test: peak HR 172, no angina, no ST changes, upsloping ST segment.",
     {"Age": 41, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 118, "Cholesterol": 185, "FastingBS": None, "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N"}),
    ("H-AR-1", "heart_disease",
     "مريض ذكر عمره 58 سنة، يعاني من ألم صدري نموذجي عند المجهود منذ ثلاثة أسابيع. ضغط الدم 150/95. الكوليسترول الكلي 265 ملغ/دل. سكر الصيام 135.",
     {}),
    ("H-AR-2", "heart_disease",
     "مريضة عمرها 67 سنة، لديها HTN و DM منذ سنوات، تشتكي من ألم صدر غير نموذجي أثناء الراحة. BP 132/84، النبض 88. Cholesterol 230.",
     {"RestingBP": 132, "Cholesterol": 230, "MaxHR": None, "FastingBS": None}),
    ("H-AR-3", "heart_disease",
     "شاب عمره 41 عاماً، غير مدخن، حضر لفحص ما قبل التوظيف. ينفي وجود ألم في الصدر. ضغط الدم 118/76، الكوليسترول 185، النبض 72 أثناء الراحة.",
     {}),
]


def _mapped(disease, text):
    return map_to_disease_schema(parse_clinical_note(text, use_bert=False), disease)


@pytest.mark.parametrize("nid,disease,text,truth", NOTES, ids=[n[0] for n in NOTES])
def test_no_wrong_values(nid, disease, text, truth):
    mapped = _mapped(disease, text)
    wrong = []
    for field, value in mapped.items():
        if field not in truth:
            continue
        expected = truth[field]
        if expected is None:
            wrong.append(f"{field}={value} (not in the note)")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if float(value) != float(expected):
                wrong.append(f"{field}: {value} != {expected}")
        elif value != expected:
            wrong.append(f"{field}: {value} != {expected}")
    assert not wrong, f"{nid}: {wrong}"


@pytest.mark.parametrize("nid,disease,text,truth", [n for n in NOTES if n[0].endswith(("EN-1", "EN-2", "EN-3"))],
                         ids=[n[0] for n in NOTES if n[0].endswith(("EN-1", "EN-2", "EN-3"))])
def test_english_notes_recall(nid, disease, text, truth):
    mapped = _mapped(disease, text)
    expected = {f for f, v in truth.items() if v is not None}
    found = expected & set(mapped)
    assert len(found) >= 0.75 * len(expected), f"{nid}: only {sorted(found)} of {sorted(expected)}"


@pytest.mark.parametrize("text,field,value", [
    ("No stroke.", "stroke_flag", 0),
    ("Denies smoking.", "smoking_flag", 0),
    ("Non-smoker.", "smoking_flag", 0),
    ("Patient without diabetes.", "diabetes_flag", 0),
    ("History of stroke.", "stroke_flag", 1),
    ("No chest pain, but known hypertension.", "hypertension", 1),
    ("Negative for CAD.", "heart_disease_flag", 0),
])
def test_negation(text, field, value):
    assert parse_clinical_note(text, use_bert=False).get(field) == value


def test_bert_status_reports_why():
    status = bert_status()
    assert set(status) == {"available", "model", "reason"}
    if not status["available"]:
        assert status["reason"].startswith("not installed")
