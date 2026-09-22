"""
Tests — LLM report guardrails
=============================
The prompt forbids medication/dose advice and diagnosis wording; any output
that still contains them is replaced by the deterministic report. The "bad"
sentences below are verbatim from DeepSeek reports generated on the live
Space on 2026-09-22 (docs/FEATURE_VERIFICATION.md §6).
"""

import asyncio
import sys
import types

import pytest

from backend.llm import report_generator as rg

BAD = [
    "Optimize lipid management: high-intensity statin ± add-on therapy in view of LDL burden",
    "Guideline-directed medical therapy: antiplatelet and anti-ischemic agents per cardiology",
    "Initiate/optimize antihypertensive therapy (target BP <130/80 mmHg).",
    "significant ST depression indicates inducible ischemia",
    "The patient is diagnosed with type 2 diabetes.",
    "Findings are consistent with coronary artery disease.",
    "Start metformin 500 mg twice daily.",
    "Offer pharmacotherapy and behavioral support.",
]
GOOD = [
    "Estimated risk 49.4% against a 10.8% decision threshold; confirmatory glycemic testing is recommended.",
    "Cholesterol 340 mg/dL is the second-largest risk driver.",
    "Refer to cardiology for further evaluation; follow up in 4 weeks.",
    "The model flags elevated risk, not a diagnosis.",
]


@pytest.mark.parametrize("text", BAD)
def test_forbidden_text_is_caught(text):
    assert rg.forbidden_content(text), text


@pytest.mark.parametrize("text", GOOD)
def test_acceptable_text_passes(text):
    assert rg.forbidden_content(text) == [], text


@pytest.mark.parametrize("p", [0.05, 0.30, 0.55, 0.95])
def test_rule_based_report_passes_its_own_check(p):
    report = rg._rule_based_report(
        "Diabetes Risk Assessment", p, "Positive",
        [{"feature": "BMI", "shap_value": 0.4}, {"feature": "Age", "shap_value": -0.2}],
        {"BMI": 33}, {"high": 0.42, "moderate": 0.17},
    )
    assert rg.forbidden_content(report) == []


def _fake_openai(monkeypatch, text):
    class _Completions:
        async def create(self, **_):
            msg = types.SimpleNamespace(content=text)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_Client))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


def _run():
    return asyncio.run(rg.generate_report(
        disease_display="Coronary Artery Disease Risk",
        probability_corrected=0.965, label="Positive",
        shap_values=[{"feature": "ChestPainType", "shap_value": 0.7}],
        features={"Age": 62}, decision_threshold=0.3695,
    ))


def test_llm_output_with_medication_falls_back(monkeypatch):
    _fake_openai(monkeypatch, "Recommended: high-intensity statin and aspirin 81 mg.")
    out = _run()
    assert out["source"] == "rule_based"
    assert "medication" in out["fallback_reason"] or "dose" in out["fallback_reason"]
    assert rg.forbidden_content(out["report"]) == []


def test_llm_output_with_diagnosis_falls_back(monkeypatch):
    _fake_openai(monkeypatch, "The patient has coronary artery disease.")
    out = _run()
    assert out["source"] == "rule_based"


def test_clean_llm_output_is_kept(monkeypatch):
    _fake_openai(monkeypatch, GOOD[2])
    out = _run()
    assert out["source"] == "llm"
    assert out["report"] == GOOD[2]
