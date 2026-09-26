"""
14 — what the report sends out, and what it is allowed to call things.

Three defects, all in the prompt or the band:

  * every report posted the first 20 raw patient values to a third-party API.
    That is how the narrative came to read "62-year-old female" and
    "cholesterol 340". Nothing in a SHAP-driven report needs them.
  * the narrative described serum TOTAL cholesterol as "LDL burden" — a
    different lipid fraction, and one this platform never measures.
  * a Positive patient could be labelled LOW risk and handed "rescreen in 12
    months": D-004 sits at 14.4% against a 10.8% decision threshold and a
    17.2% moderate cut-point.
"""

import pytest

from backend.llm import report_generator as rg

DIABETES_BANDS = {"high": 0.4202127659574468, "moderate": 0.1715526601520087}
DIABETES_THRESHOLD = 0.108184

# Deliberately memorable values — easy to spot if they leak into a prompt.
RAW_FEATURES = {
    "Age": 62, "Sex": "F", "RestingBP": 187, "Cholesterol": 341,
    "BMI": 43, "GenHlth": 5, "Income": 3, "Education": 2,
}

SHAP = [
    {"feature": "Cholesterol", "shap_value": 0.91},
    {"feature": "RestingBP", "shap_value": 0.44},
    {"feature": "MaxHR", "shap_value": -0.22},
    {"feature": "Oldpeak", "shap_value": 0.18},
    {"feature": "Age", "shap_value": 0.09},
]


def build_prompt(features=RAW_FEATURES, shap=SHAP, output_type=None,
                 probability_scale=None, decision=None, lower=None, upper=None):
    """
    The user prompt exactly as generate_report() assembles it.

    Since Gate 8.4 the how-it-decided part of the prompt is a block chosen by
    the module's output_type, so this builds it the same way generate_report
    does. The default arguments keep the threshold wording, which is what the
    privacy tests below were written against.
    """
    return rg._USER_PROMPT_TEMPLATE.format(
        disease_display="Coronary Artery Disease Risk",
        decision_block=rg._decision_block(
            0.63, "Positive", "HIGH",
            "0.3695 — at or above this the patient is classified Positive",
            output_type, probability_scale, decision, lower, upper,
        ),
        shap_summary=rg._format_shap(shap),
        glossary_summary=rg._format_glossary(shap),
    )


class TestNoRawValuesLeaveTheServer:
    def test_the_prompt_template_has_no_patient_features_slot(self):
        assert "{features_summary}" not in rg._USER_PROMPT_TEMPLATE, (
            "the prompt still has a slot for raw patient values"
        )

    @pytest.mark.parametrize("value", ["62", "187", "341", "43"])
    def test_no_measured_value_appears_in_the_prompt(self, value):
        prompt = build_prompt()
        # The probability and threshold are numbers too, so look only at the
        # lines that would have carried patient data.
        assert value not in prompt, (
            f"the measured value {value} reached the outbound prompt"
        )

    def test_the_prompt_still_names_the_factors(self):
        """Stripping values must not strip the explanation."""
        prompt = build_prompt()
        for name in ("Cholesterol", "RestingBP", "MaxHR"):
            assert name in prompt
        assert "SHAP=" in prompt

    def test_the_system_prompt_forbids_inventing_a_value_or_a_demographic(self):
        assert "never" in rg._SYSTEM_PROMPT.lower()
        lowered = rg._SYSTEM_PROMPT.lower()
        assert "not given the patient's measured values" in lowered

    def test_features_are_still_accepted_so_existing_clients_keep_working(self):
        """The parameter stays; it is simply not forwarded."""
        import inspect
        assert "features" in inspect.signature(rg.generate_report).parameters


class TestFeatureGlossary:
    def test_cholesterol_is_described_as_total_not_ldl(self):
        meaning = rg._FEATURE_GLOSSARY["Cholesterol"]
        assert "TOTAL" in meaning
        assert "not LDL" in meaning

    def test_the_glossary_covers_the_factors_being_discussed(self):
        text = rg._format_glossary(SHAP)
        for name in ("Cholesterol", "RestingBP", "MaxHR", "Oldpeak", "Age"):
            assert name in text

    def test_an_unknown_feature_is_not_given_an_invented_meaning(self):
        text = rg._format_glossary([{"feature": "SES_Composite", "shap_value": 0.5}])
        assert "model-internal" in text

    def test_fasting_blood_sugar_is_described_as_a_flag_not_a_value(self):
        assert "flag" in rg._FEATURE_GLOSSARY["FastingBS"]


class TestLabMislabelGuard:
    @pytest.mark.parametrize("text", [
        "The estimate is driven largely by LDL burden.",
        "Elevated HDL is protective here.",
        "Consider HbA1c trends.",
        "Triglycerides contribute to the estimate.",
    ])
    def test_a_report_naming_an_unmeasured_analyte_is_rejected(self, text):
        violations = rg.forbidden_content(text)
        assert violations, f"{text!r} passed the guard"
        assert any("does not measure" in v for v in violations)

    def test_total_cholesterol_wording_is_accepted(self):
        text = "Total cholesterol is the largest upward contributor to this estimate."
        assert rg.forbidden_content(text) == []


class TestBandNeverContradictsTheLabel:
    def test_a_flagged_patient_is_not_low_risk(self):
        """D-004: 14.4% Positive, moderate cut-point 17.2%."""
        band = rg.band_for_report(0.1438, DIABETES_BANDS, DIABETES_THRESHOLD)
        assert band != "LOW", (
            "a patient above the decision threshold was reported as LOW risk, "
            "which is the Positive/LOW contradiction (14)"
        )
        assert band == "MODERATE"

    def test_a_patient_below_the_threshold_is_still_low(self):
        band = rg.band_for_report(0.03, DIABETES_BANDS, DIABETES_THRESHOLD)
        assert band == "LOW"

    def test_high_is_untouched(self):
        assert rg.band_for_report(0.63, DIABETES_BANDS, DIABETES_THRESHOLD) == "HIGH"

    def test_exactly_at_the_threshold_counts_as_flagged(self):
        band = rg.band_for_report(DIABETES_THRESHOLD, DIABETES_BANDS, DIABETES_THRESHOLD)
        assert band != "LOW"

    def test_without_a_threshold_the_bands_are_unchanged(self):
        assert rg.band_for_report(0.1438, DIABETES_BANDS, None) == "LOW"

    def test_the_rule_based_report_does_not_tell_a_flagged_patient_to_wait_a_year(self):
        report = rg._rule_based_report(
            "Diabetes Risk Assessment", 0.1438, "Positive", SHAP, RAW_FEATURES,
            DIABETES_BANDS, DIABETES_THRESHOLD,
        )
        assert "Rescreen in 12 months" not in report
        assert "follow-up within 4 weeks" in report

    def test_the_rule_based_report_does_not_echo_raw_values_either(self):
        report = rg._rule_based_report(
            "Diabetes Risk Assessment", 0.1438, "Positive", SHAP, RAW_FEATURES,
            DIABETES_BANDS, DIABETES_THRESHOLD,
        )
        for value in ("187", "341", "62"):
            assert value not in report


@pytest.mark.asyncio
class TestEveryFallbackPathUsesTheFlooredBand:
    """
    generate_report() has three rule-based fallbacks (no key, guard
    violation, API error). One of them was left passing no decision
    threshold, so a report that fell back after a guard rejection came out
    reporting "MODERATE" in its metadata and "low priority ... rescreen in 12
    months" in its text — the exact contradiction this was meant to remove,
    reintroduced on the path most likely to run.
    """

    async def _report(self, monkeypatch, **overrides):
        kwargs = dict(
            disease_display="Diabetes Risk Assessment",
            probability_corrected=0.1438,
            label="Positive",
            shap_values=SHAP,
            features=RAW_FEATURES,
            risk_bands=DIABETES_BANDS,
            decision_threshold=DIABETES_THRESHOLD,
        )
        kwargs.update(overrides)
        return await rg.generate_report(**kwargs)

    async def test_missing_api_key_path(self, monkeypatch):
        monkeypatch.setattr(rg, "_get_api_key", lambda: "")
        result = await self._report(monkeypatch)
        assert result["source"] == "rule_based"
        assert result["risk_band"] == "MODERATE"
        assert "Rescreen in 12 months" not in result["report"]
        assert "low priority" not in result["report"]

    async def test_guard_violation_path(self, monkeypatch):
        """The path that actually fired against the live API."""
        monkeypatch.setattr(rg, "_get_api_key", lambda: "test-key")

        class _Msg:
            content = "The diagnosis of diabetes is supported by HbA1c trends."

        class _Choice:
            message = _Msg()

        class _Completions:
            async def create(self, **_kwargs):
                return type("R", (), {"choices": [_Choice()]})()

        class _Chat:
            completions = _Completions()

        class _FakeClient:
            def __init__(self, **_kwargs):
                self.chat = _Chat()

        import openai
        monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)

        result = await self._report(monkeypatch)
        assert result["source"] == "rule_based"
        assert "fallback_reason" in result
        assert result["risk_band"] == "MODERATE"
        assert "Rescreen in 12 months" not in result["report"], (
            "the guard-violation fallback dropped the decision threshold and "
            "reported a flagged patient as low priority"
        )

    async def test_api_error_path(self, monkeypatch):
        monkeypatch.setattr(rg, "_get_api_key", lambda: "test-key")

        class _Boom:
            def __init__(self, **_kwargs):
                raise RuntimeError("network down")

        import openai
        monkeypatch.setattr(openai, "AsyncOpenAI", _Boom)

        result = await self._report(monkeypatch)
        assert result["source"] == "rule_based"
        assert result["risk_band"] == "MODERATE"
        assert "Rescreen in 12 months" not in result["report"]


class TestConformalModuleReportSaysWhatTheModelActuallyDid:
    """
    Gate 8.4. Until this gate, a heart report was generated from a prompt built
    for a threshold model on a prevalence-corrected probability, and every
    premise in it was false for the shipped model: it was told the probability
    was "calibrated to real-world prevalence" (it is calibrated to the training
    hospitals' mix), that the label said which side of a decision threshold the
    patient fell on (there is no threshold), and it was handed a
    HIGH/MODERATE/LOW band that heart configures none of. The rule-based
    fallback -- which is what runs with no API key, and so what a demo shows --
    told the patient their result was "<n>% probability on this population's
    prevalence, classified against the module's own risk bands".

    This is the text a reviewer reads, so it is pinned.
    """

    CONFORMAL = dict(
        output_type="conformal_decision",
        probability_scale="ivap_calibrated_training_mix",
    )
    SHAP = [{"feature": "cp_anginal", "shap_value": 0.31},
            {"feature": "Age", "shap_value": 0.12}]

    def _report(self, decision="uncertain", label="Uncertain — refer for further evaluation"):
        return rg._rule_based_report(
            "Heart Disease", 0.4763, label, self.SHAP, None, None, None,
            self.CONFORMAL["output_type"], self.CONFORMAL["probability_scale"],
            decision, 0.4643, 0.4872,
        )

    @pytest.mark.parametrize("decision", ["referral", "no_referral", "uncertain"])
    def test_no_band_language_reaches_the_report(self, decision):
        text = self._report(decision=decision).lower()
        for claim in ("high priority", "moderate priority", "low priority",
                      "risk band", "risk stratification note"):
            assert claim not in text, f"{decision}: report still says {claim!r}"

    @pytest.mark.parametrize("decision", ["referral", "no_referral", "uncertain"])
    def test_the_report_does_not_claim_a_population_prevalence(self, decision):
        text = self._report(decision=decision).lower()
        assert "this population's prevalence" not in text
        assert "real-world prevalence" not in text
        # It must say what the number IS calibrated to, not merely omit the lie.
        assert "four teaching hospitals" in text

    def test_the_interval_is_in_the_report(self):
        assert "46.4% to 48.7%" in self._report()

    def test_an_uncertain_result_is_not_described_as_middling_risk(self):
        text = self._report(decision="uncertain")
        assert "did not separate the two groups" in text
        assert "moderate" not in text.lower()

    def test_actions_follow_the_decision_not_a_band(self):
        assert "Routine follow-up" in self._report(
            decision="no_referral", label="Negative")
        assert "Specialist referral" in self._report(decision="referral", label="Positive")

    def test_the_prompt_for_a_conformal_module_states_no_threshold(self):
        prompt = build_prompt(
            output_type="conformal_decision",
            probability_scale="ivap_calibrated_training_mix",
            decision="uncertain", lower=0.4643, upper=0.4872,
        )
        assert "does NOT compare a probability against a cut-off" in prompt
        assert "Risk Band:" not in prompt
        assert "Decision Threshold:" not in prompt
        assert "calibrated to real-world prevalence" not in prompt

    def test_a_threshold_module_keeps_its_wording(self):
        """Diabetes must be untouched: its prompt still names both."""
        prompt = build_prompt()
        assert "Decision Threshold:" in prompt
        assert "Risk Band: HIGH" in prompt
