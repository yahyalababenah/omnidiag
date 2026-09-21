"""
Tests — LLM Report Generator (backend/llm/report_generator.py)
Uses mocked Anthropic client so no real API calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.llm.report_generator as rg
from backend.llm.report_generator import (
    generate_report,
    _format_shap,
    _rule_based_report,
)

_SHAP_VALUES = [
    {"feature": "Age", "shap_value": 0.35},
    {"feature": "BMI", "shap_value": -0.22},
    {"feature": "HighBP", "shap_value": 0.18},
    {"feature": "Smoker", "shap_value": 0.05},
    {"feature": "PhysActivity", "shap_value": -0.01},
]

_FEATURES = {"Age": 55, "BMI": 34.0, "HighBP": 1, "Smoker": 0}


class TestRuleBasedFallback:
    async def test_fallback_used_when_no_api_key(self):
        """L-1: When ANTHROPIC_API_KEY is empty, rule-based report is returned."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Coronary Artery Disease",
                probability_corrected=0.75,
                label="Positive",
                confidence_band="CERTAIN",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
            )
        assert result["source"] == "rule_based"
        assert len(result["report"]) > 0

    async def test_fallback_report_contains_disease_name(self):
        """L-2: Rule-based report includes the disease name."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Coronary Artery Disease Risk",
                probability_corrected=0.75,
                label="Positive",
                confidence_band="CERTAIN",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
            )
        assert "Coronary Artery Disease Risk" in result["report"]

    async def test_fallback_report_contains_probability(self):
        """L-2b: Rule-based report includes the probability percentage."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.82,
                label="Positive",
                confidence_band="CERTAIN",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
            )
        # 0.82 formatted as 82.0%
        assert "82" in result["report"]

    async def test_fallback_report_contains_top_features(self):
        """L-2c: Rule-based report lists top SHAP features."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.82,
                label="Positive",
                confidence_band="CERTAIN",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
            )
        # Top feature by |SHAP| is Age
        assert "Age" in result["report"]

    async def test_low_probability_fallback_includes_low_actions(self):
        """L-2d: Low-probability rule-based report includes low-priority guidance."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.2,
                label="Negative",
                confidence_band="CONFIDENT",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
            )
        # LOW band → "Routine follow-up" or "preventive"
        report_lower = result["report"].lower()
        assert any(term in report_lower for term in ("routine", "preventive", "rescreen", "low"))


class TestMockedAPICall:
    """
    The generator calls DeepSeek through the OpenAI SDK
    (`from openai import AsyncOpenAI`, imported lazily inside generate_report),
    so these tests patch `openai.AsyncOpenAI` rather than the Anthropic SDK.
    """

    async def test_llm_path_returns_api_response(self):
        """L-3: When API key is set and client returns content, use that content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test clinical report from LLM"

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        MockAsyncOpenAI = MagicMock(return_value=mock_client_instance)

        with patch.object(rg, "_get_api_key", lambda: "fake-api-key"):
            with patch("openai.AsyncOpenAI", MockAsyncOpenAI):
                result = await generate_report(
                    disease_display="Heart Disease",
                    probability_corrected=0.75,
                    label="Positive",
                    confidence_band="CERTAIN",
                    shap_values=_SHAP_VALUES,
                    features=_FEATURES,
                )

        assert result["source"] == "llm"
        assert result["report"] == "Test clinical report from LLM"
        assert result["llm_model"] == "deepseek-chat"
        assert "latency_ms" in result

    async def test_llm_exception_falls_back_to_rule_based(self):
        """L-3b: If the LLM call raises an exception, fall back gracefully."""
        with patch.object(rg, "_get_api_key", lambda: "fake-key"):
            with patch("openai.AsyncOpenAI", side_effect=Exception("API failure")):
                result = await generate_report(
                    disease_display="Diabetes",
                    probability_corrected=0.5,
                    label="Positive",
                    confidence_band="UNCERTAIN",
                    shap_values=_SHAP_VALUES,
                    features=_FEATURES,
                )
        # Falls back to rule-based
        assert result["source"] == "rule_based"
        assert len(result["report"]) > 0


class TestFormatShap:
    async def test_format_shap_includes_feature_names(self):
        """L-4: _format_shap includes feature names in the output."""
        result = _format_shap(_SHAP_VALUES)
        assert "Age" in result
        assert "BMI" in result

    async def test_format_shap_includes_values(self):
        """L-4b: _format_shap formats numeric SHAP values."""
        result = _format_shap(_SHAP_VALUES)
        # Should show the SHAP value for Age (+0.350)
        assert "+0.35" in result or "0.350" in result

    async def test_format_shap_shows_direction(self):
        """L-4c: _format_shap includes risk direction indicators."""
        result = _format_shap(_SHAP_VALUES)
        assert "increases risk" in result or "decreases risk" in result

    async def test_format_shap_top_n_limits_output(self):
        """L-4d: _format_shap respects the top_n parameter."""
        result = _format_shap(_SHAP_VALUES, top_n=2)
        # Only top 2 features (Age and BMI by |SHAP|) should appear
        lines = [line for line in result.split("\n") if line.strip().startswith("-")]
        assert len(lines) == 2

    async def test_format_shap_empty_returns_fallback(self):
        """L-4e: _format_shap with empty input returns the fallback message."""
        result = _format_shap([])
        assert "No SHAP data" in result

    async def test_rule_based_report_direct_call(self):
        """L-4f: _rule_based_report can be called directly and returns a non-empty string."""
        result = _rule_based_report(
            disease_display="Diabetes",
            probability_corrected=0.65,
            label="Positive",
            shap_values=_SHAP_VALUES,
            features=_FEATURES,
        )
        assert isinstance(result, str)
        assert len(result) > 50
        assert "Diabetes" in result


# ── Corrected-scale bands (diabetes) ──────────────────────────────────────────
# Diabetes probabilities are on the deployment prior (23.7%, Jordan's actual
# diabetes prevalence -- was ~14%, a US/BRFSS placeholder, until 2026-09-21),
# so its bands are the corrected twins of the raw 0.70 / 0.40 cut-points.
# Literal 0.70 / 0.40 made HIGH unreachable and filed thousands of Positive
# patients under LOW.
_DIABETES_BANDS = {"high": 0.4202127659574468, "moderate": 0.1715526601520087}


class TestCorrectedScaleBands:
    async def test_diabetes_positive_is_high_on_corrected_bands(self):
        """L-5a: 0.64 corrected (raw 0.85) is HIGH, not MODERATE."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.6377,
                label="Positive",
                shap_values=_SHAP_VALUES,
                features=_FEATURES,
                risk_bands=_DIABETES_BANDS,
                decision_threshold=0.108184,
            )
        assert result["risk_band"] == "HIGH"
        assert "Urgent specialist referral" in result["report"]

    async def test_diabetes_moderate_band(self):
        """L-5b: 0.193 corrected (raw 0.435) is MODERATE, not LOW."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.1930,
                label="Positive",
                risk_bands=_DIABETES_BANDS,
            )
        assert result["risk_band"] == "MODERATE"

    async def test_caller_band_is_advisory(self):
        """L-5c: a contradicting caller-supplied band is discarded."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Diabetes",
                probability_corrected=0.6377,
                label="Positive",
                confidence_band="LOW",
                risk_bands=_DIABETES_BANDS,
            )
        assert result["risk_band"] == "HIGH"

    async def test_heart_default_bands_unchanged(self):
        """L-5d: no bands supplied -> the historical 0.70 / 0.40 cut-points."""
        with patch.object(rg, "_get_api_key", lambda: ""):
            result = await generate_report(
                disease_display="Heart", probability_corrected=0.55, label="Positive",
            )
        assert result["risk_band"] == "MODERATE"

    async def test_legacy_probability_keyword_is_gone(self):
        """L-5e: the scaleless `probability=` alias no longer exists internally."""
        with pytest.raises(TypeError):
            await generate_report(
                disease_display="Diabetes", probability=0.5, label="Positive",
            )
