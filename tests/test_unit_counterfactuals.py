"""
Unit Tests — CounterfactualGenerator (backend/counterfactual_generator.py)
Tests the DiCE-inspired generator without ML models.
"""

import pytest
import pandas as pd

from backend.counterfactual_generator import (
    CounterfactualGenerator,
    IMMUTABLE_FEATURES,
    CLINICAL_BOUNDS,
    BINARY_FEATURES,
    MUTABLE_FEATURES,
)


# Full BRFSS patient dict with high-risk profile
_HIGH_RISK_PATIENT = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 35.0,
    "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
    "DiffWalk": 0, "Sex": 1, "Age": 8, "Education": 3, "Income": 4,
}

_RAW_FEATURES = list(_HIGH_RISK_PATIENT.keys())


def _make_generator(first_confidence=0.9, subsequent_confidence=0.1):
    """
    Creates a generator whose predict_fn returns first_confidence on the first
    call (baseline) and subsequent_confidence on all later calls (candidates).
    """
    _first = [True]

    def predict_fn(df: pd.DataFrame):
        if _first[0]:
            _first[0] = False
            return {"confidence": first_confidence}
        return {"confidence": subsequent_confidence}

    def pipeline_fn(df: pd.DataFrame) -> pd.DataFrame:
        return df

    return CounterfactualGenerator(
        predict_fn=predict_fn,
        pipeline_fn=pipeline_fn,
        feature_names=_RAW_FEATURES,
        raw_feature_names=_RAW_FEATURES,
        n_samples=100,
        n_counterfactuals=3,
        random_state=42,
    )


class TestCounterfactualGeneratorInit:
    async def test_instantiation_succeeds(self):
        gen = _make_generator()
        assert gen is not None
        assert gen.n_samples == 100
        assert gen.n_counterfactuals == 3

    async def test_default_feature_names(self):
        gen = _make_generator()
        assert gen.feature_names == _RAW_FEATURES

    async def test_immutable_features_not_empty(self):
        assert len(IMMUTABLE_FEATURES) > 0

    async def test_clinical_bounds_include_bmi(self):
        assert "BMI" in CLINICAL_BOUNDS
        lo, hi = CLINICAL_BOUNDS["BMI"]
        assert lo == 15.0
        assert hi == 50.0


class TestCounterfactualGeneration:
    async def test_generate_returns_list(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        assert isinstance(result, list)

    async def test_generate_returns_counterfactuals(self):
        gen = _make_generator(first_confidence=0.9, subsequent_confidence=0.1)
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        # With predict_fn always returning 0.1 for candidates, should flip
        assert len(result) > 0

    async def test_generate_result_has_required_keys(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        if result:
            cf = result[0]
            for key in ("scenario", "changes", "new_probability", "risk_reduction", "feasibility"):
                assert key in cf, f"Missing key: {key}"

    async def test_generate_returns_empty_if_already_low_risk(self):
        # Baseline returns 0.1 (already low risk for desired_class=0)
        gen = _make_generator(first_confidence=0.1)
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        assert result == []

    async def test_new_probability_in_range(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            assert 0.0 <= cf["new_probability"] <= 1.0

    async def test_feasibility_valid_value(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            assert cf["feasibility"] in ("high", "medium", "low")

    async def test_risk_reduction_is_percentage_string(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            assert cf["risk_reduction"].endswith("%")


class TestImmutableFeaturesNotChanged:
    async def test_immutable_features_absent_from_changes(self):
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            for feat in IMMUTABLE_FEATURES:
                assert feat not in cf["changes"], (
                    f"Immutable feature '{feat}' appeared in counterfactual changes"
                )

    async def test_sex_not_in_changes(self):
        # Sex is the canonical immutable feature
        assert "Sex" in IMMUTABLE_FEATURES
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            assert "Sex" not in cf["changes"]

    async def test_age_not_in_changes(self):
        assert "Age" in IMMUTABLE_FEATURES
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            assert "Age" not in cf["changes"]


class TestClinicalBounds:
    async def test_bmi_within_bounds_if_changed(self):
        lo, hi = CLINICAL_BOUNDS["BMI"]
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            if "BMI" in cf["changes"]:
                bmi_val = cf["changes"]["BMI"]
                assert lo <= bmi_val <= hi, (
                    f"BMI {bmi_val} out of clinical bounds [{lo}, {hi}]"
                )

    async def test_binary_feature_values_are_zero_or_one(self):
        mutable_binary = BINARY_FEATURES & MUTABLE_FEATURES - IMMUTABLE_FEATURES
        gen = _make_generator()
        result = gen.generate(_HIGH_RISK_PATIENT, desired_class=0)
        for cf in result:
            for feat, val in cf["changes"].items():
                if feat in mutable_binary:
                    assert val in (0.0, 1.0), (
                        f"Binary feature '{feat}' has non-binary value {val}"
                    )
