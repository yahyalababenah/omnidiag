"""
Tests — model-family registry (backend/model_backends/)
========================================================
Sections
  A. registry: fail-fast on unknown / missing families
  B. the two built-in families: the model-level interface agrees with the
     router-facing output it wraps (heart = sklearn_pipeline,
     diabetes = stacking_ensemble)
  C. PROOF: a third disease, `demo_logreg`, in a NON-tree family
     (sklearn_generic), registered by YAML only in a temporary configs dir
     and served through the real router and the real FastAPI app
  D. PROOF: nothing under backend/ other than the one backend class knows
     about that family or that disease

Nothing here writes to configs/ or models/; everything for demo_logreg lives
in pytest's tmp dir.
"""

import os
import shutil
import subprocess
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import BaseModel, Field

from backend.model_backends import (
    UnknownModelFamilyError,
    get_backend,
    registered_families,
)
# Captured at import time, before conftest's `app` fixture patches the class.
from backend.router import OmniDiagRouter as _RealOmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_ROOT, "configs")
_HEART_DATA = os.path.join(_ROOT, "data", "heart_disease", "processed", "uci_heart_by_site.csv")

# LogisticRegression needs numeric input; the heart CSV's categorical columns
# are strings, so the demo model uses the numeric ones only.
DEMO_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


class DemoLogregInput(BaseModel):
    """Input schema for the test-only demo_logreg disease (referenced from its YAML)."""

    Age: int = Field(..., ge=20, le=100)
    RestingBP: Optional[int] = Field(None, ge=80, le=220)
    Cholesterol: Optional[int] = Field(None, ge=100, le=600)
    FastingBS: Optional[int] = Field(None, ge=0, le=1)
    MaxHR: Optional[int] = Field(None, ge=60, le=220)
    Oldpeak: Optional[float] = Field(None, ge=-3.0, le=10.0)


DEMO_PATIENT = {"Age": 63, "RestingBP": 145, "Cholesterol": 233, "FastingBS": 1,
                "MaxHR": 108, "Oldpeak": 2.6}
DEMO_PATIENT_SPARSE = {"Age": 41, "RestingBP": None, "Cholesterol": None,
                       "FastingBS": 0, "MaxHR": 170, "Oldpeak": None}

HEART_PATIENT = {
    "Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
    "Cholesterol": 233, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 108,
    "ExerciseAngina": "Y", "Oldpeak": 2.6, "ST_Slope": "Flat",
}
DIABETES_PATIENT = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.0, "Smoker": 1,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 1,
    "Age": 10, "Education": 4, "Income": 3,
}


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

def _write_configs(directory, extra: dict):
    os.makedirs(directory, exist_ok=True)
    for name in ("heart_disease.yaml", "diabetes.yaml"):
        shutil.copy(os.path.join(_CONFIGS_DIR, name), directory)
    for filename, cfg in extra.items():
        with open(os.path.join(directory, filename), "w") as f:
            yaml.safe_dump(cfg, f)
    return str(directory)


@pytest.fixture(scope="module")
def demo_logreg_configs(tmp_path_factory):
    """Train Pipeline(imputer -> scaler -> LogisticRegression) on the heart
    training data and register it as `demo_logreg` via YAML only."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv(_HEART_DATA)
    X = data[DEMO_FEATURES].astype(float)
    y = data["HeartDisease"].astype(int)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)

    root = tmp_path_factory.mktemp("demo_logreg")
    weights = root / "demo_logreg.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "features": DEMO_FEATURES,
        "background": X.sample(n=100, random_state=0),
    }, weights)

    config = {
        "disease": {
            "name": "demo_logreg",
            "display_name": "Demo — Logistic Regression",
            "description": "Test-only disease proving the model-family registry",
            "version": "0.0.1",
        },
        "schema": {"module": __name__, "class": "DemoLogregInput"},
        "model": {
            "family": "sklearn_generic",
            "type": "logistic_regression",
            "weights_path": str(weights),
            "inference_threshold": 0.5,
        },
    }
    configs_dir = _write_configs(root / "configs", {"demo_logreg.yaml": config})
    return {"configs_dir": configs_dir, "pipeline": pipeline}


@pytest.fixture(scope="module")
def demo_router(demo_logreg_configs):
    from backend.schemas import DISEASE_SCHEMA_REGISTRY

    router = _RealOmniDiagRouter(configs_dir=demo_logreg_configs["configs_dir"])
    yield router
    DISEASE_SCHEMA_REGISTRY.pop("demo_logreg", None)


@pytest.fixture(scope="module")
def real_router():
    return _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)


@pytest.fixture(scope="module")
def demo_app(app, demo_router):
    """The shared app with the demo router swapped in and the limiter off."""
    import backend.main as main_module

    previous_router = main_module.router
    limiter = main_module.app.state.limiter
    previous_enabled = limiter.enabled
    main_module.router = demo_router
    limiter.enabled = False
    yield main_module.app
    main_module.router = previous_router
    limiter.enabled = previous_enabled


@pytest.fixture
async def demo_client(demo_app, db_tables):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=demo_app), base_url="http://test") as c:
        yield c


# ═════════════════════════════════════════════════════════════════════════════
# A. Registry
# ═════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_builtin_families_registered(self):
        assert {"sklearn_pipeline", "stacking_ensemble", "sklearn_generic"} <= set(
            registered_families()
        )

    def test_unknown_family_lists_registered_ones(self):
        with pytest.raises(UnknownModelFamilyError) as exc:
            get_backend("gradient_magic")
        for family in registered_families():
            assert family in str(exc.value)

    @pytest.mark.parametrize("family", ["gradient_magic", None], ids=["unknown", "missing"])
    def test_router_fails_fast_on_bad_family(self, tmp_path, family):
        model = {"weights_path": "nowhere.pkl"}
        if family is not None:
            model["family"] = family
        configs_dir = _write_configs(tmp_path / "configs", {
            "broken.yaml": {"disease": {"name": "broken"}, "model": model},
        })
        with pytest.raises(UnknownModelFamilyError) as exc:
            _RealOmniDiagRouter(configs_dir=configs_dir)
        assert "broken.yaml" in str(exc.value)
        assert "sklearn_pipeline" in str(exc.value)

    def test_builtin_configs_dispatch_by_family(self, real_router):
        assert real_router._get_loader("heart_disease").family == "sklearn_pipeline"
        assert real_router._get_loader("diabetes").family == "stacking_ensemble"


# ═════════════════════════════════════════════════════════════════════════════
# B. Built-in families: model-level interface == what the router returns
# ═════════════════════════════════════════════════════════════════════════════

class TestBuiltinFamiliesInterface:
    def test_heart_capabilities(self, real_router):
        caps = real_router._get_loader("heart_disease").capabilities
        assert caps.supports_tree_shap and caps.supports_vectorized_batch
        assert caps.explainer == "tree"

    def test_diabetes_capabilities(self, real_router):
        caps = real_router._get_loader("diabetes").capabilities
        assert caps.supports_tree_shap and not caps.supports_vectorized_batch
        assert caps.supports_counterfactuals and caps.explainer == "tree"

    def test_heart_predict_proba_matches_predict(self, real_router):
        backend = real_router._get_loader("heart_disease")
        proba = backend.predict_proba(pd.DataFrame([HEART_PATIENT]))
        assert float(proba[0]) == real_router.predict("heart_disease", dict(HEART_PATIENT))["confidence"]
        assert backend.feature_names == list(backend.model["features"])

    def test_heart_shap_values_match_explain(self, real_router):
        backend = real_router._get_loader("heart_disease")
        sr = backend.shap_values(pd.DataFrame([HEART_PATIENT]))
        explained = real_router.explain("heart_disease", dict(HEART_PATIENT))
        by_name = {c["feature"]: c["shap_value"] for c in explained["chart_data"]}
        assert sr.feature_names == backend.feature_names
        assert [by_name[f] for f in sr.feature_names] == sr.values.tolist()
        assert sr.base_value == explained["base_value"]

    def test_diabetes_predict_proba_is_the_raw_scale(self, real_router):
        backend = real_router._get_loader("diabetes")
        out = real_router.predict("diabetes", dict(DIABETES_PATIENT))
        proba = backend.predict_proba(pd.DataFrame([DIABETES_PATIENT]))
        assert float(proba[0]) == out["probability_raw"]
        assert len(backend.feature_names) == 21

    def test_diabetes_shap_values_match_explain(self, real_router):
        backend = real_router._get_loader("diabetes")
        sr = backend.shap_values(pd.DataFrame([DIABETES_PATIENT]))
        explained = real_router.explain("diabetes", dict(DIABETES_PATIENT))
        by_name = {c["feature"]: c["shap_value"] for c in explained["chart_data"]}
        assert sorted(sr.feature_names) == sorted(by_name)
        assert [by_name[f] for f in sr.feature_names] == sr.values.tolist()
        assert sr.base_value == explained["base_value_raw"]


# ═════════════════════════════════════════════════════════════════════════════
# C. Third disease in a non-tree family, served through the real app
# ═════════════════════════════════════════════════════════════════════════════

class TestDemoLogregThroughRealApp:
    def test_registered_with_generic_explainer(self, demo_router):
        assert set(demo_router.get_available_diseases()) == {
            "heart_disease", "diabetes", "demo_logreg",
        }
        backend = demo_router._get_loader("demo_logreg")
        assert backend.family == "sklearn_generic"
        assert backend.capabilities.explainer == "generic"
        assert not backend.capabilities.supports_tree_shap

    async def test_schema(self, demo_client):
        resp = await demo_client.get("/api/v4/demo_logreg/schema")
        assert resp.status_code == 200, resp.text
        assert set(resp.json()["properties"]) == set(DEMO_FEATURES)

    @pytest.mark.parametrize("patient", [DEMO_PATIENT, DEMO_PATIENT_SPARSE], ids=["full", "sparse"])
    async def test_predict(self, demo_client, demo_logreg_configs, patient):
        resp = await demo_client.post(
            "/api/v4/demo_logreg/predict", json=patient,
            params={"patient_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        expected = float(demo_logreg_configs["pipeline"].predict_proba(
            pd.DataFrame([patient])[DEMO_FEATURES].astype(float)
        )[0, 1])
        assert data["confidence"] == pytest.approx(expected, abs=1e-12)
        assert data["prediction"] == int(expected >= 0.5)
        assert data["diagnosis"] == ("Positive" if expected >= 0.5 else "Negative")
        assert data["inference_threshold"] == 0.5

    @pytest.mark.parametrize("patient", [DEMO_PATIENT, DEMO_PATIENT_SPARSE], ids=["full", "sparse"])
    async def test_explain_goes_through_generic_explainer(self, demo_client, patient):
        resp = await demo_client.post("/api/v4/demo_logreg/explain", json=patient)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # A value for every feature...
        assert sorted(c["feature"] for c in data["chart_data"]) == sorted(DEMO_FEATURES)
        values = [c["shap_value"] for c in data["chart_data"]]
        assert all(np.isfinite(values))
        assert any(abs(v) > 1e-6 for v in values)
        # ...and they are real Shapley values of predict_proba: the generic
        # (exact, probability-space) explainer is additive.
        assert data["base_value"] + sum(values) == pytest.approx(data["confidence"], abs=1e-6)

    async def test_counterfactuals_not_supported(self, demo_client):
        resp = await demo_client.post("/api/v4/demo_logreg/counterfactuals", json=DEMO_PATIENT)
        assert resp.status_code == 501, resp.text

    async def test_listed_by_diseases_endpoint(self, demo_client):
        resp = await demo_client.get("/api/v4/diseases")
        assert resp.status_code == 200, resp.text
        info = {d["name"]: d["info"] for d in resp.json()["diseases"]}
        assert info["demo_logreg"]["available"] is True
        assert info["demo_logreg"]["supports_counterfactuals"] is False
        assert info["heart_disease"]["supports_counterfactuals"] is True
        assert info["diabetes"]["supports_counterfactuals"] is True


# ═════════════════════════════════════════════════════════════════════════════
# D. Nothing under backend/ except the backend class knows about any of this
# ═════════════════════════════════════════════════════════════════════════════

_BACKEND_DIR = os.path.join(_ROOT, "backend")
_NEW_FAMILY_FILE = os.path.join("backend", "model_backends", "sklearn_generic.py")


def _source_files(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in (".venv", "__pycache__")]
        for name in filenames:
            if name.endswith((".py", ".yaml", ".yml")):
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    yield os.path.relpath(path, _ROOT), f.read()


class TestZeroBackendEditsForNewFamily:
    def test_disease_name_appears_nowhere_in_backend_or_configs(self):
        hits = [
            path for d in (_BACKEND_DIR, _CONFIGS_DIR)
            for path, text in _source_files(d) if "demo_logreg" in text
        ]
        assert hits == []

    def test_family_name_lives_in_exactly_one_backend_file(self):
        hits = [path for path, text in _source_files(_BACKEND_DIR) if "sklearn_generic" in text]
        assert hits == [_NEW_FAMILY_FILE]

    @pytest.mark.parametrize("filename", ["router.py", "main.py"])
    def test_router_and_main_name_no_family(self, filename):
        with open(os.path.join(_BACKEND_DIR, filename)) as f:
            text = f.read()
        assert [fam for fam in registered_families() if fam in text] == []
        for cls in ("ModelLoader(", "EnsembleModelLoader(", "SklearnGenericBackend"):
            assert cls not in text

    def test_commit_adding_the_family_touched_only_that_backend_file(self):
        try:
            added = subprocess.run(
                ["git", "log", "--diff-filter=A", "--format=%H", "--", _NEW_FAMILY_FILE],
                cwd=_ROOT, capture_output=True, text=True, check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git unavailable")
        if not added:
            pytest.skip(f"{_NEW_FAMILY_FILE} is not committed yet")
        changed = subprocess.run(
            ["git", "show", "--name-only", "--format=", added[-1]],
            cwd=_ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert [p for p in changed if p.startswith("backend/")] == [_NEW_FAMILY_FILE]
