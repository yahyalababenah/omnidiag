"""
OmniDiag — ModelBackend interface + model-family registry
==========================================================
A *model family* is one way of turning a weights file into predictions and
SHAP explanations (a single sklearn Pipeline, a stacking ensemble, ...). Each
family is one ModelBackend subclass registered under a name; a disease picks
its family with `model.family` in its YAML config, and the router
instantiates whatever class is registered under that name.

    @register_backend("my_family")
    class MyBackend(ModelBackend):
        def predict_proba(self, df): ...
        @property
        def feature_names(self): ...

Two layers:
  - model-level  (predict_proba, feature_names, shap_values): what a new
    family has to supply.
  - service-level (predict, predict_batch, explain): the dicts the router
    returns to the API. Defaults here are built on the model-level methods;
    the two built-in families override them with their original, unchanged
    implementations.

See docs/MODEL_FAMILY_REGISTRY_DESIGN.md.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd

from backend.shap_service import generate_shap_explanation

log = logging.getLogger("omnidiag.model_backends")


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do. Read by the router and main.py, never guessed."""

    supports_tree_shap: bool = False
    # True only when predict_batch() is a genuinely vectorised override. The
    # default predict_batch() loops, and /batch then keeps its per-row path
    # (a failing row only fails itself).
    supports_vectorized_batch: bool = False
    supports_counterfactuals: bool = False
    # "tree" (shap.TreeExplainer, fast, exact) | "generic" (shap.Explainer
    # over a background sample, model-agnostic, much slower).
    explainer: str = "generic"


@dataclass
class ShapResult:
    """One patient's SHAP decomposition, in feature_names order."""

    values: np.ndarray        # shape (n_features,)
    base_value: float
    feature_names: List[str]


class _ShapContainer:
    """The two attributes generate_shap_explanation() reads off a shap.Explanation."""

    def __init__(self, values, base_value):
        self.values = np.array([values])
        self.base_values = np.array([base_value])


class ModelBackend(ABC):
    """Interface every model family implements. See module docstring."""

    family: str = ""
    capabilities: BackendCapabilities = BackendCapabilities()

    # Generic-explainer knobs. Small on purpose: the generic path costs
    # O(background_rows * evaluations) model calls per explanation.
    generic_background_rows: int = 50

    def __init__(self, config: dict):
        self.config = config
        self._generic_explainer = None

    # ── model-level ──────────────────────────────────────────────────────

    def load(self) -> "ModelBackend":
        """Force every artifact into memory now (they load lazily otherwise)."""
        _ = self.feature_names
        return self

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Positive-class probability per row, on the model's OWN scale
        (before any deployment-time correction a family may apply)."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Raw input columns the model expects, in order."""

    def shap_background(self) -> pd.DataFrame:
        """Background sample for the generic explainer. Families that use it
        must override this; tree families never call it."""
        raise NotImplementedError(
            f"Model family '{self.family}' uses the generic SHAP explainer but "
            f"provides no background sample (override shap_background())."
        )

    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        """
        Generic, model-agnostic SHAP: shap.Explainer over predict_proba with
        an Independent masker on a small background sample. Values are in
        probability units, base value = mean prediction on the background.
        Tree families override this with TreeExplainer.
        """
        import shap

        names = list(self.feature_names)
        if getattr(self, "_generic_explainer", None) is None:
            background = self.shap_background()[names]
            background = background.sample(
                n=min(len(background), self.generic_background_rows), random_state=0
            )
            numeric = [c for c in names if pd.api.types.is_numeric_dtype(background[c])]

            def to_frame(x) -> pd.DataFrame:
                frame = pd.DataFrame(x, columns=names)
                for c in numeric:
                    frame[c] = pd.to_numeric(frame[c], errors="coerce")
                return frame

            self._generic_to_frame = to_frame
            self._generic_explainer = shap.Explainer(
                lambda x: self.predict_proba(to_frame(x)),
                shap.maskers.Independent(background, max_samples=len(background)),
                feature_names=names,
            )
        sv = self._generic_explainer(self._generic_to_frame(df[names].to_numpy(dtype=object)))
        return ShapResult(
            values=np.asarray(sv.values[0], dtype=float),
            base_value=float(np.ravel(sv.base_values)[0]),
            feature_names=names,
        )

    # ── service-level (router-facing) ────────────────────────────────────

    @property
    def decision_threshold(self) -> float:
        return float(self.config.get("model", {}).get("inference_threshold", 0.5))

    def _frame(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([patient_data])[list(self.feature_names)]

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        threshold = self.decision_threshold
        proba = float(self.predict_proba(self._frame(patient_data))[0])
        positive = proba >= threshold
        return {
            "prediction": 1 if positive else 0,
            "confidence": proba,
            "diagnosis": "Positive" if positive else "Negative",
            "inference_threshold": threshold,
        }

    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(p) for p in patients_data]

    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        sr = self.shap_values(self._frame(patient_data))
        result = generate_shap_explanation(
            _ShapContainer(sr.values, sr.base_value), sr.feature_names
        )
        pred = self.predict(patient_data)
        result["prediction"] = pred["prediction"]
        result["confidence"] = pred["confidence"]
        result["diagnosis"] = pred["diagnosis"]
        return result

    def invalidate(self) -> None:
        self._generic_explainer = None


# ── Registry ─────────────────────────────────────────────────────────────────

class UnknownModelFamilyError(ValueError):
    """A config names a model family no backend is registered for."""


_REGISTRY: Dict[str, Type[ModelBackend]] = {}


def register_backend(family: str) -> Callable[[Type[ModelBackend]], Type[ModelBackend]]:
    """Class decorator: make `cls` the backend for `model.family: <family>`."""

    def decorate(cls: Type[ModelBackend]) -> Type[ModelBackend]:
        if not (isinstance(cls, type) and issubclass(cls, ModelBackend)):
            raise TypeError(f"{cls!r} must subclass ModelBackend to be registered")
        existing = _REGISTRY.get(family)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model family '{family}' is already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        cls.family = family
        _REGISTRY[family] = cls
        return cls

    return decorate


def registered_families() -> List[str]:
    return sorted(_REGISTRY)


def get_backend(family: Any) -> Type[ModelBackend]:
    """The backend class for `family`; fails fast, naming what is registered."""
    if not family or family not in _REGISTRY:
        raise UnknownModelFamilyError(
            f"Unknown model family {family!r}. Set model.family in the disease "
            f"YAML to one of the registered families: {registered_families()}"
        )
    return _REGISTRY[family]
