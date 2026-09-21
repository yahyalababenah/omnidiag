"""
Family "sklearn_generic" — any fitted scikit-learn estimator (typically a
Pipeline) that takes the raw feature columns and exposes predict_proba.

Not assumed to be tree-based, so /explain goes through ModelBackend's
generic SHAP explainer (shap.Explainer over a background sample): model-
agnostic, values in probability units, and much slower than TreeExplainer.

Weights file: a joblib bundle
    {"pipeline": fitted estimator, "features": [...], "background": DataFrame}
Decision threshold: model.inference_threshold in the YAML (default 0.5).
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd

from backend.model_backends.base import BackendCapabilities, ModelBackend, register_backend

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@register_backend("sklearn_generic")
class SklearnGenericBackend(ModelBackend):

    capabilities = BackendCapabilities(explainer="generic")

    def __init__(self, config: dict):
        super().__init__(config)
        self._bundle = None

    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            path = self.config["model"]["weights_path"]
            self._bundle = joblib.load(os.path.join(_PROJECT_ROOT, path))
        return self._bundle

    @property
    def feature_names(self) -> List[str]:
        return list(self.bundle["features"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.bundle["pipeline"].predict_proba(df[self.feature_names])[:, 1]

    def shap_background(self) -> pd.DataFrame:
        return self.bundle["background"]

    def invalidate(self) -> None:
        super().invalidate()
        self._bundle = None
