"""
Family "stacking_ensemble" — base models + meta-learner (or soft voting),
with Bayes prior-shift correction (diabetes).

Wraps backend.ensemble_loader.EnsembleModelLoader without changing it:
predict(), explain(), generate_counterfactuals() and invalidate() all resolve
to EnsembleModelLoader's own methods (it comes first in the MRO), so the
prevalence correction, thresholds and risk bands are applied exactly once,
there. predict_batch() is the ModelBackend default loop, and
supports_vectorized_batch is False, so /batch keeps its per-row path.
"""

from typing import List

import numpy as np
import pandas as pd

from backend.ensemble_loader import EnsembleModelLoader
from backend.model_backends.base import (
    BackendCapabilities,
    ModelBackend,
    ShapResult,
    register_backend,
)


@register_backend("stacking_ensemble")
class StackingEnsembleBackend(EnsembleModelLoader, ModelBackend):

    capabilities = BackendCapabilities(
        supports_tree_shap=True,
        supports_vectorized_batch=False,
        supports_counterfactuals=True,
        explainer="tree",
    )

    def load(self) -> "StackingEnsembleBackend":
        _ = self.preprocessors
        _ = self.base_models
        _ = self.meta_learner
        return self

    @property
    def feature_names(self) -> List[str]:
        features = self.config.get("features", {})
        return list(features.get("binary_columns", [])) + list(
            features.get("numerical_columns", [])
        )

    @property
    def decision_threshold(self) -> float:
        return self._inference_threshold

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Raw (training-prior) scale: the model's own output. predict() is
        # where the deployment-prior correction happens, once.
        return np.array([
            self.predict(row)["probability_raw"]
            for row in df.to_dict(orient="records")
        ])

    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        row = df.to_dict(orient="records")[0]
        result = self.explain(row)
        engineered = list(
            self._engineer_features(self._apply_preprocessors(pd.DataFrame([row]))).columns
        )
        by_name = {item["feature"]: item["shap_value"] for item in result["chart_data"]}
        return ShapResult(
            values=np.array([by_name[name] for name in engineered], dtype=float),
            base_value=float(result["base_value_raw"]),
            feature_names=engineered,
        )
