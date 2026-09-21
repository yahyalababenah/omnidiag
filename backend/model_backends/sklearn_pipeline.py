"""
Family "sklearn_pipeline" — one bundle {pipeline, features, threshold, ...}
whose Pipeline has a "prep" ColumnTransformer and a tree "clf" step (heart).

Wraps backend.model_loader.ModelLoader without changing it: predict(),
predict_batch(), explain(), generate_counterfactuals() and invalidate() all
resolve to ModelLoader's own methods (it comes first in the MRO). This class
only adds the model-level ModelBackend methods, delegating to the same
bundle and TreeExplainer ModelLoader already uses.
"""

from typing import List

import numpy as np
import pandas as pd

from backend.model_backends.base import (
    BackendCapabilities,
    ModelBackend,
    ShapResult,
    register_backend,
)
from backend.model_loader import ModelLoader


@register_backend("sklearn_pipeline")
class SklearnPipelineBackend(ModelLoader, ModelBackend):

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_tree_shap=True,
            supports_vectorized_batch=True,          # ModelLoader.predict_batch
            supports_counterfactuals=bool(
                self.config.get("model", {}).get("counterfactuals", False)
            ),
            explainer="tree",
        )

    def load(self) -> "SklearnPipelineBackend":
        _ = self.model
        return self

    @property
    def feature_names(self) -> List[str]:
        return self.get_feature_names()

    @property
    def decision_threshold(self) -> float:
        return float(self.model["threshold"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        bundle = self.model
        return bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]

    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        bundle = self.model
        features = bundle["features"]
        transformed = bundle["pipeline"].named_steps["prep"].transform(df[features])
        sv = self.explainer(transformed)
        return ShapResult(
            values=np.asarray(sv.values[0], dtype=float),
            base_value=float(sv.base_values[0]),
            feature_names=list(features),
        )
