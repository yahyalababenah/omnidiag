"""
OmniDiag — Model Loader
=======================
Lazy-loads the model, preprocessors, and SHAP explainer for a given disease.
Supports both TreeExplainer (for XGBoost, RandomForest, etc.) and
DeepExplainer (for neural networks like TabNet, PyTorch).

The loader is instantiated per disease and caches loaded objects
so they are only loaded once (on first request).
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Optional, Dict, Any, List


class ModelLoader:
    """
    Lazy loader for a single disease's model artifacts.
    
    Attributes:
        config: The disease configuration dictionary.
        _model: Cached model object (loaded on first access).
        _explainer: Cached SHAP explainer (loaded on first access).
        _preprocessors: Cached preprocessor objects (loaded on first access).
    """
    
    def __init__(self, config: dict):
        """
        Initialize the loader with a disease config.
        
        Args:
            config: Parsed YAML config dict for the disease.
                    Must contain 'model' key with 'weights_path' and 'explainer_type'.
        """
        self.config = config
        self._model: Optional[object] = None
        self._explainer: Optional[object] = None
        self._preprocessors: Optional[Dict[str, object]] = None
        self._feature_names: Optional[List[str]] = None
    
    # ------------------------------------------------------------------
    # Properties with lazy loading
    # ------------------------------------------------------------------
    
    @property
    def model(self) -> object:
        """Lazy-load and cache the trained model."""
        if self._model is None:
            weights_path = self._resolve_weights_path()
            self._model = joblib.load(weights_path)
        return self._model
    
    @property
    def explainer(self) -> object:
        """Lazy-load and cache the SHAP explainer."""
        if self._explainer is None:
            explainer_type = self.config.get("model", {}).get("explainer_type", "tree")
            if explainer_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif explainer_type == "deep":
                self._explainer = shap.DeepExplainer(self.model)
            else:
                raise ValueError(
                    f"Unknown explainer_type '{explainer_type}' for disease "
                    f"'{self.config.get('disease', {}).get('name', 'unknown')}'. "
                    f"Supported: 'tree', 'deep'."
                )
        return self._explainer
    
    @property
    def preprocessors(self) -> Dict[str, object]:
        """Lazy-load and cache preprocessors (label encoders, scaler)."""
        if self._preprocessors is None:
            preprocessors_path = self.config.get("model", {}).get("preprocessors_path", "")
            self._preprocessors = {}
            if preprocessors_path and os.path.isdir(preprocessors_path):
                for filename in os.listdir(preprocessors_path):
                    if filename.endswith(".pkl"):
                        filepath = os.path.join(preprocessors_path, filename)
                        key = filename.replace(".pkl", "")
                        self._preprocessors[key] = joblib.load(filepath)
        return self._preprocessors
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on a single patient's data.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
        """
        df = pd.DataFrame([patient_data])
        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0][1])
        return {
            "prediction": pred,
            "confidence": proba,
            "diagnosis": "Positive" if pred == 1 else "Negative"
        }
    
    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SHAP explanation on a single patient's data.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - shap_values: List of SHAP values per feature.
                - base_value: Base (expected) value from the explainer.
                - feature_names: List of feature names in order.
        """
        df = pd.DataFrame([patient_data])
        shap_values = self.explainer(df)
        return {
            "shap_values": shap_values[0].values.tolist(),
            "base_value": float(shap_values[0].base_values),
            "feature_names": list(df.columns)
        }
    
    def get_feature_names(self) -> List[str]:
        """Return the feature names expected by the model."""
        if self._feature_names is None:
            if hasattr(self.model, 'feature_names_in_'):
                self._feature_names = list(self.model.feature_names_in_)
            else:
                # Fallback: try to infer from the booster
                try:
                    self._feature_names = self.model.get_booster().feature_names
                except Exception:
                    self._feature_names = []
        return self._feature_names
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    def _resolve_weights_path(self) -> str:
        """
        Resolve the model weights path, trying primary then fallback.
        
        Returns:
            The path to the model weights file.
        
        Raises:
            FileNotFoundError: If neither primary nor fallback path exists.
        """
        primary = self.config.get("model", {}).get("weights_path", "")
        fallback = self.config.get("model", {}).get("fallback_weights_path", "")
        
        if os.path.exists(primary):
            return primary
        elif fallback and os.path.exists(fallback):
            return fallback
        else:
            raise FileNotFoundError(
                f"Model weights not found. Tried:\n"
                f"  Primary: {primary}\n"
                f"  Fallback: {fallback}\n"
                f"Please ensure the model file exists at one of these paths."
            )
