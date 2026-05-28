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
import sys
import json
import logging
import traceback
import joblib
import pandas as pd
import numpy as np
import shap
import importlib
from typing import Optional, Dict, Any, List

from backend.shap_service import generate_shap_explanation

log = logging.getLogger("omnidiag.model_loader")

# Project root: resolve relative paths from the config
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        self._project_root = _PROJECT_ROOT
        self._feature_engineer: Optional[object] = None
    
    # ------------------------------------------------------------------
    # Properties with lazy loading
    # ------------------------------------------------------------------
    
    @property
    def model(self) -> object:
        """Lazy-load and cache the trained model."""
        if self._model is None:
            weights_path = self._resolve_weights_path()
            log.debug(f"Loading model weights from: {weights_path}")
            log.debug(f"File exists: {os.path.exists(weights_path)}")
            if os.path.exists(weights_path):
                log.debug(f"File size: {os.path.getsize(weights_path)} bytes")
            self._model = joblib.load(weights_path)
            log.debug(f"Model loaded successfully. Type: {type(self._model).__name__}")
        return self._model
    
    @property
    def explainer(self) -> object:
        """Lazy-load and cache the SHAP explainer.

        Includes a fallback for XGBoost 3.x compatibility where ``base_score``
        is stored as a bracket-wrapped string (e.g. ``'[5.85041E-1]'``) that
        older SHAP versions cannot parse. If the initial ``TreeExplainer``
        creation fails with a ``ValueError`` from the base_score, we patch the
        model's internal parameter before retrying.
        """
        if self._explainer is None:
            explainer_type = self.config.get("model", {}).get("explainer_type", "tree")
            if explainer_type == "tree":
                try:
                    self._explainer = shap.TreeExplainer(self.model)
                except ValueError as e:
                    # XGBoost 3.x stores base_score as a bracket-wrapped string
                    # e.g. '[5.85041E-1]'. Older SHAP can't parse this with
                    # float(), so we patch the booster's config and retry.
                    log.warning(
                        "TreeExplainer creation failed — attempting base_score "
                        "bracket patch: %s", e
                    )
                    try:
                        booster = self.model.get_booster()
                        cfg = json.loads(booster.save_config())
                        raw = cfg["learner"]["learner_model_param"]["base_score"]
                        # Strip surrounding brackets from e.g. '[0.585]'
                        raw_clean = raw.strip("[]")
                        cfg["learner"]["learner_model_param"]["base_score"] = raw_clean
                        booster.load_config(json.dumps(cfg))
                        self._explainer = shap.TreeExplainer(self.model)
                    except Exception as patch_e:
                        # If patching also fails, re-raise the original error
                        raise e from patch_e
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
            # Resolve relative to project root
            if preprocessors_path and not os.path.isabs(preprocessors_path):
                preprocessors_path = os.path.join(self._project_root, preprocessors_path)
            self._preprocessors = {}
            if preprocessors_path and os.path.isdir(preprocessors_path):
                for filename in os.listdir(preprocessors_path):
                    if filename.endswith(".pkl"):
                        filepath = os.path.join(preprocessors_path, filename)
                        key = filename.replace(".pkl", "")
                        self._preprocessors[key] = joblib.load(filepath)
        return self._preprocessors
    
    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    
    def _get_feature_engineer(self):
        """
        Lazy-load the feature engineer class specified in the config.
        
        Returns:
            An instance of the feature engineer (subclass of BaseFeatureEngineer).
        """
        if self._feature_engineer is None:
            module_path = self.config.get("features", {}).get("module", "")
            class_name = self.config.get("features", {}).get("class", "")
            if module_path and class_name:
                try:
                    module = importlib.import_module(module_path)
                    engineer_class = getattr(module, class_name)
                    self._feature_engineer = engineer_class(self.config)
                except (ImportError, AttributeError) as e:
                    raise ImportError(
                        f"Could not load feature engineer '{class_name}' from "
                        f"'{module_path}': {e}"
                    )
        return self._feature_engineer
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to a DataFrame.
        
        Runs the heuristic feature engineering pipeline (the winning
        experiment at 88.98% accuracy). The clinical path is available
        for experimentation but is not used by the production model.
        
        Args:
            df: Raw patient DataFrame.
        
        Returns:
            DataFrame with engineered features appended.
        """
        engineer = self._get_feature_engineer()
        if engineer:
            df = engineer.engineer_heuristic(df)
        return df
    
    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    
    def _apply_preprocessors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoders to categorical columns and scaler to numeric columns.
        
        Args:
            df: Raw DataFrame with string categoricals and raw numeric values.
        
        Returns:
            DataFrame with encoded categoricals and scaled numerics.
        """
        df = df.copy()
        preprocessors = self.preprocessors
        
        # Apply label encoders to categorical columns
        label_encoders = preprocessors.get("label_encoders", {})
        if isinstance(label_encoders, dict):
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))
        
        # Apply standard scaler to numeric columns
        scaler = preprocessors.get("standard_scaler")
        if scaler is not None:
            numeric_cols = self.config.get("features", {}).get("numerical_columns", [])
            numeric_cols_present = [c for c in numeric_cols if c in df.columns]
            if numeric_cols_present:
                df[numeric_cols_present] = scaler.transform(df[numeric_cols_present])
        
        return df
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on a single patient's data.
        
        Applies feature engineering FIRST on raw data, then preprocessing
        (label encoding + scaling), then model inference.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
        """
        df = pd.DataFrame([patient_data])
        df = self._engineer_features(df)
        df = self._apply_preprocessors(df)
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
        
        Applies feature engineering FIRST on raw data, then preprocessing
        (label encoding + scaling), then SHAP explanation. Returns structured
        chart data and a human-readable textual explanation — no images or
        matplotlib.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - chart_data: List of {"feature": str, "shap_value": float}
                  sorted by |shap_value| descending, for ShapBarChart.jsx.
                - text_explanation: Human-readable string identifying the
                  top 3 most impactful features with direction labels.
                - base_value: Base (expected) value from the explainer.
        """
        try:
            df = pd.DataFrame([patient_data])
            log.debug(f"Explain: raw data columns={list(df.columns)}")
            df = self._engineer_features(df)
            log.debug(f"Explain: after engineering columns={list(df.columns)}, shape={df.shape}")
            df = self._apply_preprocessors(df)
            log.debug(f"Explain: after preprocessors columns={list(df.columns)}, shape={df.shape}")
            
            log.debug("Creating SHAP explainer...")
            explainer = self.explainer
            log.debug(f"SHAP explainer ready: {type(explainer).__name__}")
            
            log.debug("Computing SHAP values...")
            shap_values = explainer(df)
            log.debug(f"SHAP values computed, shape={shap_values.values.shape}")
            
            feature_names = list(df.columns)
            
            result = generate_shap_explanation(shap_values, feature_names)
            log.debug("SHAP explanation generated successfully")
            return result
        except Exception as e:
            log.error(f"Explain failed: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            raise
    
    def get_feature_names(self) -> List[str]:
        """Return the feature names expected by the model."""
        if self._feature_names is None:
            # Try booster feature names first
            try:
                booster = self.model.get_booster()
                if booster.feature_names and all(n != '' for n in booster.feature_names):
                    self._feature_names = list(booster.feature_names)
            except Exception:
                pass
            # Fallback: try sklearn's feature_names_in_
            if self._feature_names is None:
                try:
                    self._feature_names = list(self.model.feature_names_in_)
                except (AttributeError, Exception):
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
        
        # Resolve relative paths to absolute from project root
        primary_abs = primary if os.path.isabs(primary) else os.path.join(self._project_root, primary)
        fallback_abs = fallback if os.path.isabs(fallback) else os.path.join(self._project_root, fallback) if fallback else ""
        
        log.debug(f"Resolving weights path...")
        log.debug(f"  Project root: {self._project_root}")
        log.debug(f"  Primary (config): {primary}")
        log.debug(f"  Primary (abs): {primary_abs}")
        log.debug(f"  Fallback (config): {fallback}")
        log.debug(f"  Fallback (abs): {fallback_abs}")
        log.debug(f"  Primary exists: {os.path.exists(primary_abs)}")
        log.debug(f"  Fallback exists: {os.path.exists(fallback_abs) if fallback_abs else 'N/A'}")
        
        if os.path.exists(primary_abs):
            return primary_abs
        elif fallback_abs and os.path.exists(fallback_abs):
            return fallback_abs
        else:
            raise FileNotFoundError(
                f"Model weights not found. Tried:\n"
                f"  Primary: {primary_abs}\n"
                f"  Fallback: {fallback_abs}\n"
                f"Please ensure the model file exists at one of these paths."
            )
