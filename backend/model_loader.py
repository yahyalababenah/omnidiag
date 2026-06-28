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

            # XGBoost 3.x compatibility patch: base_score may be stored as a
            # bracket-wrapped string (e.g. '[5.85041E-1]') in the model's raw
            # UBJSON serialization (save_raw()). Older SHAP versions parse this
            # with float() which fails on the bracket-wrapped format.
            #
            # Neither booster.load_config() nor booster.set_attr() modify the
            # save_raw() output, because base_score lives in the C-level
            # learner_model_param that is serialized from internal state.
            # Therefore we monkey-patch booster.save_raw() to strip the
            # brackets from the base_score string in the UBJSON output.
            import types as _types

            try:
                booster = self._model.get_booster()
                cfg = json.loads(booster.save_config())
                raw = cfg["learner"]["learner_model_param"]["base_score"]
                if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
                    raw_clean = raw.strip("[]")

                    _original_save_raw = booster.save_raw

                    def _patched_save_raw(self, raw_format="ubj"):
                        raw = _original_save_raw(raw_format=raw_format)
                        raw_bytes = bytes(raw)
                        marker = b"base_score"
                        idx = raw_bytes.find(marker)
                        if idx >= 0:
                            s_pos = idx + len(marker)
                            if (
                                raw_bytes[s_pos] == 0x53
                                and raw_bytes[s_pos + 1] == 0x4C
                            ):
                                len_bytes = raw_bytes[
                                    s_pos + 2 : s_pos + 2 + 8
                                ]
                                old_len = int.from_bytes(len_bytes, "big")
                                content_start = s_pos + 2 + 8
                                old_content = raw_bytes[
                                    content_start : content_start + old_len
                                ]
                                if old_content.startswith(b"[") and old_content.endswith(b"]"):
                                    new_content = old_content[1:-1]
                                    new_len = len(new_content)
                                    patched = bytearray(raw_bytes)
                                    patched[
                                        s_pos + 2 : s_pos + 2 + 8
                                    ] = new_len.to_bytes(8, "big")
                                    patched = (
                                        patched[:content_start]
                                        + new_content
                                        + patched[content_start + old_len :]
                                    )
                                    return bytearray(patched)
                        return raw

                    booster.save_raw = _types.MethodType(
                        _patched_save_raw, booster
                    )
                    log.info(
                        "Patched XGBoost base_score via save_raw() monkey-patch: "
                        "%s -> %s",
                        raw,
                        raw_clean,
                    )
            except Exception:
                log.debug(
                    "base_score save_raw() patch skipped "
                    "(not an XGBoost model or already clean)"
                )

        return self._model
    
    @property
    def explainer(self) -> object:
        """Lazy-load and cache the SHAP explainer.

        The model's ``base_score`` is already patched at load time (see
        ``model`` property) so ``TreeExplainer`` should never encounter the
        XGBoost 3.x bracket-wrapped string format.
        """
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
        
        Runs two feature engineering pipelines in dependency order:
            1. Heuristic (Statistical): Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio
            2. Medical (Cardiology): RPP, Exercise_Risk_Index
        
        Note: Age_Bins and Global_Risk_Score were tested (v5.1 beta) but did NOT
        improve accuracy — removed per A/B diagnostic (diagnose_v5_drop.py).
        
        Args:
            df: Raw patient DataFrame (12 base features).
        
        Returns:
            DataFrame with engineered features appended (up to 16 total features).
        """
        engineer = self._get_feature_engineer()
        if engineer:
            df = engineer.engineer_heuristic(df)     # Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio
            df = engineer.engineer_medical(df)       # RPP, Exercise_Risk_Index
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

        Pipeline order must match the training pipeline exactly:
          1. Apply label encoders + StandardScaler (preprocessing)
          2. Engineer features FROM the already-scaled/encoded values

        This matches how final_ready_data.csv was built (encode → scale → save),
        and how the training script loaded it before calling engineer_*_features().
        Reversing the order causes ~5% accuracy loss because engineered features
        like Age_BP_Interaction are computed at completely different magnitudes.

        Args:
            patient_data: Dictionary of feature_name -> value.

        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
        """
        df = pd.DataFrame([patient_data])
        df = self._apply_preprocessors(df)   # encode → scale first
        df = self._engineer_features(df)     # then engineer from scaled values
        raw_pred = int(self.model.predict(df)[0])
        raw_proba = self.model.predict_proba(df)[0]
        # The heart disease training CSV has inverted labels:
        # HeartDisease=0 means "has disease", HeartDisease=1 means "healthy".
        # Verified via feature correlations (Oldpeak, Age, MaxHR are all sign-flipped
        # vs clinical expectations). We correct here so the API returns clinical truth.
        has_disease = (raw_pred == 0)
        confidence = float(raw_proba[0])  # P(class=0) = P(disease)
        return {
            "prediction": 1 if has_disease else 0,
            "confidence": confidence,
            "diagnosis": "Positive" if has_disease else "Negative"
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
            df = self._apply_preprocessors(df)   # encode → scale first (matches training pipeline)
            log.debug(f"Explain: after preprocessors columns={list(df.columns)}, shape={df.shape}")
            df = self._engineer_features(df)     # then engineer from scaled values
            log.debug(f"Explain: after engineering columns={list(df.columns)}, shape={df.shape}")
            
            # Cast any object dtype columns to category for SHAP TreeExplainer compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
                    log.debug(f"Cast column '{col}' from object to category for SHAP compatibility")

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
    
    def generate_counterfactuals(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate What-If counterfactual scenarios for heart disease.

        Uses random perturbation of mutable clinical features to find the
        minimal changes that flip the prediction from Positive to Negative.
        """
        import random

        baseline_result = self.predict(patient_data)
        baseline_pred = baseline_result["prediction"]
        baseline_prob = baseline_result["confidence"]

        if baseline_pred == 0:
            return {
                "status": "not_applicable",
                "counterfactuals": [],
                "baseline_probability": baseline_prob,
                "message": "Patient is already at low risk. No counterfactuals needed.",
            }

        # Heart disease mutable features and their perturbation ranges
        MUTABLE = {
            "RestingBP":   (90,  180,  False),   # (min, max, is_binary)
            "Cholesterol": (100, 400,  False),
            "MaxHR":       (60,  200,  False),
            "Oldpeak":     (0.0, 6.2,  False),
            "FastingBS":   (0,   1,    True),
            "ExerciseAngina": (None, None, True),  # Y/N toggle
        }

        rng = random.Random(42)
        candidates = []

        for _ in range(800):
            cf = dict(patient_data)
            changed: List[str] = []

            for feat, (lo, hi, is_binary) in MUTABLE.items():
                if feat not in cf:
                    continue
                if rng.random() < 0.4:
                    original = cf[feat]
                    if feat == "ExerciseAngina":
                        cf[feat] = "N" if str(original).upper() == "Y" else "Y"
                    elif is_binary:
                        cf[feat] = 1 - int(original)
                    else:
                        cf[feat] = round(rng.uniform(lo, hi), 1)
                    if cf[feat] != original:
                        changed.append(feat)

            if not changed:
                continue

            try:
                result = self.predict(cf)
                if result["prediction"] == 0:
                    candidates.append({
                        "features": cf,
                        "changed": changed,
                        "probability": result["confidence"],
                        "distance": len(changed),
                    })
            except Exception:
                continue

        # Sort by fewest changes, then by lowest probability
        candidates.sort(key=lambda c: (c["distance"], c["probability"]))

        # Pick top 3 with diversity (different primary change)
        selected = []
        seen_primary = set()
        for c in candidates:
            primary = c["changed"][0] if c["changed"] else ""
            if primary not in seen_primary:
                seen_primary.add(primary)
                selected.append(c)
            if len(selected) >= 3:
                break

        counterfactuals = []
        for c in selected:
            scenario_changes = []
            for feat in c["changed"]:
                original_val = patient_data.get(feat)
                new_val = c["features"].get(feat)
                scenario_changes.append({
                    "feature": feat,
                    "original_value": original_val,
                    "counterfactual_value": new_val,
                    "direction": "decrease" if (
                        isinstance(new_val, (int, float)) and isinstance(original_val, (int, float))
                        and new_val < original_val
                    ) else "increase",
                })
            counterfactuals.append({
                "scenario_id": len(counterfactuals) + 1,
                "probability": c["probability"],
                "changes": scenario_changes,
            })

        return {
            "status": "success",
            "counterfactuals": counterfactuals,
            "baseline_probability": baseline_prob,
        }

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
