"""
OmniDiag — Ensemble Model Loader (Stacking / Voting)
======================================================
Wraps multiple base models + a meta-learner for ensemble inference.

Key Design:
    - NEW class (does NOT modify existing ModelLoader)
    - Each base model is loaded as a standalone sklearn-compatible estimator
    - Feature engineering + preprocessing is applied ONCE, shared across models
    - Supports both stacking (meta-learner) and voting (equal-weight average)
    - SHAP explanations are weighted averages of per-model SHAP values

Usage:
    loader = EnsembleModelLoader(config)
    result = loader.predict(patient_data)
    explanation = loader.explain(patient_data)
"""

import os
import json
import logging
import traceback
import importlib
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from backend.shap_service import generate_shap_explanation
from backend.counterfactual_generator import CounterfactualGenerator

log = logging.getLogger("omnidiag.ensemble_loader")

# Project root for resolving relative paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EnsembleModelLoader:
    """
    Lazy-loading ensemble inference wrapper.

    Supports two modes:
      - 'stacking': base model probabilities → Logistic Regression meta-learner
      - 'voting':   average of base model probabilities (equal weight)

    Attributes:
        config: Full disease configuration dict with 'ensemble' section.
        _base_models: Dict of model_name → loaded model object.
        _meta_learner: Loaded Logistic Regression (stacking) or None (voting).
        _feature_engineer: Cached feature engineer instance.
        _preprocessors: Cached dict of preprocessor objects.
        _feature_names: Cached list of feature names expected by the models.
        _ensemble_type: 'stacking' or 'voting'.
    """

    def __init__(self, config: dict):
        """
        Initialize the ensemble loader with a disease config.

        Args:
            config: Parsed YAML config dict. Must contain a 'model.ensemble' key
                    with 'base_models' list and optionally 'meta_learner'.
        """
        self.config = config
        self.ensemble_config = config.get("model", {}).get("ensemble", {})
        self._ensemble_type = self.ensemble_config.get("type", "stacking")

        # Clinical inference threshold — read from config, default 0.5
        # Updated by training script after clinical threshold optimisation.
        # FN cost (2×) vs FP cost (1×) shifts threshold below 0.5 to catch
        # more true diabetics at the cost of increased false alarms — a
        # clinically justified trade-off (HbA1c follow-up resolves FPs).
        self._inference_threshold: float = float(
            config.get("model", {}).get("inference_threshold", 0.5)
        )

        # Cached objects
        self._base_models: Optional[Dict[str, object]] = None
        self._meta_learner: Optional[object] = None
        self._feature_engineer: Optional[object] = None
        self._preprocessors: Optional[Dict[str, object]] = None
        self._feature_names: Optional[List[str]] = None
        self._meta_feature_names: Optional[List[str]] = None

        log.info(
            f"EnsembleModelLoader initialized: type={self._ensemble_type}, "
            f"base_models={[m['name'] for m in self.ensemble_config.get('base_models', [])]}, "
            f"inference_threshold={self._inference_threshold}"
        )

    # ------------------------------------------------------------------
    # Properties with lazy loading
    # ------------------------------------------------------------------

    @property
    def base_models(self) -> Dict[str, object]:
        """Lazy-load all base models from their configured paths."""
        if self._base_models is None:
            self._base_models = {}
            base_configs = self.ensemble_config.get("base_models", [])

            for bcfg in base_configs:
                name = bcfg["name"]
                weights_path = self._resolve_path(bcfg["weights_path"])

                log.debug(f"Loading base model '{name}' from: {weights_path}")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"Base model '{name}' weights not found at: {weights_path}"
                    )

                model = joblib.load(weights_path)
                self._base_models[name] = model
                log.info(f"Loaded base model '{name}': {type(model).__name__}")

        return self._base_models

    @property
    def meta_learner(self) -> Optional[object]:
        """Lazy-load the meta-learner (only for stacking)."""
        if self._ensemble_type != "stacking":
            return None

        if self._meta_learner is None:
            meta_config = self.ensemble_config.get("meta_learner", {})
            weights_path = self._resolve_path(meta_config.get("weights_path", ""))

            log.debug(f"Loading meta-learner from: {weights_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Meta-learner weights not found at: {weights_path}"
                )

            self._meta_learner = joblib.load(weights_path)
            log.info(
                f"Loaded meta-learner: {type(self._meta_learner).__name__}"
            )

        return self._meta_learner

    @property
    def preprocessors(self) -> Dict[str, object]:
        """Lazy-load preprocessors (standard_scaler, label_encoders, feature_names)."""
        if self._preprocessors is None:
            preprocessors_path = self.config.get("model", {}).get(
                "preprocessors_path", ""
            )
            if preprocessors_path and not os.path.isabs(preprocessors_path):
                preprocessors_path = os.path.join(
                    _PROJECT_ROOT, preprocessors_path
                )

            self._preprocessors = {}
            if preprocessors_path and os.path.isdir(preprocessors_path):
                for filename in os.listdir(preprocessors_path):
                    if filename.endswith(".pkl"):
                        filepath = os.path.join(preprocessors_path, filename)
                        key = filename.replace(".pkl", "")
                        self._preprocessors[key] = joblib.load(filepath)

                # Load feature names from JSON
                fn_path = os.path.join(preprocessors_path, "feature_names.json")
                if os.path.exists(fn_path):
                    with open(fn_path) as f:
                        self._feature_names = json.load(f)

                # Load meta-learner feature names
                mfn_path = os.path.join(
                    preprocessors_path, "meta_feature_names.json"
                )
                if os.path.exists(mfn_path):
                    with open(mfn_path) as f:
                        self._meta_feature_names = json.load(f)

        return self._preprocessors

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _get_feature_engineer(self):
        """Lazy-load the feature engineer from config."""
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
                        f"Could not load feature engineer '{class_name}' "
                        f"from '{module_path}': {e}"
                    )
        return self._feature_engineer

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply heuristic + medical feature engineering (NOT clinical)."""
        engineer = self._get_feature_engineer()
        if engineer:
            df = engineer.engineer_heuristic(df)
            df = engineer.engineer_medical(df)
        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _apply_preprocessors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoders + standard scaler.
        Same logic as ModelLoader._apply_preprocessors().
        """
        df = df.copy()
        preprocessors = self.preprocessors

        # Label encoders
        label_encoders = preprocessors.get("label_encoders", {})
        if isinstance(label_encoders, dict):
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))

        # Standard scaler
        scaler = preprocessors.get("standard_scaler")
        if scaler is not None:
            numeric_cols = self.config.get("features", {}).get(
                "numerical_columns", []
            )
            numeric_cols_present = [c for c in numeric_cols if c in df.columns]
            if numeric_cols_present:
                df[numeric_cols_present] = scaler.transform(
                    df[numeric_cols_present]
                )

        return df

    # ------------------------------------------------------------------
    # Public API: predict
    # ------------------------------------------------------------------

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ensemble prediction on a single patient's data.

        For stacking: each base model predicts → meta-learner combines them.
        For voting:   each base model predicts → average probabilities.

        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
                - model_contributions: Dict of model_name → probability
                - ensemble_variance: float (std of base model probabilities)
                - model_agreement: str ("high", "moderate", "low")
        """
        # ── Feature engineering + preprocessing (shared) ──────────────
        df = pd.DataFrame([patient_data])
        df = self._engineer_features(df)
        df = self._apply_preprocessors(df)

        # ── Get probabilities from each base model ────────────────────
        probas = {}
        for name, model in self.base_models.items():
            try:
                proba = float(model.predict_proba(df)[0][1])
                probas[name] = proba
            except Exception as e:
                log.warning(
                    f"Base model '{name}' prediction failed: {e}. Using 0.5"
                )
                probas[name] = 0.5

        # ── Combine predictions ───────────────────────────────────────
        if self._ensemble_type == "stacking" and self.meta_learner is not None:
            # Stacking: meta-learner combines base model probabilities
            X_meta = np.array(
                [probas[name] for name in self.base_models.keys()]
            ).reshape(1, -1)
            final_proba = float(
                self.meta_learner.predict_proba(X_meta)[0][1]
            )
        else:
            # Voting: simple average
            final_proba = float(np.mean(list(probas.values())))

        final_pred = int(final_proba >= self._inference_threshold)

        # ── Ensemble variance (P1) ──────────────────────────────────────
        proba_values = list(probas.values())
        ensemble_variance = float(np.std(proba_values)) if len(proba_values) > 0 else 0.0
        # model_agreement: high (<0.05), moderate (0.05-0.15), low (>0.15)
        if ensemble_variance < 0.05:
            model_agreement = "high"
        elif ensemble_variance < 0.15:
            model_agreement = "moderate"
        else:
            model_agreement = "low"

        return {
            "prediction": final_pred,
            "confidence": final_proba,
            "diagnosis": "Positive" if final_pred == 1 else "Negative",
            "model_contributions": probas,
            "ensemble_type": self._ensemble_type,
            "inference_threshold": self._inference_threshold,
            "ensemble_variance": round(ensemble_variance, 4),
            "model_agreement": model_agreement,
        }

    # ------------------------------------------------------------------
    # Public API: generate_counterfactuals (P2: DiCE-inspired)
    # ------------------------------------------------------------------

    def generate_counterfactuals(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate diverse counterfactual explanations for a patient.
        
        Uses the DiCE-inspired CounterfactualGenerator with random sampling
        and diversity selection to produce 3 actionable "what-if" scenarios
        that flip Positive → Negative predictions.
        
        Args:
            patient_data: Raw patient features dict (e.g., 21 BRFSS fields).
        
        Returns:
            Dictionary with:
                - counterfactuals: List of counterfactual scenario dicts
                - status: "generated" or "not_applicable" (if already low risk)
                - baseline_probability: Current probability of positive class
        """
        # Get baseline probability for metadata
        baseline_pred = self.predict(patient_data)
        baseline_proba = baseline_pred.get("confidence", 0.5)
        
        # If patient is already low risk (< threshold), no counterfactuals needed
        if baseline_proba < self._inference_threshold:
            return {
                "counterfactuals": [],
                "status": "not_applicable",
                "baseline_probability": baseline_proba,
                "message": "Patient is already below the clinical threshold. "
                          "Counterfactuals focus on reducing high-risk predictions.",
            }
        
        # Build the pipeline function that raw → engineered → preprocessed → predict
        def pipeline_fn(df: pd.DataFrame) -> pd.DataFrame:
            """Apply feature engineering + preprocessing to raw DataFrame."""
            df = self._engineer_features(df)
            df = self._apply_preprocessors(df)
            return df
        
        # Build the predict function for the generator (wraps predict)
        def predict_fn(df: pd.DataFrame) -> Dict[str, Any]:
            """Get the ensemble's predict_proba output for a DataFrame."""
            # df is already engineered+preprocessed, so we skip that in predict
            # We need a method that works on preprocessed data directly
            probas = {}
            for name, model in self.base_models.items():
                try:
                    proba = float(model.predict_proba(df)[0][1])
                    probas[name] = proba
                except Exception:
                    probas[name] = 0.5
            
            if self._ensemble_type == "stacking" and self.meta_learner is not None:
                X_meta = np.array(
                    [probas[name] for name in self.base_models.keys()]
                ).reshape(1, -1)
                final_proba = float(
                    self.meta_learner.predict_proba(X_meta)[0][1]
                )
            else:
                final_proba = float(np.mean(list(probas.values())))
            
            return {"confidence": final_proba}
        
        # Get feature names from preprocessed data
        sample_df = pipeline_fn(pd.DataFrame([patient_data]))
        feature_names = list(sample_df.columns)
        
        # Get raw feature names (all non-engineered features)
        raw_feature_names = [
            c for c in list(patient_data.keys())
            if c not in (
                "BMI_Age_Interaction", "Health_Index", "Lifestyle_Score",
                "SES_Composite", "Diabetes_Clinical_Risk"
            )
        ]
        
        # Create and run the counterfactual generator
        generator = CounterfactualGenerator(
            predict_fn=predict_fn,
            pipeline_fn=pipeline_fn,
            feature_names=feature_names,
            raw_feature_names=raw_feature_names,
            n_samples=500,
            n_counterfactuals=3,
            random_state=42,
            inference_threshold=self._inference_threshold,
        )
        
        counterfactuals = generator.generate(
            patient_data=patient_data,
            desired_class=0,  # Flip Positive → Negative
        )
        
        return {
            "counterfactuals": counterfactuals,
            "status": "generated" if counterfactuals else "no_valid_counterfactuals",
            "baseline_probability": baseline_proba,
            "message": None if counterfactuals else (
                "Could not find valid counterfactuals for this patient. "
                "The patient may need more significant lifestyle or medical changes "
                "than typical recommendations can provide."
            ),
        }
    
    # ------------------------------------------------------------------
    # Public API: explain
    # ------------------------------------------------------------------

    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a weighted-average SHAP explanation across all base models.

        For stacking: weights = meta-learner coefficients (absolute value).
        For voting:   weights = 1 / n_models (equal).

        If a base model's TreeExplainer fails (e.g. sklearn version incompatibility),
        that model's weight is redistributed to the working models.

        Returns:
            Dictionary with keys:
                - chart_data: List of {"feature": str, "shap_value": float}
                - text_explanation: Human-readable top-3 feature impacts
                - base_value: Weighted average base value
                - per_model_shap: Dict of model_name → individual SHAP result
                - shap_weights: Dict of model_name → weight used in average
                - ensemble_variance: float (std of base model probabilities)
                - model_agreement: str ("high", "moderate", "low")
        """
        try:
            df = pd.DataFrame([patient_data])
            df = self._engineer_features(df)
            df = self._apply_preprocessors(df)
            feature_names = list(df.columns)

            # ── Ensemble variance from base model predictions (P1) ─────
            probas = {}
            for name, model in self.base_models.items():
                try:
                    proba = float(model.predict_proba(df)[0][1])
                    probas[name] = proba
                except Exception:
                    probas[name] = 0.5
            proba_values = list(probas.values())
            ensemble_variance = float(np.std(proba_values)) if len(proba_values) > 0 else 0.0
            if ensemble_variance < 0.05:
                model_agreement = "high"
            elif ensemble_variance < 0.15:
                model_agreement = "moderate"
            else:
                model_agreement = "low"

            # ── Compute per-model SHAP values (P0: with robust fallback) ─
            # Cast any object dtype columns to category for SHAP TreeExplainer compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
                    log.debug(f"Cast column '{col}' from object to category for SHAP compatibility")

            # Cast to float64 for SHAP compatibility with sklearn 1.8.0
            df_float = df.astype(np.float64)
            per_model_shap = {}
            failed_models = set()

            # Build a mapping of model name → weights_path for fresh reload
            shap_model_paths = {}
            for bcfg in self.ensemble_config.get("base_models", []):
                shap_model_paths[bcfg["name"]] = self._resolve_path(
                    bcfg["weights_path"]
                )

            for name in self.base_models:
                try:
                    # Freshly reload model for SHAP (avoids cached-object issues
                    # with sklearn 1.8.0 + SHAP TreeExplainer)
                    fresh_model = joblib.load(shap_model_paths[name])

                    # XGBoost 3.x compatibility: base_score may be stored as a
                    # bracket-wrapped string (e.g. '[5.000088E-1]') in UBJSON.
                    # Safely parse using ast.literal_eval before float conversion.
                    import ast as _ast
                    try:
                        booster = fresh_model.get_booster()
                        cfg = json.loads(booster.save_config())
                        raw_bs = cfg["learner"]["learner_model_param"]["base_score"]
                        if isinstance(raw_bs, str):
                            stripped = raw_bs.strip()
                            if stripped.startswith("[") and stripped.endswith("]"):
                                parsed = _ast.literal_eval(stripped)
                                if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                                    clean_bs = float(parsed[0])
                                    import types as _types
                                    # Patch save_raw() to return clean UBJSON without brackets
                                    _original_save_raw = booster.save_raw
                                    def _patched_save_raw(self, raw_format="ubj"):
                                        raw = _original_save_raw(raw_format=raw_format)
                                        raw_bytes = bytes(raw)
                                        marker = b"base_score"
                                        idx = raw_bytes.find(marker)
                                        if idx >= 0:
                                            s_pos = idx + len(marker)
                                            if raw_bytes[s_pos] == 0x53 and raw_bytes[s_pos + 1] == 0x4C:
                                                len_bytes = raw_bytes[s_pos + 2 : s_pos + 2 + 8]
                                                old_len = int.from_bytes(len_bytes, "big")
                                                content_start = s_pos + 2 + 8
                                                old_content = raw_bytes[content_start : content_start + old_len]
                                                if old_content.startswith(b"[") and old_content.endswith(b"]"):
                                                    new_content = old_content[1:-1]
                                                    new_len = len(new_content)
                                                    patched = bytearray(raw_bytes)
                                                    patched[s_pos + 2 : s_pos + 2 + 8] = new_len.to_bytes(8, "big")
                                                    patched = patched[:content_start] + new_content + patched[content_start + old_len:]
                                                    return bytearray(patched)
                                        return raw
                                    booster.save_raw = _types.MethodType(_patched_save_raw, booster)
                                    log.info(
                                        "Patched XGBoost base_score via save_raw() monkey-patch "
                                        "for ensemble model '%s': '%s' -> %s",
                                        name, raw_bs, clean_bs,
                                    )
                    except Exception:
                        log.debug(
                            "base_score save_raw() patch skipped for ensemble model '%s' "
                            "(not XGBoost or already clean)",
                            name,
                        )

                    explainer = shap.TreeExplainer(fresh_model)
                    sv = explainer(df_float)
                    # For binary classifiers, sv.values shape is (1, n_features, 2)
                    # Take class 1 (positive) values from the last axis
                    vals = sv.values
                    if vals.ndim == 3:
                        vals = vals[:, :, -1]
                    # base_values shapes differ by model:
                    #   XGBoost/LightGBM: (1,)        — single class 1 value
                    #   Random Forest:    (1, 2)       — [class0, class1] both outputs
                    # np.ravel flattens any shape to 1D; [-1] takes the positive class
                    bv_raw = sv.base_values if hasattr(sv, "base_values") else 0.0
                    bv_scalar = float(np.ravel(bv_raw)[-1])
                    per_model_shap[name] = {
                        "values": vals.flatten().tolist(),
                        "base_value": bv_scalar,
                    }
                    nz = sum(1 for v in vals.flatten() if abs(v) > 1e-10)
                    log.debug(
                        f"SHAP for '{name}' succeeded: "
                        f"non-zero={nz}/{len(feature_names)}"
                    )
                except Exception as e:
                    log.warning(
                        f"SHAP explanation for '{name}' failed: "
                        f"{type(e).__name__}: {e}. Skipping."
                    )
                    log.warning(traceback.format_exc())
                    per_model_shap[name] = {
                        "values": [0.0] * len(feature_names),
                        "base_value": 0.0,
                    }
                    failed_models.add(name)

            # ── Determine weights ─────────────────────────────────────
            if (
                self._ensemble_type == "stacking"
                and self.meta_learner is not None
            ):
                # Use absolute meta-learner coefficients as weights
                raw_weights = np.abs(self.meta_learner.coef_[0])
                # Ensure same order as base_models
                model_names = list(self.base_models.keys())
                if len(raw_weights) == len(model_names):
                    shap_weights = {
                        name: float(w)
                        for name, w in zip(model_names, raw_weights)
                    }
                else:
                    # Fallback: equal weights
                    n = len(self.base_models)
                    shap_weights = {
                        name: 1.0 / n for name in self.base_models
                    }
            else:
                # Voting: equal weights
                n = len(self.base_models)
                shap_weights = {name: 1.0 / n for name in self.base_models}

            # ── Zero-out weights for failed models and re-normalize ───
            for failed in failed_models:
                shap_weights[failed] = 0.0

            # Normalize weights to sum to 1
            total_weight = sum(shap_weights.values())
            if total_weight > 0:
                shap_weights = {
                    k: v / total_weight for k, v in shap_weights.items()
                }
            else:
                # Catastrophic: all models failed — equal fallback
                n = len(self.base_models)
                shap_weights = {name: 1.0 / n for name in self.base_models}

            # ── Weighted average of SHAP values ───────────────────────
            n_features = len(feature_names)
            weighted_shap_values = np.zeros(n_features)
            weighted_base_value = 0.0

            for name, weight in shap_weights.items():
                sv = per_model_shap.get(name, {})
                values = np.array(sv.get("values", [0.0] * n_features))
                # Ensure length matches
                if len(values) != n_features:
                    values = np.array(
                        values[:n_features]
                        if len(values) > n_features
                        else list(values)
                        + [0.0] * (n_features - len(values))
                    )
                weighted_shap_values += weight * values
                weighted_base_value += weight * sv.get("base_value", 0.0)

            # ── Build structured explanation ──────────────────────────
            # Create a SHAP-like object for generate_shap_explanation
            class ShapValuesContainer:
                """Mimics shap.Explanation for use with generate_shap_explanation."""

                def __init__(self, values, base_values, feature_names):
                    self.values = np.array([values])
                    self.base_values = np.array([base_values])
                    self.data = None
                    self.feature_names = feature_names

            shap_container = ShapValuesContainer(
                weighted_shap_values, weighted_base_value, feature_names
            )

            result = generate_shap_explanation(shap_container, feature_names)

            # Attach prediction + confidence so /explain mirrors /predict output
            pred_result = self.predict(patient_data)
            result["prediction"] = pred_result["prediction"]
            result["confidence"] = pred_result["confidence"]
            result["diagnosis"] = pred_result["diagnosis"]

            # Add ensemble-specific metadata
            result["per_model_shap"] = per_model_shap
            result["shap_weights"] = shap_weights
            result["ensemble_variance"] = round(ensemble_variance, 4)
            result["model_agreement"] = model_agreement

            log.debug(
                f"Ensemble SHAP generated: {len(result.get('chart_data', []))} features, "
                f"weights={shap_weights}, "
                f"ensemble_variance={ensemble_variance:.4f}"
            )
            return result

        except Exception as e:
            log.error(
                f"Ensemble explain failed: {type(e).__name__}: {e}"
            )
            log.error(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """
        Clear cached base models / meta-learner so the next property access
        reloads them from disk. Mirrors ModelLoader.invalidate() — called
        after retraining to hot-swap without a process restart.
        """
        self._base_models = None
        self._meta_learner = None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path against the project root."""
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        return os.path.join(_PROJECT_ROOT, path)
