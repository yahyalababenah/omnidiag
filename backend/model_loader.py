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
import shap
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
        _model: Cached model bundle dict (loaded on first access).
        _explainer: Cached SHAP explainer (loaded on first access).
    """

    def __init__(self, config: dict):
        """
        Initialize the loader with a disease config.
        
        Args:
            config: Parsed YAML config dict for the disease.
                    Must contain 'model' key with 'weights_path' and 'explainer_type'.
        """
        self.config = config
        self._model: Optional[Dict[str, Any]] = None
        self._explainer: Optional[object] = None
        self._feature_names: Optional[List[str]] = None
        self._project_root = _PROJECT_ROOT
    
    # ------------------------------------------------------------------
    # Properties with lazy loading
    # ------------------------------------------------------------------
    
    @property
    def model(self) -> Dict[str, Any]:
        """
        Lazy-load and cache the model bundle.

        The weights file is a dict — {"pipeline": sklearn Pipeline,
        "features": [...], "threshold": float, ...} — not a bare estimator.
        The Pipeline's ColumnTransformer does all encoding/scaling/imputation
        internally, so there is no separate preprocessors file to load.
        """
        if self._model is None:
            weights_path = self._resolve_weights_path()
            log.debug(f"Loading model weights from: {weights_path}")
            log.debug(f"File exists: {os.path.exists(weights_path)}")
            if os.path.exists(weights_path):
                log.debug(f"File size: {os.path.getsize(weights_path)} bytes")
            self._model = joblib.load(weights_path)
            log.debug(f"Model loaded successfully. Keys: {list(self._model.keys())}")

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
                clf = self._model["pipeline"].named_steps["clf"]
                booster = clf.get_booster()
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

        TreeExplainer wraps the XGBClassifier step alone — it cannot explain
        a full sklearn Pipeline. The ColumnTransformer step runs separately
        in explain() to produce the numeric matrix TreeExplainer needs.

        The model's ``base_score`` is already patched at load time (see
        ``model`` property) so ``TreeExplainer`` should never encounter the
        XGBoost 3.x bracket-wrapped string format.
        """
        if self._explainer is None:
            explainer_type = self.config.get("model", {}).get("explainer_type", "tree")
            if explainer_type == "tree":
                clf = self.model["pipeline"].named_steps["clf"]
                self._explainer = shap.TreeExplainer(clf)
            elif explainer_type == "deep":
                self._explainer = shap.DeepExplainer(self.model)
            else:
                raise ValueError(
                    f"Unknown explainer_type '{explainer_type}' for disease "
                    f"'{self.config.get('disease', {}).get('name', 'unknown')}'. "
                    f"Supported: 'tree', 'deep'."
                )
        return self._explainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on a single patient's data.

        The Pipeline (imputation, scaling, ordinal-encoding) runs internally
        via ``pipeline.predict_proba`` — the caller passes raw feature values
        and nothing is preprocessed here. The decision is the probability of
        class 1 (disease) compared against the model's own threshold, not
        sklearn's argmax(0.5).

        Args:
            patient_data: Dictionary of feature_name -> value.

        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
                - inference_threshold: float (decision cut-point, model's own scale)
        """
        bundle = self.model
        features = bundle["features"]
        threshold = float(bundle["threshold"])
        df = pd.DataFrame([patient_data])[features]
        proba = float(bundle["pipeline"].predict_proba(df)[0, 1])
        has_disease = proba >= threshold
        return {
            "prediction": 1 if has_disease else 0,
            "confidence": proba,
            "diagnosis": "Positive" if has_disease else "Negative",
            "inference_threshold": threshold,
        }

    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Vectorized prediction for multiple patients in one Pipeline call.

        Same math as calling predict() once per patient -- verified to
        return identical probabilities (float equality) -- but the
        Pipeline's ColumnTransformer (IterativeImputer + StandardScaler +
        OrdinalEncoder) and XGBClassifier each run once on the whole batch
        instead of once per single-row DataFrame. IterativeImputer is the
        dominant cost of a single predict() call regardless of whether
        that row has any missing values, so calling it 303 times on one
        row each is far slower than calling it once on 303 rows (measured
        ~300x on the real batch endpoint's traffic shape; see
        WEAKNESS_REGISTER.md HM-2). Used by /batch (backend/main.py) when
        the loader supports it; predict() is unchanged and still used by
        /predict.

        Args:
            patients_data: List of feature_name -> value dicts, one per
                patient. This call itself provides no per-patient isolation
                -- a single row that reaches here with an inf/nan/invalid
                value can fail the whole group (StandardScaler/check_array
                reject the entire matrix on one such value; confirmed by
                direct test, see WEAKNESS_REGISTER.md HM-3). The caller
                (backend/main.py::batch_predict) is expected to validate
                every patient with the disease's pydantic schema first, so
                that no value reaching this call can trigger that failure
                -- see HeartDiseaseInput.Oldpeak in backend/schemas.py,
                the field that was previously unbounded and let inf/nan
                through.

        Returns:
            List of result dicts (same shape as predict()'s return value),
            in the same order as patients_data.
        """
        if not patients_data:
            return []
        bundle = self.model
        features = bundle["features"]
        threshold = float(bundle["threshold"])
        df = pd.DataFrame(patients_data)[features]
        probas = bundle["pipeline"].predict_proba(df)[:, 1]
        results = []
        for proba in probas:
            proba = float(proba)
            has_disease = proba >= threshold
            results.append({
                "prediction": 1 if has_disease else 0,
                "confidence": proba,
                "diagnosis": "Positive" if has_disease else "Negative",
                "inference_threshold": threshold,
            })
        return results

    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SHAP explanation on a single patient's data.

        The ColumnTransformer step runs on the raw input to produce the
        numeric matrix TreeExplainer needs; feature names are read from the
        model bundle so they line up with that matrix's column order (no
        one-hot expansion — OrdinalEncoder keeps one column per feature).
        Returns structured chart data and a human-readable textual
        explanation — no images or matplotlib.

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
            bundle = self.model
            features = bundle["features"]
            threshold = float(bundle["threshold"])
            df = pd.DataFrame([patient_data])[features]
            log.debug(f"Explain: raw data columns={list(df.columns)}")

            transformed = bundle["pipeline"].named_steps["prep"].transform(df)
            log.debug(f"Explain: transformed shape={transformed.shape}")

            log.debug("Creating SHAP explainer...")
            explainer = self.explainer
            log.debug(f"SHAP explainer ready: {type(explainer).__name__}")

            log.debug("Computing SHAP values...")
            shap_values = explainer(transformed)
            log.debug(f"SHAP values computed, shape={shap_values.values.shape}")

            result = generate_shap_explanation(shap_values, features)

            proba = float(bundle["pipeline"].predict_proba(df)[0, 1])
            has_disease = proba >= threshold
            result["prediction"] = 1 if has_disease else 0
            result["confidence"] = proba
            result["diagnosis"] = "Positive" if has_disease else "Negative"

            log.debug("SHAP explanation generated successfully")
            return result
        except Exception as e:
            log.error(f"Explain failed: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            raise
    
    def invalidate(self) -> None:
        """
        Clear cached model/explainer/feature-name state so the next property
        access reloads them from disk.

        Called after retraining writes new weights to the model's file path
        (see backend/active_learning/retrain.py) to hot-swap the running
        process onto the updated model without a restart.
        """
        self._model = None
        self._explainer = None
        self._feature_names = None

    def get_feature_names(self) -> List[str]:
        """Return the feature names expected by the model.

        Read directly from the model bundle's own ``features`` list — the
        booster carries no feature names (the Pipeline fits it on a bare
        numpy array from the ColumnTransformer, not a named DataFrame).
        """
        if self._feature_names is None:
            self._feature_names = list(self.model.get("features", []))
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
