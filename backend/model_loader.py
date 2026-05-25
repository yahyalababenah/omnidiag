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
import importlib
from typing import Optional, Dict, Any, List

# Project root: resolve relative paths from the config
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arabic feature name translations for clinical summaries
_ARABIC_FEATURE_NAMES = {
    "Age": "العمر",
    "Sex": "الجنس",
    "ChestPainType": "نوع ألم الصدر",
    "RestingBP": "ضغط الدم أثناء الراحة",
    "Cholesterol": "الكوليسترول",
    "FastingBS": "سكر الدم الصائم",
    "RestingECG": "تخطيط القلب أثناء الراحة",
    "MaxHR": "الحد الأقصى لمعدل ضربات القلب",
    "ExerciseAngina": "الذبحة الصدرية الناتجة عن الجهد",
    "Oldpeak": "انخفاض ST",
    "ST_Slope": "انحدار ST",
    "Age_BP_Interaction": "تفاعل العمر مع ضغط الدم",
    "HR_Age_Ratio": "نسبة معدل ضربات القلب إلى العمر",
    "Chol_Age_Ratio": "نسبة الكوليسترول إلى العمر",
    "Clinical_Risk_Score": "درجة الخطر السريري",
}


def _get_arabic_feature_name(name: str) -> str:
    """Translate a feature name to Arabic for clinical summaries."""
    return _ARABIC_FEATURE_NAMES.get(name, name)


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
        
        Applies preprocessing (label encoding + scaling), then feature
        engineering, then model inference.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - prediction: int (0 = Negative, 1 = Positive)
                - confidence: float (probability of positive class)
                - diagnosis: str ("Positive" or "Negative")
        """
        df = pd.DataFrame([patient_data])
        df = self._apply_preprocessors(df)
        df = self._engineer_features(df)
        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0][1])
        return {
            "prediction": pred,
            "confidence": proba,
            "diagnosis": "Positive" if pred == 1 else "Negative"
        }
    
    def _generate_clinical_summary(
        self,
        patient_data: Dict[str, Any],
        feature_names: List[str],
        shap_values_list: List[float],
    ) -> Dict[str, str]:
        """
        Generate a bilingual (English/Arabic) clinical summary from SHAP values.
        
        Identifies the top risk factors (features with highest positive SHAP)
        and protective factors (features with lowest negative SHAP), then
        produces a human-readable note for doctors.
        
        Args:
            patient_data: Raw patient input dictionary (original values).
            feature_names: List of feature names in order.
            shap_values_list: List of SHAP values for each feature.
        
        Returns:
            Dictionary with 'en' and 'ar' keys containing clinical notes.
        """
        # Build list of (feature_name, shap_value, raw_value) tuples
        feature_impact = []
        for name, shap_val in zip(feature_names, shap_values_list):
            raw_val = patient_data.get(name, "N/A")
            feature_impact.append((name, shap_val, raw_val))
        
        # Sort by absolute SHAP value (descending) to find top drivers
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Separate risk factors (positive SHAP) and protective factors (negative SHAP)
        risk_factors = [(n, s, v) for n, s, v in feature_impact if s > 0]
        protective_factors = [(n, s, v) for n, s, v in feature_impact if s < 0]
        
        # Top 2-3 risk factors
        top_risks = risk_factors[:3]
        # Top 1-2 protective factors
        top_protective = protective_factors[:2]
        
        # --- English summary ---
        en_parts = []
        if top_risks:
            risk_descriptions = [
                f"{name} ({val})" for name, _, val in top_risks
            ]
            if len(risk_descriptions) == 1:
                en_parts.append(
                    f"The primary factor increasing risk is {risk_descriptions[0]}."
                )
            else:
                en_parts.append(
                    "The primary factors increasing risk are "
                    + ", ".join(risk_descriptions[:-1])
                    + f" and {risk_descriptions[-1]}."
                )
        if top_protective:
            prot_descriptions = [
                f"{name} ({val})" for name, _, val in top_protective
            ]
            if len(prot_descriptions) == 1:
                en_parts.append(
                    f"A protective factor reducing risk is {prot_descriptions[0]}."
                )
            else:
                en_parts.append(
                    "Protective factors reducing risk are "
                    + ", ".join(prot_descriptions[:-1])
                    + f" and {prot_descriptions[-1]}."
                )
        en_summary = " ".join(en_parts) if en_parts else "No significant factors identified."
        
        # --- Arabic summary ---
        ar_parts = []
        if top_risks:
            ar_risk_descriptions = [
                f"{_get_arabic_feature_name(name)} ({val})" for name, _, val in top_risks
            ]
            if len(ar_risk_descriptions) == 1:
                ar_parts.append(
                    f"العامل الرئيسي الذي يزيد من الخطر هو {ar_risk_descriptions[0]}."
                )
            else:
                ar_parts.append(
                    "العوامل الرئيسية التي تزيد من الخطر هي "
                    + " و".join(ar_risk_descriptions)
                    + "."
                )
        if top_protective:
            ar_prot_descriptions = [
                f"{_get_arabic_feature_name(name)} ({val})" for name, _, val in top_protective
            ]
            if len(ar_prot_descriptions) == 1:
                ar_parts.append(
                    f"العامل الوقائي الذي يقلل الخطر هو {ar_prot_descriptions[0]}."
                )
            else:
                ar_parts.append(
                    "العوامل الوقائية التي تقلل الخطر هي "
                    + " و".join(ar_prot_descriptions)
                    + "."
                )
        ar_summary = " ".join(ar_parts) if ar_parts else "لم يتم تحديد عوامل مهمة."
        
        return {"en": en_summary, "ar": ar_summary}
    
    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SHAP explanation on a single patient's data.
        
        Applies preprocessing (label encoding + scaling), then feature
        engineering, then SHAP explanation. Returns both raw SHAP values
        and a bilingual clinical summary for doctors.
        
        Args:
            patient_data: Dictionary of feature_name -> value.
        
        Returns:
            Dictionary with keys:
                - shap_values: List of SHAP values per feature.
                - base_value: Base (expected) value from the explainer.
                - feature_names: List of feature names in order.
                - clinical_summary: Dict with 'en' and 'ar' clinical notes.
        """
        df = pd.DataFrame([patient_data])
        df = self._apply_preprocessors(df)
        df = self._engineer_features(df)
        shap_values = self.explainer(df)
        
        shap_values_list = shap_values[0].values.tolist()
        feature_names = list(df.columns)
        
        clinical_summary = self._generate_clinical_summary(
            patient_data, feature_names, shap_values_list
        )
        
        return {
            "shap_values": shap_values_list,
            "base_value": float(shap_values[0].base_values),
            "feature_names": feature_names,
            "clinical_summary": clinical_summary,
        }
    
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
