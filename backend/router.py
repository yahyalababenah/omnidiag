"""
OmniDiag — Dynamic Disease Router
==================================
Scans the configs/ directory at initialization, loads all disease
configurations, and provides a unified interface for routing prediction
and explanation requests to the correct disease model.

Key Design:
    - Zero hardcoded disease references: adding a new disease = drop a YAML
      file in configs/ and implement a feature engineer in features/.
    - Lazy model loading: models are loaded on first request, not at startup.
    - Consistent API: all diseases use the same predict() and explain() interface.
    - Config-driven model families: `model.family` in the YAML names a
      ModelBackend registered in backend/model_backends/. The router never
      names a family itself. An unknown or missing family fails at startup,
      naming the registered ones.

Usage:
    router = OmniDiagRouter()
    result = router.predict("heart_disease", patient_data)
    explanation = router.explain("heart_disease", patient_data)
"""

import os
import logging
import importlib
import yaml
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from backend.model_backends import ModelBackend, UnknownModelFamilyError, get_backend

log = logging.getLogger("omnidiag.router")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OmniDiagRouter:
    """
    Dynamic router that maps disease names to their model loaders.
    
    Attributes:
        configs_dir: Path to the directory containing YAML config files.
        disease_configs: Dict mapping disease_name -> parsed config dict.
        model_loaders: Dict mapping disease_name -> ModelBackend instance.
    """
    
    def __init__(self, configs_dir: str = "configs"):
        """
        Initialize the router by scanning the configs directory.
        
        Args:
            configs_dir: Path to the directory containing YAML config files.
                         Defaults to "configs" relative to the project root.
        """
        self.configs_dir = configs_dir
        self.disease_configs: Dict[str, dict] = {}
        self.model_loaders: Dict[str, ModelBackend] = {}
        self._load_all_configs()
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_available_diseases(self) -> List[str]:
        """
        Return the list of registered disease names.
        
        Returns:
            List of disease name strings (e.g., ["heart_disease"]).
        """
        return list(self.disease_configs.keys())
    
    def get_disease_info(self, disease: str) -> Optional[dict]:
        """
        Get metadata about a specific disease.
        
        Args:
            disease: The disease name.
        
        Returns:
            Dictionary with disease metadata, or None if not found.
        """
        config = self.disease_configs.get(disease)
        if config is None:
            return None
        weights_path = os.path.join(
            _PROJECT_ROOT if hasattr(self, '_project_root') else ".",
            config.get("model", {}).get("weights_path", "")
        )
        has_model = os.path.isfile(weights_path)
        # Display bands for the risk badge, on the same scale as the
        # probabilities this disease returns. A disease that configures none
        # (heart) reports None and the UI keeps its own default — nothing
        # about that path changes.
        model_cfg = config.get("model", {})
        risk_bands = model_cfg.get("risk_bands") or None
        if risk_bands and {"prevalence_train", "prevalence_deploy"} <= set(model_cfg):
            from backend.prevalence_correction import apply_prevalence_correction

            risk_bands = {
                band: float(apply_prevalence_correction(
                    value, model_cfg["prevalence_train"], model_cfg["prevalence_deploy"]
                ))
                for band, value in risk_bands.items()
            }

        return {
            "name": config.get("disease", {}).get("name"),
            "display_name": config.get("disease", {}).get("display_name"),
            "description": config.get("disease", {}).get("description"),
            "version": config.get("disease", {}).get("version"),
            "model_type": config.get("model", {}).get("type"),
            "explainer_type": config.get("model", {}).get("explainer_type"),
            "available": has_model,
            "risk_bands": risk_bands,
            # Decision threshold on the same scale as the probabilities this
            # disease returns (already the deployment scale for diabetes;
            # configs/diabetes.yaml states it there). None for a disease that
            # configures none — heart takes sklearn's argmax — and the UI
            # falls back to its own documented constant in that case.
            "inference_threshold": model_cfg.get("inference_threshold"),
            "supports_counterfactuals": self._supports_counterfactuals(disease),
        }
    
    def predict(self, disease: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a prediction request to the correct disease model.
        
        Args:
            disease: The disease name (must match a YAML config filename).
            patient_data: Dictionary of feature_name -> value for the patient.
        
        Returns:
            Prediction result dict with 'prediction', 'confidence', 'diagnosis'.
            For ensemble models, also includes 'model_contributions'.
        
        Raises:
            HTTPException 404: If the disease is not registered.
        """
        loader = self._get_loader(disease)
        return loader.predict(patient_data)
    
    def explain(self, disease: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route an explanation request to the correct disease SHAP explainer.
        
        Args:
            disease: The disease name (must match a YAML config filename).
            patient_data: Dictionary of feature_name -> value for the patient.
        
        Returns:
            Explanation dict with 'chart_data', 'text_explanation', 'base_value'.
            For ensemble models, also includes 'per_model_shap' and 'shap_weights'.
        
        Raises:
            HTTPException 404: If the disease is not registered.
        """
        loader = self._get_loader(disease)
        return loader.explain(patient_data)
    
    def counterfactuals(self, disease: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a counterfactual generation request to the correct disease model.
        
        Generates diverse "what-if" scenarios showing what features a patient
        could change to alter their diagnosis (DiCE-inspired).
        
        Args:
            disease: The disease name (must match a YAML config filename).
            patient_data: Dictionary of feature_name -> value for the patient.
        
        Returns:
            Dictionary with 'counterfactuals' list, 'status', 'baseline_probability'.
        
        Raises:
            HTTPException 404: If the disease is not registered.
            HTTPException 400: If the loader doesn't support counterfactuals.
        """
        loader = self._get_loader(disease)
        
        # Check if loader has generate_counterfactuals method
        if not hasattr(loader, "generate_counterfactuals"):
            raise HTTPException(
                status_code=501,
                detail={
                    "error": f"Counterfactual generation is not implemented for disease '{disease}'.",
                    "code": "COUNTERFACTUALS_NOT_SUPPORTED",
                    "hint": "This feature is only available for ensemble models (e.g. diabetes). "
                            "Heart disease uses a single XGBoost model without a counterfactual generator.",
                }
            )
        
        return loader.generate_counterfactuals(patient_data)
    
    def reload_configs(self) -> int:
        """
        Reload all configs and model loaders from disk.
        Useful when a new disease config is added without restarting the server.
        
        Returns:
            Number of disease configs loaded.
        """
        self.disease_configs.clear()
        self.model_loaders.clear()
        self._load_all_configs()
        return len(self.disease_configs)
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    
    def _load_all_configs(self):
        """Scan configs/ directory and load all YAML config files."""
        if not os.path.isdir(self.configs_dir):
            log.warning("Configs directory '%s' not found. No diseases registered.", self.configs_dir)
            return

        for filename in sorted(os.listdir(self.configs_dir)):
            if filename.endswith((".yaml", ".yml")):
                config_path = os.path.join(self.configs_dir, filename)
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)

                    disease_name = config.get("disease", {}).get("name")
                    if not disease_name:
                        log.warning("Skipping %s: missing 'disease.name' field.", filename)
                        continue

                    self.disease_configs[disease_name] = config
                    self.model_loaders[disease_name] = self._create_loader(config)
                    self._register_schema(disease_name, config)
                    display = config.get("disease", {}).get("display_name", disease_name)
                    log.info("Registered disease: %s (%s)", display, disease_name)

                except yaml.YAMLError as e:
                    log.error("Error parsing %s: %s", filename, e)
                except UnknownModelFamilyError as e:
                    # Fail fast: a disease whose family has no backend would
                    # otherwise sit registered and 404 on every request.
                    raise UnknownModelFamilyError(f"{filename}: {e}") from e
                except Exception as e:
                    log.error("Error loading %s: %s", filename, e)

        if not self.disease_configs:
            log.warning("No disease configs loaded. The API will return 404 for all diseases.")
    
    def _create_loader(self, config: dict) -> ModelBackend:
        """
        Instantiate the ModelBackend registered for `model.family`.

        Args:
            config: Parsed YAML config dict.

        Returns:
            A ModelBackend instance (artifacts load lazily, on first request).

        Raises:
            UnknownModelFamilyError: model.family is missing or unregistered.
        """
        family = config.get("model", {}).get("family")
        backend_cls = get_backend(family)
        log.info("Using model family '%s' (%s)", family, backend_cls.__name__)
        return backend_cls(config)

    def _supports_counterfactuals(self, disease: str) -> bool:
        loader = self.model_loaders.get(disease)
        return bool(loader is not None and loader.capabilities.supports_counterfactuals)

    def _register_schema(self, disease_name: str, config: dict) -> None:
        """
        Register the disease's pydantic input schema when the YAML declares
        one (`schema: {module: ..., class: ...}`). Diseases that declare none
        keep whatever backend/schemas.py registers for them.
        """
        schema_cfg = config.get("schema")
        if not schema_cfg:
            return
        from backend.schemas import register_schema

        module = importlib.import_module(schema_cfg["module"])
        register_schema(disease_name, getattr(module, schema_cfg["class"]))

    def _get_loader(self, disease: str) -> ModelBackend:
        """
        Get the ModelBackend for a disease,
        raising HTTPException if not found.
        
        Args:
            disease: The disease name.
        
        Returns:
            The loader instance.
        
        Raises:
            HTTPException 404: If disease is not registered.
        """
        if disease not in self.model_loaders:
            available = self.get_available_diseases()
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Disease '{disease}' is not registered.",
                    "available_diseases": available,
                    "message": f"Available diseases: {', '.join(available) if available else 'None'}. "
                               f"Ensure a YAML config file exists in the configs/ directory."
                }
            )
        return self.model_loaders[disease]
