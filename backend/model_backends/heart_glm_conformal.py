"""
Family "glm_ivap_conformal" — Spline-GLM + Venn-Abers + Mondrian conformal.

The model is a bundle built by `scripts/train_heart_glm.py` at image build time
(no binary is committed). Its contents and the decisions behind them are
documented in `backend/heart_glm/stack.py`.

What makes this family different from `sklearn_pipeline`:

  * The decision is a CONFORMAL SET, not a probability compared with a
    threshold. `decision` is one of referral / no_referral / uncertain, and
    there is no `inference_threshold` in the response -- a single threshold is
    exactly what produced the sex gap this model exists to close (HF-13).
  * `uncertain` counts as a referral for further evaluation. `prediction` is 1
    for it, because reading only the confident referrals as positives drops
    sensitivity to 0.60 for women and 0.43 for men (measured, Phase 7).
  * The reported probability carries its own interval [p_lower, p_upper] from
    the Venn-Abers layer, and is calibrated to the TRAINING hospitals' mix --
    not to the hospital reading it (D-32).
  * SHAP is linear and exact on the raw log-odds score, not tree-based.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.heart_glm import stack
from backend.model_backends.base import (
    BackendCapabilities,
    ModelBackend,
    ShapResult,
    register_backend,
)

log = logging.getLogger("omnidiag.heart_glm")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@register_backend("glm_ivap_conformal")
class HeartGlmConformalBackend(ModelBackend):

    capabilities = BackendCapabilities(
        supports_tree_shap=False,
        supports_vectorized_batch=True,
        supports_counterfactuals=True,
        explainer="linear",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self._bundle: Dict[str, Any] | None = None

    # ── artifacts ────────────────────────────────────────────────────────

    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            path = self.config.get("model", {}).get("weights_path", "")
            path = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Heart model bundle not found at {path}. It is built at image build "
                    f"time by scripts/train_heart_glm.py and is never committed."
                )
            with open(path, "rb") as handle:
                self._bundle = pickle.load(handle)
            card = self._bundle.get("model_card", {})
            log.info(
                "Loaded heart bundle: %s, built %s, data sha256 %s",
                card.get("model"), card.get("built"), card.get("training_csv_sha256", "")[:12],
            )
        return self._bundle

    def load(self) -> "HeartGlmConformalBackend":
        _ = self.bundle
        return self

    def invalidate(self) -> None:
        super().invalidate()
        self._bundle = None

    @property
    def feature_names(self) -> List[str]:
        """The inputs this model reads. The schema accepts four more
        (MaxHR, Oldpeak, ExerciseAngina, ST_Slope); this model is the
        pre-stress-test one (D-25) and ignores them."""
        return list(self.bundle["features_input"])

    # ── model level ──────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw GLM score — the model's own scale, before calibration."""
        encoded = stack.encode_for_inference(df[self.feature_names])
        return self.bundle["pipeline"].predict_proba(encoded)[:, 1]

    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        encoded = stack.encode_for_inference(df[self.feature_names])
        values, base = stack.shap_log_odds(self.bundle, encoded)
        return ShapResult(
            values=values, base_value=base, feature_names=list(stack.MODEL_FEATURES)
        )

    # ── service level ────────────────────────────────────────────────────

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([patient_data])[0]

    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not patients_data:
            return []
        bundle = self.bundle
        frame = pd.DataFrame(patients_data)
        encoded = stack.encode_for_inference(frame[self.feature_names])
        raw = bundle["pipeline"].predict_proba(encoded)[:, 1]
        probability, lower, upper = stack.ivap(
            bundle["cal_scores"], bundle["cal_labels"], raw
        )

        results = []
        for patient, score, p, p0, p1 in zip(patients_data, raw, probability, lower, upper):
            decision, conformal = stack.decide(
                float(score), patient.get("Sex"), bundle["conformal_cells"]
            )
            referred = decision in (stack.DECISION_REFERRAL, stack.DECISION_UNCERTAIN)
            result = {
                # 1 for referral AND for uncertain: both mean "this patient
                # goes on for further evaluation". Counting only the confident
                # referrals as positives is the silent sensitivity drop this
                # family exists to avoid.
                "prediction": 1 if referred else 0,
                "confidence": float(p),
                "diagnosis": (
                    stack.UNCERTAIN_DIAGNOSIS
                    if decision == stack.DECISION_UNCERTAIN
                    else "Positive" if referred else "Negative"
                ),
            }
            warning = self._completeness_warning(patient)
            if warning:
                result["data_completeness_warning"] = warning
            results.append(result)
        return results

    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        result = super().explain(patient_data)
        result["shap_scale"] = "log_odds_raw_score"
        encoded = stack.encode_for_inference([patient_data])
        imputed = {
            column: bool(pd.isna(encoded.iloc[0][column])) for column in stack.MODEL_FEATURES
        }
        for item in result.get("chart_data", []):
            item["imputed"] = imputed.get(item["feature"], False)
        return result

    def generate_counterfactuals(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """What-if scenarios over the levers this model actually reads.

        Same policy shape as the previous heart model, minus the levers this
        feature set does not contain. Every other input -- Age, Sex,
        ChestPainType, RestingECG -- is immutable.
        """
        from backend.counterfactual_generator import (
            NO_IMPROVEMENT_MESSAGE, all_improvements, lowest_achievable, policy_violations,
        )

        policy = {
            "RestingBP": ("decrease", 110),
            "Cholesterol": ("decrease", 150),
            "FastingBS": ("to", 0),
        }
        baseline = self.predict(patient_data)
        if baseline["prediction"] == 0:
            return {
                "status": "not_applicable",
                "counterfactuals": [],
                "baseline_probability": baseline["confidence"],
                "message": "Patient is already at low risk. No counterfactuals needed.",
            }

        levers = [
            feature
            for feature, (kind, bound) in policy.items()
            if patient_data.get(feature) is not None
            and (
                (kind == "decrease" and float(patient_data[feature]) > bound)
                or (kind == "to" and float(patient_data[feature]) != float(bound))
            )
        ]

        def changes_of(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [
                {
                    "feature": feature,
                    "original_value": patient_data.get(feature),
                    "counterfactual_value": candidate[feature],
                    "direction": "decrease",
                }
                for feature in policy
                if candidate.get(feature) != patient_data.get(feature)
            ]

        counterfactuals: List[Dict[str, Any]] = []
        best_achievable = None
        if levers:
            improved = all_improvements(patient_data, policy)
            outcome = self.predict(improved)
            changed = changes_of(improved)
            if not policy_violations(
                patient_data,
                {c["feature"]: c["counterfactual_value"] for c in changed},
                policy,
            ):
                if outcome["prediction"] == 0:
                    counterfactuals.append(
                        {
                            "scenario_id": 1,
                            "probability": outcome["confidence"],
                            "changes": changed,
                            "crosses_threshold": True,
                        }
                    )
                else:
                    found = lowest_achievable(
                        patient_data, policy,
                        lambda row: self.predict(row)["confidence"],
                        baseline["confidence"],
                    )
                    if found is not None:
                        rows, after = found
                        best_achievable = {
                            "scenario_id": 1,
                            "probability": after,
                            "changes": changes_of(rows),
                            "risk_reduction_relative_pct": round(
                                (baseline["confidence"] - after)
                                / max(baseline["confidence"], 0.001) * 100, 2
                            ),
                            "risk_reduction_absolute_pp": round(
                                (baseline["confidence"] - after) * 100, 2
                            ),
                            "crosses_threshold": False,
                        }

        return {
            "status": "success" if counterfactuals else "no_valid_counterfactuals",
            "counterfactuals": counterfactuals,
            "crosses_threshold": bool(counterfactuals),
            "best_achievable": best_achievable,
            "baseline_probability": baseline["confidence"],
            # Same three cases, and the same wording, as the previous heart
            # model: something helped / nothing helped / there was no lever to
            # pull. Only the reason for the referral changed.
            "message": None if counterfactuals else (
                "Even with every modifiable factor improved, this patient is still "
                "referred for further evaluation. The dominant factors are not "
                "modifiable. Referral is recommended."
                if best_achievable else
                NO_IMPROVEMENT_MESSAGE
                if levers else
                "No modifiable factor is available for this patient (none was "
                "supplied, or each is already at its target). This patient is still "
                "referred for further evaluation. Referral is recommended."
            ),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _completeness_warning(self, patient_data: Dict[str, Any]) -> str | None:
        """Warn when an input the model reads was left blank and imputed.

        Read from the bundle, not from a hand-maintained constant: the previous
        list was copied from a SHAP file belonging to a model that was never
        shipped (F0-1), and it omitted the one field whose blank value moved
        risk the most (HF-11).
        """
        missing = [
            feature
            for feature in self.config.get("model", {}).get("high_impact_features", [])
            if patient_data.get(feature) is None
        ]
        if not missing:
            return None
        plural = len(missing) > 1
        return (
            f"Input{'s' if plural else ''} missing and imputed from the training data: "
            f"{', '.join(missing)}. This prediction used a statistical stand-in rather than "
            f"the patient's actual value{'s' if plural else ''} — treat with extra caution."
        )
