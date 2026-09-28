"""
OmniDiag — model family `ebm_platt_conformal`: the NHANES dysglycaemia module.
==============================================================================
Registered by existing as a file in this package. Nothing outside this file is
edited to add it -- that is the point of the registry (docs/ADDING_A_MODEL_FAMILY.md),
and it is why the heart module could ship without touching the diabetes one.

What makes this family different from the two tree families
-----------------------------------------------------------
  * The model is an Explainable Boosting Machine. It is additive BY CONSTRUCTION:

        logit(raw score) = intercept + sum_j f_j(x_j) + sum_jk f_jk(x_j, x_k)

    so the per-term contributions ARE the model, not a post-hoc approximation of
    it. There is no SHAP explainer here and there must never be one. The identity
    is asserted at serve time in shap_values() to < 1e-8, the same way the heart
    GLM's additivity is checked.

  * The decision is a CONFORMAL SET, not a probability compared with a threshold,
    so this module exposes no `inference_threshold` and no `risk_bands` -- the
    same shape the heart module settled on (D-32). `decision` is one of
    referral / no_referral / uncertain.

  * The conformal layer is GROUP-conditional over age band, not merely
    class-conditional (D9-05). The class-conditional layer measured 0.917 / 0.856
    pooled coverage while covering only 0.578 of dysglycaemic 20-39 year olds and
    0.574 of healthy 60+ (F9-31). Two failures in opposite directions cancelled in
    the pooled figure. Age band is used because the model legitimately holds it;
    race is never an input and never a conditioning variable.

  * The probability layer is Platt scaling, and the interval is a bootstrap of
    that fit (D9-04). It describes uncertainty in the CALIBRATION MAP. It is not
    a distribution-free predictive interval and no consumer may describe it as one.

  * Six fields are MANDATORY and may never reach the model as NaN (D9-06). The
    blank-field audit measured that leaving PAQ650 empty shifts median risk by
    +0.108 and flips 31% of decisions (F9-28). This backend refuses such a row
    rather than scoring it; the schema and the form refuse it earlier.

Every number above is measured in docs/phase9/DISCOVERY_RECORD.md and recorded in
the bundle's own model card. The bundle is built by
`train_diabetes_nhanes_ebm_v2.py` in the experiments repo and committed here,
because unlike the heart stack it carries no patient data -- only fitted shape
functions.
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.model_backends.base import (
    BackendCapabilities,
    ModelBackend,
    ShapResult,
    register_backend,
)

log = logging.getLogger("omnidiag.model_backends.diabetes_ebm")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DECISION_REFERRAL = "referral"
DECISION_NO_REFERRAL = "no_referral"
DECISION_UNCERTAIN = "uncertain"
UNCERTAIN_DIAGNOSIS = "Uncertain — order an HbA1c test"

# F9-29 / D9-03. A complete blood count raises AUC by +0.0218 against this HbA1c
# label and by -0.0040 against a glucose label: the entire gain is the red-cell /
# HbA1c assay artefact, not risk information. The ban is asserted in the training
# script and again here, so a bundle built elsewhere cannot smuggle one in.
CBC_BANNED = frozenset({
    "LBXRDW", "LBXMC", "LBXMCHSI", "LBXMCVSI", "LBXWBCSI", "LBXPLTSI", "LBXHGB",
    "LBXHCT", "LBXRBCSI", "LBDLYMNO", "LBDNENO", "LBDMONO", "LBDEONO", "LBDBANO",
    "LBXLYPCT", "LBXNEPCT", "LBXMOPCT", "LBXEOPCT", "LBXBAPCT", "LBXMPSI", "LBXSCK",
})


class MandatoryFieldMissing(ValueError):
    """A D9-06 field arrived null. Raised instead of scoring the row.

    Carries the field list so the API can name them rather than return a generic
    422. Scoring the row anyway is the failure mode this exception exists to
    prevent: a blank would be read as a learned value, not as "no information".
    """

    def __init__(self, fields: List[str]):
        self.fields = list(fields)
        super().__init__(
            "These fields are required and cannot be left empty, because leaving "
            "one blank moves the predicted risk measurably rather than being "
            "treated as unknown: " + ", ".join(self.fields)
        )


def age_band(age: float) -> int:
    """The conformal conditioning variable (D9-05). 0: 20-39, 1: 40-59, 2: 60+."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return 1  # the middle band; a null age is refused upstream by D9-06 anyway
    return 0 if age < 40 else (1 if age < 60 else 2)


@register_backend("ebm_platt_conformal")
class DiabetesEbmConformalBackend(ModelBackend):

    capabilities = BackendCapabilities(
        # The EBM's own term contributions replace SHAP entirely; neither the tree
        # nor the generic explainer is ever used, so both flags stay false and
        # `explainer` is declared for what it actually is.
        supports_tree_shap=False,
        supports_vectorized_batch=True,
        supports_counterfactuals=True,
        explainer="additive",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self._bundle: Dict[str, Any] | None = None

    # ── artifacts ────────────────────────────────────────────────────────

    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            import joblib

            path = self.config.get("model", {}).get("weights_path", "")
            path = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Diabetes NHANES bundle not found at {path}. It is built by "
                    f"train_diabetes_nhanes_ebm_v2.py in the experiments repo and "
                    f"committed under models/diabetes_nhanes/."
                )
            bundle = joblib.load(path)

            leaked = CBC_BANNED.intersection(bundle["features"])
            if leaked:
                raise ValueError(
                    f"D9-03 VIOLATION: this bundle's feature list contains complete-blood-count "
                    f"columns {sorted(leaked)}. F9-29 measured that their apparent gain is an "
                    f"HbA1c assay artefact, not risk information. Refusing to load."
                )

            self._bundle = bundle
            card = bundle.get("model_card", {})
            log.info(
                "Loaded diabetes NHANES bundle: %s features, alpha %s, %s, built %s",
                len(bundle["features"]),
                bundle["conformal"]["alpha"],
                bundle["conformal"]["layer"],
                card.get("generated"),
            )
        return self._bundle

    def load(self) -> "DiabetesEbmConformalBackend":
        _ = self.bundle
        return self

    def invalidate(self) -> None:
        super().invalidate()
        self._bundle = None

    @property
    def feature_names(self) -> List[str]:
        return list(self.bundle["features"])

    @property
    def mandatory_fields(self) -> List[str]:
        return list(self.bundle["mandatory_fields"])

    # ── input discipline (D9-06) ─────────────────────────────────────────

    def _model_matrix(self, patients_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build the model matrix, refusing any row with a mandatory field blank.

        Deliberately NOT named `_frame`: the base class already has a `_frame`
        that takes ONE patient dict, and explain() calls it. Shadowing it with a
        list-taking override broke explain() the first time this was written.
        """
        frame = pd.DataFrame(patients_data)
        for column in self.feature_names:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[self.feature_names].apply(pd.to_numeric, errors="coerce")

        missing = [c for c in self.mandatory_fields if frame[c].isna().any()]
        if missing:
            raise MandatoryFieldMissing(missing)
        return frame

    def _optional_blanks(self, frame: pd.DataFrame, row: int) -> List[str]:
        optional = [c for c in self.feature_names if c not in self.mandatory_fields]
        return [c for c in optional if pd.isna(frame.iloc[row][c])]

    # ── probability layer (D9-04) ────────────────────────────────────────

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _platt(self, raw: np.ndarray) -> np.ndarray:
        clf = self.bundle["calibrator"]
        return clf.predict_proba(self._logit(raw).reshape(-1, 1))[:, 1]

    def _interval(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Percentile interval over B bootstrap refits of the Platt map.

        The draws are precomputed in the bundle, so this is two matrix products
        rather than a thousand logistic fits per request. It states uncertainty in
        the calibration map only -- see the module docstring.
        """
        boot = self.bundle["platt_bootstrap"]
        lp = self._logit(raw)[:, None]                         # (n, 1)
        draws = 1.0 / (1.0 + np.exp(-(lp * boot["a"][None, :] + boot["b"][None, :])))
        return np.percentile(draws, 2.5, axis=1), np.percentile(draws, 97.5, axis=1)

    # ── decision layer (D9-05) ───────────────────────────────────────────

    def _decide(self, raw: np.ndarray, bands: np.ndarray) -> tuple[List[str], List[List[int]]]:
        q = self.bundle["conformal"]["q_group"]
        q1 = np.array([q[f"{b}|1"] for b in bands])
        q0 = np.array([q[f"{b}|0"] for b in bands])
        in1 = (1.0 - raw) <= q1
        in0 = raw <= q0

        decisions, sets = [], []
        for one, zero in zip(in1, in0):
            if one and zero:
                decisions.append(DECISION_UNCERTAIN); sets.append([0, 1])
            elif one:
                decisions.append(DECISION_REFERRAL); sets.append([1])
            elif zero:
                decisions.append(DECISION_NO_REFERRAL); sets.append([0])
            else:
                # Empty set: the score is outside both calibration quantiles. Never
                # observed on the test cycle, but an empty set means "neither label
                # is plausible", which for a screening tool resolves to ordering the
                # test rather than to clearing the patient.
                decisions.append(DECISION_UNCERTAIN); sets.append([])
        return decisions, sets

    # ── model level ──────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw EBM score — the model's own scale, before Platt."""
        return self.bundle["model"].predict_proba(df[self.feature_names])[:, 1]

    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        """Per-feature contributions in log-odds, read out of the EBM itself.

        A pairwise term f_jk contributes to two features at once, so its value is
        split evenly between them. The split is a presentation choice; the SUM is
        exact either way, and the exactness is what gets asserted. The per-term
        breakdown before splitting is available in explain() under
        `term_contributions`, so the engineering view can show the interaction as
        its own row rather than as two halves.
        """
        model = self.bundle["model"]
        names = self.feature_names
        frame = df[names].iloc[:1]

        terms = np.asarray(model.eval_terms(frame))[0]
        intercept = float(np.asarray(model.intercept_).ravel()[0])

        index = {name: i for i, name in enumerate(names)}
        values = np.zeros(len(names), dtype=float)
        for term_name, contribution in zip(model.term_names_, terms):
            parts = [p.strip() for p in term_name.split("&")]
            share = float(contribution) / len(parts)
            for part in parts:
                if part in index:
                    values[index[part]] += share

        total = intercept + float(values.sum())
        expected = float(np.asarray(model.decision_function(frame)).ravel()[0])
        if abs(total - expected) > 1e-8:
            # An EBM that fails its own additivity identity is not an EBM any more;
            # serving an explanation that does not sum to the score is worse than
            # serving none. Gate 9.0b measured 2.7e-15 over the whole test cycle.
            raise AssertionError(
                f"EBM additivity violated: intercept + contributions = {total:.12f} "
                f"but decision_function = {expected:.12f}."
            )
        return ShapResult(values=values, base_value=intercept, feature_names=list(names))

    # ── service level ────────────────────────────────────────────────────

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([patient_data])[0]

    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not patients_data:
            return []
        model_config = self.config.get("model", {}) or {}
        frame = self._model_matrix(patients_data)

        raw = self.predict_proba(frame)
        probability = self._platt(raw)
        lower, upper = self._interval(raw)
        bands = np.array([age_band(a) for a in frame["RIDAGEYR"].values])
        decisions, sets = self._decide(raw, bands)

        results = []
        for i, (score, p, lo, hi, decision, cset) in enumerate(
            zip(raw, probability, lower, upper, decisions, sets)
        ):
            # uncertain counts as a referral: both mean "this patient goes on for
            # an HbA1c test". Counting only the confident referrals as positives is
            # the silent sensitivity drop this family exists to avoid (heart Gate 8.4).
            referred = decision in (DECISION_REFERRAL, DECISION_UNCERTAIN)
            result = {
                "prediction": 1 if referred else 0,
                "confidence": float(p),
                "diagnosis": (
                    UNCERTAIN_DIAGNOSIS
                    if decision == DECISION_UNCERTAIN
                    else "Positive" if referred else "Negative"
                ),
                "decision": decision,
                "conformal_set": cset,
                "decision_is_referral": bool(referred),
                "probability_lower": float(lo),
                "probability_upper": float(hi),
                "score_raw": float(score),
                "output_type": model_config.get("output_type"),
                "probability_scale": model_config.get("probability_scale"),
            }
            blanks = self._optional_blanks(frame, i)
            if blanks:
                result["data_completeness_warning"] = (
                    f"{len(blanks)} optional field(s) left empty: {', '.join(blanks)}. "
                    f"The model reads a blank as a learned value, not as 'unknown', so the "
                    f"result is not the same as it would be with these measured."
                )
            results.append(result)
        return results

    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        result = super().explain(patient_data)
        result["shap_scale"] = "log_odds_raw_score"
        result["shap_note"] = (
            "These are the model's own additive term contributions, not a post-hoc "
            "approximation, and they explain the raw score rather than the calibrated "
            "probability."
        )

        frame = self._model_matrix([patient_data])
        model = self.bundle["model"]
        terms = np.asarray(model.eval_terms(frame))[0]
        result["term_contributions"] = [
            {"term": name, "contribution": float(value), "interaction": "&" in name}
            for name, value in sorted(
                zip(model.term_names_, terms), key=lambda t: -abs(t[1])
            )
        ]
        result["intercept"] = float(np.asarray(model.intercept_).ravel()[0])
        return result

    def generate_counterfactuals(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """What-if over the levers a patient can actually move.

        Age, sex and prior cardiovascular disease are immutable; family history is a
        fact, not a lever. Lab values move only within the ranges the model was
        trained on, which the bundle's feature_meta already records.
        """
        from backend.counterfactual_generator import (
            NO_IMPROVEMENT_MESSAGE_DECISION,
            all_improvements,
            lowest_achievable,
            policy_violations,
        )

        # Same policy shape as the heart module: feature -> (kind, bound).
        # Clinical targets, not model-derived ones. ADIPOSITY_BAND is a lever
        # because losing central fat is exactly what the patient is being asked
        # to do; age, sex, family history and prior CVD are facts, not levers,
        # and so are the renal and hepatic markers a patient cannot act on
        # directly (LBXSCR, LBXSBU, LBXSAL).
        policy = {
            "BMXBMI": ("decrease", 24.9),
            "ADIPOSITY_BAND": ("decrease", 0),
            "SBP": ("decrease", 120),
            "DBP": ("decrease", 80),
            "LBXSTR": ("decrease", 150),
            "LBXSCH": ("decrease", 200),
            "LBXSGTSI": ("decrease", 40),
            "LBXSATSI": ("decrease", 40),
            "LBXSUA": ("decrease", 6.0),
            "PAQ650": ("to", 1),
            "PAQ665": ("to", 1),
        }
        # F9-32: raising HDL is a real lever and it is NOT here, because the shared
        # counterfactual_generator only understands "decrease" and "to".
        # `all_improvements` would set LBDHDD to the bound unconditionally, pulling a
        # healthy HDL of 70 DOWN to 40, and `policy_violations` has no branch that
        # would catch it. Adding an "increase" kind means editing a module the heart
        # module also uses, so it is deferred to Gate 9.3 with its own regression test.
        # The exercise levers (PAQ650/PAQ665) already carry most of the same advice.
        baseline = self.predict(patient_data)
        if baseline["prediction"] == 0:
            return {
                "status": "not_applicable",
                "counterfactuals": [],
                "baseline_probability": baseline["confidence"],
                "message": "This patient is already cleared. No counterfactuals needed.",
            }

        def engaged(feature: str, kind: str, bound: float) -> bool:
            value = patient_data.get(feature)
            if value is None:
                return False
            if kind == "decrease":
                return float(value) > float(bound)
            if kind == "increase":
                return float(value) < float(bound)
            return float(value) != float(bound)

        levers = [f for f, (k, b) in policy.items() if engaged(f, k, b)]

        def changes_of(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [
                {
                    "feature": feature,
                    "original_value": patient_data.get(feature),
                    "counterfactual_value": candidate[feature],
                    "direction": policy[feature][0],
                }
                for feature in policy
                if candidate.get(feature) != patient_data.get(feature)
            ]

        counterfactuals: List[Dict[str, Any]] = []
        best_achievable = None
        if levers:
            improved = all_improvements(patient_data, policy)
            changed = changes_of(improved)
            if not policy_violations(
                patient_data,
                {c["feature"]: c["counterfactual_value"] for c in changed},
                policy,
            ):
                outcome = self.predict(improved)
                if outcome["prediction"] == 0:
                    counterfactuals.append({
                        "scenario_id": 1,
                        "probability": outcome["confidence"],
                        "changes": changed,
                        "crosses_threshold": True,
                    })
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
            "probability_scale": (self.config.get("model", {}) or {}).get("probability_scale"),
            "message": None if counterfactuals else (
                "Even with every modifiable factor at its target, this patient is still "
                "sent for an HbA1c test. The dominant factors — age above all — are not "
                "modifiable."
                if best_achievable else
                NO_IMPROVEMENT_MESSAGE_DECISION
                if levers else
                "No modifiable factor is available for this patient (none was supplied, "
                "or each is already at its target). An HbA1c test is still recommended."
            ),
        }
