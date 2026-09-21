"""
OmniDiag — DiCE-inspired Counterfactual Generator
===================================================
Generates diverse "what-if" scenarios showing what features a patient could
change to reduce their diabetes risk. Uses random sampling with diversity
selection — no external dependencies beyond numpy/pandas.

Algorithm (DiCE-inspired):
    1. Sample 500+ random perturbations of mutable patient features
    2. Pass each perturbation through the full pipeline (engineer + preprocess + predict)
    3. Filter perturbations that flip Positive → Negative (y_target = 0)
    4. Score by proximity (L1 distance) with diversity penalty
    5. Select top 3 most diverse counterfactuals

Clinical constraints (mutability policy — DIABETES_POLICY below):
    - Only BMI, PhysActivity, Fruits, Veggies and HvyAlcoholConsump may change,
      each in ONE clinically beneficial direction (BMI: decrease only, floor
      18.5). Everything else is immutable — including HighBP, HighChol,
      Smoker, Stroke, HeartDiseaseorAttack and CholCheck, which in BRFSS 2015
      record whether the patient was EVER told / EVER did something, so
      "undoing" them is not an intervention.
    - Engineered features: never perturbed; recomputed by pipeline_fn from
      the changed base features.
    - The policy is enforced twice: when candidates are sampled, and again
      by policy_violations() on every scenario right before it is returned.

Reference:
    - DiCE (Diverse Counterfactual Explanations): https://arxiv.org/abs/1905.07697
    - Implementation inspired by dice-ml but avoids pandas 3.0 compatibility issues
"""

import logging
import random
from typing import Dict, Any, List, Optional, Callable, Tuple, Set

import numpy as np
import pandas as pd

log = logging.getLogger("omnidiag.counterfactual")

# =============================================================================
# Diabetes BRFSS feature definitions
# =============================================================================

# Binary features (0/1) — can be toggled in counterfactuals
BINARY_FEATURES: Set[str] = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
}

# Continuous/ordinal features — can be varied within bounds
CONTINUOUS_FEATURES: Set[str] = {
    "BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income",
}

# Engineered features — auto-computed, cannot be directly changed
ENGINEERED_FEATURES: Set[str] = {
    "BMI_Age_Interaction", "Health_Index", "Lifestyle_Score",
    "SES_Composite", "Diabetes_Clinical_Risk",
}

# Mutability policy. Each entry is the ONLY way that feature may change:
#   ("decrease", floor)  — continuous, may only go down, never below floor
#   ("to", value)        — binary, may only move to `value`
# A feature not listed here is immutable. policy_violations() enforces this.
DIABETES_POLICY: Dict[str, Tuple[str, float]] = {
    "BMI": ("decrease", 18.5),
    "PhysActivity": ("to", 1),
    "Fruits": ("to", 1),
    "Veggies": ("to", 1),
    "HvyAlcoholConsump": ("to", 0),
}

# Immutable features — never changed in a counterfactual. Listed explicitly
# (it is exactly "everything not in DIABETES_POLICY") so the reason for each
# is on record.
IMMUTABLE_FEATURES: Set[str] = {
    "HighBP",        # BRFSS: EVER told by a health professional — history
    "HighChol",      # BRFSS: EVER told by a health professional — history
    "CholCheck",     # Already had a cholesterol check — can't undo
    "Stroke",        # Past medical history — can't undo
    "HeartDiseaseorAttack",  # Past medical history — can't undo
    "Smoker",        # BRFSS: smoked >= 100 cigarettes in ENTIRE life — history
    "DiffWalk",      # Mobility limitation — not a lever
    "Age",           # Cannot reverse age
    "Sex",           # Cannot change biological sex
    "Education",     # Socioeconomic — cannot realistically change
    "Income",        # Socioeconomic — cannot realistically change
    "AnyHealthcare", # Access/insurance — systemic, not clinical
    "NoDocbcCost",   # Cost/access barrier — systemic, not clinical
    "GenHlth",       # Self-rated health — an outcome, not a lever
    "MentHlth",      # Poor-health days — an outcome, not a lever
    "PhysHlth",      # Poor-health days — an outcome, not a lever
}

# Clinical feasibility bounds
CLINICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "BMI": (15.0, 50.0),
    "MentHlth": (0.0, 30.0),
    "PhysHlth": (0.0, 30.0),
    "GenHlth": (1.0, 5.0),
    "Age": (1.0, 13.0),
    "Education": (1.0, 6.0),
    "Income": (1.0, 8.0),
}

# Features that can realistically be modified — exactly the policy's keys.
MUTABLE_FEATURES: Set[str] = set(DIABETES_POLICY)

# Perturbation scales (std as fraction of range) for each mutable feature
PERTURB_SCALES: Dict[str, float] = {
    "BMI": 0.15,           # 15% of range (35 BMI units)
    "PhysActivity": 0.9,   # Binary — high flip probability
    "Fruits": 0.7,         # Binary
    "Veggies": 0.7,        # Binary
    "HvyAlcoholConsump": 0.6,  # Binary
}

# Directional constraints for binary features, derived from the policy:
# feature -> the set of allowed target values.
DIRECTIONAL_CONSTRAINTS: Dict[str, Set[int]] = {
    feat: {int(target)}
    for feat, (kind, target) in DIABETES_POLICY.items()
    if kind == "to"
}


def policy_violations(
    original: Dict[str, Any],
    changes: Dict[str, Any],
    policy: Dict[str, Tuple[str, float]],
) -> List[str]:
    """
    Every way `changes` (feature -> new value) breaks `policy`. Empty list =
    allowed. Shared by the diabetes generator and the heart loader, and run
    on every scenario immediately before it is returned, so a violating
    scenario cannot be emitted even if candidate generation changes later.
    """
    problems = []
    for feat, new in sorted(changes.items()):
        if feat not in policy:
            problems.append(f"{feat}: immutable")
            continue
        old = original.get(feat)
        if old is None or new is None:
            problems.append(f"{feat}: missing value cannot be a lever")
            continue
        kind, bound = policy[feat]
        old, new = float(old), float(new)
        if new == old:
            continue
        if kind == "decrease" and not (new < old and new >= bound):
            problems.append(f"{feat}: {old} -> {new} (decrease only, floor {bound})")
        elif kind == "to" and new != float(bound):
            problems.append(f"{feat}: {old} -> {new} (may only move to {bound})")
    return problems


def all_improvements(
    patient_data: Dict[str, Any], policy: Dict[str, Tuple[str, float]]
) -> Dict[str, Any]:
    """The patient with every allowed lever pushed to its most favourable
    value (continuous to its floor, binary to its target). Levers the
    patient is missing (None) or already satisfies are left as they are."""
    improved = dict(patient_data)
    for feat, (kind, bound) in sorted(policy.items()):
        value = patient_data.get(feat)
        if value is None:
            continue
        if kind == "decrease":
            if float(value) > bound:
                improved[feat] = bound
        else:
            improved[feat] = bound
    return improved


class CounterfactualGenerator:
    """
    DiCE-inspired counterfactual generator using random sampling + diversity selection.
    
    Args:
        predict_fn: Callable that takes a preprocessed DataFrame and returns
                    probability of positive class for each row.
        pipeline_fn: Callable that takes a raw patient DataFrame and returns
                     an engineered + preprocessed DataFrame ready for predict_fn.
        feature_names: List of feature names in the preprocessed space.
        raw_feature_names: List of feature names in the raw input space.
        n_samples: Number of random perturbations to generate (default: 500).
        n_counterfactuals: Number of diverse counterfactuals to return (default: 3).
        random_state: Random seed for reproducibility (default: 42).
    """
    
    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], float],
        pipeline_fn: Callable[[pd.DataFrame], pd.DataFrame],
        feature_names: List[str],
        raw_feature_names: Optional[List[str]] = None,
        n_samples: int = 500,
        n_counterfactuals: int = 3,
        random_state: int = 42,
        inference_threshold: float = 0.5,
    ):
        self.predict_fn = predict_fn
        self.pipeline_fn = pipeline_fn
        self.feature_names = feature_names
        self.raw_feature_names = raw_feature_names or feature_names
        self.n_samples = n_samples
        self.n_counterfactuals = n_counterfactuals
        self.rng = random.Random(random_state)
        self.np_rng = np.random.default_rng(random_state)
        self.inference_threshold = inference_threshold
    
    def generate(
        self,
        patient_data: Dict[str, Any],
        desired_class: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse counterfactual explanations for a patient.
        
        Args:
            patient_data: Raw patient features dict (21 BRFSS fields).
            desired_class: Target class (0 = Negative / lower risk).
        
        Returns:
            List of counterfactual dicts, each containing:
                - scenario: Human-readable description of changes (text only)
                - changes: Dict of feature_name → new_value (only changed features)
                - new_probability: Float probability after the changes, CORRECTED
                  scale (kept under its original name for existing clients)
                - new_probability_corrected: the same value, explicitly named
                - baseline_probability_corrected: probability before the changes
                - risk_reduction: relative reduction as a string, e.g. "74%"
                - risk_reduction_relative_pct: float, (base - after) / base * 100
                - risk_reduction_absolute_pp: float, (base - after) * 100
                - probability_scale: always "corrected" here
                - feasibility: "high", "medium", or "low"

        Every probability in the returned dicts is on the deployment
        (prevalence-corrected) scale, the same scale as inference_threshold.
        """
        # Get baseline prediction
        baseline_df = self.pipeline_fn(pd.DataFrame([patient_data]))
        baseline_proba_corrected = self._get_proba_corrected(baseline_df)
        
        log.debug(
            f"Generating counterfactuals: baseline_proba_corrected="
            f"{baseline_proba_corrected:.4f}, "
            f"desired_class={desired_class}, n_samples={self.n_samples}"
        )
        
        # If patient is already Negative (low risk), no counterfactuals needed
        if (desired_class == 0 and baseline_proba_corrected < self.inference_threshold) or \
           (desired_class == 1 and baseline_proba_corrected >= self.inference_threshold):
            log.debug("Patient already in desired class — no counterfactuals generated")
            return []
        
        # Generate candidate perturbations. The all-levers-improved patient is
        # always evaluated too, so a crossing that exists is not missed by
        # random sampling alone.
        candidates = self._sample_candidates(patient_data)
        candidates.append(all_improvements(patient_data, DIABETES_POLICY))
        
        # Evaluate each candidate through the full pipeline
        valid_candidates = []
        for cand_raw in candidates:
            try:
                cand_df = self.pipeline_fn(pd.DataFrame([cand_raw]))
                cand_proba_corrected = self._get_proba_corrected(cand_df)
                
                # Check if prediction flips to desired class
                if (desired_class == 0 and cand_proba_corrected < self.inference_threshold) or \
                   (desired_class == 1 and cand_proba_corrected >= self.inference_threshold):
                    
                    # Calculate changes relative to baseline
                    changes = self._compute_changes(patient_data, cand_raw)
                    if not changes or policy_violations(patient_data, changes, DIABETES_POLICY):
                        continue
                    
                    # Calculate proximity score (L1 distance normalized)
                    proximity = self._proximity_score(patient_data, cand_raw)
                    
                    valid_candidates.append({
                        "raw": cand_raw,
                        "proba_corrected": cand_proba_corrected,
                        "baseline_proba_corrected": baseline_proba_corrected,
                        "changes": changes,
                        "proximity": proximity,
                    })
            except Exception as e:
                log.debug(f"Candidate evaluation failed: {e}")
                continue
        
        if not valid_candidates:
            log.debug("No valid counterfactuals found — try increasing n_samples")
            return []
        
        # One candidate per set of changed features — the closest one. Three
        # scenarios that all say "lower BMI" (to 22.7, 21.4, 21.3) are one
        # scenario, not three alternatives; with few levers, fewer than
        # n_counterfactuals scenarios is the honest answer.
        best_per_set: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for cand in valid_candidates:
            key = tuple(sorted(cand["changes"]))
            if key not in best_per_set or cand["proximity"] > best_per_set[key]["proximity"]:
                best_per_set[key] = cand
        valid_candidates = [best_per_set[k] for k in sorted(best_per_set)]

        # Sort by proximity (closest first), then select diverse subset
        selected = self._select_diverse(valid_candidates)
        
        # Build response.
        #
        # Two different numbers can both be called "risk reduction", and they
        # are far apart on the corrected scale: a patient going from 0.145 to
        # 0.038 has dropped 74% of their risk but only 10.7 percentage points.
        # Both are emitted, each named for what it is, so no consumer has to
        # infer which one it is holding. `risk_reduction` keeps the relative
        # reading it has always had.
        counterfactuals = []
        for cand in selected:
            baseline = cand["baseline_proba_corrected"]
            after = cand["proba_corrected"]
            relative_pct = max(0.0, (baseline - after) / max(baseline, 0.001) * 100)
            absolute_pp = max(0.0, (baseline - after) * 100)
            feasibility = self._assess_feasibility(cand["changes"])
            
            counterfactuals.append({
                "scenario": self._build_scenario(cand["changes"]),
                "changes": cand["changes"],
                "new_probability": round(after, 4),
                "new_probability_corrected": round(after, 4),
                "baseline_probability_corrected": round(baseline, 4),
                "risk_reduction": f"{int(round(relative_pct))}%",
                "risk_reduction_relative_pct": round(relative_pct, 2),
                "risk_reduction_absolute_pp": round(absolute_pp, 2),
                "probability_scale": "corrected",
                "feasibility": feasibility,
                "crosses_threshold": True,
            })

        # Final filter (defence in depth): nothing that breaks the policy
        # leaves this function, whatever generated it.
        counterfactuals = [
            cf for cf in counterfactuals
            if not policy_violations(patient_data, cf["changes"], DIABETES_POLICY)
        ]
        log.debug(f"Generated {len(counterfactuals)} counterfactuals")
        return counterfactuals

    def best_achievable(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        What the model estimates with EVERY allowed lever improved at once,
        whether or not that crosses the threshold. None when the patient has
        no lever left to move. Same fields as a generate() scenario, with
        `crosses_threshold` saying whether it gets below the threshold.
        """
        improved = all_improvements(patient_data, DIABETES_POLICY)
        changes = self._compute_changes(patient_data, improved)
        if not changes or policy_violations(patient_data, changes, DIABETES_POLICY):
            return None
        baseline = self._get_proba_corrected(self.pipeline_fn(pd.DataFrame([patient_data])))
        after = self._get_proba_corrected(self.pipeline_fn(pd.DataFrame([improved])))
        relative_pct = max(0.0, (baseline - after) / max(baseline, 0.001) * 100)
        absolute_pp = max(0.0, (baseline - after) * 100)
        return {
            "scenario": self._build_scenario(changes),
            "changes": changes,
            "new_probability": round(after, 4),
            "new_probability_corrected": round(after, 4),
            "baseline_probability_corrected": round(baseline, 4),
            "risk_reduction": f"{int(round(relative_pct))}%",
            "risk_reduction_relative_pct": round(relative_pct, 2),
            "risk_reduction_absolute_pp": round(absolute_pp, 2),
            "probability_scale": "corrected",
            "feasibility": self._assess_feasibility(changes),
            "crosses_threshold": bool(after < self.inference_threshold),
        }
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    
    def _get_proba_corrected(self, df: pd.DataFrame) -> float:
        """
        Positive-class probability from the predictor, on the deployment scale.

        `predict_fn` is EnsembleModelLoader's, which applies the prevalence
        correction before returning, so every probability in this module is
        corrected — the same scale as `self.inference_threshold`.

        A missing value falls back to the decision threshold, not to 0.5. On
        the raw prior those coincided; on the deployment prior 0.5 is roughly
        eight times the threshold, so the old fallback turned a failed lookup
        into a confident Positive.
        """
        result = self.predict_fn(df)
        if isinstance(result, dict):
            value_corrected = result.get("confidence")
            if value_corrected is None:
                log.warning(
                    "predict_fn returned no 'confidence'; falling back to the "
                    "decision threshold (%s) as the neutral point",
                    self.inference_threshold,
                )
                return float(self.inference_threshold)
            return float(value_corrected)
        return float(result)
    
    def _is_illegal_flip(self, feat: str, from_val: float, to_val: float) -> bool:
        """
        Clinical Firewall — return True if changing `feat` from `from_val` to
        `to_val` would be a clinically absurd recommendation (e.g. advising a
        patient to start smoking, raise blood pressure, or reduce physical
        activity).

        Only features listed in DIRECTIONAL_CONSTRAINTS are restricted; each
        entry names the only clinically safe target value(s) for that
        feature. Features not listed may flip in either direction.
        """
        if to_val == from_val:
            return False  # no-op, never illegal
        if feat not in DIRECTIONAL_CONSTRAINTS:
            return False
        return to_val not in DIRECTIONAL_CONSTRAINTS[feat]

    def _sample_candidates(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random perturbations of mutable features."""
        candidates = []
        
        for _ in range(self.n_samples):
            cand = dict(patient_data)  # Copy all features
            
            # Perturb mutable features — sorted, so the RNG is consumed in
            # the same order in every process (iterating the set directly
            # followed PYTHONHASHSEED and made results differ per process).
            for feat in sorted(MUTABLE_FEATURES):
                if feat not in patient_data:
                    continue
                
                original = float(patient_data[feat])
                
                if feat in BINARY_FEATURES:
                    flip_prob = PERTURB_SCALES.get(feat, 0.5)
                    proposed = original if self.rng.random() >= flip_prob else 1.0 - original

                    # Clinical Firewall: never emit a flip that the directional
                    # constraints forbid (e.g. Smoker 0→1, DiffWalk 0→1).
                    # Features without a constraint entry can flip either way.
                    if self._is_illegal_flip(feat, original, proposed):
                        cand[feat] = original
                    else:
                        cand[feat] = proposed
                
                elif feat in CLINICAL_BOUNDS:
                    lo, hi = CLINICAL_BOUNDS[feat]
                    scale = PERTURB_SCALES.get(feat, 0.15)
                    
                    # Half-normal step in the one allowed direction: a
                    # "decrease" lever never goes up (the old two-sided
                    # normal proposed weight GAIN), and never below its floor.
                    std = (hi - lo) * scale
                    step = abs(self.np_rng.normal(loc=0.0, scale=std))
                    kind, floor = DIABETES_POLICY.get(feat, ("decrease", lo))
                    perturbed = max(original - step, max(lo, floor))
                    perturbed = min(round(perturbed, 1), original)
                    
                    # For ordinal features, round to integer
                    if feat in {"MentHlth", "PhysHlth", "GenHlth",
                                "Age", "Education", "Income"}:
                        perturbed = round(perturbed)
                    
                    cand[feat] = float(perturbed)
                
                else:
                    # Unknown feature — leave unchanged
                    cand[feat] = original
            
            candidates.append(cand)
        
        return candidates
    
    def _compute_changes(
        self, original: Dict[str, Any], candidate: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute the changes between original and candidate features."""
        changes = {}
        for key in candidate:
            # Only report changes for raw (not engineered) features
            if key in ENGINEERED_FEATURES:
                continue
            # Every raw change is reported — including an immutable one, if a
            # perturbation ever slipped through — so policy_violations() can
            # reject the scenario instead of it silently hiding the change.
            
            orig_val = float(original.get(key, 0))
            cand_val = float(candidate[key])
            
            if abs(cand_val - orig_val) > 0.001:
                changes[key] = cand_val
        
        return changes
    
    def _proximity_score(
        self, original: Dict[str, Any], candidate: Dict[str, Any]
    ) -> float:
        """
        Compute proximity score (inverse of normalized L1 distance).
        
        Higher score = closer to original (more minimal change).
        Uses only mutable features for the distance calculation.
        """
        total_distance = 0.0
        n_features = 0
        
        for feat in sorted(MUTABLE_FEATURES):
            if feat not in original or feat not in candidate:
                continue
            if feat in ENGINEERED_FEATURES:
                continue
            
            orig_val = float(original[feat])
            cand_val = float(candidate[feat])
            
            # Normalize by feature range
            if feat in CLINICAL_BOUNDS:
                lo, hi = CLINICAL_BOUNDS[feat]
                range_size = max(hi - lo, 0.001)
            elif feat in BINARY_FEATURES:
                range_size = 1.0
            else:
                range_size = max(abs(orig_val), 1.0)
            
            distance = abs(cand_val - orig_val) / range_size
            total_distance += distance
            n_features += 1
        
        if n_features == 0:
            return 0.0
        
        avg_distance = total_distance / n_features
        # Convert to proximity (1 = identical, 0 = maximally different)
        return 1.0 / (1.0 + avg_distance)
    
    def _select_diverse(
        self, candidates: List[Dict], k: Optional[int] = None
    ) -> List[Dict]:
        """
        Select k diverse counterfactuals using greedy diversity selection.
        
        Algorithm:
            1. Sort candidates by proximity (closest first)
            2. Take the closest candidate as the first selection
            3. For each subsequent selection, pick the candidate that maximizes:
               score = proximity - λ * max_diversity_penalty(existing)
        
        This ensures diverse action plans (e.g., one focused on BMI,
        another on physical activity, another on blood pressure).
        """
        if k is None:
            k = self.n_counterfactuals
        
        if len(candidates) <= k:
            return candidates
        
        # Sort by proximity descending (closest first)
        sorted_cands = sorted(candidates, key=lambda c: -c["proximity"])
        
        selected = [sorted_cands[0]]
        remaining = sorted_cands[1:]
        
        diversity_lambda = 0.5  # Weight for diversity vs proximity
        
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = float("-inf")
            
            for i, cand in enumerate(remaining):
                # Proximity score
                prox = cand["proximity"]
                
                # Diversity penalty: max similarity to any selected candidate
                max_sim = max(
                    self._change_similarity(cand["changes"], sel["changes"])
                    for sel in selected
                )
                
                score = prox - diversity_lambda * max_sim
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _change_similarity(
        self, changes_a: Dict[str, float], changes_b: Dict[str, float]
    ) -> float:
        """
        Compute similarity between two change sets (0 = different, 1 = identical).
        
        Two counterfactuals are "similar" if they change the same features.
        Diversity requires changes in different feature subsets.
        """
        if not changes_a or not changes_b:
            return 0.0
        
        features_a = set(changes_a.keys())
        features_b = set(changes_b.keys())
        
        if not features_a or not features_b:
            return 0.0
        
        # Jaccard similarity of changed feature sets
        intersection = features_a & features_b
        union = features_a | features_b
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _assess_feasibility(self, changes: Dict[str, float]) -> str:
        """
        Assess clinical feasibility of the proposed changes.
        
        Rules:
            - "high": Only lifestyle changes (BMI, PhysActivity, Fruits, Veggies)
            - "medium": Requires medical intervention (HighBP, HighChol medication)
            - "low": Unrealistic changes (large BMI drop >10 units, multiple medical)
        """
        if not changes:
            return "high"
        
        # Check for large BMI changes
        bmi_change = abs(changes.get("BMI", 0))
        if bmi_change > 10:
            return "low"
        
        # Check for multiple medical interventions
        medical_changes = {"HighBP", "HighChol"}
        medical_count = sum(1 for f in medical_changes if f in changes)
        
        lifestyle_changes = {"PhysActivity", "Fruits", "Veggies", "BMI", "Smoker"}
        lifestyle_count = sum(1 for f in lifestyle_changes if f in changes)
        
        mental_changes = {"MentHlth", "PhysHlth", "GenHlth"}
        mental_count = sum(1 for f in mental_changes if f in changes)
        
        if medical_count >= 2:
            return "low"
        
        if medical_count == 1:
            if lifestyle_count >= 2 or mental_count >= 2:
                return "low"
            return "medium"
        
        if mental_count >= 2:
            return "medium"
        
        # Only lifestyle changes
        return "high"
    
    def _build_scenario(self, changes: Dict[str, float]) -> str:
        """
        Build a human-readable scenario description.

        Text only — deliberately no percentage. The reduction is carried by
        `risk_reduction_relative_pct` / `risk_reduction_absolute_pp`, so the
        sentence cannot drift out of step with the numbers beside it. (The
        old signature took a percentage it never rendered, while the docstring
        showed examples containing one.)

        Examples:
            - "If BMI drops to 27.0"
            - "If starts physical activity and increases fruit intake"
            - "If blood pressure is controlled"
        """
        if not changes:
            return "No changes needed"
        
        parts = []
        for feat, val in sorted(changes.items()):
            if feat == "BMI":
                parts.append(f"BMI drops to {val:.1f}")
            elif feat == "HighBP":
                parts.append("blood pressure is controlled" if val == 0 else "blood pressure elevates")
            elif feat == "HighChol":
                parts.append("cholesterol is controlled" if val == 0 else "cholesterol elevates")
            elif feat == "Smoker":
                parts.append("stops smoking" if val == 0 else "starts smoking")
            elif feat == "PhysActivity":
                parts.append("starts physical activity" if val == 1 else "stops physical activity")
            elif feat == "Fruits":
                parts.append("increases fruit intake" if val == 1 else "decreases fruit intake")
            elif feat == "Veggies":
                parts.append("increases vegetable intake" if val == 1 else "decreases vegetable intake")
            elif feat == "HvyAlcoholConsump":
                parts.append("limits alcohol consumption")
            elif feat == "MentHlth" and val == 0:
                parts.append("mental health improves (0 poor days)")
            elif feat == "MentHlth":
                parts.append(f"mental health days drop to {int(val)}")
            elif feat == "PhysHlth" and val == 0:
                parts.append("physical health improves (0 poor days)")
            elif feat == "PhysHlth":
                parts.append(f"physical health days drop to {int(val)}")
            elif feat == "GenHlth" and val < 3:
                parts.append(f"general health improves to level {int(val)}")
            elif feat == "NoDocbcCost":
                parts.append("removes cost barriers")
            elif feat == "DiffWalk":
                parts.append("maintains or improves mobility")
            else:
                parts.append(f"{feat} changes to {val}")
        
        if not parts:
            return "No changes needed"
        
        scenario = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
        
        return f"If {scenario}"
