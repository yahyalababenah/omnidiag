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

Clinical constraints:
    - BMI: 15.0–50.0 (cannot go below or above biologically plausible bounds)
    - Binary features: only 0 or 1 (no fractional values)
    - Ordinal features: must remain integer within their range
    - Immutable features: Sex, Age, Education cannot be changed
    - Engineered features: auto-computed from raw features

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

# Immutable features — cannot be changed in realistic counterfactuals
IMMUTABLE_FEATURES: Set[str] = {
    "Sex",           # Cannot change biological sex
    "Age",           # Cannot reverse age
    "Income",        # Socioeconomic — cannot realistically change
    "Education",     # Socioeconomic — cannot realistically change
    "NoDocbcCost",   # Cost/access barrier — systemic, not clinical
    "AnyHealthcare", # Already has insurance — can't undo
    "CholCheck",     # Already had cholesterol check — can't undo
    "Stroke",        # Past medical history — can't undo
    "HeartDiseaseorAttack",  # Past medical history — can't undo
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

# Features that can realistically be modified
# NOTE: NoDocbcCost, Income, Education, AnyHealthcare are excluded
# because they are social/access barriers — clinically inappropriate to suggest changing.
MUTABLE_FEATURES: Set[str] = {
    "BMI", "HighBP", "HighChol", "Smoker", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth",
    "PhysHlth", "GenHlth", "DiffWalk",
}

# Perturbation scales (std as fraction of range) for each mutable feature
PERTURB_SCALES: Dict[str, float] = {
    "BMI": 0.15,           # 15% of range (35 BMI units)
    "HighBP": 1.0,         # Binary — flip probability
    "HighChol": 1.0,       # Binary — flip probability
    "Smoker": 0.8,         # Binary — high flip probability
    "PhysActivity": 0.9,   # Binary — high flip probability
    "Fruits": 0.7,         # Binary
    "Veggies": 0.7,        # Binary
    "HvyAlcoholConsump": 0.6,  # Binary
    "MentHlth": 0.2,       # 20% of 30-day range
    "PhysHlth": 0.2,       # 20% of 30-day range
    "GenHlth": 0.25,       # 25% of 4-unit range
    "DiffWalk": 0.5,       # Binary
}

# Directional constraints: prevent clinically harmful perturbations.
# Key   = feature name
# Value = set of allowed target values (the only clinically safe values)
# Features NOT listed here can flip bidirectionally (original behavior).
DIRECTIONAL_CONSTRAINTS: Dict[str, Set[int]] = {
    "HvyAlcoholConsump": {0},   # Only allow stopping/reducing alcohol, never starting
    "Smoker": {0},              # Only allow stopping smoking, never starting
    "Veggies": {1},             # Only allow adopting vegetable intake, never dropping
    "Fruits": {1},              # Only allow adopting fruit intake, never dropping
    "PhysActivity": {1},        # Only allow adopting physical activity, never dropping
    "DiffWalk": {0},            # Never advise decreasing mobility (0→1 forbidden)
}


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
    ):
        self.predict_fn = predict_fn
        self.pipeline_fn = pipeline_fn
        self.feature_names = feature_names
        self.raw_feature_names = raw_feature_names or feature_names
        self.n_samples = n_samples
        self.n_counterfactuals = n_counterfactuals
        self.rng = random.Random(random_state)
        self.np_rng = np.random.default_rng(random_state)
    
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
                - scenario: Human-readable description of changes
                - changes: Dict of feature_name → new_value (only changed features)
                - new_probability: Float probability of positive class (0-1)
                - risk_reduction: String like "44%"
                - feasibility: "high", "medium", or "low"
        """
        # Get baseline prediction
        baseline_df = self.pipeline_fn(pd.DataFrame([patient_data]))
        baseline_proba = self._get_proba(baseline_df)
        
        log.debug(
            f"Generating counterfactuals: baseline_proba={baseline_proba:.4f}, "
            f"desired_class={desired_class}, n_samples={self.n_samples}"
        )
        
        # If patient is already Negative (low risk), no counterfactuals needed
        if (desired_class == 0 and baseline_proba < 0.5) or \
           (desired_class == 1 and baseline_proba >= 0.5):
            log.debug("Patient already in desired class — no counterfactuals generated")
            return []
        
        # Generate candidate perturbations
        candidates = self._sample_candidates(patient_data)
        
        # Evaluate each candidate through the full pipeline
        valid_candidates = []
        for cand_raw in candidates:
            try:
                cand_df = self.pipeline_fn(pd.DataFrame([cand_raw]))
                cand_proba = self._get_proba(cand_df)
                
                # Check if prediction flips to desired class
                if (desired_class == 0 and cand_proba < 0.5) or \
                   (desired_class == 1 and cand_proba >= 0.5):
                    
                    # Calculate changes relative to baseline
                    changes = self._compute_changes(patient_data, cand_raw)
                    
                    # Calculate proximity score (L1 distance normalized)
                    proximity = self._proximity_score(patient_data, cand_raw)
                    
                    valid_candidates.append({
                        "raw": cand_raw,
                        "proba": cand_proba,
                        "baseline_proba": baseline_proba,
                        "changes": changes,
                        "proximity": proximity,
                    })
            except Exception as e:
                log.debug(f"Candidate evaluation failed: {e}")
                continue
        
        if not valid_candidates:
            log.debug("No valid counterfactuals found — try increasing n_samples")
            return []
        
        # Sort by proximity (closest first), then select diverse subset
        selected = self._select_diverse(valid_candidates)
        
        # Build response
        counterfactuals = []
        for cand in selected:
            risk_reduction_pct = max(
                0, (cand["baseline_proba"] - cand["proba"]) / max(cand["baseline_proba"], 0.001) * 100
            )
            feasibility = self._assess_feasibility(cand["changes"])
            
            counterfactuals.append({
                "scenario": self._build_scenario(cand["changes"], risk_reduction_pct),
                "changes": cand["changes"],
                "new_probability": round(cand["proba"], 4),
                "risk_reduction": f"{int(round(risk_reduction_pct))}%",
                "feasibility": feasibility,
            })
        
        log.debug(f"Generated {len(counterfactuals)} counterfactuals")
        return counterfactuals
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    
    def _get_proba(self, df: pd.DataFrame) -> float:
        """Get positive-class probability from the predictor function."""
        # predict_fn returns dict with 'confidence' key
        result = self.predict_fn(df)
        if isinstance(result, dict):
            return float(result.get("confidence", 0.5))
        return float(result)
    
    def _sample_candidates(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random perturbations of mutable features."""
        candidates = []
        
        for _ in range(self.n_samples):
            cand = dict(patient_data)  # Copy all features
            
            # Perturb mutable features
            for feat in MUTABLE_FEATURES:
                if feat not in patient_data:
                    continue
                
                original = float(patient_data[feat])
                
                if feat in BINARY_FEATURES:
                    # Check directional constraint first
                    if feat in DIRECTIONAL_CONSTRAINTS:
                        allowed = DIRECTIONAL_CONSTRAINTS[feat]
                        safe_val = float(list(allowed)[0])
                        if original == safe_val:
                            # Already at clinically safe value — lock it
                            cand[feat] = safe_val
                        else:
                            # Original is unsafe (e.g. Smoker=1, safe=0).
                            # Use flip_prob to decide whether to switch to safe target,
                            # but NEVER allow illegal flips (e.g. 0→1 for smoking).
                            flip_prob = PERTURB_SCALES.get(feat, 0.5)
                            if self.rng.random() < flip_prob:
                                cand[feat] = safe_val
                            else:
                                cand[feat] = original
                    else:
                        # Standard binary flip (original behavior)
                        flip_prob = PERTURB_SCALES.get(feat, 0.5)
                        if self.rng.random() < flip_prob:
                            cand[feat] = 1.0 - original  # Flip 0→1 or 1→0
                        else:
                            cand[feat] = original
                
                elif feat in CLINICAL_BOUNDS:
                    lo, hi = CLINICAL_BOUNDS[feat]
                    scale = PERTURB_SCALES.get(feat, 0.15)
                    
                    # Sample from truncated normal around original value
                    std = (hi - lo) * scale
                    perturbed = self.np_rng.normal(loc=original, scale=std)
                    perturbed = np.clip(perturbed, lo, hi)
                    
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
            # Defensive: skip immutable/systemic features — they should never
            # appear as actionable changes, even if a perturbation slipped through
            if key in IMMUTABLE_FEATURES:
                continue
            
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
        
        for feat in MUTABLE_FEATURES:
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
    
    def _build_scenario(
        self, changes: Dict[str, float], risk_reduction_pct: float
    ) -> str:
        """
        Build a human-readable scenario description.
        
        Examples:
            - "If BMI drops from 34 to 27 (—44% risk)"
            - "If PhysActivity increases and Fruits increases (—34% risk)"
            - "If HighBP is controlled with medication (—55% risk)"
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
