"""
Guard test for the probability-scale contract (see backend/probability_scale.py).

The architectural rule this file enforces:

    Every probability leaving EnsembleModelLoader.predict() is CORRECTED.
    A raw value must say so in its name (`_raw`) everywhere it travels.
    No probability may be compared against a bare float literal — cut-points
    come from configs/diabetes.yaml, on a named scale.

The checks are AST-based rather than grep-based: `p >= 0.85` and
`p>=0.85` and a line-wrapped comparison are the same defect, and a string
containing "0.7" is not. AST also lets a rule say "the *argument* of this
call", which is exactly what the contract is about.

Scope is the diabetes surface only. backend/model_loader.py and everything
under models/heart_disease/ are deliberately NOT scanned — the heart module
has no prevalence correction and is out of scope for this work.

When this test fails it prints file:line for every violation. Fixing a
violation means naming the scale, not silencing the rule.
"""

import ast
import os
from typing import List, NamedTuple

import pytest

from backend.probability_scale import (
    Scale,
    ScaledProbability,
    classify_band,
    scale_of_disease_config,
    scale_of_result,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# What gets scanned
# ---------------------------------------------------------------------------
# Every backend module that reads, compares or forwards a diabetes probability.
# Heart-only modules are intentionally absent.
SCANNED_MODULES = (
    "backend/ensemble_loader.py",
    "backend/counterfactual_generator.py",
    "backend/active_learning/sampler.py",
    "backend/llm/report_generator.py",
    "backend/main.py",
    "backend/cache.py",
    "backend/monitoring/metrics.py",
    "backend/monitoring/routes.py",
    "backend/admin/routes.py",
    "backend/active_learning/routes.py",
)

# Identifiers that denote a probability. Substring match, plus a few exact
# short names that are common locals in this codebase.
_PROB_SUBSTRINGS = ("confidence", "probability", "proba", "risk_score")
_PROB_EXACT = {"p", "prob", "conf"}

# A name satisfies the contract when it says which scale it is on.
_SCALE_SUFFIXES = ("_raw", "_corrected")

# Functions whose result depends on WHICH scale the argument is on. Passing an
# unnamed probability to one of these is the exact defect that moved the
# review queue off the decision boundary.
SCALE_SENSITIVE_FUNCS = {
    "should_queue_for_review",
    "prediction_entropy",
    "uncertainty_band",
    "classify_band",
}

# Dict keys that hold a probability.
_PROB_KEYS = {"confidence", "probability", "proba"}


class Violation(NamedTuple):
    rule: str
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}:{self.line} — {self.detail}"


def _is_probability_name(name: str) -> bool:
    low = name.lower()
    if low in _PROB_EXACT:
        return True
    return any(s in low for s in _PROB_SUBSTRINGS)


def _declares_scale(name: str) -> bool:
    return name.lower().endswith(_SCALE_SUFFIXES)


def _name_of(node: ast.AST) -> str | None:
    """Best-effort identifier for a Name / Attribute / simple Subscript node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, str):
            return node.slice.value
    return None


def _is_unit_interval_literal(node: ast.AST) -> bool:
    """A float strictly inside (0, 1) — i.e. something shaped like a cut-point."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and 0.0 < node.value < 1.0
    )


def _unwrap(node: ast.AST) -> ast.AST:
    """Strip float(...) / round(...) wrappers to reach the real expression."""
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "round", "abs"}
        and node.args
    ):
        node = node.args[0]
    return node


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def _rule_literal_cutpoint(tree: ast.AST, path: str) -> List[Violation]:
    """R1 — a probability compared against a bare float literal in (0, 1).

    Cut-points belong in configs/diabetes.yaml on a declared scale. A literal
    is silently a *raw*-scale number written before the correction existed.
    """
    out: List[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        has_prob = any(
            (n := _name_of(op)) is not None and _is_probability_name(n)
            for op in operands
        )
        if not has_prob:
            continue
        for op in operands:
            if _is_unit_interval_literal(op):
                out.append(
                    Violation(
                        "R1-literal-cutpoint",
                        path,
                        node.lineno,
                        f"probability compared against the bare literal {op.value!r}; "
                        f"read the cut-point from config on a named scale",
                    )
                )
    return out


def _rule_literal_default(tree: ast.AST, path: str) -> List[Violation]:
    """R2 — `.get("confidence", 0.5)`-style defaults.

    0.5 is only neutral on the raw (50/50) prior. On the deployment prior the
    neutral point is the decision threshold, which lives in config.
    """
    out: List[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        if len(node.args) != 2:
            continue
        key = node.args[0]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if key.value.lower() not in _PROB_KEYS:
            continue
        if _is_unit_interval_literal(node.args[1]):
            out.append(
                Violation(
                    "R2-literal-default",
                    path,
                    node.lineno,
                    f'.get("{key.value}", {node.args[1].value!r}) — the fallback is a '
                    f"raw-scale constant; use the configured threshold instead",
                )
            )
    return out


def _rule_scale_sensitive_arg(tree: ast.AST, path: str) -> List[Violation]:
    """R3 — an unnamed probability handed to a scale-sensitive function."""
    out: List[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if fname not in SCALE_SENSITIVE_FUNCS or not node.args:
            continue
        arg = _unwrap(node.args[0])
        ident = _name_of(arg)
        if ident is None:
            # A call or computed expression: nothing to read a scale off.
            out.append(
                Violation(
                    "R3-unnamed-argument",
                    path,
                    node.lineno,
                    f"{fname}() receives an expression with no scale in its name",
                )
            )
            continue
        if not _declares_scale(ident):
            out.append(
                Violation(
                    "R3-unnamed-argument",
                    path,
                    node.lineno,
                    f"{fname}({ident}) — argument does not declare its scale; "
                    f"expected a name ending in _raw or _corrected",
                )
            )
    return out


def _rule_probability_binding(tree: ast.AST, path: str) -> List[Violation]:
    """R4 — extracting a probability out of a result dict without naming its scale.

    `confidence = result["confidence"]` loses the one fact the reader needs.
    By contract that value is corrected, so it must be bound as such.
    """
    out: List[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _unwrap(node.value)
        key: str | None = None
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
            if isinstance(value.slice.value, str):
                key = value.slice.value
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            key = value.args[0].value
        if key is None or key.lower() not in _PROB_KEYS:
            continue
        if not _declares_scale(target.id):
            out.append(
                Violation(
                    "R4-unnamed-binding",
                    path,
                    node.lineno,
                    f'{target.id} = ...["{key}"] — a predict() probability is '
                    f"corrected by contract; name it {target.id}_corrected",
                )
            )
    return out


_RULES = (
    _rule_literal_cutpoint,
    _rule_literal_default,
    _rule_scale_sensitive_arg,
    _rule_probability_binding,
)


def collect_violations() -> List[Violation]:
    found: List[Violation] = []
    for rel in SCANNED_MODULES:
        path = os.path.join(ROOT, rel)
        assert os.path.isfile(path), f"scanned module is missing: {rel}"
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=rel)
        for rule in _RULES:
            found.extend(rule(tree, rel))
    return sorted(found, key=lambda v: (v.path, v.line, v.rule))


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
class TestScaleContractGuard:
    def test_heart_modules_are_out_of_scope(self):
        """The scanner must never reach the heart module."""
        assert "backend/model_loader.py" not in SCANNED_MODULES
        assert not any("heart" in m for m in SCANNED_MODULES)

    def test_no_unscaled_probability_handling(self):
        violations = collect_violations()
        if violations:
            report = "\n".join(f"  {v}" for v in violations)
            pytest.fail(
                f"{len(violations)} probability-scale contract violation(s):\n{report}\n\n"
                "Each one hands a probability to a consumer that cannot tell which\n"
                "scale it is on. See backend/probability_scale.py for the contract."
            )


# ---------------------------------------------------------------------------
# The carrier type itself
# ---------------------------------------------------------------------------
PI_TRAIN, PI_DEPLOY = 0.50, 0.14


class TestScaledProbability:
    def test_from_raw_round_trips(self):
        p = ScaledProbability.from_raw(0.8, PI_TRAIN, PI_DEPLOY)
        back = ScaledProbability.from_corrected(p.corrected, PI_TRAIN, PI_DEPLOY)
        assert back.raw == pytest.approx(0.8, abs=1e-12)

    def test_corrected_is_below_raw_when_deploy_prior_is_lower(self):
        p = ScaledProbability.from_raw(0.5, PI_TRAIN, PI_DEPLOY)
        assert p.corrected < p.raw
        assert p.corrected == pytest.approx(0.14, abs=1e-12)

    def test_fixed_points_are_preserved(self):
        for edge in (0.0, 1.0):
            p = ScaledProbability.from_raw(edge, PI_TRAIN, PI_DEPLOY)
            assert p.corrected == pytest.approx(edge, abs=1e-12)

    def test_on_requires_a_real_scale(self):
        p = ScaledProbability.from_raw(0.3, PI_TRAIN, PI_DEPLOY)
        assert p.on(Scale.RAW) == pytest.approx(0.3)
        assert p.on(Scale.CORRECTED) == pytest.approx(p.corrected)
        with pytest.raises(TypeError):
            p.on("corrected")

    def test_cannot_be_compared_to_a_bare_float(self):
        """The whole point: a scaleless comparison must not silently succeed."""
        p = ScaledProbability.from_raw(0.9, PI_TRAIN, PI_DEPLOY)
        with pytest.raises(TypeError):
            _ = p >= 0.5

    def test_scale_enum_values_match_db_column(self):
        assert Scale.RAW.value == "raw"
        assert Scale.CORRECTED.value == "corrected"


class TestClassifyBand:
    BANDS = {"high": 0.2745, "moderate": 0.1176}  # corrected twins of 0.70 / 0.40

    def test_bands_are_ordered(self):
        assert classify_band(0.30, self.BANDS) == "HIGH"
        assert classify_band(0.15, self.BANDS) == "MODERATE"
        assert classify_band(0.05, self.BANDS) == "LOW"

    def test_boundaries_are_inclusive(self):
        assert classify_band(0.2745, self.BANDS) == "HIGH"
        assert classify_band(0.1176, self.BANDS) == "MODERATE"

    def test_missing_bands_fall_back_to_low(self):
        assert classify_band(0.99, {}) == "LOW"


class TestScaleOfResult:
    """Stored rows must be stamped with the scale that was actually used."""

    def test_diabetes_result_is_corrected(self):
        assert scale_of_result({"confidence": 0.1, "prevalence_correction_applied": True}) is Scale.CORRECTED

    def test_heart_result_is_raw(self):
        # Heart applies no correction; calling its output "corrected" would
        # claim a transformation that never happened.
        assert scale_of_result({"confidence": 0.36, "prediction": 0, "diagnosis": "Negative"}) is Scale.RAW

    def test_truthy_but_not_true_is_not_enough(self):
        assert scale_of_result({"prevalence_correction_applied": "yes"}) is Scale.RAW

    def test_config_with_both_priors_is_corrected(self):
        cfg = {"model": {"prevalence_train": 0.5, "prevalence_deploy": 0.14}}
        assert scale_of_disease_config(cfg) is Scale.CORRECTED

    def test_config_without_priors_is_raw(self):
        assert scale_of_disease_config({"model": {}}) is Scale.RAW
        assert scale_of_disease_config(None) is Scale.RAW

    def test_shipped_configs(self):
        import yaml
        with open(os.path.join(ROOT, "configs", "diabetes.yaml")) as f:
            assert scale_of_disease_config(yaml.safe_load(f)) is Scale.CORRECTED
        with open(os.path.join(ROOT, "configs", "heart_disease.yaml")) as f:
            assert scale_of_disease_config(yaml.safe_load(f)) is Scale.RAW
