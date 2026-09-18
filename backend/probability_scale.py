"""
OmniDiag — Probability scale contract
=====================================
Every probability the diabetes module produces exists on exactly one of two
scales, and the two are NOT interchangeable:

    raw        the ensemble's own output, on the training prior (50/50 resample)
    corrected  the same quantity mapped to the deployment prior (~14% BRFSS),
               via backend/prevalence_correction.py

The governing rule for this codebase:

    *Every probability leaving EnsembleModelLoader.predict() is CORRECTED by
    default. A raw value must carry an explicit `_raw` suffix everywhere it
    travels — variable, JSON field, database column, function name.*

A bare `float` cannot carry that information, so the critical internal paths
pass a `ScaledProbability` instead. It is a tuple subclass on purpose: it has
no `__float__` and no ordering against numbers, so

    p >= 0.5                      # TypeError
    should_queue_for_review(p)    # TypeError inside, unless .corrected is used

fail loudly rather than silently comparing two different scales. Call sites
must name the scale they want: `p.corrected` or `p.raw`.

This module is deliberately dependency-light (only prevalence_correction) so
it can be imported from loaders, routes and tests alike.
"""

from enum import Enum
from typing import Mapping, NamedTuple

from backend.prevalence_correction import (
    apply_prevalence_correction,
    invert_prevalence_correction,
)


class Scale(str, Enum):
    """The two probability scales in the system. Values match the DB enum."""

    RAW = "raw"
    CORRECTED = "corrected"


class ScaledProbability(NamedTuple):
    """
    One probability, known on both scales, with the priors that relate them.

    Built through `from_raw` / `from_corrected` so the pair can never drift
    apart. Read with `.raw` / `.corrected`, or `.on(scale)` when the scale is
    itself a variable.
    """

    raw: float
    corrected: float
    prevalence_train: float
    prevalence_deploy: float

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def from_raw(
        cls, raw: float, prevalence_train: float, prevalence_deploy: float
    ) -> "ScaledProbability":
        corrected = float(
            apply_prevalence_correction(raw, prevalence_train, prevalence_deploy)
        )
        return cls(float(raw), corrected, float(prevalence_train), float(prevalence_deploy))

    @classmethod
    def from_corrected(
        cls, corrected: float, prevalence_train: float, prevalence_deploy: float
    ) -> "ScaledProbability":
        raw = float(
            invert_prevalence_correction(corrected, prevalence_train, prevalence_deploy)
        )
        return cls(raw, float(corrected), float(prevalence_train), float(prevalence_deploy))

    # ── accessors ───────────────────────────────────────────────────────
    def on(self, scale: Scale) -> float:
        """Return the value on `scale`. Raises on anything that is not a Scale."""
        if scale is Scale.RAW:
            return self.raw
        if scale is Scale.CORRECTED:
            return self.corrected
        raise TypeError(f"expected a Scale, got {scale!r}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ScaledProbability(raw={self.raw:.6f}, corrected={self.corrected:.6f}, "
            f"pi={self.prevalence_train}->{self.prevalence_deploy})"
        )


def classify_band(probability_corrected: float, risk_bands: Mapping[str, float]) -> str:
    """
    Display band (HIGH / MODERATE / LOW) for a CORRECTED probability.

    `risk_bands` must already be on the corrected scale — that is what
    `EnsembleModelLoader.predict()` returns in `risk_bands` and what
    `GET /api/v4/diseases` returns in `info.risk_bands`. This is the backend
    twin of `classifyRisk()` in frontend/src/constants/thresholds.js; the two
    must stay in step.

    Bands are optional: a disease that configures none gets MODERATE-free
    behaviour driven purely by whatever caller-supplied cut-points exist.
    """
    high = risk_bands.get("high")
    moderate = risk_bands.get("moderate")
    if high is not None and probability_corrected >= high:
        return "HIGH"
    if moderate is not None and probability_corrected >= moderate:
        return "MODERATE"
    return "LOW"


def scale_of_result(result: Mapping) -> Scale:
    """
    The scale a loader's predict()/explain() result is stated on.

    A module that applied the prevalence correction says so with
    `prevalence_correction_applied: True`; its probabilities are CORRECTED.
    A module that did not (heart_disease) reports the model's own output,
    which is RAW by definition — there is no other scale for it to be on.
    Stamping heart rows as "corrected" would claim a correction that never
    happened.
    """
    return Scale.CORRECTED if result.get("prevalence_correction_applied") is True else Scale.RAW


def scale_of_disease_config(disease_config: Mapping | None) -> Scale:
    """
    The scale a disease reports, read from its config: CORRECTED when both
    prevalence priors are declared (EnsembleModelLoader then refuses to run
    without applying the correction), RAW otherwise.
    """
    model_cfg = (disease_config or {}).get("model", {}) or {}
    if {"prevalence_train", "prevalence_deploy"} <= set(model_cfg):
        return Scale.CORRECTED
    return Scale.RAW
