"""
OmniDiag — Probability scale contract
=====================================
Every stored probability says which scale it is on, in the `probability_scale`
column of predictions and the `uncertainty_scale` column of the review queue:

    raw        the model's own output, on the scale it was calibrated to
    corrected  a prior-shift (prevalence) corrected probability

Every live module reports RAW: heart (`ivap_calibrated_training_mix`) and the
NHANES dysglycaemia module (`platt_calibrated_nhanes_2015_2016`). CORRECTED is
kept as a legal value because rows written by the retired BRFSS diabetes module
carry it; they are read back as stored strings and never re-scaled.

Prevalence correction itself was removed with that module (gate B5): the router
no longer maps a config's risk bands onto a deployment prior, and nothing
corrects a probability before it is returned. So a module that *declared*
priors or bands would be labelled CORRECTED while nothing corrected it -- a
silent mislabel across the API, the database and the report. Instead such a
config is refused when it is loaded (`PrevalenceCorrectionUnsupported`), and a
result that claims a correction was applied is refused when it is stamped.
"""

from enum import Enum
from typing import Mapping


class Scale(str, Enum):
    """The two probability scales. Values match the DB column values."""

    RAW = "raw"
    CORRECTED = "corrected"


class PrevalenceCorrectionUnsupported(ValueError):
    """A config or a result asks for a prevalence correction, which no longer exists."""


UNSUPPORTED_MESSAGE = "prevalence correction is not supported (removed in B5)"

#: Config keys that only meant something together with the correction.
_CORRECTION_KEYS = ("prevalence_train", "prevalence_deploy", "risk_bands")


def scale_of_result(result: Mapping) -> Scale:
    """
    The scale a module's predict()/explain() result is stated on: RAW.

    A result claiming `prevalence_correction_applied: True` is refused rather
    than stamped CORRECTED -- no code path can produce one any more, so it
    would be a mislabel.
    """
    if result.get("prevalence_correction_applied") is True:
        raise PrevalenceCorrectionUnsupported(UNSUPPORTED_MESSAGE)
    return Scale.RAW


def scale_of_disease_config(disease_config: Mapping | None) -> Scale:
    """
    The scale a disease reports, read from its config: RAW.

    Called when the router loads a config, so a config declaring prevalence
    priors or risk bands fails at load instead of being served with a scale
    label nothing honours. `risk_bands: null` (heart, NHANES) declares none.
    """
    model_cfg = (disease_config or {}).get("model", {}) or {}
    declared = [k for k in _CORRECTION_KEYS if model_cfg.get(k) is not None]
    if declared:
        raise PrevalenceCorrectionUnsupported(
            f"{UNSUPPORTED_MESSAGE}: model.{', model.'.join(declared)} declared"
        )
    return Scale.RAW
