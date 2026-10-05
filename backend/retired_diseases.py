"""
Retired disease modules — the single source of truth.

A retired module has no config in configs/, so the router no longer registers it.
Its records are not deleted: predictions, visits and review items written while
it was live stay in the database and stay readable (history, export, the review
queue, a report rendered from a stored row).

The rule every route follows:

    reading an existing record of a retired disease  -> allowed
    anything that scores, writes or retrains for it  -> 410 Gone, with the reason

410 rather than 404 because the module existed and was withdrawn on purpose, and
a client should stop asking rather than retry. Without this list, a retired name
falls through to whatever each route does with an unknown disease: today that is
a 404 on predict, a 200 on /schema (a static schema registry entry), a silent
write on POST visits, and on /admin/retrain the legacy XGBoost path, which in an
image that still holds the old weights would overwrite them (W-08).
"""

from typing import Any, Dict

from fastapi import HTTPException

RETIRED_DISEASES: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "display_name": "Diabetes Risk Assessment (BRFSS 2015, retired)",
        "retired_on": "2026-10-05",
        "replaced_by": "diabetes_nhanes",
        "replaced_by_display": "NHANES dysglycaemia module",
        "reason": (
            "The BRFSS 2015 diabetes module has been retired and replaced by the "
            "NHANES dysglycaemia module (diabetes_nhanes). Existing records remain "
            "readable; new predictions, labels and retraining are not accepted."
        ),
    },
}


def is_retired(disease: Any) -> bool:
    return isinstance(disease, str) and disease in RETIRED_DISEASES


def retired_error(disease: str) -> HTTPException:
    """The 410 every route raises for a write or a score on a retired disease."""
    info = RETIRED_DISEASES[disease]
    return HTTPException(
        status_code=410,
        detail={
            "error": f"Disease '{disease}' has been retired.",
            "code": "DISEASE_RETIRED",
            "reason": info["reason"],
            "replaced_by": info["replaced_by"],
            "retired_on": info["retired_on"],
        },
    )


def reject_if_retired(disease: Any) -> None:
    if is_retired(disease):
        raise retired_error(disease)
