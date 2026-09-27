"""
Gate 8.7c — the falsifying case for "the drift reference is anchored to the
exact training data" (reviewer protocol rule 7).

A matching `--verify` proves nothing by itself: a checker that always reports
"matches" would pass that case too. What proves the anchor is real is that a
CHANGED source, compared against the untouched committed profile, is reported
as a mismatch -- the same shape as
test_heart_glm_backend.py::test_build_refuses_a_training_csv_with_one_row_changed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_drift_reference import build_profiles, verify_profiles

ROOT = Path(__file__).resolve().parents[1]
HEART_CSV = ROOT / "data/heart_disease/processed/uci_heart_by_site.csv"
COMMITTED_PROFILE = ROOT / "models/heart_disease/drift_reference.json"


@pytest.mark.skipif(not HEART_CSV.exists(), reason="training CSV not present")
class TestVerifyCatchesADriftedSource:
    def test_the_untouched_source_matches_the_committed_profile(self):
        """The positive case, so the negative one below means something."""
        built = build_profiles({"heart_disease": HEART_CSV})
        failures = verify_profiles({"heart_disease": built["heart_disease"]},
                                   {"heart_disease": COMMITTED_PROFILE})
        assert failures == []

    def test_one_changed_cell_in_the_training_csv_fails_verify(self, tmp_path):
        """The falsifying case itself: change the data, keep the committed
        profile untouched, and --verify must refuse it -- the same one-cell
        tamper test_heart_glm_backend.py uses for the model's own fingerprint."""
        frame = pd.read_csv(HEART_CSV)
        frame.loc[0, "Cholesterol"] = (frame.loc[0, "Cholesterol"] or 0) + 1
        tampered = tmp_path / "tampered.csv"
        frame.to_csv(tampered, index=False)

        built = build_profiles({"heart_disease": tampered})
        failures = verify_profiles({"heart_disease": built["heart_disease"]},
                                   {"heart_disease": COMMITTED_PROFILE})
        assert failures, "a changed training CSV must fail verification, not pass it"
        assert "differs from the committed one" in failures[0]

    def test_a_missing_committed_profile_is_reported_by_name(self, tmp_path):
        built = build_profiles({"heart_disease": HEART_CSV})
        missing = tmp_path / "does_not_exist.json"
        failures = verify_profiles({"heart_disease": built["heart_disease"]},
                                   {"heart_disease": missing})
        assert failures and str(missing) in failures[0]
