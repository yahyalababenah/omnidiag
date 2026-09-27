"""
Gate 8.7c — input drift on scipy: KS, chi-square, PSI.

The seeded-drift test and its negative control are the point of this file. A
function that answers "drift" for everything passes the seeded test alone, so the
control that must NOT flag is what makes the seeded one mean anything.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.monitoring.drift_stats import (
    MIN_CURRENT_ROWS, PSI_MAJOR, bin_counts, chi2_test, combine_p, compare_feature,
    expand_ecdf, flag_features, holm_bonferroni, ks_test, missingness_test, psi,
    quantile_edges,
)

ROOT = Path(__file__).resolve().parents[1]
HEART_CSV = ROOT / "data/heart_disease/processed/uci_heart_by_site.csv"

MONITORED_KINDS = {
    "Age": "numeric", "RestingBP": "numeric", "Cholesterol": "numeric",
    "Sex_m": "categorical", "cp_anginal": "categorical",
    "FastingBS_cat": "categorical", "RestingECG": "categorical",
}


# ── PSI, by hand ──────────────────────────────────────────────────────────────

class TestPSI:
    def test_matches_a_hand_computed_value(self):
        # reference 50/30/20, current 20/30/50, on 100 each
        ref, cur = [50, 30, 20], [20, 30, 50]
        expected = sum((c / 100 - r / 100) * math.log((c / 100) / (r / 100))
                       for r, c in zip(ref, cur))
        assert psi(ref, cur) == pytest.approx(expected, abs=1e-12)
        assert psi(ref, cur) == pytest.approx(0.549774, abs=1e-5)   # 2 * 0.3 * ln(2.5)

    def test_is_zero_for_identical_distributions(self):
        assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0, abs=1e-12)

    def test_is_symmetric(self):
        assert psi([50, 30, 20], [20, 30, 50]) == pytest.approx(psi([20, 30, 50], [50, 30, 20]))

    def test_is_scale_free(self):
        """Doubling one side's sample size must not change PSI."""
        assert psi([50, 30, 20], [40, 60, 100]) == pytest.approx(psi([50, 30, 20], [20, 30, 50]))

    def test_an_empty_reference_bin_is_floored_not_dropped(self):
        """A bin the reference never filled is the drift worth seeing."""
        value = psi([100, 0], [50, 50])
        assert value > PSI_MAJOR and math.isfinite(value)

    def test_refuses_an_empty_distribution(self):
        with pytest.raises(ValueError):
            psi([0, 0], [1, 1])

    def test_refuses_mismatched_bins(self):
        with pytest.raises(ValueError):
            psi([1, 2, 3], [1, 2])


# ── the wrappers do not change scipy's numbers ────────────────────────────────

class TestWrappersMatchScipy:
    def test_ks_matches_scipy_directly(self):
        from scipy import stats
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=300), rng.normal(0.5, size=200)
        mine = ks_test(a, b)
        theirs = stats.ks_2samp(a, b)
        assert mine["statistic"] == pytest.approx(theirs.statistic)
        assert mine["p_value"] == pytest.approx(theirs.pvalue)

    def test_chi2_matches_scipy_directly(self):
        from scipy import stats
        ref = ["a"] * 60 + ["b"] * 40
        cur = ["a"] * 20 + ["b"] * 80
        mine = chi2_test(ref, cur)
        theirs = stats.chi2_contingency(np.array([[60, 40], [20, 80]], dtype=float))
        assert mine["p_value"] == pytest.approx(theirs[1])

    def test_ks_reports_none_rather_than_guessing_on_too_few_values(self):
        out = ks_test([1.0], [2.0])
        assert out["p_value"] is None and "too few" in out["detail"]

    def test_missing_is_its_own_category_not_dropped(self):
        out = chi2_test(["a"] * 90 + [None] * 10, ["a"] * 10 + [None] * 90)
        assert "__missing__" in out["categories"]
        assert out["p_value"] < 1e-20

    def test_low_expected_cells_are_reported_not_hidden(self):
        out = chi2_test(["a"] * 100 + ["b"] * 2, ["a"] * 10 + ["b"] * 1)
        assert out["low_expected_cells"] >= 1
        assert "unreliable" in out["detail"]

    def test_expand_ecdf_rebuilds_the_exact_sample(self):
        got = expand_ecdf([1.0, 2.0, 3.0], [2, 1, 3])
        assert got.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]

    def test_expand_ecdf_refuses_a_mismatch(self):
        with pytest.raises(ValueError):
            expand_ecdf([1.0, 2.0], [1])


# ── multiple testing ──────────────────────────────────────────────────────────

class TestHolmBonferroni:
    def test_one_marginal_p_among_seven_is_not_significant(self):
        """The false alarm the correction exists to stop."""
        ps = {"a": 0.04, **{k: 0.9 for k in "bcdefg"}}
        out = holm_bonferroni(ps, alpha=0.05)
        assert out["a"]["p_adjusted"] == pytest.approx(0.28)
        assert not out["a"]["significant"]

    def test_a_strong_p_survives_the_correction(self):
        out = holm_bonferroni({"a": 1e-9, **{k: 0.9 for k in "bcdefg"}}, alpha=0.05)
        assert out["a"]["significant"]

    def test_adjusted_values_never_decrease_down_the_ranking(self):
        out = holm_bonferroni({"a": 0.001, "b": 0.002, "c": 0.003, "d": 0.9})
        adj = [out[k]["p_adjusted"] for k in ("a", "b", "c", "d")]
        assert adj == sorted(adj)

    def test_untested_features_are_carried_through_not_counted(self):
        out = holm_bonferroni({"a": 0.01, "b": None})
        assert out["b"]["untested"] and out["b"]["p_value"] is None
        assert out["a"]["p_adjusted"] == pytest.approx(0.01)   # family of one

    def test_combine_p_penalises_taking_the_minimum_of_two_tests(self):
        assert combine_p([0.01, 0.4]) == pytest.approx(0.02)
        assert combine_p([0.01, None]) == pytest.approx(0.01)
        assert combine_p([None, None]) is None
        assert combine_p([0.9, 0.8]) == pytest.approx(1.0)


# ── the decision rule needs BOTH conditions ───────────────────────────────────

class TestTwoConditionRule:
    def _comp(self, name, p, psi_value):
        return {"feature": name, "kind": "numeric", "p_value": p, "psi": psi_value}

    def test_significant_but_tiny_effect_is_not_flagged(self):
        out = flag_features([self._comp("a", 1e-12, 0.01)])
        assert out["n_drifted"] == 0 and out["drift_share"] == 0.0

    def test_large_effect_without_significance_is_not_flagged(self):
        out = flag_features([self._comp("a", 0.9, 5.0)])
        assert out["n_drifted"] == 0

    def test_both_conditions_flag(self):
        out = flag_features([self._comp("a", 1e-12, 5.0)])
        assert out["drifted_features"] == ["a"] and out["drift_share"] == 1.0

    def test_an_untested_feature_is_never_flagged(self):
        out = flag_features([self._comp("a", None, 99.0)])
        assert out["n_drifted"] == 0

    def test_share_is_over_monitored_features(self):
        out = flag_features([self._comp("a", 1e-12, 5.0), self._comp("b", 0.9, 0.0),
                             self._comp("c", 0.9, 0.0), self._comp("d", 0.9, 0.0)])
        assert out["drift_share"] == pytest.approx(0.25)
        assert out["n_monitored"] == 4


# ── missingness: the fault the seeded test exposed ────────────────────────────

class TestMissingnessIsDrift:
    def test_a_feature_that_stopped_being_measured_is_caught(self):
        """KS sees finite values only, so this is invisible to it."""
        ref_counts = [20, 20, 20, 20, 20, 0]      # last bin = missing
        cur_counts = [0, 0, 0, 0, 0, 100]
        out = missingness_test(ref_counts, cur_counts)
        assert out["p_value"] < 1e-20
        assert out["current_missing_share"] == pytest.approx(1.0)

    def test_numeric_comparison_flags_all_missing_even_when_ks_cannot_run(self):
        ref = [float(x) for x in range(100)]
        edges = quantile_edges(ref)
        comp = compare_feature("Chol", "numeric", ref, [float("nan")] * 100,
                               edges=edges, reference_counts=bin_counts(ref, edges))
        assert comp["ks"]["p_value"] is None          # KS could not run
        assert comp["p_value"] is not None            # the feature still has one
        assert flag_features([comp])["n_drifted"] == 1

    def test_no_missingness_on_either_side_is_reported_as_untestable(self):
        out = missingness_test([50, 50, 0], [10, 10, 0])
        assert out["p_value"] is None and "no variation" in out["detail"]


# ── the seeded drift, and the control that gives it meaning ───────────────────

def _encoded_by_site():
    from backend.heart_glm.stack import encode_for_training
    frame = pd.read_csv(HEART_CSV)
    encoded = encode_for_training(frame)
    encoded["site"] = frame["site"].to_numpy()
    return encoded


def _compare(reference: pd.DataFrame, current: pd.DataFrame):
    comps = []
    for name, kind in MONITORED_KINDS.items():
        if kind == "numeric":
            edges = quantile_edges(reference[name].tolist())
            comps.append(compare_feature(name, kind, reference[name].tolist(),
                                         current[name].tolist(), edges=edges,
                                         reference_counts=bin_counts(reference[name].tolist(), edges)))
        else:
            comps.append(compare_feature(name, kind, reference[name].tolist(),
                                         current[name].tolist()))
    return flag_features(comps)


@pytest.mark.skipif(not HEART_CSV.exists(), reason="training CSV not present")
class TestSeededDrift:
    def test_zurich_against_cleveland_is_detected(self):
        e = _encoded_by_site()
        out = _compare(e[e.site == "cleveland"], e[e.site == "switzerland"])
        assert out["n_drifted"] >= 3, out["drifted_features"]
        # The two shifts that are documented facts about Zurich: 94 % prevalence
        # and cholesterol recorded for almost nobody.
        assert "Cholesterol" in out["drifted_features"]
        assert "FastingBS_cat" in out["drifted_features"]

    def test_cleveland_against_its_own_other_half_is_not_flagged(self):
        """The control. Without it, a function that always says "drift" passes above."""
        e = _encoded_by_site()
        cle = e[e.site == "cleveland"]
        idx = np.random.default_rng(42).permutation(len(cle))
        a, b = cle.iloc[idx[: len(cle) // 2]], cle.iloc[idx[len(cle) // 2:]]
        out = _compare(a, b)
        assert out["n_drifted"] == 0, out["drifted_features"]
        assert out["drift_share"] == 0.0


# ── the monitor, end to end ───────────────────────────────────────────────────

def _api_rows(site: str) -> pd.DataFrame:
    """Rows as the API would receive them: CLINICAL chest-pain coding."""
    from backend.heart_glm.stack import CP_MAP_CLINICAL, CP_MAP_UCI_RAW
    frame = pd.read_csv(HEART_CSV)
    rows = frame[frame.site == site].copy()
    inverse = {v: k for k, v in CP_MAP_CLINICAL.items()}
    rows["ChestPainType"] = rows["ChestPainType"].map(CP_MAP_UCI_RAW).map(inverse)
    return rows


@pytest.mark.skipif(not HEART_CSV.exists(), reason="training CSV not present")
class TestProfileMonitor:
    def test_the_monitor_is_ready_and_reports_the_reference_it_used(self):
        from backend.monitoring.drift import get_monitor
        m = get_monitor("heart_disease")
        assert m.is_ready and m.unavailable_reason() is None
        report = m.run(_api_rows("switzerland"))
        assert report["status"] == "ok"
        assert report["reference_rows"] == 920
        assert report["metrics"]["dataset_drift"]["number_of_columns"] == 7

    def test_the_reference_is_the_csv_the_model_was_built_from(self):
        import hashlib
        profile = json.load(open(ROOT / "models/heart_disease/drift_reference.json"))
        digest = hashlib.sha256(HEART_CSV.read_bytes()).hexdigest()
        assert profile["source"]["sha256"] == digest
        assert profile["source"]["rows"] == 920

    def test_only_the_seven_features_the_model_reads_are_monitored(self):
        from backend.heart_glm.stack import MODEL_FEATURES
        profile = json.load(open(ROOT / "models/heart_disease/drift_reference.json"))
        assert set(profile["monitored"]) == set(MODEL_FEATURES)
        assert set(profile["profiled_but_not_monitored"]) == {
            "MaxHR", "Oldpeak", "ExerciseAngina", "ST_Slope"}

    def test_a_field_the_model_ignores_cannot_move_the_drift_share(self):
        """Blanking these changes 0 of 920 decisions (Gate 8.3), so they must not alarm."""
        from backend.monitoring.drift import get_monitor
        rows = _api_rows("cleveland")
        baseline = get_monitor("heart_disease").run(rows)
        wrecked = rows.copy()
        for f in ("MaxHR", "Oldpeak", "ExerciseAngina", "ST_Slope"):
            wrecked[f] = None
        after = get_monitor("heart_disease").run(wrecked)
        assert (after["metrics"]["dataset_drift"]["drift_share"]
                == baseline["metrics"]["dataset_drift"]["drift_share"])

    def test_too_few_rows_says_so_instead_of_reporting_no_drift(self):
        """A share of 0.0 on twelve rows is the lie `count: 0` told about MLflow."""
        from backend.monitoring.drift import get_monitor
        report = get_monitor("heart_disease").run(_api_rows("cleveland").head(12))
        assert report["status"] == "insufficient_data"
        assert report["metrics"]["dataset_drift"]["drift_share"] is None
        assert "12" in report["detail"]

    def test_rows_that_cannot_be_encoded_are_named_not_swallowed(self):
        from backend.monitoring.drift import get_monitor
        rows = _api_rows("cleveland").drop(columns=["ChestPainType"])
        report = get_monitor("heart_disease").run(rows)
        assert report["status"] == "cannot_encode"
        assert "ChestPainType" in report["detail"]

    def test_the_report_states_that_it_is_input_drift_only(self):
        from backend.monitoring.drift import get_monitor
        report = get_monitor("heart_disease").run(_api_rows("cleveland"))
        assert "not model degradation" in report["measures"]


class TestDiabetesReferenceIsReweighted:
    def test_the_reference_declares_its_reweighting_and_its_limit(self):
        profile = json.load(open(ROOT / "models/diabetes/drift_reference.json"))
        rw = profile["reweighting"]
        assert rw["applied"] is True
        assert rw["prevalence_of_sample"] == 0.50
        assert rw["prevalence_of_reference"] == 0.237
        assert "not BRFSS" in rw["declared_limit"].replace("NOT BRFSS", "not BRFSS")
        assert "0.108184" in rw["unchanged"]

    def test_reweighting_moves_a_correlated_feature_off_the_5050_value(self):
        """The false drift the 50/50 reference would have reported."""
        profile = json.load(open(ROOT / "models/diabetes/drift_reference.json"))
        counts = profile["monitored"]["HighBP"]["counts"]
        reweighted_share = counts[1] / sum(counts)
        raw = pd.read_csv(ROOT / "data/diabetes/raw/"
                          "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
                          usecols=["HighBP", "Diabetes_binary"])
        assert (raw.HighBP == 1).mean() == pytest.approx(0.5635, abs=1e-3)
        assert reweighted_share == pytest.approx(0.4639, abs=1e-3)
        assert abs(reweighted_share - (raw.HighBP == 1).mean()) > 0.09
