"""
Gate 8.10 — heart retraining produces a reviewed CANDIDATE, not a replacement.

What each test here exists to stop:

  * A parallel training path. The candidate must be built by the same
    `stack.build_bundle()` the shipped image build calls. A retraining routine
    that reimplements the spline knots, the C, the 5-fold OOF or the IVAP fit
    agrees with the shipped model on the day it is written and drifts afterwards,
    silently, because nothing compares them again. Two tests hold this: one at
    source level (no estimator is imported here), and the CONTROL below, which
    is the one that would catch real drift.

  * The control. A candidate built with ZERO added rows is the shipped model.
    Every decision must be identical — zero movement off the diagonal of the
    transition matrix. This is the test that fails if the candidate path ever
    stops being the shipped path, and it fails by running rather than by review.

  * Writing where it must not write. The shipped bundle
    (`heart_l3_glm_stack.pkl`) and `reference_scores.json` are read-only on this
    path: the first because promotion is a human decision, the second because it
    is the record of what the shipped model decided (D-78) and a path that
    rewrites its own yardstick can never disagree with itself.

  * HF-1 coming back through the review queue. Review rows are stored in the
    API's CLINICAL chest-pain coding and the training file uses the inverted raw
    UCI coding. A row merged without translation trains on an inverted chest
    pain for that patient.

  * A patient field reaching MLflow. The run is the one part of this that leaves
    the machine. Provenance and aggregates only.

  * An invented version. `config_version` is the configured version or the
    explicit string `version_unavailable`, never a plausible-looking default.

These tests do not need mlflow: the measurement and assembly functions are
importable on their own, and only the orchestrator in
`backend.active_learning.retrain` opens a run.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from backend.heart_glm import candidate as candidate_mod
from backend.heart_glm import stack

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING_CSV = os.path.join(_ROOT, "data/heart_disease/processed/uci_heart_by_site.csv")
_REFERENCE = os.path.join(_ROOT, "backend/heart_glm/reference_scores.json")
_SHIPPED = os.path.join(_ROOT, "models/heart_disease/heart_l3_glm_stack.pkl")

def code_of(obj) -> str:
    """
    The CODE of a module or function, with docstrings and comments removed.

    Every source-level assertion below is about what the code does, and prose
    that WARNS against a pattern necessarily contains that pattern. Reading raw
    source made `_config_version`'s own docstring — which exists to say a version
    must never be invented — read as an invented version.
    """
    tree = ast.parse(inspect.getsource(obj).lstrip() if inspect.isfunction(obj)
                     else inspect.getsource(obj))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


#: A review-queue row as the database actually stores it: the API's own field
#: names, in the CLINICAL chest-pain coding. Copied in shape from a real row.
REVIEW_ROW = {
    "Age": 56, "Sex": "M", "ChestPainType": "NAP", "RestingBP": 140,
    "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Flat",
}


@pytest.fixture(scope="module")
def control_card(tmp_path_factory):
    """
    A candidate built with ZERO added rows — which is the shipped model.

    Module-scoped: the build refits the pipeline five times for the OOF plus
    once for the model, and measures blank impact over six fields, so it is paid
    for once and shared. Written to a tmp directory, never to the real
    candidates directory.
    """
    if not os.path.exists(_SHIPPED):
        pytest.skip("shipped bundle not built in this environment")
    config = {
        "disease": {"version": "7.0.0"},
        "data": {"training_csv_sha256": stack.sha256_of(_TRAINING_CSV)},
        "model": {"conformal_alpha": stack.ALPHA},
    }
    return candidate_mod.build_candidate(
        run_id="control_zero_rows",
        features_list=[],
        labels=[],
        config=config,
        candidates_dir=tmp_path_factory.mktemp("candidates"),
    )


# ── The control ──────────────────────────────────────────────────────────────

class TestControlReproducesTheShippedModel:
    def test_zero_added_rows_moves_no_patient(self, control_card):
        """
        The proof that the candidate path IS the shipped path.

        With nothing added, the candidate is a rebuild of the shipped bundle on
        the same 920 rows, so every one of the 920 decisions must be the one
        `reference_scores.json` recorded. Any movement here means the retrain
        path has diverged from `scripts/train_heart_glm.py` — which is a bug to
        find, not a tolerance to widen.
        """
        matrix = control_card["transition_matrix"]
        assert matrix["n"] == 920
        assert matrix["decisions_changed_total"] == 0, (
            f"the candidate path no longer reproduces the shipped model: "
            f"{matrix['counts']}"
        )
        assert matrix["moved_out_of_uncertain"] == 0
        assert matrix["moved_into_uncertain"] == 0
        for decision in candidate_mod.DECISIONS:
            assert matrix["shipped_totals"][decision] == matrix["candidate_totals"][decision]

    def test_the_diagonal_accounts_for_every_patient(self, control_card):
        counts = control_card["transition_matrix"]["counts"]
        diagonal = sum(counts[f"{d}__to__{d}"] for d in candidate_mod.DECISIONS)
        assert diagonal == 920
        assert sum(counts.values()) == 920

    def test_control_coverage_equals_the_shipped_model(self, control_card):
        """A rebuild of the same rows has the same cells, so the same coverage."""
        coverage = control_card["conformal_coverage"]
        for candidate_key, shipped_key in (
            ("candidate_in_sample", "shipped_in_sample"),
            ("candidate_cross_conformal", "shipped_cross_conformal"),
        ):
            assert coverage[candidate_key] == coverage[shipped_key]


# ── The candidate is a candidate ─────────────────────────────────────────────

class TestNothingShippedIsTouched:
    def test_build_leaves_the_shipped_bundle_and_reference_byte_identical(self, control_card):
        """Measured before and after the build, by the build itself."""
        assert control_card["shipped_bundle_unchanged"] is True
        assert control_card["reference_scores_unchanged"] is True
        assert control_card["shipped_bundle_sha256"] == stack.sha256_of(_SHIPPED)
        assert control_card["reference_scores_sha256"] == stack.sha256_of(_REFERENCE)

    def test_the_candidate_is_not_written_over_the_shipped_bundle(self, control_card):
        assert control_card["bundle_file"] != os.path.basename(_SHIPPED)
        assert control_card["bundle_sha256"] != control_card["shipped_bundle_sha256"]

    def test_writing_a_protected_path_is_refused(self):
        from pathlib import Path

        for protected in (candidate_mod.SHIPPED_BUNDLE_PATH, candidate_mod.REFERENCE_SCORES_PATH):
            with pytest.raises(RuntimeError, match="read-only"):
                candidate_mod._assert_writes_nothing_protected(Path(protected))

    def test_no_module_on_this_path_writes_a_protected_file(self):
        """
        Source-level: neither the candidate builder nor the pipeline that calls
        it names the shipped bundle or the reference file as a write target.

        A runtime guard catches the call that happens; this catches the call that
        was added and not exercised.
        """
        from backend.active_learning import retrain

        for module in (candidate_mod, retrain):
            source = code_of(module)
            for node in ast.walk(ast.parse(source)):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id not in ("open",):
                    continue
                mode = ""
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = keyword.value.value
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                if "w" in mode or "a" in mode:
                    target = ast.unparse(node.args[0])
                    assert "SHIPPED_BUNDLE_PATH" not in target, (
                        f"{module.__name__} opens the shipped bundle for writing: {target}"
                    )
                    assert "REFERENCE" not in target.upper(), (
                        f"{module.__name__} opens the reference scores for writing: {target}"
                    )

    def test_the_candidate_path_does_not_reload_the_live_model(self):
        """
        Promotion is not implemented, and neither is the half of promotion that
        would make a candidate live without copying a file: invalidating the
        loader so the next request lazy-loads from disk.
        """
        from backend.active_learning import retrain

        assert "invalidate" not in code_of(candidate_mod.build_candidate)
        assert "invalidate" not in code_of(retrain.retrain_candidate)


# ── The candidate is built by the shipped path, not a copy of it ─────────────

class TestNoParallelTrainingPath:
    def test_the_builder_calls_build_bundle(self):
        assert "stack.build_bundle(" in code_of(candidate_mod)

    def test_the_builder_imports_no_estimator_of_its_own(self):
        """
        Every constant that defines this model lives in stack.py. A module that
        imports LogisticRegression, SplineTransformer or IsotonicRegression here
        is reimplementing the model, which is the drift the control test would
        eventually catch and this catches at source.

        StratifiedKFold is allowed and is not model-defining: it is used to hold
        out the conformal CELL for the cross-conformal coverage estimate, with
        stack.CV_SEED and stack.N_FOLDS.
        """
        source = code_of(candidate_mod)
        for estimator in (
            "LogisticRegression", "SplineTransformer", "IsotonicRegression",
            "ColumnTransformer", "IterativeImputer", "OneHotEncoder",
        ):
            assert estimator not in source, (
                f"{estimator} is imported or named here — the candidate must be "
                f"built by stack.build_bundle(), not by a second pipeline"
            )

    def test_alpha_and_folds_come_from_the_stack(self):
        source = code_of(candidate_mod)
        assert "stack.ALPHA" in source and "stack.CV_SEED" in source
        assert "0.10" not in source


# ── Review rows become training rows correctly ───────────────────────────────

class TestReviewRowsAreTranslated:
    def test_chest_pain_is_rewritten_into_the_raw_uci_coding(self):
        """
        HF-1, arriving through the review queue.

        The queue stores what the API received: CLINICAL codes, where "TA" means
        typical angina. `encode_for_training` reads CP_MAP_UCI_RAW, in which "TA"
        means NO anginal features. Merging a row untranslated inverts that
        patient's chest pain in the training set.
        """
        frame, rejected = candidate_mod.review_rows_to_training_frame([REVIEW_ROW], [1])
        assert rejected == []
        assert len(frame) == 1
        clinical = REVIEW_ROW["ChestPainType"]
        raw = frame.loc[0, "ChestPainType"]
        assert raw != clinical or stack.CP_RAW_TO_CLINICAL[raw] == clinical
        # What must survive the rewrite: the anginal-feature COUNT.
        assert stack.CP_MAP_UCI_RAW[raw] == stack.CP_MAP_CLINICAL[clinical]

    def test_every_clinical_code_round_trips(self):
        for clinical in stack.CP_MAP_CLINICAL:
            row = dict(REVIEW_ROW, ChestPainType=clinical)
            frame, rejected = candidate_mod.review_rows_to_training_frame([row], [0])
            assert rejected == []
            raw = frame.loc[0, "ChestPainType"]
            assert stack.CP_MAP_UCI_RAW[raw] == stack.CP_MAP_CLINICAL[clinical]
        # And the translated frame is encodable by the TRAINING encoder, which is
        # the actual consumer.
        rows = [dict(REVIEW_ROW, ChestPainType=c) for c in stack.CP_MAP_CLINICAL]
        frame, _ = candidate_mod.review_rows_to_training_frame(rows, [0] * len(rows))
        encoded = stack.encode_for_training(frame)
        assert list(encoded["cp_anginal"]) == [
            stack.CP_MAP_CLINICAL[c] for c in stack.CP_MAP_CLINICAL
        ]

    def test_the_expert_label_becomes_the_target(self):
        frame, _ = candidate_mod.review_rows_to_training_frame([REVIEW_ROW, REVIEW_ROW], [1, 0])
        assert list(frame["HeartDisease"]) == [1, 0]

    def test_review_rows_get_their_own_site(self):
        """
        They have no hospital, and the OOF split stratifies on site. Folding them
        into a hospital they did not come from would put them in that hospital's
        strata and misreport the split.
        """
        frame, _ = candidate_mod.review_rows_to_training_frame([REVIEW_ROW], [1])
        assert frame.loc[0, "site"] == candidate_mod.REVIEW_SITE
        base_sites = set(pd.read_csv(_TRAINING_CSV)["site"].unique())
        assert candidate_mod.REVIEW_SITE not in base_sites

    @pytest.mark.parametrize(
        "row,label,reason",
        [
            ({k: v for k, v in REVIEW_ROW.items() if k != "Sex"}, 1, "missing"),
            (dict(REVIEW_ROW, Sex="X"), 1, "Sex"),
            (dict(REVIEW_ROW, ChestPainType="nonsense"), 1, "ChestPainType"),
            (dict(REVIEW_ROW), 7, "label"),
        ],
    )
    def test_an_unusable_row_is_rejected_not_imputed(self, row, label, reason):
        """
        An imputed Sex or ChestPainType is an invented patient, and the whole
        point of these rows is that a human looked at a real one.
        """
        frame, rejected = candidate_mod.review_rows_to_training_frame([row], [label])
        assert frame.empty
        assert len(rejected) == 1 and reason in rejected[0]["reason"]

    def test_a_blank_optional_field_is_kept_and_imputed(self):
        """RestingBP blank is ordinary: 60 of the 920 UCI rows have it blank."""
        row = {k: v for k, v in REVIEW_ROW.items() if k != "RestingBP"}
        frame, rejected = candidate_mod.review_rows_to_training_frame([row], [1])
        assert rejected == []
        assert pd.isna(frame.loc[0, "RestingBP"])

    def test_mismatched_lengths_fail(self):
        with pytest.raises(ValueError):
            candidate_mod.review_rows_to_training_frame([REVIEW_ROW], [1, 0])


class TestMergeGuards:
    def test_a_changed_base_csv_is_refused(self):
        with pytest.raises(ValueError, match="sha256"):
            candidate_mod.merge_training_data(
                _TRAINING_CSV, pd.DataFrame(), expected_base_sha256="0" * 64
            )

    def test_review_rows_are_appended_after_the_base_rows(self):
        """
        The transition matrix reads the first 920 rows against
        reference_scores.json positionally, so the base rows must keep their
        order and stay first.
        """
        frame, _ = candidate_mod.review_rows_to_training_frame([REVIEW_ROW], [1])
        merged = candidate_mod.merge_training_data(_TRAINING_CSV, frame)
        base = pd.read_csv(_TRAINING_CSV)
        assert len(merged) == len(base) + 1
        pd.testing.assert_frame_equal(merged.iloc[: len(base)], base)
        assert merged.iloc[-1]["site"] == candidate_mod.REVIEW_SITE

    def test_a_base_csv_that_already_has_review_rows_is_refused(self, tmp_path):
        base = pd.read_csv(_TRAINING_CSV)
        base.loc[0, "site"] = candidate_mod.REVIEW_SITE
        path = tmp_path / "polluted.csv"
        base.to_csv(path, index=False)
        with pytest.raises(ValueError, match=candidate_mod.REVIEW_SITE):
            candidate_mod.merge_training_data(path, pd.DataFrame())


# ── The reference file is a yardstick, read only ─────────────────────────────

class TestReferenceIsReadOnly:
    def test_a_reference_for_different_data_is_refused(self, tmp_path):
        reference = json.load(open(_REFERENCE))
        reference["training_csv_sha256"] = "0" * 64
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(reference))
        with pytest.raises(ValueError, match="different training"):
            candidate_mod.load_reference(path, stack.sha256_of(_TRAINING_CSV), 920)

    def test_a_reference_with_the_wrong_row_count_is_refused(self, tmp_path):
        reference = json.load(open(_REFERENCE))
        reference["n"] = 919
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(reference))
        with pytest.raises(ValueError, match="n=919"):
            candidate_mod.load_reference(path, reference["training_csv_sha256"], 920)


# ── The transition matrix ───────────────────────────────────────────────────

class TestTransitionMatrix:
    def test_uncertain_movement_is_counted_in_both_directions_separately(self):
        """
        Never netted. One patient the candidate commits to and one it withdraws
        from is not "no change": those are two different clinical events, and a
        net of zero would report the model as unchanged.
        """
        shipped = ["uncertain", "referral", "no_referral", "uncertain"]
        candidate = ["referral", "uncertain", "no_referral", "uncertain"]
        matrix = candidate_mod.transition_matrix(shipped, candidate)
        assert matrix["moved_out_of_uncertain"] == 1
        assert matrix["moved_into_uncertain"] == 1
        assert matrix["decisions_changed_total"] == 2

    def test_a_referral_flipping_straight_to_no_referral_is_its_own_cell(self):
        matrix = candidate_mod.transition_matrix(["referral"], ["no_referral"])
        assert matrix["counts"]["referral__to__no_referral"] == 1
        assert matrix["moved_out_of_uncertain"] == 0
        assert matrix["moved_into_uncertain"] == 0

    def test_all_nine_cells_are_always_present(self):
        matrix = candidate_mod.transition_matrix(["referral"], ["referral"])
        assert len(matrix["counts"]) == 9

    def test_an_unknown_decision_fails(self):
        with pytest.raises(ValueError):
            candidate_mod.transition_matrix(["maybe"], ["referral"])

    def test_length_mismatch_fails(self):
        with pytest.raises(ValueError):
            candidate_mod.transition_matrix(["referral"], ["referral", "referral"])


# ── Coverage ────────────────────────────────────────────────────────────────

class TestCoverageIsLabelled:
    def test_the_two_kinds_are_named_and_not_interchangeable(self, control_card):
        coverage = control_card["conformal_coverage"]
        assert coverage["candidate_in_sample"]["kind"] == "in_sample_by_construction"
        assert coverage["candidate_cross_conformal"]["kind"] == "cross_conformal_held_out_cells"

    def test_in_sample_coverage_meets_one_minus_alpha_by_construction(self, control_card):
        """
        This is the assertion that says what the figure is worth. The conformal
        quantile is the (1-alpha) quantile of exactly these scores, so this
        holding proves the quantile was taken correctly and nothing about how
        the model generalises.
        """
        coverage = control_card["conformal_coverage"]["candidate_in_sample"]
        target = control_card["conformal_coverage"]["declared_target"]
        assert coverage["marginal"] >= target - 1e-9
        for cell in ("F0", "F1", "M0", "M1"):
            assert coverage[cell] >= target - 1e-9, f"cell {cell} below 1-alpha in sample"

    def test_every_mondrian_cell_is_reported_with_its_size(self, control_card):
        """
        Per cell, not only marginal: Gate 6 met the marginal figure while
        diseased women at Hungarian sat at 0.58. And with its n, because a cell
        of 50 diseased women cannot support a coverage figure read to two places.
        """
        for which in ("candidate_in_sample", "candidate_cross_conformal"):
            coverage = control_card["conformal_coverage"][which]
            assert "marginal" in coverage
            for cell in ("F0", "F1", "M0", "M1"):
                assert cell in coverage
                assert coverage[f"n_{cell}"] > 0
            assert sum(coverage[f"n_{cell}"] for cell in ("F0", "F1", "M0", "M1")) == 920

    def test_the_declared_target_is_one_minus_the_bundles_own_alpha(self, control_card):
        assert control_card["conformal_coverage"]["declared_target"] == pytest.approx(
            1.0 - control_card["conformal_alpha"]
        )
        assert control_card["conformal_alpha"] == stack.ALPHA


# ── What reaches MLflow ─────────────────────────────────────────────────────

class TestMlflowPayloadCarriesNoPatient:
    def test_no_patient_field_name_appears_in_any_key_or_value(self, control_card):
        """
        No key and no value names a patient field.

        Matched on word boundaries rather than as a substring: `coverage`
        contains "age", and a test that cannot tell those apart is one that gets
        relaxed the first time it cries wolf.
        """
        import re

        params, metrics = candidate_mod.mlflow_payload(control_card)
        subjects = [("key", k) for k in list(params) + list(metrics)]
        subjects += [
            ("value", str(v)) for k, v in params.items() if k != "scientific_limit"
        ]
        for field in stack.INPUT_FEATURES + stack.UNUSED_INPUT_FEATURES:
            pattern = re.compile(rf"\b{re.escape(field)}\b", re.IGNORECASE)
            for kind, text in subjects:
                assert not pattern.search(text), (
                    f"patient field {field} reached the MLflow payload as a {kind}: {text}"
                )

    def test_no_per_patient_sequence_reaches_the_payload(self, control_card):
        """
        The shape that would leak 920 rows: a list or dict of per-patient values
        forwarded as a param. Every param is a scalar string or number.
        """
        params, _ = candidate_mod.mlflow_payload(control_card)
        for key, value in params.items():
            assert isinstance(value, (str, int, float)), f"{key} is {type(value)}"
            assert not isinstance(value, (list, dict, tuple))

    def test_every_metric_is_a_float_aggregate(self, control_card):
        _, metrics = candidate_mod.mlflow_payload(control_card)
        assert metrics
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is {type(value)}"

    def test_the_payload_is_a_closed_set_built_by_naming_fields(self):
        """
        Nothing iterates over the card and forwards what it finds: a field added
        to the card later must be added here deliberately, or it does not leave
        the machine. Guarded at source because the failure mode is a future edit.
        """
        source = code_of(candidate_mod.mlflow_payload)
        assert "card.items()" not in source
        assert "**card" not in source

    def test_the_two_uncertain_movements_are_logged_as_independent_metrics(self, control_card):
        _, metrics = candidate_mod.mlflow_payload(control_card)
        assert "moved_out_of_uncertain" in metrics
        assert "moved_into_uncertain" in metrics

    def test_the_whole_matrix_is_logged_cell_by_cell(self, control_card):
        _, metrics = candidate_mod.mlflow_payload(control_card)
        cells = [k for k in metrics if k.startswith("transition_")]
        assert len(cells) == 9

    def test_both_coverage_kinds_are_logged_for_candidate_and_shipped(self, control_card):
        _, metrics = candidate_mod.mlflow_payload(control_card)
        for which in (
            "candidate_in_sample", "candidate_cross_conformal",
            "shipped_in_sample", "shipped_cross_conformal",
        ):
            assert f"coverage_{which}_marginal" in metrics
            for cell in ("F0", "F1", "M0", "M1"):
                assert f"coverage_{which}_{cell}" in metrics
        assert "coverage_declared_target_1_minus_alpha" in metrics

    def test_the_lineage_hashes_are_params(self, control_card):
        params, _ = candidate_mod.mlflow_payload(control_card)
        for key in (
            "candidate_bundle_sha256", "training_data_sha256",
            "base_training_csv_sha256", "shipped_bundle_sha256",
            "reference_scores_sha256", "config_version", "conformal_alpha",
        ):
            assert key in params

    def test_the_run_says_it_is_a_candidate(self, control_card):
        params, _ = candidate_mod.mlflow_payload(control_card)
        assert params["is_candidate_not_replacement"] == "yes"
        assert params["shipped_bundle_replaced"] == "no"
        assert params["shipped_bundle_unchanged"] == "yes"
        assert params["reference_scores_unchanged"] == "yes"


# ── The scientific limit travels with the numbers ───────────────────────────

class TestTheLimitIsInTheArtifact:
    def test_the_card_carries_it(self, control_card):
        assert control_card["limit"] == candidate_mod.CANDIDATE_LIMIT

    def test_the_mlflow_run_carries_it_as_its_own_field(self, control_card):
        params, _ = candidate_mod.mlflow_payload(control_card)
        assert params["scientific_limit"] == candidate_mod.CANDIDATE_LIMIT

    def test_it_states_the_selection_and_the_consequence(self):
        """
        Not a vague disclaimer. It must name WHY the rows are not a sample
        (uncertainty selection, no catheterisation cohort) and WHAT that costs
        (the conformal guarantee is over a different mix).
        """
        limit = candidate_mod.CANDIDATE_LIMIT.lower()
        assert "uncertainty" in limit
        assert "catheterisation" in limit
        assert "conformal" in limit and "mix" in limit


# ── Version provenance ──────────────────────────────────────────────────────

class TestVersionIsNeverInvented:
    def test_a_missing_version_is_stated_explicitly(self):
        for config in ({}, {"disease": {}}, {"disease": {"version": ""}}, None):
            assert candidate_mod._config_version(config) == candidate_mod.VERSION_UNAVAILABLE

    def test_a_present_version_is_used_verbatim(self):
        assert candidate_mod._config_version({"disease": {"version": "9.9.9"}}) == "9.9.9"

    def test_no_version_shaped_default_appears_in_the_module(self):
        """
        The pattern this must not repeat: `card.get("version", "v7.0.0")`, which
        invents a version because the dict it reads has none. A default that
        looks like a real version is worse than a missing field, because it is
        indistinguishable from provenance.
        """
        import re

        source = code_of(candidate_mod)
        for match in re.finditer(r"'v?\d+\.\d+\.\d+'", source):
            line = source[: match.start()].count("\n") + 1
            pytest.fail(f"version-shaped literal {match.group()} at line {line}")

    def test_the_real_config_supplies_one(self):
        from backend.active_learning.retrain import load_disease_config

        config = load_disease_config("heart_disease")
        version = candidate_mod._config_version(config)
        assert version != candidate_mod.VERSION_UNAVAILABLE
        assert version == config["disease"]["version"]


# ── The pipeline routes by family, not by disease name ──────────────────────

class TestRoutingIsByModelFamily:
    def test_heart_is_routed_to_the_candidate_path(self):
        from backend.active_learning.retrain import CANDIDATE_FAMILIES, _model_family

        assert _model_family("heart_disease") in CANDIDATE_FAMILIES

    def test_nhanes_is_not(self):
        """BRFSS diabetes is retired (gate B3) and has no family at all; NHANES is
        the live module that is not on the candidate path."""
        from backend.active_learning.retrain import CANDIDATE_FAMILIES, _model_family

        assert _model_family("diabetes_nhanes") == "ebm_platt_conformal"
        assert _model_family("diabetes_nhanes") not in CANDIDATE_FAMILIES
        assert _model_family("diabetes") is None

    def test_the_branch_reads_the_config_and_not_the_disease_name(self):
        """
        `retrain_xgb` was wrong because it assumed the family from nothing. A
        branch on `disease == "heart_disease"` would be the same mistake spelled
        differently: the next module of this family would land on the XGBoost
        path by default.
        """
        from backend.active_learning import retrain

        source = code_of(retrain.run_retrain_pipeline)
        assert "_model_family" in source
        assert 'disease == "heart_disease"' not in source
        assert "CANDIDATE_FAMILIES" in source

    def test_the_candidate_path_reports_that_it_promoted_nothing(self):
        from backend.active_learning import retrain

        source = code_of(retrain.retrain_candidate)
        assert "candidate_built_not_promoted" in source
        assert "promote" not in source.replace("candidate_built_not_promoted", "")

    @pytest.mark.parametrize("disease,family", [
        ("diabetes_nhanes", None),                 # its own config: ebm_platt_conformal
        ("heart_disease", "sklearn_pipeline"),     # the two-line revert to the old XGBoost
    ], ids=["nhanes", "heart-revert-sklearn_pipeline"])
    async def test_every_other_family_is_refused_before_reading(self, monkeypatch, disease, family):
        """
        Gate B4 removed `retrain_xgb`, the writer every non-candidate family used
        to reach: no live module read what it wrote (the W-08 pattern). Such a
        family is now refused with a stated reason, before a single review row
        is read -- which is also what the admin page displays (status + reason).
        """
        from backend.active_learning import retrain

        if family is not None:   # simulate configs/heart_disease.yaml reverted
            real = retrain.load_disease_config

            def reverted(d):
                cfg = real(d)
                if d == disease:
                    cfg = {**cfg, "model": {**cfg["model"], "family": family}}
                return cfg

            monkeypatch.setattr(retrain, "load_disease_config", reverted)
        expected = family or "ebm_platt_conformal"

        def boom(*a, **k):
            raise AssertionError("an unsupported family read samples or built a model")

        monkeypatch.setattr(retrain, "get_annotated_samples", boom)
        monkeypatch.setattr(retrain, "retrain_candidate", boom)
        result = await retrain.run_retrain_pipeline(None, disease, 1)
        assert result["status"] == "unsupported"
        assert result["reason"] == f"retrain not yet supported for {expected}"
        assert result["samples_used"] == 0

    def test_the_legacy_xgboost_writer_is_gone(self):
        from backend.active_learning import retrain

        assert not hasattr(retrain, "retrain_xgb")
        assert "omni_diag_xgb_optimized.pkl" not in code_of(retrain.run_retrain_pipeline)


# ── The candidate bundle is a working bundle ─────────────────────────────────

class TestTheCandidateIsUsable:
    def test_it_loads_and_scores_through_the_ordinary_inference_path(self, control_card, tmp_path_factory):
        """
        A candidate that cannot be loaded and scored is not reviewable. Scored
        through `encode_for_inference` — the clinical coding — because that is
        what would score a patient if it were ever promoted.
        """
        directory = str(control_card["candidate_dir"])
        if not os.path.isabs(directory):
            directory = os.path.join(_ROOT, directory)
        path = os.path.join(directory, control_card["bundle_file"])
        with open(path, "rb") as handle:
            bundle = pickle.load(handle)

        encoded = stack.encode_for_inference([REVIEW_ROW])
        raw = bundle["pipeline"].predict_proba(encoded)[:, 1]
        assert 0.0 <= float(raw[0]) <= 1.0
        decision, _ = stack.decide(float(raw[0]), REVIEW_ROW["Sex"], bundle["conformal_cells"])
        assert decision in candidate_mod.DECISIONS

    def test_its_card_counts_the_rows_it_was_trained_on(self, control_card):
        """
        The model card's row count is counted from the frame, not typed. A
        hardcoded 920 would have made every candidate card state a row count the
        candidate was not trained on.
        """
        assert control_card["rows_total"] == 920
        assert control_card["rows_base_uci"] == 920
        assert control_card["rows_from_review_queue"] == 0

    def test_the_model_card_row_count_follows_the_data(self, tmp_path):
        """stack.build_bundle's card, on a smaller file, must say the smaller number."""
        frame = pd.read_csv(_TRAINING_CSV)
        subset = frame.groupby(["site", "Sex", "HeartDisease"], group_keys=False).head(12)
        path = tmp_path / "subset.csv"
        subset.to_csv(path, index=False)
        bundle = stack.build_bundle(path)
        card = bundle["model_card"]
        assert f"{len(subset)} patients" in card["training_data"]
        assert f"fit on all {len(subset)}" in card["split"]
        assert "920" not in card["training_data"]
