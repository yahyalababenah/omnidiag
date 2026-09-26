"""
Gate 8.1 — the shipped heart model.

What these tests protect, and why each one exists:

  * HF-1 (S1). UCI's chest-pain codes are inverted with respect to their own
    documentation, and the previous model inherited that: a clinician choosing
    "Typical Angina" got a LOWER risk than "Asymptomatic" for 100% of patients,
    flipping 26.4% of decisions. The API keeps the clinical meaning, so the fix
    lives in the bundle's two maps — and the test runs through the API, not the
    model, because that is where a clinician meets it.

    It asserts ONLY typical > no-anginal-features. The full clinical order does
    NOT hold and is not claimed: the unconstrained GLM learns
    typical 0.736 > none 0.444 > atypical 0.389 > non-anginal 0.242 (measured
    on all 920), because "no anginal features yet catheterised anyway" carries
    real referral signal in this cohort. That is inherited from how the data
    were collected, not a clinical ordering, and it is stated in the config and
    the defence document rather than constrained away.

  * The two maps never mix. A single crossed import would restore HF-1
    silently, so it is checked at source level.

  * The bundle reproduces the research artifact. The build script's fingerprint
    check is the gate; this is the same check from the test suite's side.

  * SHAP is exact, not approximate. Shipping an approximate explanation while
    calling it SHAP is the same class of defect as F0-1.

  * Importing every backend module keeps startup alive. The package imports its
    folder, so one broken module takes down BOTH diseases.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import pkgutil
import time

import numpy as np
import pandas as pd
import pytest

from backend.heart_glm import stack
from backend.model_backends import registered_families
from backend.router import OmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING_CSV = os.path.join(_ROOT, "data/heart_disease/processed/uci_heart_by_site.csv")
_REFERENCE = os.path.join(_ROOT, "backend/heart_glm/reference_scores.json")

PATIENT = {
    "Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
    "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Flat",
}


@pytest.fixture(scope="module")
def router() -> OmniDiagRouter:
    return OmniDiagRouter(configs_dir=os.path.join(_ROOT, "configs"))


@pytest.fixture(scope="module")
def backend(router):
    return router._get_loader("heart_disease")


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    return pd.read_csv(_TRAINING_CSV)


# ═════════════════════════════════════════════════════════════════════════════
# Startup
# ═════════════════════════════════════════════════════════════════════════════

def test_every_backend_module_imports():
    """The package imports its whole folder; a broken module kills both diseases."""
    import backend.model_backends as package

    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{module.name}")
    assert "glm_ivap_conformal" in registered_families()


def test_heart_dispatches_to_the_new_family(backend):
    assert backend.family == "glm_ivap_conformal"
    assert backend.capabilities.explainer == "linear"
    assert not backend.capabilities.supports_tree_shap


def test_heart_exposes_no_threshold_and_no_risk_bands(router):
    """A single cut-point is what produced HF-13; this module must expose none."""
    info = router.get_disease_info("heart_disease")
    assert info["inference_threshold"] is None
    assert info["risk_bands"] is None


# ═════════════════════════════════════════════════════════════════════════════
# HF-1 — chest pain, through the API surface
# ═════════════════════════════════════════════════════════════════════════════

def test_typical_angina_outranks_no_anginal_features_for_every_patient(router, training_frame):
    """The HF-1 assertion, on all 920 patients, raw score and calibrated alike."""
    backend = router._get_loader("heart_disease")
    frame = training_frame[backend.feature_names].copy()

    scores = {}
    for code in ("TA", "ASY"):
        variant = frame.copy()
        variant["ChestPainType"] = code
        raw = backend.predict_proba(variant)
        calibrated, _, _ = stack.ivap(
            backend.bundle["cal_scores"], backend.bundle["cal_labels"], raw
        )
        scores[code] = (raw, calibrated)

    assert np.all(scores["TA"][0] > scores["ASY"][0]), "raw score: HF-1 is back"
    assert np.all(scores["TA"][1] >= scores["ASY"][1]), "calibrated: HF-1 is back"


def test_typical_angina_outranks_no_anginal_features_through_predict(router):
    """The same property one patient at a time, through router.predict()."""
    typical = router.predict("heart_disease", {**PATIENT, "ChestPainType": "TA"})
    asymptomatic = router.predict("heart_disease", {**PATIENT, "ChestPainType": "ASY"})
    assert typical["confidence"] > asymptomatic["confidence"]


def test_the_two_chest_pain_maps_are_never_crossed():
    """Source-level: each encoder reads exactly one map."""
    source = ast.parse(open(stack.__file__).read())
    functions = {
        node.name: ast.dump(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "CP_MAP_CLINICAL" not in functions["encode_for_training"]
    assert "CP_MAP_UCI_RAW" not in functions["encode_for_inference"]
    assert "CP_MAP_UCI_RAW" in functions["encode_for_training"]
    assert "CP_MAP_CLINICAL" in functions["encode_for_inference"]


def test_inference_rejects_an_unknown_chest_pain_code():
    with pytest.raises(ValueError, match="clinical meaning"):
        stack.encode_for_inference([{**PATIENT, "ChestPainType": "unknown"}])


# ═════════════════════════════════════════════════════════════════════════════
# The bundle is the measured model
# ═════════════════════════════════════════════════════════════════════════════

def test_bundle_matches_the_recorded_fingerprint(backend, training_frame):
    """Same check the Docker build runs: decisions absolute, probabilities to 1e-6."""
    reference = json.load(open(_REFERENCE))
    assert stack.sha256_of(_TRAINING_CSV) == reference["training_csv_sha256"]

    encoded = stack.encode_for_training(training_frame)
    raw = backend.bundle["pipeline"].predict_proba(encoded)[:, 1]
    probability, _, _ = stack.ivap(
        backend.bundle["cal_scores"], backend.bundle["cal_labels"], raw
    )
    decisions = [
        stack.decide(float(s), g, backend.bundle["conformal_cells"])[0]
        for s, g in zip(raw, training_frame["Sex"])
    ]
    assert decisions == reference["decision"]
    assert np.max(np.abs(raw - np.array(reference["raw"]))) <= 1e-6
    assert np.max(np.abs(probability - np.array(reference["ivap"]))) <= 1e-6


def test_uncertain_is_a_referral(backend, training_frame):
    """`prediction` must count uncertain patients as referred.

    Reading only the confident referrals as positives drops sensitivity to 0.60
    for women and 0.43 for men (Phase 7). Any consumer counting positives reads
    this field, so this is where the property is pinned.
    """
    rows = training_frame.head(200).to_dict(orient="records")
    results = backend.predict_batch(rows)

    for row, result in zip(rows, results):
        raw = backend.predict_proba(pd.DataFrame([row])[backend.feature_names])[0]
        decision, _ = stack.decide(float(raw), row["Sex"], backend.bundle["conformal_cells"])
        if decision == stack.DECISION_UNCERTAIN:
            assert result["prediction"] == 1
            assert result["diagnosis"] == stack.UNCERTAIN_DIAGNOSIS
        elif decision == stack.DECISION_REFERRAL:
            assert result["prediction"] == 1 and result["diagnosis"] == "Positive"
        else:
            assert result["prediction"] == 0 and result["diagnosis"] == "Negative"


def test_predict_batch_matches_predict_row_by_row(backend, training_frame):
    rows = training_frame.head(25).to_dict(orient="records")
    batch = backend.predict_batch(rows)
    for row, batched in zip(rows, batch):
        single = backend.predict(row)
        assert single["prediction"] == batched["prediction"]
        assert single["confidence"] == pytest.approx(batched["confidence"], abs=1e-12)


def test_blank_fasting_bs_does_not_silently_raise_risk(backend, training_frame):
    """HF-11: the previous model added 7.6 points for a blank FastingBS, with
    no warning. D-26 removed the mechanism; this holds the line."""
    rows = training_frame[training_frame["FastingBS"].notna()].head(150)
    with_value = backend.predict_proba(rows[backend.feature_names])
    blanked = rows.copy()
    blanked["FastingBS"] = None
    without = backend.predict_proba(blanked[backend.feature_names])
    assert float(np.mean(without - with_value)) * 100 <= 1.0

    warned = backend.predict(blanked.iloc[0].to_dict())
    assert "FastingBS" in warned.get("data_completeness_warning", "")


# ═════════════════════════════════════════════════════════════════════════════
# SHAP — exact, and said to be on the raw scale
# ═════════════════════════════════════════════════════════════════════════════

def test_shap_is_additive_on_the_raw_log_odds(backend, training_frame):
    rows = training_frame.head(50)
    raw = backend.predict_proba(rows[backend.feature_names])
    logit = np.log(raw / (1 - raw))
    for i in range(len(rows)):
        result = backend.shap_values(pd.DataFrame([rows.iloc[i][backend.feature_names]]))
        assert abs(result.base_value + result.values.sum() - logit[i]) < 1e-8


def test_shap_matches_shaps_own_linear_and_permutation_explainers(backend, training_frame):
    shap = pytest.importorskip("shap")
    bundle = backend.bundle
    pipeline = bundle["pipeline"]
    rows = training_frame.head(10)[backend.feature_names]
    encoded = stack.encode_for_inference(rows)
    design = np.asarray(pipeline.named_steps["prep"].transform(encoded), dtype=float)
    background = np.asarray(bundle["shap_background_mean"], dtype=float)
    groups = bundle["shap_column_groups"]

    linear = shap.LinearExplainer(
        pipeline.named_steps["lr"], (background, np.zeros((len(background), len(background))))
    ).shap_values(design)

    def raw_logit(matrix: np.ndarray) -> np.ndarray:
        proba = pipeline.named_steps["lr"].predict_proba(matrix)[:, 1]
        return np.log(proba / (1 - proba))

    permutation = shap.PermutationExplainer(
        raw_logit, background.reshape(1, -1)
    ).shap_values(design, silent=True)

    for i in range(len(rows)):
        ours = backend.shap_values(pd.DataFrame([rows.iloc[i]])).values
        for j, feature in enumerate(stack.MODEL_FEATURES):
            columns = groups[feature]
            assert abs(ours[j] - linear[i][columns].sum()) < 1e-6
            assert abs(ours[j] - permutation[i][columns].sum()) < 1e-6


def test_explain_declares_the_scale_and_flags_imputed_values(router):
    blank = {**PATIENT, "Cholesterol": None}
    result = router.explain("heart_disease", blank)
    assert result["shap_scale"] == "log_odds_raw_score"
    flags = {item["feature"]: item["imputed"] for item in result["chart_data"]}
    assert flags["Cholesterol"] is True
    assert flags["Age"] is False
    # Inputs this model does not read never appear in an explanation of it.
    assert set(flags) == set(stack.MODEL_FEATURES)


# ═════════════════════════════════════════════════════════════════════════════
# Cost
# ═════════════════════════════════════════════════════════════════════════════

def test_single_patient_latency_is_reasonable(backend):
    backend.predict(dict(PATIENT))  # warm
    start = time.perf_counter()
    for _ in range(20):
        backend.predict(dict(PATIENT))
    elapsed_ms = (time.perf_counter() - start) / 20 * 1000
    assert elapsed_ms < 250, f"{elapsed_ms:.0f} ms per patient"


# ═════════════════════════════════════════════════════════════════════════════
# The build gate must actually refuse
# ═════════════════════════════════════════════════════════════════════════════
#
# A check that has never been seen to fail is not known to work. Each test here
# breaks one input and asserts the build refuses, so the Docker build step
# cannot quietly degrade into a no-op.

def test_build_refuses_a_training_csv_with_one_row_changed(tmp_path, training_frame):
    """Condition 1: the CSV hash. One cell is enough."""
    tampered = training_frame.copy()
    tampered.loc[0, "Cholesterol"] = float(tampered.loc[0, "Cholesterol"]) + 1
    path = tmp_path / "tampered.csv"
    tampered.to_csv(path, index=False)

    bundle = stack.build_bundle(path)
    with pytest.raises(ValueError, match="sha256 does not match"):
        stack.verify_against_reference(bundle, path, _REFERENCE, 1e-6)


def test_build_refuses_a_model_whose_decisions_moved(backend, monkeypatch):
    """Condition 2: decisions are absolute, whatever the probability delta.

    The conformal cells are nudged so that a few patients change side. No
    tolerance may absorb that — the message says so in as many words.
    """
    tampered = dict(backend.bundle)
    tampered["conformal_cells"] = {
        key: value * 0.97 for key, value in backend.bundle["conformal_cells"].items()
    }
    with pytest.raises(ValueError, match="not a tolerance to relax"):
        stack.verify_against_reference(tampered, _TRAINING_CSV, _REFERENCE, 1e-6)


def test_build_refuses_probabilities_outside_the_tolerance(backend):
    """Condition 3: the probability tolerance, with the decisions left alone."""
    reference = json.load(open(_REFERENCE))
    shifted = dict(reference)
    shifted["raw"] = [min(1.0, v + 1e-4) for v in reference["raw"]]
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(shifted, handle)
        path = handle.name
    with pytest.raises(ValueError, match="exceeds the declared tolerance"):
        stack.verify_against_reference(backend.bundle, _TRAINING_CSV, path, 1e-6)


def test_the_build_gate_passes_on_the_real_inputs(backend):
    """Guards the three tests above against passing because everything fails."""
    result = stack.verify_against_reference(
        backend.bundle, _TRAINING_CSV, _REFERENCE, 1e-6
    )
    assert result["decision_mismatches"] == 0
    assert max(result["max_abs_delta"].values()) <= 1e-6


# ── Gate 8.3 — the blank-input warning list is derived, not typed ────────────


def test_bundle_carries_the_blank_impact_measurement(backend):
    """
    The numbers behind the warning list travel with the artifact. Measured at
    build time on the artifact itself, for every Optional input.
    """
    impact = backend.bundle["blank_impact"]
    assert set(impact) == set(stack.OPTIONAL_INPUT_FEATURES)
    for field, row in impact.items():
        assert row["n"] > 0, f"{field}: no patient has it recorded"
        for key in ("decision_changed", "decision_changed_share",
                    "mean_delta_points", "p90_delta_points"):
            assert key in row, f"{field}: missing {key}"


def test_only_the_inputs_the_model_reads_can_change_a_decision(backend):
    """
    The sharpest evidence that the shipped artifact really is L3 (D-25) rather
    than a claim in a document: blanking MaxHR, Oldpeak or ST_Slope changes
    EXACTLY nothing -- no decision, no probability -- because the model does not
    read them. A model that read them could not produce exact zeros.
    """
    impact = backend.bundle["blank_impact"]
    read = {"FastingBS": "FastingBS_cat"}
    for field, row in impact.items():
        encoded_name = read.get(field, field)
        is_read = encoded_name in stack.MODEL_FEATURES
        moves = row["decision_changed"] > 0 or row["mean_delta_points"] != 0.0
        assert moves == is_read, (
            f"{field}: read_by_model={is_read} but moves={moves} ({row})"
        )
        if not is_read:
            assert row["decision_changed"] == 0
            assert row["mean_delta_points"] == 0.0
            assert row["p90_delta_points"] == 0.0


def test_warning_list_is_derived_and_matches_the_config_cross_check(backend, router):
    """
    The config keeps the expected list as a cross-check, not as the source. If a
    retrain ever changes the facts, this fails and says so, instead of letting
    the config quietly describe a model that no longer exists -- which is the
    F0-1 failure mode one level up.
    """
    model_config = router.disease_configs["heart_disease"]["model"]
    derived = stack.blank_warning_features(
        backend.bundle["blank_impact"], model_config["blank_warning_min_decision_share"]
    )
    assert derived == model_config["high_impact_features"], (
        f"derived {derived} != config cross-check {model_config['high_impact_features']}"
    )
    # And it is what the backend actually uses.
    assert backend._warned_blank_features() == derived


def test_the_list_does_not_hinge_on_the_chosen_threshold(backend):
    """
    Answers the obvious objection to a threshold picked after seeing the data.
    The gap between the lowest warned field and the highest unwarned one runs
    from ~2.8% to exactly 0%, so every threshold in that range gives the same
    list. Measured rather than asserted -- if a retrain narrowed the gap, this
    test would start failing and the threshold would need a real argument.
    """
    impact = backend.bundle["blank_impact"]
    baseline = stack.blank_warning_features(impact, 0.01)
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.027):
        assert stack.blank_warning_features(impact, threshold) == baseline, threshold

    shares = {f: r["decision_changed_share"] for f, r in impact.items()}
    warned = [shares[f] for f in baseline]
    unwarned = [s for f, s in shares.items() if f not in baseline]
    assert min(warned) > max(unwarned), "the gap this test relies on has closed"


def test_an_unread_blank_input_is_not_warned_about(backend):
    """
    Warning about a field that cannot move anything is not extra caution, it is
    noise that costs the warning its meaning. ST_Slope is the field that
    surfaced HM-5 in the first place, and the shipped model does not read it.
    """
    patient = {
        "Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
        "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": None,
    }
    assert backend._completeness_warning(patient) is None

    blanked = dict(patient, ST_Slope="Flat", Cholesterol=None)
    warning = backend._completeness_warning(blanked)
    assert warning is not None and "Cholesterol" in warning


def test_the_warning_does_not_claim_a_direction(backend):
    """
    Blanking these fields does not simply raise risk: measured on this bundle,
    RestingBP -0.47 points and FastingBS -0.72 on average, Cholesterol +0.49. A
    directional wording would be wrong for two of the three, so the message must
    not carry one.
    """
    impact = backend.bundle["blank_impact"]
    directions = {f: impact[f]["mean_delta_points"] > 0 for f in backend._warned_blank_features()}
    assert len(set(directions.values())) > 1, (
        "the measured directions agree now, so re-check whether the wording "
        f"should say one: {directions}"
    )
    warning = backend._completeness_warning(
        {"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": None,
         "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal"}
    )
    assert warning is not None
    lowered = warning.lower()
    for claim in ("raise", "raises", "higher risk", "increase"):
        assert claim not in lowered, f"the warning claims a direction: {warning!r}"


def test_a_pre_8_3_bundle_falls_back_to_the_config_list(backend, router):
    """
    An artifact built before this gate carries no blank_impact. It must degrade
    to the config's list rather than lose the warning altogether -- a silent
    loss of the data-completeness warning is worse than an unmeasured list.
    """
    model_config = router.disease_configs["heart_disease"]["model"]
    original = backend.bundle.pop("blank_impact")
    try:
        assert backend._warned_blank_features() == model_config["high_impact_features"]
    finally:
        backend.bundle["blank_impact"] = original
    assert backend._warned_blank_features() == stack.blank_warning_features(
        original, model_config["blank_warning_min_decision_share"]
    )


# ── Gate 8.4 — the decision is consistent down the whole live path ───────────


def test_o3_identity_holds_for_every_one_of_the_920(backend, training_frame):
    """
    The O3 rule as a DEFINITION, over every patient, through the service-level
    call the API uses: a patient is counted positive exactly when the decision is
    `referral` or `uncertain`. Counting only the confident referrals is the
    silent sensitivity drop this model family exists to prevent, so this is
    checked as an identity with no tolerance rather than as a rate.

    Not compared with p7's O3 numbers: p7 scored each patient with a model fitted
    without them, this scores them with the model fitted on all 920. Different
    estimands -- see results/p8_4_live_o3.json in the research repo.
    """
    frame = training_frame.drop(columns=["HeartDisease", "site"])
    # The CSV carries RAW UCI chest-pain codes and this path reads clinical
    # ones, so they are converted first -- feeding them straight in is HF-1.
    patients = stack.translate_raw_codes(frame.to_dict("records"))
    results = backend.predict_batch(patients)

    assert len(results) == len(training_frame)
    for i, r in enumerate(results):
        goes_forward = r["decision"] in (stack.DECISION_REFERRAL, stack.DECISION_UNCERTAIN)
        assert (r["prediction"] == 1) == goes_forward, (i, r)
        assert r["decision_is_referral"] == goes_forward, (i, r)
        # A response that omitted these would satisfy the identity vacuously.
        assert r["probability_lower"] <= r["confidence"] <= r["probability_upper"], (i, r)
        assert r["output_type"] == "conformal_decision"
        assert r["probability_scale"] == "ivap_calibrated_training_mix"

    # All three decisions must actually occur, or the identity is being checked
    # against a degenerate case.
    seen = {r["decision"] for r in results}
    assert seen == {
        stack.DECISION_REFERRAL, stack.DECISION_NO_REFERRAL, stack.DECISION_UNCERTAIN
    }, seen


def test_the_response_publishes_no_threshold_and_no_bands(backend):
    """
    Consumers were fabricating both. The response must not offer either, so
    nothing can read one by accident, and must say what it DOES decide by.
    """
    result = backend.predict({
        "Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
        "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
    })
    assert "inference_threshold" not in result
    assert "risk_bands" not in result
    assert result["output_type"] == "conformal_decision"


def test_no_consumer_needs_to_read_the_diagnosis_sentence(backend):
    """
    `diagnosis` is prose for a human. The machine-readable answer is `decision`,
    and this asserts the two agree so nothing has an excuse to parse the text.
    """
    cases = [
        {"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
         "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal"},
        {"Age": 29, "Sex": "F", "ChestPainType": "NAP", "RestingBP": 100,
         "Cholesterol": 180, "FastingBS": 0, "RestingECG": "Normal"},
    ]
    for result in backend.predict_batch(cases):
        if result["decision"] == stack.DECISION_UNCERTAIN:
            assert result["diagnosis"] == stack.UNCERTAIN_DIAGNOSIS
        elif result["decision"] == stack.DECISION_REFERRAL:
            assert result["diagnosis"] == "Positive"
        else:
            assert result["diagnosis"] == "Negative"


def test_the_review_queue_reads_the_model_s_own_uncertainty(backend):
    """
    Gate 8.4, and the sharpest fault this gate fixed.

    Auto-queueing for human review used to run every module through
    `should_queue_for_review`, which scores entropy around a decision threshold.
    Heart has no threshold, so it was given the 0.5 default -- and the result was
    wrong in both directions: a patient the model itself called UNCERTAIN at
    p=0.95 (an empty conformal set: atypical for BOTH classes) was not queued,
    while a confidently decided patient at p=0.52 was. The consumer whose only
    job is catching uncertainty was ignoring the model's own statement of it.

    This pins the arithmetic that made it wrong, so the rule cannot quietly be
    routed back through it.
    """
    from backend.active_learning.sampler import should_queue_for_review

    # The old path, at probabilities a conformal module really can produce.
    assert should_queue_for_review(0.95, decision_threshold=0.5) is False
    assert should_queue_for_review(0.52, decision_threshold=0.5) is True

    # That the new rule is actually wired into the endpoint is asserted end to
    # end, against the database, in
    # tests/test_diabetes_calibration.py::TestReviewQueueUsesTheModelsDecision --
    # restating the rule as a comparison here would be a tautology.


def test_no_module_branches_on_a_disease_name_for_its_decision_shape():
    """
    The rule this gate enforces: behaviour follows what the config declares, not
    what the disease is called. Two branches in the frontend read the disease
    name to decide what to display, and both were producing numbers the model
    does not have -- a 0.5 threshold, and 0.7/0.4 bands the config sets to null.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    watched = [
        "frontend/src/constants/thresholds.js",
        "frontend/src/components/BatchUpload.jsx",
        "backend/llm/report_generator.py",
    ]
    offenders = []
    for rel in watched:
        with open(os.path.join(root, rel)) as fh:
            for lineno, line in enumerate(fh, 1):
                code = line.split("//")[0].split("#")[0]
                if "'heart_disease'" in code or '"heart_disease"' in code:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
                if "'diabetes'" in code or '"diabetes"' in code:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "disease-name branch in a decision-shape path:\n" + "\n".join(offenders)
