"""
OmniDiag — Heart CAD retraining as a CANDIDATE, never a replacement
===================================================================
The active-learning cycle for this disease used to call `retrain_xgb`, which
looks for an XGBoost file this module has not shipped since Gate 8.1 and builds
`X` from raw predict-time strings. It could not succeed, and if it had, it would
have written an incrementally-boosted tree model over the path a Spline-GLM
bundle is loaded from, then hot-reloaded it into the running process. That is
three separate problems: the wrong family, no recomputed calibration, and an
automatic promotion with nobody reviewing it.

What this module does instead:

    920 UCI rows + expert-labelled review-queue rows
        -> one merged CSV in the raw UCI chest-pain coding
        -> stack.build_bundle()          <-- the SAME call the shipped build makes
        -> models/heart_disease/candidates/<mlflow_run_id>/

The shipped bundle is never written. `reference_scores.json` is never written --
it is read, as the record of what the shipped model decided, so the candidate can
be compared against it. Promotion does not exist here: a human reads the MLflow
run and the candidate card, and nothing in this file can make a candidate live.

── Why build_bundle() and not a retraining routine of its own ────────────────
A parallel training path is a path that agrees with the shipped one on the day
it is written and diverges silently afterwards. Every constant that defines this
model -- the spline knots, C, the 5-fold OOF stratified by site x sex x label,
the IVAP calibrator, the Mondrian cells -- lives in `stack.py` and is read from
there by exactly one function. This module assembles a CSV and calls it. The
only thing it knows about the model is the shape of the training file.

`tests/test_heart_candidate_retrain.py` holds that to the source: it asserts
this module calls `stack.build_bundle` and imports no estimator of its own, and
it runs a CONTROL candidate with zero added rows, which must reproduce the
shipped decisions exactly -- zero movement off the diagonal of the transition
matrix. A parallel path fails that by running, not by review.

── The scientific limit, stated in the artifact and not only in a report ─────
The review-queue rows are not a random sample of anything. They are there
because the model was uncertain about them (that is what puts a prediction in
the queue), and the patients behind them were not selected into a
catheterisation cohort the way all 920 UCI patients were. Adding them moves the
mix the conformal guarantee is marginal over, and the guarantee follows the
mix -- it is not a property the model carries with it. `CANDIDATE_LIMIT` says so,
and it is written into the candidate card and logged as its own field on the
MLflow run, so it cannot be separated from the numbers it qualifies.
"""

from __future__ import annotations

import json
import logging
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.heart_glm import stack

log = logging.getLogger("omnidiag.heart_candidate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Read-only inputs. Neither is ever opened for writing in this module, and
#: `_assert_writes_nothing_protected` re-checks that at run time.
SHIPPED_BUNDLE_PATH = PROJECT_ROOT / "models/heart_disease/heart_l3_glm_stack.pkl"
REFERENCE_SCORES_PATH = PROJECT_ROOT / "backend/heart_glm/reference_scores.json"

#: Where a candidate goes. One directory per MLflow run id; git-ignored, because
#: the merged CSV inside it carries patient rows out of the database.
CANDIDATES_DIR = PROJECT_ROOT / "models/heart_disease/candidates"

#: The site label given to review-queue rows. They have no hospital, and the
#: OOF split stratifies on site, so they need one of their own rather than
#: being folded into a hospital they did not come from.
REVIEW_SITE = "review_queue"

#: Columns the training CSV has, in order. A merged file that does not match
#: this exactly is not handed to build_bundle().
TRAINING_COLUMNS: List[str] = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
    "HeartDisease", "site",
]

#: Inputs a review-queue row must carry to be usable. The rest of
#: TRAINING_COLUMNS may be blank: the pipeline imputes them, and L3 does not
#: read MaxHR / Oldpeak / ExerciseAngina / ST_Slope at all (D-25).
REQUIRED_REVIEW_FIELDS: List[str] = ["Age", "Sex", "ChestPainType", "RestingECG"]

#: API (clinical) chest-pain code -> raw UCI code. Derived by inverting
#: stack.CP_RAW_TO_CLINICAL, which is itself derived from the two maps by
#: anginal-feature count. Written out by hand nowhere: a third hand-written map
#: is a third place for HF-1 to come back.
CP_CLINICAL_TO_RAW: Dict[str, str] = {
    clinical: raw for raw, clinical in stack.CP_RAW_TO_CLINICAL.items()
}
if len(CP_CLINICAL_TO_RAW) != len(stack.CP_RAW_TO_CLINICAL):
    raise ImportError(
        f"CP_CLINICAL_TO_RAW lost a code inverting {stack.CP_RAW_TO_CLINICAL}"
    )

#: Written into candidate_card.json and logged on the MLflow run as its own
#: field. Not a caveat for a report -- a property of every number produced here.
CANDIDATE_LIMIT = (
    "The review-queue rows added here were selected BY MODEL UNCERTAINTY, not "
    "sampled, and the patients behind them were not selected into a "
    "catheterisation cohort as all 920 UCI patients were. The Mondrian "
    "conformal guarantee is marginal over the mix it was calibrated on, so with "
    "these rows included it is a guarantee over a DIFFERENT mix than the "
    "shipped model's, and coverage measured here does not transfer to either "
    "the UCI hospitals or the hospital reading it (D-32, HF-15). The candidate "
    "is evidence for a human decision, not a validated model."
)

#: Used when the config carries no version. Never a made-up version number.
VERSION_UNAVAILABLE = "version_unavailable"

DECISIONS: Tuple[str, str, str] = (
    stack.DECISION_REFERRAL,
    stack.DECISION_NO_REFERRAL,
    stack.DECISION_UNCERTAIN,
)


# ── Data assembly ────────────────────────────────────────────────────────────

def review_rows_to_training_frame(
    features_list: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Turn expert-labelled review-queue rows into rows of the TRAINING file.

    Two things change and nothing else: the chest-pain code is rewritten from
    the clinical coding the API stores into the raw UCI coding the training file
    uses, and the expert label becomes `HeartDisease`. The coding rewrite is the
    whole reason this function exists -- `encode_for_training` reads
    CP_MAP_UCI_RAW, so handing it a clinically-coded row inverts chest pain for
    that patient, which is HF-1 arriving through the back door.

    Returns (frame, rejected). A row missing a required field is rejected with a
    reason rather than imputed into the training set: an imputed Sex or
    ChestPainType is an invented patient, and the point of these rows is that a
    human looked at them.
    """
    if len(features_list) != len(labels):
        raise ValueError(
            f"{len(features_list)} feature rows against {len(labels)} labels"
        )

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for index, (features, label) in enumerate(zip(features_list, labels)):
        row = {column: np.nan for column in TRAINING_COLUMNS}
        source = dict(features or {})

        missing = [f for f in REQUIRED_REVIEW_FIELDS if source.get(f) in (None, "")]
        if missing:
            rejected.append({"index": index, "reason": f"missing {','.join(missing)}"})
            continue
        if int(label) not in (0, 1):
            rejected.append({"index": index, "reason": f"label {label!r} not 0 or 1"})
            continue
        if source.get("Sex") not in ("M", "F"):
            rejected.append({"index": index, "reason": f"Sex {source.get('Sex')!r}"})
            continue

        clinical_cp = source.get("ChestPainType")
        if clinical_cp not in CP_CLINICAL_TO_RAW:
            rejected.append({"index": index, "reason": f"ChestPainType {clinical_cp!r}"})
            continue

        for column in TRAINING_COLUMNS:
            if column in source and source[column] is not None:
                row[column] = source[column]
        # The two deliberate rewrites.
        row["ChestPainType"] = CP_CLINICAL_TO_RAW[clinical_cp]
        row["HeartDisease"] = int(label)
        row["site"] = REVIEW_SITE
        kept.append(row)

    frame = pd.DataFrame(kept, columns=TRAINING_COLUMNS)
    return frame, rejected


def merge_training_data(
    base_csv: Path,
    review_frame: pd.DataFrame,
    expected_base_sha256: Optional[str] = None,
) -> pd.DataFrame:
    """
    Base training file plus the review rows, in the training file's own schema.

    The base file's sha256 is checked against the value the config records when
    one is given: a candidate built on a different 920 rows is not comparable
    with the shipped model, and `reference_scores.json` -- which is what it gets
    compared against -- is keyed to that hash.
    """
    actual = stack.sha256_of(base_csv)
    if expected_base_sha256 and actual != expected_base_sha256:
        raise ValueError(
            "base training CSV sha256 does not match the configured value:\n"
            f"  configured {expected_base_sha256}\n"
            f"  actual     {actual}"
        )

    base = pd.read_csv(base_csv)
    if list(base.columns) != TRAINING_COLUMNS:
        raise ValueError(
            f"base CSV columns changed: {list(base.columns)} != {TRAINING_COLUMNS}"
        )
    if REVIEW_SITE in set(base["site"].unique()):
        raise ValueError(
            f"base CSV already contains site {REVIEW_SITE!r} — it is not the "
            f"pristine UCI file, and the transition matrix would not line up "
            f"with reference_scores.json"
        )

    if review_frame.empty:
        return base.copy()
    merged = pd.concat([base, review_frame[TRAINING_COLUMNS]], ignore_index=True)
    return merged


# ── Comparison against the shipped model ─────────────────────────────────────

def load_reference(reference_path: Path, base_csv_sha256: str, n_base: int) -> Dict[str, Any]:
    """The shipped model's recorded decisions, READ ONLY (D-78).

    Generated locally from `p7_final` alone and never rewritten from here. The
    two checks below are what makes it usable as the left-hand side of a
    transition matrix: same rows, same order, same file.
    """
    with open(reference_path) as handle:
        reference = json.load(handle)
    if reference["training_csv_sha256"] != base_csv_sha256:
        raise ValueError(
            "reference_scores.json was recorded against a different training "
            f"CSV:\n  reference {reference['training_csv_sha256']}\n"
            f"  base      {base_csv_sha256}"
        )
    if int(reference["n"]) != n_base:
        raise ValueError(
            f"reference_scores.json has n={reference['n']} against {n_base} base rows"
        )
    return reference


def decisions_of(bundle: Mapping[str, Any], frame: pd.DataFrame) -> List[str]:
    """Conformal decision per row, through the bundle's own decision layer."""
    raw = bundle["pipeline"].predict_proba(stack.encode_for_training(frame))[:, 1]
    return [
        stack.decide(float(score), sex, bundle["conformal_cells"])[0]
        for score, sex in zip(raw, frame["Sex"])
    ]


def transition_matrix(shipped: Sequence[str], candidate: Sequence[str]) -> Dict[str, Any]:
    """
    Where the 920 patients moved, shipped -> candidate.

    The full 3x3 count, plus the two numbers that carry the clinical meaning on
    their own: how many patients the candidate pulled OUT of `uncertain` (it
    committed to a decision the shipped model would not make) and how many it
    pushed INTO it (it withdrew a decision the shipped model made). A net figure
    would hide one inside the other, so neither is netted.
    """
    if len(shipped) != len(candidate):
        raise ValueError(f"{len(shipped)} shipped decisions against {len(candidate)}")

    counts = {
        f"{before}__to__{after}": 0 for before in DECISIONS for after in DECISIONS
    }
    for before, after in zip(shipped, candidate):
        key = f"{before}__to__{after}"
        if key not in counts:
            raise ValueError(f"unknown decision pair {before!r} -> {after!r}")
        counts[key] += 1

    uncertain = stack.DECISION_UNCERTAIN
    moved_out = sum(
        counts[f"{uncertain}__to__{after}"] for after in DECISIONS if after != uncertain
    )
    moved_in = sum(
        counts[f"{before}__to__{uncertain}"] for before in DECISIONS if before != uncertain
    )
    changed = sum(
        count for key, count in counts.items()
        if key.split("__to__")[0] != key.split("__to__")[1]
    )
    return {
        "n": len(shipped),
        "counts": counts,
        "moved_out_of_uncertain": moved_out,
        "moved_into_uncertain": moved_in,
        "decisions_changed_total": changed,
        "shipped_totals": {d: sum(1 for x in shipped if x == d) for d in DECISIONS},
        "candidate_totals": {d: sum(1 for x in candidate if x == d) for d in DECISIONS},
    }


# ── Conformal coverage ───────────────────────────────────────────────────────

def _covered(scores: np.ndarray, sexes: np.ndarray, y: np.ndarray,
             cells: Mapping[Any, float]) -> np.ndarray:
    """Per patient: is the TRUE class in the conformal prediction set?"""
    out = np.empty(len(y), dtype=bool)
    for i, (score, sex, label) in enumerate(zip(scores, sexes, y)):
        in1, in0 = stack.conformal_set(float(score), sex, cells)
        out[i] = bool(in1) if int(label) == 1 else bool(in0)
    return out


def _coverage_by_cell(covered: np.ndarray, sexes: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Marginal coverage plus one figure per Mondrian cell (sex x class).

    Per cell is the number that matters and the number that broke: the marginal
    figure was met while diseased women at Hungarian sat at 0.58 (Gate 6). A
    marginal-only report would have shown nothing wrong.
    """
    out = {"marginal": round(float(np.mean(covered)), 6)}
    for sex in ("F", "M"):
        for klass in (0, 1):
            mask = (sexes == sex) & (y == klass)
            out[f"{sex}{klass}"] = (
                round(float(np.mean(covered[mask])), 6) if mask.any() else None
            )
            out[f"n_{sex}{klass}"] = int(mask.sum())
    return out


def coverage_in_sample(bundle: Mapping[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    """
    Coverage of the bundle's own cells on the rows they were computed from.

    BY CONSTRUCTION >= 1 - alpha. The conformal quantile is the (1-alpha)
    quantile of exactly these nonconformity scores, so this measures that the
    quantile was taken correctly and nothing else. It is reported because it is
    the figure an unlabelled "coverage" number usually turns out to be, and
    labelling it is the only way a reader can tell it apart from the one below.
    """
    oof = np.asarray(bundle["cal_scores"], dtype=float)
    y = np.asarray(bundle["cal_labels"], dtype=int)
    sexes = frame["Sex"].to_numpy(dtype=object)
    covered = _covered(oof, sexes, y, bundle["conformal_cells"])
    result = _coverage_by_cell(covered, sexes, y)
    result["kind"] = "in_sample_by_construction"
    return result


def coverage_cross_conformal(
    bundle: Mapping[str, Any], frame: pd.DataFrame, n_folds: int = stack.N_FOLDS,
) -> Dict[str, Any]:
    """
    Coverage on rows whose Mondrian cells were computed WITHOUT them.

    The cells are recomputed from the OOF scores of n-1 folds and the held-out
    fold is scored against them, for every fold. This is the only coverage
    figure here that is an estimate of anything: it is measured on rows that did
    not set the quantile they are judged by. It is still marginal over the
    training mix and still says nothing about a new hospital (D-32).

    Reuses the bundle's existing OOF scores rather than refitting: the scores
    are already out-of-fold, so what is being held out here is the CELL, which
    is the quantity whose in-sample optimism is at issue.
    """
    from sklearn.model_selection import StratifiedKFold

    oof = np.asarray(bundle["cal_scores"], dtype=float)
    y = np.asarray(bundle["cal_labels"], dtype=int)
    sexes = frame["Sex"].to_numpy(dtype=object)
    sites = frame["site"].to_numpy(dtype=object)

    strata = np.char.add(
        np.char.add(sites.astype(str), sexes.astype(str)), y.astype(str)
    )
    covered = np.zeros(len(y), dtype=bool)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=stack.CV_SEED)
    for fit_idx, held_idx in splitter.split(np.zeros((len(y), 1)), strata):
        cells = {
            f"{sex}{klass}": stack.conformal_quantile(
                (1 - oof[fit_idx][(sexes[fit_idx] == sex) & (y[fit_idx] == 1)])
                if klass == 1
                else oof[fit_idx][(sexes[fit_idx] == sex) & (y[fit_idx] == 0)],
                alpha=float(bundle["alpha"]),
            )
            for sex in ("F", "M")
            for klass in (0, 1)
        }
        covered[held_idx] = _covered(
            oof[held_idx], sexes[held_idx], y[held_idx], cells
        )

    result = _coverage_by_cell(covered, sexes, y)
    result["kind"] = "cross_conformal_held_out_cells"
    result["n_folds"] = int(n_folds)
    return result


# ── Guards ───────────────────────────────────────────────────────────────────

def _assert_writes_nothing_protected(*paths: Path) -> None:
    """No output of this module may be the shipped bundle or the reference file.

    A belt-and-braces check on resolved paths: the constraint is that those two
    files are not written, and a constraint worth stating is worth asserting
    where it would be violated rather than only in a comment.
    """
    protected = {SHIPPED_BUNDLE_PATH.resolve(), REFERENCE_SCORES_PATH.resolve()}
    for path in paths:
        if path.resolve() in protected:
            raise RuntimeError(
                f"refusing to write {path} — the shipped bundle and the "
                f"reference scores are read-only on the retrain path"
            )


def _relative_to_root(path: Path) -> str:
    """Project-relative when the path is inside the project, absolute otherwise.

    Candidates normally live under `models/`, and a relative path is what a card
    should record. A test builds into a tmp directory, and `relative_to` raises
    there rather than falling back -- which would make a path formatting detail
    fail the build it is describing.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _config_version(config: Optional[Mapping[str, Any]]) -> str:
    """The configured module version, or an explicit statement that there is none.

    Never a default that looks like a real version. A silently-invented "v7.0.0"
    is how a candidate card comes to claim provenance it does not have.
    """
    version = ((config or {}).get("disease") or {}).get("version")
    if version in (None, ""):
        return VERSION_UNAVAILABLE
    return str(version)


# ── The candidate build ──────────────────────────────────────────────────────

def build_candidate(
    run_id: str,
    features_list: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    config: Optional[Mapping[str, Any]] = None,
    base_csv: Optional[Path] = None,
    candidates_dir: Optional[Path] = None,
    reference_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build one candidate bundle and measure it against the shipped model.

    Writes `models/heart_disease/candidates/<run_id>/` and nothing else. Returns
    the candidate card, which is also written into that directory, plus the
    metrics and params for the MLflow run.

    `run_id` comes from an MLflow run that is already open, so that the
    directory name and the run are the same identifier and neither can exist
    without the other.
    """
    base_csv = Path(base_csv or (PROJECT_ROOT / "data/heart_disease/processed/uci_heart_by_site.csv"))
    candidates_dir = Path(candidates_dir or CANDIDATES_DIR)
    reference_path = Path(reference_path or REFERENCE_SCORES_PATH)

    configured_sha = ((config or {}).get("data") or {}).get("training_csv_sha256")
    base_sha = stack.sha256_of(base_csv)

    # Hashed before anything is built, and re-hashed after, so that "the shipped
    # bundle was not replaced" is a measurement this run made rather than a
    # sentence this file asserts about itself.
    shipped_sha_before = stack.sha256_of(SHIPPED_BUNDLE_PATH)
    reference_sha_before = stack.sha256_of(reference_path)

    review_frame, rejected = review_rows_to_training_frame(features_list, labels)
    merged = merge_training_data(base_csv, review_frame, configured_sha)
    n_base = len(pd.read_csv(base_csv))

    # The alpha the config declares against the alpha the artifact will use.
    # A candidate built at one alpha and reported against another is a coverage
    # claim about a model that was not built.
    configured_alpha = ((config or {}).get("model") or {}).get("conformal_alpha")
    if configured_alpha is not None and float(configured_alpha) != float(stack.ALPHA):
        raise ValueError(
            f"configured conformal_alpha {configured_alpha} does not match the "
            f"stack's alpha {stack.ALPHA} — the candidate would not be built at "
            f"the alpha it is reported against"
        )

    target_dir = candidates_dir / run_id
    if target_dir.exists():
        raise RuntimeError(f"candidate directory already exists: {target_dir}")
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Built in a temporary directory and moved into place at the end, so a
    # failed build cannot leave a half-written candidate that reads as real.
    staging = Path(tempfile.mkdtemp(prefix=f"candidate_{run_id}_", dir=candidates_dir))
    merged_csv = staging / "training_data_merged.csv"
    bundle_path = staging / "heart_l3_glm_stack_candidate.pkl"
    card_path = staging / "candidate_card.json"
    _assert_writes_nothing_protected(merged_csv, bundle_path, card_path)

    merged.to_csv(merged_csv, index=False)
    merged_sha = stack.sha256_of(merged_csv)

    log.info(
        "building heart candidate on %d rows (%d UCI + %d reviewed, %d rejected)",
        len(merged), n_base, len(review_frame), len(rejected),
    )
    bundle = stack.build_bundle(merged_csv, experiments_git_ref="")

    with open(bundle_path, "wb") as handle:
        pickle.dump(bundle, handle)
    bundle_sha = stack.sha256_of(bundle_path)

    # ── Comparison: the 920 base rows only, in their recorded order ──────────
    base_frame = merged.iloc[:n_base]
    reference = load_reference(reference_path, base_sha, n_base)
    matrix = transition_matrix(reference["decision"], decisions_of(bundle, base_frame))

    with open(SHIPPED_BUNDLE_PATH, "rb") as handle:
        shipped = pickle.load(handle)
    shipped_frame = pd.read_csv(base_csv)

    coverage = {
        "candidate_in_sample": coverage_in_sample(bundle, merged),
        "candidate_cross_conformal": coverage_cross_conformal(bundle, merged),
        "shipped_in_sample": coverage_in_sample(shipped, shipped_frame),
        "shipped_cross_conformal": coverage_cross_conformal(shipped, shipped_frame),
        "declared_target": round(1.0 - float(bundle["alpha"]), 6),
    }

    # The two read-only files, re-hashed now that the build and every
    # measurement above are done. Checked, not claimed: if either moved, this
    # run wrote something it must not write, and the candidate is destroyed
    # rather than reported -- a candidate produced by a run that also mutated
    # the shipped model is not evidence about anything.
    shipped_sha_after = stack.sha256_of(SHIPPED_BUNDLE_PATH)
    reference_sha_after = stack.sha256_of(reference_path)
    if shipped_sha_after != shipped_sha_before or reference_sha_after != reference_sha_before:
        raise RuntimeError(
            "a read-only file changed during the candidate build — "
            f"shipped {shipped_sha_before} -> {shipped_sha_after}, "
            f"reference {reference_sha_before} -> {reference_sha_after}"
        )

    card = {
        "candidate_of": "heart_disease",
        "mlflow_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_candidate_not_replacement": True,
        "shipped_bundle_unchanged": shipped_sha_after == shipped_sha_before,
        "reference_scores_unchanged": reference_sha_after == reference_sha_before,
        "limit": CANDIDATE_LIMIT,
        "config_version": _config_version(config),
        "conformal_alpha": float(bundle["alpha"]),
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha,
        "training_data_file": merged_csv.name,
        "training_data_sha256": merged_sha,
        "base_training_csv": _relative_to_root(base_csv),
        "base_training_csv_sha256": base_sha,
        "rows_total": int(len(merged)),
        "rows_base_uci": int(n_base),
        "rows_from_review_queue": int(len(review_frame)),
        "rows_rejected": int(len(rejected)),
        "rejected_reasons": [r["reason"] for r in rejected],
        "shipped_bundle_sha256": shipped_sha_after,
        "reference_scores_sha256": reference_sha_after,
        "transition_matrix": matrix,
        "conformal_coverage": coverage,
        "blank_warning_impact": bundle.get("blank_impact"),
    }
    card_path.write_text(json.dumps(card, indent=1, default=str) + "\n")

    staging.rename(target_dir)
    card["candidate_dir"] = _relative_to_root(target_dir)
    log.info("candidate written to %s", target_dir)
    return card


def mlflow_payload(card: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Split the card into MLflow params (provenance) and metrics (numbers).

    A closed set, built by naming each field. Nothing iterates over patient data
    and nothing forwards a field that was not listed here: the review rows are
    patient records, and the run is the one part of this that leaves the
    machine.
    """
    params: Dict[str, Any] = {
        "candidate_of": card["candidate_of"],
        "is_candidate_not_replacement": "yes",
        "shipped_bundle_replaced": "no",
        "config_version": card["config_version"],
        "conformal_alpha": card["conformal_alpha"],
        "candidate_bundle_sha256": card["bundle_sha256"],
        "training_data_sha256": card["training_data_sha256"],
        "base_training_csv_sha256": card["base_training_csv_sha256"],
        "shipped_bundle_sha256": card["shipped_bundle_sha256"],
        "shipped_bundle_unchanged": "yes" if card["shipped_bundle_unchanged"] else "NO",
        "reference_scores_sha256": card["reference_scores_sha256"],
        "reference_scores_unchanged": "yes" if card["reference_scores_unchanged"] else "NO",
        "candidate_dir": card.get("candidate_dir", ""),
        "scientific_limit": CANDIDATE_LIMIT,
    }

    matrix = card["transition_matrix"]
    metrics: Dict[str, float] = {
        "rows_total": float(card["rows_total"]),
        "rows_base_uci": float(card["rows_base_uci"]),
        "rows_from_review_queue": float(card["rows_from_review_queue"]),
        "rows_rejected": float(card["rows_rejected"]),
        "moved_out_of_uncertain": float(matrix["moved_out_of_uncertain"]),
        "moved_into_uncertain": float(matrix["moved_into_uncertain"]),
        "decisions_changed_total": float(matrix["decisions_changed_total"]),
    }
    for key, count in matrix["counts"].items():
        metrics[f"transition_{key}"] = float(count)
    for decision, count in matrix["shipped_totals"].items():
        metrics[f"shipped_total_{decision}"] = float(count)
    for decision, count in matrix["candidate_totals"].items():
        metrics[f"candidate_total_{decision}"] = float(count)

    coverage = card["conformal_coverage"]
    metrics["coverage_declared_target_1_minus_alpha"] = float(coverage["declared_target"])
    for which in (
        "candidate_in_sample", "candidate_cross_conformal",
        "shipped_in_sample", "shipped_cross_conformal",
    ):
        for cell, value in coverage[which].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"coverage_{which}_{cell}"] = float(value)
    return params, metrics
