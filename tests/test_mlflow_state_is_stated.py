"""
Gate 8.6 — "MLflow is empty" must be an answer, not a mystery.

The experiment was empty for three separate reasons, and the system reported
all of them identically as `count: 0`:

  1. Nothing logged to it. The only automatic caller was `retrain_xgb`, which
     fails for heart (it looks for a filename that no longer exists) and is a
     no-op for diabetes (it writes a file the live ensemble never reads). The
     script that actually builds the shipped artifact logged nothing at all.
  2. `list_recent_runs()` returned [] whether the package was missing, the
     store was unreachable, or the store was reachable and empty.
  3. mlflow is not installed in backend/.venv, though it is in requirements.txt
     and in the image (F8-4).

These tests pin the fix for (1) and (2). (3) is an environment fact, recorded
rather than changed: installing into a pinned pre-demo venv is its own risk.
"""

import importlib
import os

import pytest

from backend.monitoring import mlflow_tracker


class TestTheStateIsNamed:
    def test_every_state_names_itself(self):
        state = mlflow_tracker.tracking_state()
        assert state["status"] in {"unavailable", "unreachable", "empty", "ok"}
        assert state["tracking_uri"], "the caller cannot act without knowing where it looked"
        # Anything other than a working store with runs owes an explanation.
        if state["status"] != "ok":
            assert state["detail"], f"{state['status']} with no reason given"

    def test_unavailable_is_distinguishable_from_empty(self):
        """
        The whole point: `count: 0` used to mean both. In this environment the
        package is absent, so the status must say so rather than report an
        empty experiment that was never consulted.
        """
        state = mlflow_tracker.tracking_state()
        if not mlflow_tracker._mlflow_available:
            assert state["status"] == "unavailable"
            assert "not installed" in state["detail"]
            assert state["runs"] == []
        else:
            assert state["status"] in {"ok", "empty", "unreachable"}

    def test_the_endpoint_passes_the_status_through(self):
        """A status the API does not return is a status nobody can see."""
        import inspect

        from backend.monitoring import routes

        source = inspect.getsource(routes.mlflow_runs)
        for field in ("status", "tracking_uri", "detail", "count", "runs"):
            assert field in source, f"/mlflow/runs does not return {field}"


class TestLoggingNeverBreaksTheBuild:
    def test_logging_returns_none_instead_of_raising_when_unavailable(self):
        """
        The build must not fail because a tracking store was absent. On a
        deployment whose MLFLOW_TRACKING_URI points at a server there is no
        server to reach during an image build, which is the normal case.
        """
        run_id = mlflow_tracker.log_build_artifact(
            disease="heart_disease", model_version="test",
            params={"family": "glm_ivap_conformal"}, metrics={"bundle_size_kb": 1.0},
        )
        assert run_id is None or isinstance(run_id, str)

    def test_a_broken_store_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "MLFLOW_TRACKING_URI", "http://127.0.0.1:59999")
        state = mlflow_tracker.tracking_state()
        assert state["status"] in {"unavailable", "unreachable"}
        assert state["runs"] == []

    def test_the_trainer_logs_provenance_and_not_performance(self):
        """
        What goes into the run is the training data's hash, the artifact's hash
        and the reproducibility fingerprint. NOT an accuracy figure: every
        performance number for this model is cross-fitted or leave-one-hospital-
        out and lives in the research repo, and an in-sample number logged
        beside the artifact would read as the headline.
        """
        import ast
        import inspect
        import textwrap

        function = importlib.import_module("scripts.train_heart_glm")._log_build_run
        source = inspect.getsource(function)
        for provenance in ("training_csv_sha256", "bundle_sha256", "fingerprint_decision_mismatches"):
            assert provenance in source, f"the build run does not record {provenance}"

        # The docstring is excluded on purpose: it has to NAME the figures it
        # refuses to log, and a blunt substring check would flag the refusal
        # itself. Only the code is searched.
        tree = ast.parse(textwrap.dedent(source))
        body = tree.body[0].body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        code = "\n".join(ast.unparse(node) for node in body).lower()
        for performance in ("auc", "accuracy", "sensitivity", "roc_"):
            assert performance not in code, (
                f"the build run logs a performance figure ({performance})"
            )


class TestRetrainIsNotClaimedToWork:
    def test_the_only_retrain_path_left_is_the_candidate_builder(self):
        """
        Gate 8.6 pinned `retrain_xgb` so the retrain cycle could not be quietly
        declared fixed. Gate 8.10 routed heart to a candidate builder; gate B4
        removed `retrain_xgb` itself (no live module read what it wrote). What
        reaches MLflow from a retrain is therefore only the candidate run.
        """
        from backend.active_learning import retrain

        source = inspect_source(retrain)
        assert not hasattr(retrain, "retrain_xgb")
        assert not hasattr(retrain, "_log_to_mlflow")
        assert "omni_diag_xgb_optimized.pkl" not in source
        assert "start_candidate_run" in source


def inspect_source(module) -> str:
    import inspect

    return inspect.getsource(module)
