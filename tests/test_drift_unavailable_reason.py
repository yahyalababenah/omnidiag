"""
14b — the drift status must state the real reason it cannot run.

The Space log said "evidently not installed", which sent anyone reading it
looking for a missing dependency. Evidently IS installed there:
requirements.txt asks for `evidently>=0.4.0`, the Dockerfile installs it, and
it resolves to 0.7.x — which removed `ColumnMapping` and
`evidently.report.Report`, the 0.4 API that backend/monitoring/drift.py
imports. So the import fails with the package present.

Nothing is installed to fix this; see docs/EVIDENTLY_COST.md for why. What
changed is that the message no longer misstates the cause.

UPDATED IN GATE 8.7c. The live drift path is no longer Evidently: it is KS,
chi-square and PSI on scipy against a frozen reference profile, so the monitor
now IS ready and `drift_unavailable_reason()` answers about THAT path. The two
tests that asserted "not ready, and here is the Evidently reason" were asserting
a fault this gate removed, so they now assert the same CONTRACT against the live
path: whenever the monitor cannot run, a reason names what is wrong.

The Evidently reason is still checked, through `evidently_unavailable_reason()`,
because that path and its cost report are still on record.
"""

import pytest

from backend.monitoring import drift as drift_module


class TestReasonIsStated:
    """The Evidently path: still on record, still explains itself."""

    def test_the_evidently_reason_distinguishes_absent_from_incompatible(self):
        """
        The two cases need different answers: one is 'pip install', the other
        is 'the code targets an API this version removed'.
        """
        reason = drift_module.evidently_unavailable_reason()
        if reason is None:
            pytest.skip("evidently imports cleanly in this environment")
        assert ("not installed" in reason) != ("incompatible" in reason), (
            "the reason should say exactly one of 'not installed' or "
            f"'incompatible', got: {reason}"
        )

    def test_an_incompatible_install_points_at_the_cost_report(self):
        reason = drift_module.evidently_unavailable_reason() or ""
        if "incompatible" in reason:
            assert "EVIDENTLY_COST.md" in reason


class TestLiveMonitorReason:
    """The live path (Gate 8.7c): ready, and when it is not, it says why."""

    def test_the_live_monitor_is_ready_and_gives_no_reason(self):
        assert drift_module.drift_unavailable_reason("heart_disease") is None
        assert drift_module.get_monitor("heart_disease").is_ready

    def test_the_live_reason_does_not_blame_evidently(self):
        """The old function answered about Evidently for a path that no longer uses it."""
        reason = drift_module.drift_unavailable_reason("heart_disease")
        assert reason is None or "evidently" not in reason.lower()

    def test_a_missing_profile_names_the_file_and_how_to_build_it(self, tmp_path):
        monitor = drift_module.ProfileDriftMonitor(
            "heart_disease", tmp_path / "absent.json")
        assert not monitor.is_ready
        reason = monitor.unavailable_reason()
        assert "absent.json" in reason
        assert "build_drift_reference" in reason

    def test_an_unreadable_profile_is_distinguished_from_a_missing_one(self, tmp_path):
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        monitor = drift_module.ProfileDriftMonitor("heart_disease", bad)
        assert not monitor.is_ready
        assert "unreadable" in monitor.unavailable_reason()


@pytest.mark.asyncio
class TestStatusEndpoint:
    async def test_status_reports_ready_now_that_the_live_path_works(
        self, client, admin_token, db_tables
    ):
        """This asserted `monitor_ready is False` until Gate 8.7c made it true."""
        resp = await client.get(
            "/api/v4/admin/drift/heart_disease/status",  # diabetes (BRFSS) is retired: 410
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["monitor_ready"] is True
        assert "evidently" not in body["message"].lower(), (
            "the status message still blames Evidently for a path that no "
            f"longer uses it: {body['message']}"
        )

    async def test_drift_status_still_requires_admin(self, client, doctor_token, db_tables):
        resp = await client.get(
            "/api/v4/admin/drift/heart_disease/status",  # diabetes (BRFSS) is retired: 410
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403
