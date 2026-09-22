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
"""

import pytest

from backend.monitoring import drift as drift_module


class TestReasonIsStated:
    def test_a_reason_is_available_whenever_the_monitor_cannot_run(self):
        reason = drift_module.drift_unavailable_reason()
        if drift_module._evidently_available:
            assert reason is None
        else:
            assert reason, "drift is unavailable and no reason is given"
            assert "evidently" in reason.lower()

    def test_the_reason_distinguishes_absent_from_incompatible(self):
        """
        The two cases need different answers: one is 'pip install', the other
        is 'the code targets an API this version removed'.
        """
        reason = drift_module.drift_unavailable_reason()
        if reason is None:
            pytest.skip("evidently imports cleanly in this environment")
        assert ("not installed" in reason) != ("incompatible" in reason), (
            "the reason should say exactly one of 'not installed' or "
            f"'incompatible', got: {reason}"
        )

    def test_an_incompatible_install_points_at_the_cost_report(self):
        reason = drift_module.drift_unavailable_reason() or ""
        if "incompatible" in reason:
            assert "EVIDENTLY_COST.md" in reason


@pytest.mark.asyncio
class TestStatusEndpoint:
    async def test_status_reports_not_ready_and_says_why(
        self, client, admin_token, db_tables
    ):
        resp = await client.get(
            "/api/v4/admin/drift/diabetes/status",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["monitor_ready"] is False
        if not drift_module._evidently_available:
            assert "evidently" in body["message"].lower(), (
                "the endpoint reported 'no report run yet' when the monitor "
                "cannot run at all"
            )

    async def test_drift_status_still_requires_admin(self, client, doctor_token, db_tables):
        resp = await client.get(
            "/api/v4/admin/drift/diabetes/status",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403
