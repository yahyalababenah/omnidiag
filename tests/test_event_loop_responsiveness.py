"""
X-3 — the server must stay responsive during heavy ML compute.

The failure this pins down: /predict, /explain and /counterfactuals are
`async def` endpoints that used to call the synchronous router straight from
the coroutine body. Anything they did occupied the one event-loop thread, so
a single 9-14 s diabetes What-If froze every other visitor at a booth — a
plain GET / was measured at 17,981 ms while a batch ran.

Each test here runs a deliberately slow, *blocking* router call and asserts a
concurrent GET / still returns promptly. These assertions fail outright on the
pre-fix code, where the health check cannot be served until the model call
returns.
"""

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

# The blocking call's duration. Long enough that a blocked event loop is
# unambiguous, short enough to keep the suite quick.
BLOCK_SECONDS = 1.5

# A health check that had to wait behind the blocking call would take at
# least BLOCK_SECONDS. This budget is far below that and far above the few
# milliseconds GET / actually needs, so the test is not timing-fragile.
RESPONSIVE_BUDGET_SECONDS = 0.6

def heart_patient(age: int) -> dict:
    """
    A valid heart_disease payload. Age varies per test so each one misses the
    predict/counterfactual cache -- a cache hit never reaches the model call
    and would make these tests pass vacuously.
    """
    return {
        "Age": age, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 130,
        "Cholesterol": 245, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat",
    }


def _blocking(return_value):
    """A stand-in for synchronous model work: it really does block its thread."""
    def _call(*_args, **_kwargs):
        time.sleep(BLOCK_SECONDS)
        return return_value
    return _call


async def _assert_stays_responsive(app, client, method_name, path, payload):
    """
    Fire the slow endpoint, then immediately ask for GET / and time the whole
    thing from before the slow request was even launched.

    Timing from t0 rather than from just before the GET is the point. Sleeping
    first to "let the slow call get going" measures nothing: on blocking code
    the sleep itself does not resume until the block is over, so the GET is
    only ever timed once the event loop is free again and the test passes
    whatever the endpoint does. Measured from t0, a blocked loop shows up
    directly — the health check cannot even be dispatched for BLOCK_SECONDS.
    """
    original = getattr(app_router(app), method_name)
    setattr(app_router(app), method_name, _blocking(original.return_value))
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as probe:
            start = time.perf_counter()
            slow = asyncio.create_task(client.post(path, json=payload))
            resp = await probe.get("/")
            elapsed = time.perf_counter() - start

            assert resp.status_code == 200
            assert resp.json()["status"] == "Healthy"
            assert elapsed < RESPONSIVE_BUDGET_SECONDS, (
                f"GET / took {elapsed:.3f}s while {path} was computing — the "
                f"event loop is blocked by synchronous ML work (X-3)"
            )

            slow_resp = await slow
            assert slow_resp.status_code == 200
    finally:
        setattr(app_router(app), method_name, original)


def app_router(app):
    """The router object the endpoints actually call (mocked in conftest)."""
    import backend.main as main_module
    return main_module.router


@pytest.mark.asyncio
async def test_health_check_stays_fast_during_counterfactuals(app, client):
    """The What-If path — the 9-14 s one that froze the booth."""
    await _assert_stays_responsive(
        app, client, "counterfactuals",
        "/api/v4/heart_disease/counterfactuals", heart_patient(41),
    )


@pytest.mark.asyncio
async def test_health_check_stays_fast_during_predict(app, client):
    await _assert_stays_responsive(
        app, client, "predict",
        "/api/v4/heart_disease/predict", heart_patient(42),
    )


@pytest.mark.asyncio
async def test_health_check_stays_fast_during_explain(app, client):
    await _assert_stays_responsive(
        app, client, "explain",
        "/api/v4/heart_disease/explain", heart_patient(43),
    )


@pytest.mark.asyncio
async def test_two_slow_predictions_overlap_instead_of_queueing(app, client):
    """
    Two concurrent model calls should run side by side in the threadpool, not
    one after the other. Serial execution would take ~2x BLOCK_SECONDS.
    """
    router = app_router(app)
    original = router.counterfactuals
    router.counterfactuals = _blocking(original.return_value)
    try:
        start = time.perf_counter()
        first, second = await asyncio.gather(
            client.post("/api/v4/heart_disease/counterfactuals", json=heart_patient(44)),
            client.post("/api/v4/heart_disease/counterfactuals", json=heart_patient(45)),
        )
        elapsed = time.perf_counter() - start
    finally:
        router.counterfactuals = original

    assert first.status_code == 200
    assert second.status_code == 200
    assert elapsed < BLOCK_SECONDS * 1.8, (
        f"two concurrent model calls took {elapsed:.2f}s — they are being "
        f"serialised rather than running in the threadpool"
    )
