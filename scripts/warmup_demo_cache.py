#!/usr/bin/env python3
"""
Warm the prediction / SHAP / What-If caches for every demo patient.

Why this exists
---------------
A diabetes What-If costs 9-14 s uncached. The first judge to click it waits
that long and concludes the product is slow. Run this once before judging and
every demo patient's /predict, /explain and /counterfactuals response is
already in the server's cache, so every click is served in milliseconds.

Cache lifetime is CACHE_TTL_COUNTERFACTUALS (default 12 h), long enough that
a single morning run covers the whole day. The server's cache is in-process
memory, so a Space restart or rebuild empties it -- re-run this after any
restart.

Usage
-----
    # against the live Space (the default)
    backend/.venv/bin/python scripts/warmup_demo_cache.py

    # against a local backend
    backend/.venv/bin/python scripts/warmup_demo_cache.py \
        --base-url http://127.0.0.1:8000

    # skip the slow endpoint if you only want predictions warm
    backend/.venv/bin/python scripts/warmup_demo_cache.py --skip-counterfactuals

Exit code is 0 only when every request succeeded.

The demo patients are read from frontend/src/mockPatients.js, the same file
the UI renders, so the warmed inputs cannot drift from the ones on screen.
Requires node on PATH to read that file (the same dependency
scratch/golden_master.py already has).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = "https://yahyoha-omnidiag.hf.space"

# A cold Space plus a diabetes counterfactual generation. Generous on purpose:
# a timeout here would report a failure for a request that was going to
# succeed, and re-running is cheap.
REQUEST_TIMEOUT_SECONDS = 180


def load_demo_patients() -> dict:
    """{disease: [(patient_id, patient_data), ...]} from the frontend's own file."""
    src = os.path.join(ROOT, "frontend", "src", "mockPatients.js")
    if not os.path.isfile(src):
        sys.exit(f"demo patients not found at {src}")

    # mockPatients.js is an ES module; import it by path and print it as JSON.
    tmp = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"warmup_mock_{uuid.uuid4().hex}.mjs"
    )
    with open(src) as f, open(tmp, "w") as g:
        g.write(f.read())
    try:
        out = subprocess.run(
            ["node", "-e",
             "import(process.argv[1]).then(m => console.log(JSON.stringify(m.default)))",
             tmp],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        sys.exit("node is not on PATH — it is needed to read frontend/src/mockPatients.js")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"could not read demo patients:\n{exc.stderr}")
    finally:
        os.remove(tmp)

    return {
        disease: [(p["id"], p["data"]) for p in patients]
        for disease, patients in json.loads(out.stdout).items()
    }


def post(base_url: str, path: str, payload: dict) -> tuple[bool, float, str]:
    """POST *payload*; return (ok, elapsed_seconds, note)."""
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            resp.read()
            elapsed = time.perf_counter() - start
            # The server reports whether this response was already cached, so
            # a second run of this script visibly proves the cache is live.
            hit = resp.headers.get("Cache-Hit")
            note = {"true": "already cached", "false": "computed"}.get(hit, "")
            return True, elapsed, note
    except urllib.error.HTTPError as exc:
        return False, time.perf_counter() - start, f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}"
    except Exception as exc:
        return False, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument("--base-url", default=os.getenv("OMNIDIAG_BASE_URL", DEFAULT_BASE_URL),
                        help=f"API root to warm (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--skip-counterfactuals", action="store_true",
                        help="warm only /predict and /explain")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    patients = load_demo_patients()

    endpoints = ["predict", "explain"]
    if not args.skip_counterfactuals:
        endpoints.append("counterfactuals")

    total = sum(len(p) for p in patients.values()) * len(endpoints)
    print(f"Warming {total} responses at {base_url}\n")

    failures = []
    started = time.perf_counter()

    for disease, demo in sorted(patients.items()):
        print(f"  {disease}")
        for patient_id, data in demo:
            for endpoint in endpoints:
                ok, elapsed, note = post(
                    base_url, f"/api/v4/{disease}/{endpoint}", data
                )
                mark = "ok  " if ok else "FAIL"
                suffix = f"  ({note})" if note else ""
                print(f"    {mark} {patient_id:<8} {endpoint:<16} {elapsed:6.2f}s{suffix}")
                if not ok:
                    failures.append((disease, patient_id, endpoint, note))
        print()

    print(f"Done in {time.perf_counter() - started:.1f}s — "
          f"{total - len(failures)}/{total} warmed")

    if failures:
        print("\nFailures:")
        for disease, patient_id, endpoint, note in failures:
            print(f"  {disease}/{patient_id}/{endpoint}: {note}")
        return 1

    print("Every demo patient is cached. Re-run after any Space restart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
