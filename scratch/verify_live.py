"""
Diff a deployed OmniDiag backend against the local environment, for every
demo patient in frontend/src/mockPatients.js.

    backend/.venv/bin/python scratch/verify_live.py                      # live HF Space
    backend/.venv/bin/python scratch/verify_live.py --url http://127.0.0.1:7860
    backend/.venv/bin/python scratch/verify_live.py --save scratch/verify_live_result.json

The local side is the REAL FastAPI app called in-process (no server needed),
so both sides go through identical validation, caching and serialisation.
Compared per patient:
  /predict          prediction, diagnosis, confidence, inference_threshold
  /explain          every chart_data SHAP value, by feature name
  /counterfactuals  status, scenario count, best_achievable probability
Exit code 0 = identical within tolerance, 1 = differences.

Why this exists: the Space once resolved scikit-learn 1.7.2 (models saved
with 1.8/1.9) and served different diabetes probabilities from the same
pickles (Case C 63.7% local vs 49.4% live). requirements.txt now pins the
numeric stack; this script is the check that the pin holds.
"""
import os
import sys

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import argparse
import json
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scratch"))

from golden_master import load_demo_patients  # noqa: E402

LIVE = "https://yahyoha-omnidiag.hf.space"
TOL = 1e-6
ENDPOINTS = ("predict", "explain", "counterfactuals")


def remote_call(base, path, body, timeout=300):
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:      # rate limit: back off
                time.sleep(20)
                continue
            try:
                return e.code, json.loads(e.read())
            except Exception:
                return e.code, None


def summarise(endpoint, body):
    """The fields compared for one endpoint response."""
    if body is None:
        return None
    if endpoint == "predict":
        return {k: body.get(k) for k in ("prediction", "diagnosis", "confidence", "inference_threshold")}
    if endpoint == "explain":
        return {row["feature"]: row["shap_value"] for row in body.get("chart_data", [])}
    best = body.get("best_achievable") or {}
    return {
        "status": body.get("status"),
        "n_scenarios": len(body.get("counterfactuals") or []),
        "best_probability": next(
            (best[k] for k in ("new_probability_corrected", "new_probability", "probability")
             if isinstance(best.get(k), (int, float))), None),
        "baseline_probability": body.get("baseline_probability"),
    }


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'local' if k in a else 'remote'}")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > TOL:
            out.append(f"{path}: local {a:.6g} vs remote {b:.6g} (|d|={abs(a - b):.3g})")
    elif a != b:
        out.append(f"{path}: local {a!r} vs remote {b!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=LIVE)
    ap.add_argument("--pause", type=float, default=2.2, help="seconds between remote calls (rate limit 30/min)")
    ap.add_argument("--save")
    args = ap.parse_args()

    from fastapi.testclient import TestClient
    from backend.main import app
    import logging
    logging.disable(logging.WARNING)   # the app logs at DEBUG; keep the report readable
    import warnings
    warnings.filterwarnings("ignore")

    patients = load_demo_patients()
    report, n_diff = {}, 0
    with TestClient(app) as local:
        for disease, cases in patients.items():
            for pid, data in cases:
                row = {}
                for ep in ENDPOINTS:
                    path = f"/api/v4/{disease}/{ep}"
                    lr = local.post(path, json=data)
                    rs, rb = remote_call(args.url, path, data)
                    time.sleep(args.pause)
                    l_sum = summarise(ep, lr.json()) if lr.status_code == 200 else {"http": lr.status_code}
                    r_sum = summarise(ep, rb) if rs == 200 else {"http": rs}
                    d = diff(l_sum, r_sum)
                    n_diff += len(d)
                    row[ep] = {"local": l_sum, "remote": r_sum, "diff": d}
                report[f"{disease}/{pid}"] = row
                p = row["predict"]
                conf = lambda s: f"{s['confidence']*100:.2f}%" if s and "confidence" in s else str(s)
                status = "OK  " if not any(row[e]["diff"] for e in ENDPOINTS) else "DIFF"
                print(f"{status} {disease:14s} {pid}  predict local {conf(p['local'])} remote {conf(p['remote'])}"
                      f"  | explain diffs {len(row['explain']['diff'])}  | cf diffs {len(row['counterfactuals']['diff'])}")
                for ep in ENDPOINTS:
                    for line in row[ep]["diff"][:6]:
                        print(f"       {ep}{line}")

    print(f"\n{len(report)} patients x {len(ENDPOINTS)} endpoints against {args.url}: "
          f"{'IDENTICAL' if n_diff == 0 else f'{n_diff} difference(s)'} (tol {TOL})")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if n_diff == 0 else 1)


if __name__ == "__main__":
    main()
