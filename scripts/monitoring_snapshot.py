"""Save a timestamped JSON snapshot of OmniDiag's Prometheus metrics.

Stdlib only, so it runs from cron or a systemd timer with the system python.

    python3 scripts/monitoring_snapshot.py                 # live Space (+ local backend if up)
    python3 scripts/monitoring_snapshot.py --url http://localhost:7860/metrics

Writes reports/monitoring_history/<UTC timestamp>.json and overwrites latest.json.
A target that is down is recorded as such; the script still exits 0 so a timer
does not mark itself failed just because the Space was asleep.

What is saved is what the endpoint reported at that moment: cumulative counters
since the process started (they reset when the Space restarts), so compare
snapshots by difference, not by absolute value.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "monitoring_history"
DEFAULT_TARGETS = {
    "live_space": "https://yahyoha-omnidiag.hf.space/metrics",
    "local_backend": "http://localhost:7860/metrics",
}
SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, list[dict]]:
    """Keep only omnidiag_* series; drop *_created timestamps and histogram buckets."""
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m or not m["name"].startswith("omnidiag_"):
            continue
        if m["name"].endswith("_created") or m["name"].endswith("_bucket"):
            continue
        out.setdefault(m["name"], []).append(
            {"labels": dict(LABEL.findall(m["labels"] or "")), "value": float(m["value"])})
    return out


def fetch(url: str, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return {"url": url, "up": True, "metrics": parse(r.read().decode())}
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable right now"
        return {"url": url, "up": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="scrape only this /metrics URL")
    ap.add_argument("--timeout", type=int, default=60, help="seconds; a sleeping Space needs time to wake")
    args = ap.parse_args()

    targets = {"custom": args.url} if args.url else DEFAULT_TARGETS
    now = dt.datetime.now(dt.timezone.utc)
    snap = {"taken_at": now.isoformat(), "targets": {k: fetch(u, args.timeout) for k, u in targets.items()}}

    OUT.mkdir(parents=True, exist_ok=True)
    body = json.dumps(snap, indent=2)
    (OUT / f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}.json").write_text(body)
    (OUT / "latest.json").write_text(body)
    print({k: ("up" if v["up"] else "DOWN") for k, v in snap["targets"].items()}, "->", OUT)


if __name__ == "__main__":
    main()
