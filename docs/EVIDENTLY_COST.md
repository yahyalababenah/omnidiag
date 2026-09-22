# Cost of enabling Evidently drift monitoring in the Space image

**Date:** 2026-09-23 · **Status:** REPORT ONLY — nothing installed, nothing changed in
`requirements.txt`. Measured with `pip install --dry-run --report` against the
validated environment (`backend/.venv`, Python 3.13) and PyPI release metadata.

## The finding that changes the question

Evidently is **already installed in the Space image**. `requirements.txt:70` has
`evidently>=0.4.0`, and the Space's `Dockerfile` runs
`pip install --no-cache-dir -r requirements.txt`, so every build downloads and
installs it. The image has already paid the whole cost.

It is unusable at runtime for a different reason:
[`backend/monitoring/drift.py:35-44`](../backend/monitoring/drift.py#L35) imports the
**0.4 API** —

```python
from evidently import ColumnMapping
from evidently.report import Report
```

— and `evidently>=0.4.0` resolves to **0.7.23**, which removed both. The
`except ImportError` branch then logs *"evidently not installed — drift
monitoring unavailable"*, which is why the Space log says it is missing when it
is present. The status the API reports (`monitor_ready: false`) is correct; its
stated reason is not.

## Option A — pin `evidently==0.4.40` (the version the code targets)

**Not viable.** Release metadata for 0.4.40:

```
numpy<2.1,>=1.22.0
```

The validated environment pins `numpy==2.4.6` (`requirements.txt:19`). Installing
0.4.40 forces a numpy downgrade across the whole numeric stack — which is exactly
the class of change that produced X-2, where a library drift between LOCAL and
LIVE moved Case C from 63.7% to 49.4%. The golden master would have to be
re-baselined and every shipped metric re-verified.

Cost: unacceptable before a demo, for a feature nothing in the UI displays.

## Option B — keep 0.7.23, rewrite `drift.py` for the new API

Compatible with the pinned stack. The dry-run resolves **44 packages, and moves
nothing already installed** — numpy, pandas, scikit-learn, scipy, xgboost,
lightgbm and shap all stay exactly where they are, so `/predict` and `/explain`
outputs are untouched.

| package | version | wheel |
|---|---|---|
| pyarrow | 25.0.1 | 46.8 MB |
| plotly | 5.24.1 | 19.1 MB |
| statsmodels | 0.15.0 | 11.8 MB |
| evidently | 0.7.23 | 11.7 MB |
| uvloop | 0.22.1 | 4.3 MB |
| Faker | 40.39.0 | 2.1 MB |
| nltk | 3.10.3 | 1.8 MB |
| litestar + 36 others | — | 6.0 MB |
| **total** | **44 packages** | **103.5 MB of wheels** |

Unpacked that is roughly 250–300 MB of image, all of it already being downloaded
and installed on every Space build today. Evidently 0.7 also pulls a web-server
stack (`litestar`, `msgspec`, `multipart`, `uvloop`) for its own UI, which this
product does not use.

Remaining work: rewrite `DriftMonitor` against the 0.7 API (`ColumnMapping` →
`DataDefinition`, `Report` relocated, metric classes renamed), plus a reference
window to compare against. There is no drift panel in the frontend, so nothing
would be visible to a judge without building one.

## Option C — do nothing before 1 October (recommended)

Prometheus stays and works. No drift claim is made anywhere in the UI, and the
README and FEATURE_VERIFICATION both say drift monitoring is not deployed.

The only change worth making now costs nothing and removes a false statement
from the logs: make the `except ImportError` branch say *"evidently is installed
but its API is incompatible (drift.py targets 0.4, found 0.7)"* instead of
*"evidently not installed"*.

## Recommendation

Option C for the demo, Option B afterwards if a drift panel is actually wanted.
Option A should not be taken at any point — a numpy downgrade to enable a
feature with no UI is a bad trade against a model whose outputs are the product.
