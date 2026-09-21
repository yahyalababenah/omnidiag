# OmniDiag — Weakness Register

> ⚠️ **OPERATIONAL WARNING — read before running any batch prediction.**
> **Do not upload a diabetes batch file larger than ~20-30 rows until HM-4 is fixed.**
> Larger batches can make the system appear to hang, or fail silently, well before any
> visible error. This is a live constraint, not just a line item — see **HM-4** in Part 7.

**Purpose:** the single work-driving list for the 19 days before the judged technical evaluation.
**Built:** 2026-09-15 · evaluation ≈ 2026-10-04 · branch `deploy/v2-platform` @ `698974b`
**Nothing in this document was fixed.** Verification only.

## Sources merged

| # | Source | Status |
|---|---|---|
| 1 | `AUDIT_REPORT.md` — 16 CRITICAL / 14 WARNING + `evaluation_evidence/` | ✅ merged, not re-audited |
| 2 | `docs/OmniDiag_Proposal_Defense.md` — 31 questions (س1–س31), 49 references, 15 system-crossing claims | ✅ merged |
| 3 | Prior-session findings (6 items, supplied in the brief) | ✅ merged, **all 6 re-verified — 2 were wrong** |
| — | Gap-fill checks A–E (new work, this session) | ✅ 9 new findings |

> **Path note:** you sent `docs/DEFENSE_PROPOSAL.md`; no such file exists. The document
> matching your description (31 questions, gap table) is **`docs/OmniDiag_Proposal_Defense.md`**,
> created today 16:19. That is what I merged. It is currently **untracked** — commit it.

## Severity scale

**S1** — will be found and will cost points in a scored category.
**S2** — likely to be found; damages credibility but is survivable with a prepared answer.
**S3** — found only if a judge digs; cheap to fix.
**S4** — cosmetic / hygiene.

## Rubric keys
`Demo 25` · `Technical 20` · `Evidence 5` · `Q&A 5` · `Docs`

---

# PART 0 — Gap-fill check results (new evidence, this session)

## Check A — full-history secret scan (all 22 refs, 899 commits, 2,724 objects)

Method: `git cat-file --batch-all-objects`, every blob < 2 MB matched against 12 live-credential
patterns (`hf_`, `sk-`, `ghp_`, `gho_`, `github_pat_`, `AKIA`, `AIza`, `xox[baprs]-`, `glpat-`,
`sk_live_`, `SG.`, PEM private keys).

**One blob hit — and it is the worst finding in the entire engagement.**

```
blob 47b23249221f8d33c5b21c0d9aa9a8955fab8051  (17,232 bytes)
path experiment_files/production_audit_and_action_plan.md
  line  35:  | **Secret** | `ghp_vkVLzf…REDACTED` |        ← GitHub Personal Access Token
  line  47:  | **Secret** | `hf_aOYuxa…REDACTED`  |        ← Hugging Face write token
  line 210:  | 0.1 | **Revoke GitHub PAT** | …delete token `ghp_vkVLzf…` |
  line 211:  | 0.2 | **Revoke HF Token**   | …delete token `hf_aOYuxa…`  |
```

A previous security audit correctly found two tokens in `.git/config`, **wrote them verbatim into
a markdown file, and committed it.** The file was later deleted from the working tree
(`32cf98d Delete experiment_files/production_audit_and_action_plan.md`) — **deletion does not
remove it from history.**

**Reachable from 7 pushed refs:**

| Remote ref | Reachable via |
|---|---|
| `origin/main` | `ab4d123` |
| `origin/deploy/v2-platform` | `ab4d123` |
| `origin/feature/omni-platform-final` | `6532f05` |
| `origin/feature/diabetes-module-integration` | `6532f05` |
| `origin/production-final-v3` | `2065d4f` |
| **`hf/main` — the public Hugging Face Space** | `ab4d123` |
| `hf` | `ab4d123` |

Plus 8 local branches. Anyone who clones the GitHub repo or the HF Space gets both tokens.
This is **separate from** and **in addition to** the third token (`hf_KueNwzRHce…`) that
`AUDIT_REPORT.md` C-22 found live in `.git/config`. **Three credentials total.**

> Whether the two committed tokens were ever actually revoked is **UNVERIFIED** — confirming it
> means calling the GitHub/HF APIs with them, which I will not do. See Decision H-1.

## Check B — diabetes `/counterfactuals` raw shape vs heart vs the component

Live capture, D-002 Karim Yaghi (`scratch/cf_raw.json`):

```jsonc
// DIABETES — POST /api/v4/diabetes/counterfactuals
{ "scenario": "If BMI drops to 29.1, blood pressure is controlled and physical health improves (0 poor days)",
  "changes": { "HighBP": 0.0, "BMI": 29.06559412332036, "PhysHlth": 0.0 },   // ← OBJECT
  "new_probability": 0.2595, "risk_reduction": "40%", "feasibility": "low" }

// HEART — POST /api/v4/heart_disease/counterfactuals
{ "scenario_id": 1, "probability": 0.2555962800979614,
  "changes": [ { "feature": "Oldpeak", "original_value": 2.3,                // ← ARRAY
                 "counterfactual_value": 0.0, "direction": "decrease" } ] }
```

| Field the component reads | Heart | **Diabetes** | Where |
|---|---|---|---|
| `scenario_id` | present | **MISSING** | `WhatIfScenarioCard.jsx:166` — *the branch discriminator* |
| `probability` | present | **MISSING** (it is `new_probability`) | `:131` |
| `changes` | **array** | **object** | `:231-232`, `PDFReport.jsx:194` |
| `feature`,`current`,`proposed`,`riskReduction`,`description` | n/a | **MISSING** | `:185,201,205,209,213` |

Envelope also differs: diabetes adds `message`; heart `status` is `"success"` vs diabetes `"generated"`.
Fields diabetes emits that the frontend **never reads**: `scenario`, `new_probability`,
`risk_reduction`, `feasibility` — i.e. the backend computes correct human-readable text and a
correct "40%" and throws all of it away.

**Exact differing fields:** `scenario_id`↔`scenario`, `probability`↔`new_probability`,
`changes` array↔object, plus `risk_reduction`/`feasibility` unread and `status` value mismatch.

## Check C — is the n_samples=100 + cache optimisation actually deployed?

**Yes — confirmed on all three refs.** `backend/ensemble_loader.py:441` `n_samples=100`; cache at
`backend/main.py:510,520` (`ttl=3600`). Present on `deploy/v2-platform`,
`origin/deploy/v2-platform`, **and `hf/main`** (verified via `git show <ref>:path`).
Commit `860326e perf(counterfactuals): cut diabetes n_samples 500→100, cache results…`.

Timing on a **freshly started process with an empty cache** (`scratch/check_c_cold.py`):

| Call | Seconds | Cache-Hit |
|---|---|---|
| diabetes CF — COLD (models unloaded, cache empty) | **9.83** | false |
| diabetes CF — WARM, **same** patient | 0.01 | true |
| diabetes CF — WARM, **new** patient #1 | **9.27** | false |
| diabetes CF — WARM, **new** patient #2 | **9.57** | false |
| heart CF — COLD | **12.37** | false |
| heart CF — WARM, **new** patient | **10.76** | false |

**Three corrections to the assumptions in the brief and the defense doc:**
1. The optimisation **is** deployed — stop treating it as unverified.
2. The "~18 s on the Space" and the defense doc's "~28–30 s" are both **stale**. Current real
   cost is **~9.5 s (diabetes) / ~10.8–12.4 s (heart)** per unseen patient.
3. `n_samples=100` applies to **diabetes only**. Heart still uses a hardcoded
   `for _ in range(800)` (`backend/model_loader.py:457`) and is now the **slower** of the two.
   The optimisation was never applied to the heart path.

## Check D — claims table (every `.md`, docstring, and the proposal)

| # | Claim | Source | Matches code? |
|---|---|---|---|
| D1 | **"Every figure below is reproducible from the metrics artefacts committed in the repository (`models/heart_disease/metrics.json` and `models/diabetes/ensemble_metrics.json`). No figure in this proposal is quoted from memory or from an older model version."** | Proposal §8.1 | **NO** — that file holds `{"roc_auc_full_fit":0.957,"best_model":"svm","best_cv_roc_auc":0.919}`. **0 of the 7 heart figures appear in it.** |
| D2 | Heart accuracy 80.17% | Proposal §8.1 | **YES** — reproduced exactly |
| D3 | Heart ROC-AUC 0.856 | Proposal §8.1 | **YES** — 0.8564 |
| D4 | Heart sensitivity 80.46% | Proposal §8.1, DOCS:412,965 | **NO** — 71.11% production; no orientation yields it |
| D5 | Heart specificity 83.13% | Proposal §8.1, DOCS:966 | **NO** — 85.53% production |
| D6 | Heart inference threshold 0.420 | Proposal §8.1, DOCS:414 | **NO** — no such constant exists; argmax 0.5 |
| D7 | Diabetes accuracy 75.18% | Proposal §8.1 | **NO at the deployed threshold** — 73.12% @ 0.275 |
| D8 | Diabetes ROC-AUC 0.831 / sens 91.70% / spec 54.63% | Proposal §8.1 | **YES** — 0.8305 / 91.55% / 54.68% |
| D9 | **"raises sensitivity from 87.3% to 91.7%"** | Proposal §4.2; DOCS:482; **defense س20** | **NO** — measured sensitivity at 0.5 is **78.38%**, not 87.3%. Correct statement: 78.4% → 91.6% |
| D10 | **RF meta-learner coefficient 0.711** | **defense س19** | **NO** — shipped `meta_learner.pkl` coef = XGB 0.4362 / LGBM 3.8572 / **RF 0.9019**. 0.711 is from the stale `ensemble_metrics.json` |
| D11 | Dataset "UCI Cleveland + Z-Alizadeh Sani", features incl. "num major vessels, thalassemia" | DOCS:398-399 | **NO** — production is the 605-row merged set; `ca`/`thal` absent everywhere |
| D12 | CV accuracy 88.98% "from `models/metrics.json`" | DOCS:409; arch:74; impl:221 | **NO** — no CV field exists in that file |
| D13 | Archived copies under `models/heart_disease/_archive_v1/` | Proposal §5.4 | **NO** — folder never existed in any commit on any branch |
| D14 | XGBoost params 898 est. / lr 0.0136 | impl:24; `configs/heart_disease.yaml:89-96` | **NO** — shipped pkl is 1124 / 0.006309 / subsample 0.8331 / colsample 0.4792 |
| D15 | 16 features | Proposal §5.1 | **YES** for the shipped model; `diagnostic_v5_ab.json` says 18 (an A/B artifact) |
| D16 | **"automated retrain pipeline with AUC-regression guard"** / "promoted only if held-out AUC does not regress" | Proposal §8.2, §4.5 | **NO** — `retrain.py` contains no AUC computation, no held-out set, no promotion gate. Zero matches for `auc|roc|regress|promote|tolerance` |
| D17 | `/predict` returns `{…uncertainty_band, shap_chart_data, queued_for_review, cached}` | `README.md:548` | **NO** — live response is `['confidence','diagnosis','prediction']`. `uncertainty_band()` exists in `sampler.py:43` but is never attached |
| D18 | `/explain` returns `{shap_values[], …}` | `README.md:549` | **NO** — the field is `chart_data` |
| D19 | `/counterfactuals` returns `{counterfactuals[{changes, feasibility}]}` | `README.md:550` | **PARTIAL** — true for diabetes, false for heart (no `feasibility`) |
| D20 | retrain writes a file "**not** one of the three `EnsembleModelLoader` loads (`xgb_model.pkl`, `lgb_model.pkl`, `rf_model.pkl`) — a silent no-op" | `README.md:155`; `retrain.py:103-105` | **NO — and it is inverted.** It writes `models/diabetes/omni_diag_xgb_optimized.pkl`, which **IS** the live XGB base model (`configs/diabetes.yaml:107`). Not a no-op — a live-model overwrite |
| D21 | "Config-driven router, zero hard-coded disease references" | Proposal §6.2; README | **YES** — `backend/router.py` verified; good live demo |
| D22 | Modular monolith, 3-layer architecture | Proposal §6.3 | **YES** |
| D23 | PWA installable (manifest + service worker) | Proposal §7 | **YES** — `frontend/dist/manifest.webmanifest` (`display:standalone`) + `sw.js` present |
| D24 | SHAP 0.42+ / MLflow 2.10+ | README badges | **YES** — `requirements.txt:25,61` |
| D25 | Python 3.11+ | `README.md:13` badge | **NO** — `Dockerfile:7` is `python:3.10-slim`; local venv is 3.13.5. **Three different Pythons** |
| D26 | Heart module version 5.0.0 | `configs/heart_disease.yaml:15` → `/api/v4/diseases` | **NO** — shipped artifacts say `"version": "5.1.0"` (`models/metrics.json`, `grid_best.json`) |
| D27 | "CI/CD pipeline green on every push" | Proposal §8.2 | **TRUE BUT MISLEADING** — CI runs import check, API start, frontend build. **Zero test steps** despite 13 `tests/test_*.py` files |
| D28 | Federated learning enables cross-hospital improvement | Proposal §10.4 | **NO** — see W-07 |
| D29 | Hot reload, no restart | Proposal §4.5 | **PARTIAL** — true single-process; false under the shipped HPA (2–10 replicas) |
| D30 | K8s/Helm production deployment | Proposal §7 | **UNVERIFIABLE** — `README.md:493` already states the live demo runs on HF Spaces, not K8s. Manifests never applied to a cluster; CI does not even `--dry-run` them |

## Check E — judge-visible surfaces and their contradictions

| Surface | What a judge sees | Contradiction |
|---|---|---|
| `README.md` | No performance numbers at all | ✅ clean — **this is an asset** |
| `README.md:548-553` | API response shapes | ❌ D17/D18/D19 — contradicts the live API |
| `README.md:155` | Retrain limitation | ❌ D20 — names the wrong three files, understates severity |
| **Proposal §8.1** | The metrics table + "every figure reproducible" | ❌ D1, D4–D7, D9 |
| `plans/OMNIDIAG_TECHNICAL_DOCS.md` | 73 KB of detail | ❌ D4–D6, D9, D11, D12, wrong repo org, 58.70% spec ×3 |
| **`/docs` (Swagger)** | Title "OmniDiag Multi-Disease Diagnostic API", **version 4.0.0**, **description in Arabic only** | ⚠️ version 4.0.0 vs module 5.0.0 vs artifact 5.1.0; Arabic-only description — see Decision H-5 |
| `GET /` | `{"version":"4.0.0"}` | ⚠️ same three-way version drift |
| `GET /api/v4/diseases` | `heart_disease version=5.0.0`, **`supports_counterfactuals=True`** | ❌ contradicts `router.py:166-167` hint text claiming heart has no CF generator |
| Admin dashboard | `label="Avg Risk Probability"` (`AdminDashboard.jsx:730`) | ✅ **already correct** — defense gap #10 is stale; only the backing field is still named `avg_confidence` |
| UI prediction cards | Live `confidence` only; threshold via `getDisplayThreshold()` → **50.0%** for heart | ❌ judge sees 50.0% while the proposal says 0.420 |
| What-If card (diabetes) | `undefined → undefined`, "-0%", "Coming Soon" badge | ❌ W-05 |
| `mockPatients.js` comments | "verified live on Space: 75.8%" | ❌ measured 85.0% |

---

# PART 1 — THE REGISTER

> Status legend: **OPEN** · **FIXED** · **REGRESSED** · **PARTIALLY-FIXED**
> Effort: h = hours, d = working days.

## S1 — will be found and will cost points

| ID | What & where | Evidence | Threatens | Sev | Status | Fix type | Effort |
|---|---|---|---|---|---|---|---|
| **W-01** | **Two live credentials committed to git history, pushed to 7 refs including the public HF Space.** `experiment_files/production_audit_and_action_plan.md:35,47` (blob `47b2324`) | Check A. GitHub PAT `ghp_vkVLzf…` + HF token `hf_aOYuxa…`. Reachable from `origin/main`, `origin/deploy/v2-platform`, `hf/main`, +4. Third token live in `.git/config` (AUDIT C-22) | **Docs / Technical 20** — and real-world security | S1 | **OPEN** | decision-needed → code | 2h revoke · 0.5–1d history rewrite |
| **W-02** | **Proposal §8.1 cites `models/heart_disease/metrics.json` as the source for every figure. That file contains SVM metrics.** | Check D1 + AUDIT C-1. File = `{"roc_auc_full_fit":0.957,"best_model":"svm","best_cv_roc_auc":0.919}`. 0 of 7 heart figures present. Defense gap #2 🔴 | **Evidence 5 / Q&A 5 / Technical 20** | S1 | **OPEN** | docs-only + delete artifact | 1h |
| **W-03** | **Heart sensitivity 80.46% / specificity 83.13% / threshold 0.420 are unreproducible.** Proposal §8.1; `DOCS:412-414,965-967` | AUDIT C-10/11/12 + `evaluation_evidence/heart_disease_report.txt`. All 4 orientation×threshold variants tested: production = **71.11% / 85.53%**, argmax 0.5. `grep 0.42` → zero threshold hits. `frontend/src/constants/thresholds.js:18-26` documents argmax in writing. Defense gap #3 🔴 | **Evidence 5 / Q&A 5 / Technical 20** | S1 | **OPEN** | decision-needed (restate vs implement) | 2h restate · 1d implement+revalidate |
| **W-04** | **Diabetes accuracy 75.18% is measured at 0.5 while the product runs at 0.275.** `configs/diabetes.yaml:87`, applied `ensemble_loader.py:328` | AUDIT C-13/C-6 + `evaluation_evidence/diabetes_report.txt`. 73.12% @ 0.275. Arithmetic tell: (91.7+54.6)/2 = 73.17 ≠ 75.18. **2026-09-17 update:** the threshold is no longer 0.275 (see H-1 in Part 5). At the new raw threshold 0.2809 (deployed 0.059776) accuracy is **73.36%** [72.64, 74.09]; source `evaluation_evidence/diabetes/final_metrics_table.md` | **Evidence 5 / Q&A 5** | S1 | **OPEN** | docs-only | 1h |
| **W-05** | **Diabetes What-If renders `undefined`; PDF export throws.** `WhatIfScenarioCard.jsx:166,171`; `PDFReport.jsx:193-195` | Check B (raw JSON both paths). `scenario_id` missing on diabetes → falls to mock branch. `(cf.changes ?? []).map()` on an object → `TypeError` | **Demo 25** | S1 | **PARTIALLY-FIXED** (heart ✅ `bc1b5b4` 2026-09-10; diabetes ❌ never verified) | code | 3h + 1h test |
| **W-06** | **Demo Case C: 0 counterfactuals, 85.0% vs documented 75.8%.** `frontend/src/mockPatients.js:8-16,194` | AUDIT C-20/C-21. Live: `no_valid_counterfactuals`, 0 scenarios after 9.3 s → UI shows heart-disease mock data + "Coming Soon" on a diabetes patient | **Demo 25** | S1 | **REGRESSED** (commits `dd7095e`/`698974b` tuned it to 0.78–0.85 / 3-of-3; neither holds here) | decision-needed → code/data | 2–4h |
| **W-07** | **Federated learning cannot aggregate.** `federated/client.py:67-75` returns `[pickle.dumps(model)]`; `aggregator.py:47` uses Flower `FedAvg` | Session finding, verified. FedAvg element-wise-averages `List[np.ndarray]`; it receives one opaque pickle blob. XGBoost trees cannot be averaged — correct approach is SecureBoost. `add_dp_noise()` (`aggregator.py:96-123`) is **never called** and clips "gradients" that do not exist on this path. Defense gap #1 🔴 + س31 "most dangerous question" | **Technical 20 / Q&A 5** | S1 | **OPEN** | decision-needed (reframe as PoC vs build) | 2h reframe · ≫19d to build |
| **W-08** | **`/admin/retrain` overwrites a LIVE diabetes base model — and the documented limitation is inverted.** `retrain.py:117,143-146`; `README.md:155` | **New this session.** Writes `models/diabetes/omni_diag_xgb_optimized.pkl` = the live XGB base model (`configs/diabetes.yaml:107`). Docs claim it targets a file the ensemble never loads. Also dumps a raw `xgb.Booster` (no `predict_proba`) → next request breaks after `invalidate()` hot-reload. Rename is *after* train (`:143`), so a failure is non-destructive | **Technical 20 / Demo 25** | S1 | **OPEN** | code + docs | 4h |
| **W-09** | **Hot reload is per-process; shipped HPA runs 2–10 replicas.** `k8s/hpa.yaml:11-12`, `k8s/backend-deployment.yaml:9`, `helm/values.yaml:32-33` | Defense gap #8 🔴, verified. `invalidate()` clears one pod's cache; the other 1–9 keep serving the old model | **Technical 20** | S1 | **OPEN** | docs-only (scope honestly) or code | 1h docs · ≫3d code |
| **W-10** | **Counterfactuals ~9.5–12.4 s per unseen patient — not live-demo-able.** `model_loader.py:457` (`range(800)`), `ensemble_loader.py:441` (`n_samples=100`) | Check C, cold process. RF `predict_proba` = 71.3 ms/row ≈ **87%** of diabetes cost. Cache (`main.py:520`, 3600 s) only helps repeats | **Demo 25** | S1 | **PARTIALLY-FIXED** (diabetes 500→100 + cache deployed on `hf/main`; **heart never optimised and is now slower**) | decision-needed → code | 1h pre-warm · 1d optimise |

## S2 — likely found; survivable with a prepared answer

| ID | What & where | Evidence | Threatens | Sev | Status | Fix type | Effort |
|---|---|---|---|---|---|---|---|
| **W-11** | **`models/heart_disease/metadata.json` lists `ca` + `thal` and `best_model:"svm"`.** `:14-15,30-31,33` | AUDIT C-2. Only tracked file in the repo naming `ca`/`thal`. Not loaded, not served — **tracked**. Directly undermines defense **س26**, which the doc calls "the most mature decision in the project" | **Evidence 5 / Q&A 5** | S2 | **OPEN** | docs-only (delete) | 15m |
| **W-12** | **Heart scaler + IterativeImputer fit on all 605 rows before the split.** `clean_data.py:44,77`; split at `train_v5_xgb.py:140-142` | AUDIT C-8. Diabetes is clean (`preprocess_diabetes.py:245-250`; refit delta 0.000e+00) — the contrast makes heart harder to defend | **Technical 20 / Q&A 5** | S2 | **OPEN** | decision-needed → retrain | 2h disclose · 1–2d re-run + re-report |
| **W-13** | **`configs/heart_disease.yaml:89-96` — 4 hyperparameters disagree with the shipped pkl; 4 more missing.** | AUDIT C-7 + defense gap #5. YAML 898/0.0136/0.9637/0.5455 vs pkl 1124/0.006309/0.8331/0.4792. `grid_best.json` + `models/metrics.json` agree with the pkl. Nothing reads `best_params` at runtime. File is labelled "single source of truth" at `:4` | **Technical 20 / Docs** | S2 | **OPEN** | code (config) + docs | 30m |
| **W-14** | **Defense answer س19 quotes RF meta-learner coefficient 0.711; shipped model says 0.9019.** `docs/OmniDiag_Proposal_Defense.md:~205` | Check D10 + AUDIT W-9. `meta_learner.pkl` coef = [0.4362, 3.8572, 0.9019]. 0.711 comes from stale `ensemble_metrics.json:16-20`. **A prepared Q&A answer is factually wrong** | **Q&A 5** | S2 | **OPEN** | docs-only | 30m |
| **W-15** | **"Sensitivity from 87.3% to 91.7%" — 87.3% is wrong.** Proposal §4.2; `DOCS:482`; defense **س20** | Check D9. Measured stacking @ 0.5 sensitivity = **78.38%**. Correct claim: 78.4% → 91.6% | **Evidence 5 / Q&A 5** | S2 | **OPEN** | docs-only | 30m |
| **W-16** | **"AUC-regression guard" / "promoted only if AUC does not regress" does not exist.** Proposal §8.2, §4.5 | Check D16. `grep -n "auc\|roc\|regress\|promote\|tolerance" retrain.py` → 1 unrelated comment | **Technical 20 / Q&A 5** | S2 | **OPEN** | docs-only (or build the guard) | 30m docs · 1d code |
| **W-17** | **`OMNIDIAG_TECHNICAL_DOCS.md` describes the wrong dataset and lists `ca`/`thal` as production inputs.** `:398-399`; also `:970` "~303 samples" | AUDIT C-4 + session finding | **Docs / Q&A 5** | S2 | **OPEN** | docs-only | 1h |
| **W-18** | **"CV accuracy 88.98%" attributed to a file with no CV field.** `DOCS:403,409-410`; `arch:74`; `impl:221` | AUDIT C-5. 88.98% is the pre-production Optuna number (`experiments_log.md:15`) | **Evidence 5** | S2 | **OPEN** | docs-only | 30m |
| **W-19** | **Diabetes specificity stated as 58.70% in three places.** `DOCS:455,867,966` | AUDIT C-16. Your own claim is 54.63%; measured 54.68% | **Evidence 5 / Docs** | S2 | **OPEN** | docs-only | 15m |
| **W-20** | **Repo URL points at the wrong GitHub org.** `DOCS:7` + 7 branch links use `github.com/yahyoha/…` | AUDIT C-15. `git remote -v` → `github.com/yahyalababenah/omnidiag.git`. Judges given the doc cannot reach the code | **Docs** | S2 | **OPEN** | docs-only | 15m |
| **W-21** | **Proposal §5.4 cites `models/heart_disease/_archive_v1/` — the folder never existed.** | Check (defense gap #4). `ls` → absent; `git log --all -- 'models/heart_disease/_archive_v1'` → no commits on any branch | **Evidence 5** | S2 | **OPEN** | docs-only, or create it | 30m |
| **W-22** | **Zero model weights on the demo branch.** `.gitignore:18` (`*.pkl`) | AUDIT W-14. `git ls-tree -r deploy/v2-platform \| grep .pkl` → **0**. A judge who clones gets `FileNotFoundError` on the first prediction | **Demo 25** | S2 | **OPEN** | decision-needed (LFS vs USB vs download step) | 2h |
| **W-23** | **`README.md:548-553` API response shapes contradict the live API.** | Check D17–D19. `/predict` claims 7 fields, emits 3. `/explain` claims `shap_values[]`, emits `chart_data` | **Docs / Demo 25** | S2 | **OPEN** | docs-only | 1h |
| **W-24** | **README documents the retrain limitation using the wrong filenames and understates it.** `README.md:155` | Check D20. Names `xgb_model.pkl/lgb_model.pkl/rf_model.pkl`; the loader reads `omni_diag_*`. See W-08 | **Docs / Technical 20** | S2 | **OPEN** | docs-only | 30m |
| **W-25** | **K8s/Helm manifests never validated against a live cluster; CI does not even `--dry-run` them.** | Session finding. `README.md:493` **already discloses** the live demo is HF Spaces, not K8s — good. Proposal §7 does not | **Technical 20** | S2 | **PARTIALLY-FIXED** (README honest; proposal silent) | docs-only + optional CI step | 1h docs · 2h CI lint |
| **W-26** | **Active-learning retrain builds `X` from dict key order and raw un-engineered features.** `retrain.py:125` `np.array([list(feat.values()) …])` | Session finding, verified. Heart: un-encoded strings → `ValueError`. Diabetes: 21 raw vs 26 engineered expected. Column order = JSON key order | **Technical 20** | S2 | **OPEN** | code | 1d |

## S3 / S4 — dig-to-find, or hygiene

| ID | What & where | Evidence | Threatens | Sev | Status | Fix type | Effort |
|---|---|---|---|---|---|---|---|
| **W-27** | Ensemble `/explain` silently zeroes a failed model's SHAP and re-normalises. `ensemble_loader.py:617-627,654-663` | AUDIT C-9. `except Exception` also covers the `joblib.load` at `:536` → 200 OK with a degraded chart, no response-level signal | Technical 20 | S3 | OPEN | code | 2h |
| **W-28** | `ExplainResponse` strips `per_model_shap` + `shap_weights`. `schemas.py:133-173` vs `ensemble_loader.py:712-713` | AUDIT C-14, confirmed live (9 keys emitted) | Technical 20 / Demo 25 | S3 | OPEN | code | 1h |
| **W-29** | `diabetes /explain` re-reads **97.7 MB** from disk every request. `ensemble_loader.py:536,589` | AUDIT C-18. 1.55 s every call vs heart's 0.038 s (cached explainer) | Demo 25 | S3 | OPEN | code | 2h |
| **W-30** | `router.py:160-169` 501 hint says heart has no CF generator; it does. | Session finding, verified. `ModelLoader.generate_counterfactuals` exists (`:423`) → branch unreachable; `/api/v4/diseases` correctly reports `supports_counterfactuals=True` | Docs | S4 | OPEN | docs-only (comment) | 15m |
| **W-31** | Three Pythons: README badge 3.11+, `Dockerfile:7` 3.10-slim, venv 3.13.5 | Check D25/E | Docs | S3 | OPEN | decision-needed → docs/code | 1h |
| **W-32** | Version drift: API 4.0.0, config 5.0.0, artifacts 5.1.0 | Check E | Docs | S3 | OPEN | docs-only | 30m |
| **W-33** | CI runs zero tests despite 13 `tests/test_*.py`. `.github/workflows/ci.yml` | Check D27 | Technical 20 | S3 | OPEN | code (CI) | 2h |
| **W-34** | `omni_diag_lgb_optimized.pkl` contains the hardcoded fallback params, not the advertised 100-trial Optuna result | AUDIT W-8 | Technical 20 / Q&A 5 | S3 | OPEN | retrain or rename | 30m rename · 1d retrain |
| **W-35** | `ensemble_metrics.json:16-20` coefficients ≠ shipped `meta_learner.pkl` | AUDIT W-9 (source of W-14) | Evidence 5 | S3 | OPEN | retrain/regenerate | 1h |
| **W-36** | `eval_set=[(X_test,y_test)]` in `train_v5_xgb.py:153-157` | AUDIT W-3. No early stopping → currently harmless, reads as leakage | Q&A 5 | S3 | OPEN | code | 15m |
| **W-37** | `explain()` hardcodes the label inversion instead of reading `_LABELS_INVERTED`. `model_loader.py:380-382` | AUDIT W-13 | Technical 20 | S4 | OPEN | code | 15m |
| **W-38** | `heart11.csv` (with `ca`/`thal`) still tracked in `data/heart_disease/raw/` | AUDIT W-4 | Evidence 5 | S3 | OPEN | docs-only (delete) | 15m |
| **W-39** | `preprocess_diabetes.py:66-69` points at project root; file is in `data/diabetes/raw/` → "reproduce my training" fails immediately | AUDIT W-15 | Evidence 5 / Q&A 5 | S3 | OPEN | code | 15m |
| **W-40** | Artifacts pickled under sklearn 1.6.1, unpickled under 1.9.0 (`InconsistentVersionWarning`); `requirements.txt` unpinned | AUDIT W-7 | Technical 20 | S3 | OPEN | code (pin) | 1h |
| **W-41** | `docs/OmniDiag_Proposal_Defense.md` is **untracked** — the single most valuable prep asset is not in git | `git status --porcelain` | Docs | S3 | OPEN | code (commit) | 5m |
| **W-42** | Defense س24 (Cleveland ↔ Z-Alizadeh merge) has only a placeholder answer; the doc itself says "prepare this from the code before judging day" | `docs/OmniDiag_Proposal_Defense.md` س24 | Q&A 5 | S2 | OPEN | docs-only (derive from `merge_data.py`) | 2h |
| **W-43** | Defense س10 (calibration) — the honest answer is prepared, but no Platt/Isotonic/Brier work exists to point at | ibid. س10 | Q&A 5 | S3 | OPEN | decision-needed | 1h answer · 1d implement |
| **W-44** | Uncommitted work in tree: `backend/auth/routes.py` (+13/−2), `scripts/simulate_federated.py` (+74/−20) | `git status` | Docs | S4 | OPEN | decision-needed | 30m |

---

# PART 2 — JUDGE ATTACK MAP (top 10)

| # | ID | The question a judge will actually ask | Do we have an answer today? |
|---|---|---|---|
| 1 | **W-02** | *"Your proposal says every figure is reproducible from `models/heart_disease/metrics.json`. I opened it — it says `best_model: svm`, ROC-AUC 0.957. Explain."* | ❌ **NO.** There is no honest answer that preserves the sentence. Must be corrected before submission. The only defensible reply is: wrong file cited, here is `models/metrics.json`, and here is the reproduction script. |
| 2 | **W-03** | *"Show me where threshold 0.420 is applied in the code."* | ❌ **NO.** It is applied nowhere. The UI displays 50.0%. Requires Decision H-2 first. |
| 3 | **W-01** | *"Your repo history contains a GitHub PAT and an HF token. Were they revoked?"* | ⚠️ **PARTIAL.** We can say "found and rotated" **only after** H-1. Right now we cannot even confirm revocation. |
| 4 | **W-07** | *"How does the federated part actually work?"* (defense doc marks س31 the single most dangerous question) | ❌ **NO.** Flower `FedAvg` cannot average a pickled XGBoost model; the DP function is never called. Needs H-4: reframe as an explicit PoC. |
| 5 | **W-05 / W-06** | *"Run the diabetes What-If on your severe demo patient."* | ❌ **NO.** Case C returns 0 scenarios after 9.3 s, then the UI shows heart-disease mock values and "Coming Soon". This is a live demo failure, not a document problem. |
| 6 | **W-12** | *"Was your scaler fit before or after the train/test split?"* | ⚠️ **PARTIAL.** Diabetes: strong answer with evidence. Heart: fit on all 605 rows — needs H-3 (disclose vs re-run). |
| 7 | **W-04** | *"75.18% accuracy with 91.70% sensitivity and 54.63% specificity on a balanced set — those don't reconcile."* | ✅ **YES**, once restated. 73.1% @ 0.275 is measured and defensible as a screening trade-off. Pure docs fix. |
| 8 | **W-11** | *"Why did you remove `ca` and `thal`?"* (defense س26 — your strongest answer) | ✅ **YES** — the س26 answer is excellent (Kaufman leakage + TRIPOD). ⚠️ **But** `metadata.json` still ships both feature names and `best_model:"svm"`, which hands the judge a contradiction mid-answer. Delete the file and the answer is clean. |
| 9 | **W-09** | *"You claim hot reload with no restart, and an HPA of 2–10 replicas. Which pod gets the new model?"* | ❌ **NO.** One pod. Needs a scoping sentence (H-6). |
| 10 | **W-08** | *"What happens when I click Retrain?"* | ❌ **NO** — and worse than documented. It targets a live base model, and the README describes the opposite. |

**Answers that are already strong — lead with these:** config-driven router with zero hard-coded diseases (D21, live-demoable); `ca`/`thal` removal reasoning (س26, once W-11 is deleted); the 0.275 cost-function threshold (س20, once the 87.3% number in W-15 is corrected); the Optuna-overfitting lesson (س18); SHAP label-inversion correctness, proved by additivity to 1e-9 (`AUDIT_REPORT.md` AUDIT 7).

---

# PART 3 — DECISIONS REQUIRED FROM A HUMAN (these block tomorrow's fixes)

> **IDs renamed 2026-09-17:** these decisions are `H-1 … H-9` (formerly `D-1 … D-9`).
> `D-1 … D-8` now belong exclusively to the diabetes limitations in Part 5.
> Unrelated: `D-1 … D-5` in `tests/test_database.py` and `plans/testing_checklist.md`
> are database-test IDs and were left alone.

| ID | Decision | Why it blocks | Options |
|---|---|---|---|
| **H-1** | **Credential response.** Were `ghp_vkVLzf…` and `hf_aOYuxa…` ever actually revoked? And do we rewrite history on 7 pushed refs (incl. the public HF Space) or accept the exposure? | I will not test the tokens. History rewrite breaks every clone and is irreversible — cannot be done on my judgment. Blocks W-01. | (a) Rotate all 3 + rewrite history + force-push · (b) Rotate + leave history, document it · (c) Rotate only |
| **H-2** | **Heart metrics: restate or implement?** Publish the true 71.1%/85.5% @ argmax, or implement a real 0.420 threshold, revalidate and republish? | Determines whether W-03 is a 2h docs edit or a 1d code+revalidation task. Cascades into the proposal, DOCS, and the defense doc. | (a) Restate (honest, fast, defensible) · (b) Implement 0.420 and re-measure everything · (c) Implement a *derived* clinical threshold and justify it |
| **H-3** | **Heart leakage: disclose or re-run?** | Re-running `clean_data.py` fit-on-train-only changes every heart number in every document, including ones that currently reproduce. Nineteen days is enough, but only if started now. | (a) Disclose as a stated limitation · (b) Re-run, retrain, re-report all heart figures |
| **H-4** | **Federated learning: how do we present it?** | Cannot be made to work in 19 days (needs SecureBoost). Determines the §10.4 rewrite and the س31 answer. | (a) Explicit "architectural proof-of-concept, not functional aggregation" · (b) Remove from the proposal · (c) Demo the transport layer only, state the aggregation gap |
| **H-5** | **Demo Case C.** | W-06 is the most visible demo failure and the fix is a data/tuning judgment, not a code bug. | (a) Re-tune the patient until 3/3 generate · (b) Raise `n_samples` for the demo path · (c) Replace Case C with a patient that reliably generates · (d) Accept 0 and script the narration |
| **H-6** | **Scope claim for K8s / HPA / hot reload.** | W-09 + W-25. Either scope the claim honestly or build multi-pod invalidation (≫19d). | (a) "Single-instance deployment; multi-replica invalidation is future work" · (b) Build it |
| **H-7** | **Model weights for the judges.** | W-22. Affects whether a clone can run at all. | (a) LFS on the demo branch · (b) USB / local machine only · (c) Documented download step |
| **H-8** | **Swagger `/docs` description is Arabic-only** (`main.py:178`). | Judge-visible surface; depends on the language of the evaluation. | (a) Bilingual · (b) English · (c) Leave |
| **H-9** | **The two uncommitted files** (`backend/auth/routes.py`, `scripts/simulate_federated.py`). | W-44. Unknown whether they are wanted before the freeze. | (a) Commit · (b) Revert · (c) Leave and note |

---

# PART 4 — ALREADY FIXED / STOP RE-LITIGATING

Verified done. Do not spend the 19 days on these.

| Item | Claimed in | Verdict | Verifying evidence |
|---|---|---|---|
| **`n_samples` 500→100 + CF caching** | Session finding ("confirm it exists on the deployed branch") | ✅ **FIXED and deployed** | `ensemble_loader.py:441`; `main.py:510,520`. Present on `deploy/v2-platform`, `origin/deploy/v2-platform`, **and `hf/main`** via `git show <ref>:path`. Commit `860326e`. ⚠️ *But* the "~18 s" figure is stale (now ~9.5 s) and it never applied to heart — tracked as W-10 |
| **WhatIfScenarioCard — heart path** | Session finding | ✅ **FIXED for heart** | `bc1b5b4` (2026-09-10) "render What-If scenarios from the actual API shape"; heart returns `scenario_id`+array `changes` and renders correctly. Diabetes remains broken → W-05 |
| **Admin dashboard "Avg Confidence" mislabel** | Defense gap #10 🟡 | ✅ **FIXED** | `AdminDashboard.jsx:730` reads `label="Avg Risk Probability"`. Only the backing API field is still `avg_confidence` (cosmetic) |
| **Config-driven router, zero hard-coded diseases** | Defense gap #13 ✅ | ✅ **VERIFIED** | `backend/router.py:190-244`; `/api/v4/diseases` returns both modules from YAML alone. **Use as a live demo** |
| **Modular monolith / 3-layer architecture** | Defense gap #12 ✅ | ✅ **VERIFIED** | Router → loader → feature-engineer separation confirmed throughout AUDIT 3 |
| **PWA installable** | Defense gap #15 🟢 | ✅ **VERIFIED** | `frontend/dist/manifest.webmanifest` (`"display":"standalone"`, `"start_url":"/"`) + `sw.js` + `registerSW.js` present |
| **SHAP 0.42+ / MLflow 2.10+ pins** | Defense gap #11 🟡 | ✅ **VERIFIED** | `requirements.txt:25,61`. ⚠️ Python is **not** verified — three-way drift → W-31 |
| **`ca` / `thal` absent from the production path** | AUDIT 2 | ✅ **VERIFIED CLEAN** | Absent from YAML, Pydantic schema, feature pipeline, pkl `feature_names_in_`, and the training CSV. Only `metadata.json` (W-11) and prose (W-17) still mention them |
| **Diabetes has no data leakage** | AUDIT 2 | ✅ **VERIFIED** | `preprocess_diabetes.py:245-250` fits on train only; refitting on the reproduced split gives `max\|mean_ delta\| = 0.000e+00` |
| **SHAP label inversion applied exactly once** | AUDIT 7 | ✅ **VERIFIED** | `sigmoid(base + Σshap) == confidence` to 1e-9 on all 3 heart cases; `/explain` == `/predict` exactly. `shap_service.py` performs no sign manipulation |
| **No base model silently falls back on *load* failure** | AUDIT 3 | ✅ **VERIFIED** | `FileNotFoundError` at `ensemble_loader.py:109-112,131-134`; `RuntimeError` at `:301-313`. Only the *SHAP* path degrades → W-27 |
| **Diabetes ensemble loads the optimized artifacts** | AUDIT 3 | ✅ **VERIFIED** | `omni_diag_xgb_optimized` / `omni_diag_lgb_optimized` / `omni_diag_rf` + `meta_learner`; generic `xgb_model.pkl` has a different column order and is not loaded |
| **Feature-engineering order matches training** | AUDIT 3 | ✅ **VERIFIED** both diseases | scale→engineer at `model_loader.py:299-300` ≡ `train_v5_xgb.py:97,105-106`; `ensemble_loader.py:297-298` ≡ `train_diabetes_ensemble.py:829-835` |
| **`.env` never committed** | AUDIT git hygiene | ✅ **VERIFIED** | Not tracked; `.gitignore:184`. The real `DEEPSEEK_API_KEY` is local-only. ⚠️ Two *other* credentials **were** committed → W-01 |
| **README contains no performance claims** | AUDIT 5 | ✅ **VERIFIED** | Zero metric numbers. Its API-shape tables are wrong (W-23), but no metrics to defend |
| **Heart accuracy 80.17% + ROC-AUC 0.856** | Proposal §8.1 | ✅ **REPRODUCED EXACTLY** | `evaluation_evidence/heart_disease_report.txt` — subject to W-12 |
| **Diabetes ROC-AUC 0.831 / sens 91.70% / spec 54.63%** | Proposal §8.1 | ✅ **REPRODUCED** (0.8305 / 91.55% / 54.68%) — ⚠️ *2026-09-17:* sens/spec are **at 0.275**, which is no longer shipped; at the OOF threshold they are **91.31% / 55.42%** (H-1, Part 5) | `evaluation_evidence/diabetes_report.txt`; current: `evaluation_evidence/diabetes/final_metrics_table.md` |

---

# PART 5 — DIABETES MODULE LIMITATIONS (added 2026-09-17)

> **ID note:** `D-1 … D-8` below are the *diabetes limitation* IDs. The Part 3 human decisions were
> renamed to `H-1 … H-9` on 2026-09-17, so the prefix collision is gone.
>
> **Number source:** every figure here comes from
> `evaluation_evidence/diabetes/final_metrics_table.md` (shipped stacking ensemble, held-out test
> n = 14,139, bootstrap 2,000 resamples, seed 42, percentile 95% CI) or from
> `evaluation_evidence/diabetes/oof_threshold.json`. Regenerate with
> `scratch/diabetes_oof_threshold.py` then `scratch/generate_diabetes_evidence.py`.
>
> **Correction to earlier figures:** `scratch/audit_diabetes.py` and `scratch/prevalence_experiment.py`
> measured a *proxy* XGBoost (300 trees, depth 6, unscaled input), not the shipped ensemble. Their
> numbers (OOF threshold 0.3203, 0.7% cost gap, 1.5-pt sensitivity gap, Brier 0.176 → 0.098) are
> superseded below by measurements on the shipped models. Direction agrees; magnitudes differ.

| ID | What | Measured | Why not fixed | What would fix it | Sev | Status | Evidence |
|---|---|---|---|---|---|---|---|
| **D-1** | **Threshold was selected on the test set.** `models/train_diabetes_ensemble.py` passed `y_true=y_test` to `find_optimal_clinical_threshold`; 0.275 was shipped in `configs/diabetes.yaml`. | Raw threshold **0.275 → 0.2809** (training OOF, 5-fold stratified, seed 42). On test: cost 2·FN+FP **4398 → 4380 (−0.41%)**, sensitivity **91.55% → 91.31% (−0.24 pt)**, specificity **54.68% → 55.42% (+0.74 pt)**. 0.275 is not reproducible today: re-deriving on `y_test` with the shipped models gives **0.3203**. The OOF cost curve is flat near the minimum (grid point 0.2759 costs 17,494 vs 17,477, +0.1%). | — | — | S2 | **FIXED** (not committed) | `threshold_decision_log.md`, `threshold_cost_curve_oof.png`, `oof_threshold.json`, `confusion_matrix_{old_0.2750,new_0.2809}.*` |
| **D-2** | **Trained on a 50/50 resample; real BRFSS prevalence ≈ 14%.** Raw probabilities shown to the clinician were on the 50% prior. | At 14% prevalence (test set importance-weighted): Brier **0.1753 [0.1707, 0.1798] → 0.0974 [0.0959, 0.0989]**; ECE **0.239 → 0.0082**; mean displayed probability **0.379 → 0.142** (true 0.140). Decisions changed by the correction on test: **0 of 14,139**. | No retraining allowed in this pass; a Bayes prior-shift correction (`backend/prevalence_correction.py`) fixes the scale without touching the model. | Retrain on the full natural-prevalence file (D-8), then drop the correction. Also: the correction is only as good as `prevalence_deploy` — 0.14 is BRFSS-2015 (US); the deployment population's prevalence should be sourced and set in config. The 14% evaluation is re-weighted, not a real 14% cohort. | S1 | **MITIGATED** (correction, not retrain) | `calibration_curve.png`, `calibration.json`, `before_after.{json,png}` |
| **D-3** | **PPV at real prevalence is low.** A mathematical property of screening at Se 91% / Sp 55%, not a defect: **PPV = Se·π / (Se·π + (1−Sp)(1−π))**. | PPV **67.2% [66.3, 68.1]** on the 50/50 test set → **25.0% [24.5, 25.5]** at π = 14% (old threshold: 66.9% → **24.7%**). Per 1,000 screened at 14%: **511 referred, 128 true cases, 383 needless referrals, 12 missed**. NPV at 14%: **97.5%**. **Hidden consequence:** a threshold picked with 2:1 costs on balanced data is, at π = 14%, the Bayes-optimal threshold for an FN:FP cost ratio of **≈ 15.7 : 1** (1/0.0598 − 1), not 2 : 1. | Changing it trades away sensitivity; that is a clinical decision, not an engineering one. | (a) Position as a *rule-out / triage* tool (NPV 97.5%) with mandatory HbA1c confirmation; (b) decide the real FN:FP ratio at deployment prevalence and re-select the threshold on corrected OOF probabilities. **Decision needed.** | S1 | **OPEN — decision-needed** | `ppv_collapse_table.{md,csv}`, `pr_curve.png` |
| **D-4** | **Identical feature vectors across the split.** All 21 features are binary/coarse ordinal, so exact matches arise by chance as well as by duplication. | **564 / 14,139 = 3.99%** of test rows also appear in train. ROC-AUC **0.8305 [0.8238, 0.8371] → 0.8256 [0.8187, 0.8324]** on de-duplicated rows (**Δ 0.0049**). | Effect is inside the CI; dropping duplicates changes the split that every published figure uses. | Group-aware split on the feature vector (or de-duplicate before splitting) and re-report. | S3 | **OPEN — disclosed** | `final_metrics_table.md` (`roc_auc_dedup`, `test_rows_duplicated_in_train_pct`) |
| **D-5** | **No external validation for diabetes.** Heart has a second cohort (Tehran); diabetes is evaluated only on a split of the same BRFSS file. | Not measurable — there is no second dataset in the repo. | Candidate is NHANES (lab-confirmed HbA1c), but harmonising its variables to the 21 BRFSS items is outside the time window before 2026-10-04. | Map NHANES questionnaire items to BRFSS codings, evaluate transported AUC + calibration at NHANES prevalence. | S2 | **OPEN — out of scope** | — |
| **D-6** | **Stacking was never shown to beat a single XGBoost.** | Indicative only: training OOF ROC-AUC **XGBoost 0.8304** vs **stacking 0.8306** (Δ 0.0002); LightGBM 0.8275, RF 0.8279. No paired test, no CI on the difference. | Needs a paired comparison on identical folds; not part of this pass. | Paired bootstrap / DeLong on test for stacking vs XGBoost alone; if not significant, prefer the single model (RF is ~98 MB and dominates latency, W-10/W-29). | S2 | **OPEN** | `oof_threshold.json` (`oof_auc_base`, `oof_auc_stacking`) |
| **D-7** | **The five engineered diabetes features were never ablated** (`BMI_Age_Interaction`, `Health_Index`, `Lifestyle_Score`, `SES_Composite`, `Diabetes_Clinical_Risk`). | Not measured. | Ablation requires refitting models, which this pass forbids. | Drop-one-feature OOF AUC on identical folds with a paired bootstrap; remove any feature that does not move AUC. | S3 | **OPEN** | — |
| **D-8** | **Trained on the balanced 70,692-row file while the full 253,680-row file exists.** | Training rows used: **56,553** (80% of 70,692). The full file is **not in the repo** (`data/diabetes/raw/` holds only the 50/50 file). | Retraining forbidden in this pass; file not present locally. | Obtain the full Kaggle/CDC file, train at natural prevalence with `scale_pos_weight` or class-weighted loss, re-select the threshold on OOF, and remove the D-2 correction. | S2 | **OPEN** | `final_metrics_table.md` (`n_train`, `n_test`) |

### D-3 — open policy decision (not resolved)

The cost function `Cost = 2·FN + 1·FP` was chosen on the **balanced 50/50 scale**. Once probabilities
are moved to the 14% deployment prior, the deployed cut-point (0.059776) is the Bayes-optimal
threshold for an implied cost ratio of **FN : FP ≈ 15.7 : 1** — not the 2 : 1 that was intended and
documented. Arithmetic: a cut-point *t* on calibrated probabilities is optimal when
`c_FN / c_FP = (1 − t) / t`, and `(1 − 0.059776) / 0.059776 = 15.73`.

Reading it the other way: an honest 2 : 1 preference at 14% prevalence would put the cut-point at
`t = 1/3` on the corrected scale, i.e. a raw threshold near 0.71 — a completely different operating
point, with far fewer referrals and far more missed cases.

**This is a clinical-policy decision and it has not been made.** Nothing in the code assumes an
answer: the current behaviour is the 50/50-scale 2 : 1 choice carried over unchanged, so sensitivity
stays at 91.3% and the threshold move in D-1 is the only change to who gets flagged. The open
question for the clinical owner is: *what is the true cost of a missed diabetic relative to one
unnecessary HbA1c test, and at which prevalence is that ratio stated?* The answer changes the
threshold, every metric derived from it, and the Part 5 numbers above.

> **2026-09-21 update — `prevalence_deploy` changed from 0.14 (US/BRFSS placeholder) to 0.237
> (Jordan's actual diabetes prevalence, per `docs/OmniDiag_Proposal_Defense.md`).** Every number in
> D-2/D-3/D-8 and W-04 above is stated at the OLD 0.14 prior and is now stale; they are left as
> written rather than silently rewritten, since this register otherwise preserves what was measured
> at the time (see the Part 6 preamble's own "Correction to earlier figures" convention). At the new
> prior: deployed threshold **0.108184** (was 0.059776); implied FN:FP cost ratio **≈ 8.2 : 1**
> (was ≈ 15.7 : 1, `(1 − 0.108184) / 0.108184 = 8.244`); PPV at deployment prevalence **38.9%** (was
> 25.0%); NPV **95.4%** (was 97.5%). The open policy question itself is unchanged and still
> unresolved — only the numbers it's being asked about moved. Current source of truth:
> `evaluation_evidence/diabetes/` (regenerated 2026-09-21) and `docs/DIABETES_AUDIT_REPORT.md` §2.2/§3.3
> (updated same day). D-2/D-3/D-8/W-04 themselves were **not** rewritten in this pass — flagged here,
> not fully reworked, since they were outside the explicit scope of the 2026-09-21 prevalence change.

**Part 5 counts:** FIXED 1 · MITIGATED 1 · OPEN 6 (1 decision-needed, 1 out of scope).
These 8 rows are **not** included in the Counts table below, which describes Parts 1–4 as built on 2026-09-15.

---

## Counts

| | S1 | S2 | S3 | S4 | Total |
|---|---|---|---|---|---|
| OPEN | 8 | 14 | 14 | 3 | **39** |
| PARTIALLY-FIXED | 2 | 1 | — | — | **3** |
| REGRESSED | 1 | — | — | — | **1** |
| **Register total** | | | | | **44** |
| FIXED / verified-good (Part 4) | | | | | **17** |

**Fix-type split (44 open rows):** docs-only **18** · code **13** · decision-needed **9** ·
retrain **4**. Roughly **60% of the register is writing, not engineering** — which is the good
news with 19 days left. The 9 decision-needed rows in Part 3 gate most of the rest.

## Nothing was changed

No application code, config, model artifact, or document was modified; no commits were made.
Files created this session: `WEAKNESS_REGISTER.md`, plus (from the prior audit) `AUDIT_REPORT.md`,
`evaluation_evidence/`, `scratch/`. `backend/auth/routes.py` and `scripts/simulate_federated.py`
were already modified before this work began and were left untouched.

## Addendum 2026-09-17 — diabetes decision fixes

The "Nothing was changed" statement above describes the 2026-09-15 session. On 2026-09-17 the
diabetes module was changed (uncommitted): threshold re-selected on training OOF predictions
(`models/train_diabetes_ensemble.py`, `configs/diabetes.yaml`), Bayes prevalence correction added
(`backend/prevalence_correction.py`, `backend/ensemble_loader.py`), evidence regenerated under
`evaluation_evidence/diabetes/`, and `tests/test_diabetes_calibration.py` added. See Part 5.
Heart files were not touched.

---

# PART 6 — PROBABILITY-SCALE CONTRACT: WHAT REMAINS (added 2026-09-18)

> **Context.** The Bayes prevalence correction (Part 5, D-2) was applied inside
> `EnsembleModelLoader.predict()` without updating the code that consumed its output. A
> read-only audit on 2026-09-18 found 30+ places that read, compared, stored or displayed a
> diabetes probability on the wrong scale or with no scale at all. Layers 1–5 and 7 of the
> follow-up fixed most of them (commits `7bdeda2` … on `deploy/v2-platform`). This part
> records **what was deliberately not fixed, what could not be verified, and what was found
> along the way** — so the limits are stated here, not discovered later.
>
> **ID prefix:** `P-` (probability scale). Independent of `W-`, `H-`, `D-`.
>
> **Contract in force:** every probability leaving `predict()` is corrected by default; a raw
> value carries `_raw` in its name everywhere. Enforced by the AST guard
> `tests/test_probability_scale_contract.py` over 10 backend modules (0 violations).

| ID | What | Measured / evidence | Why not fixed | What would fix it | Sev | Status |
|---|---|---|---|---|---|---|
| **P-1** | **SHAP additivity does not hold for the diabetes ensemble — and did not hold before the correction either.** `/explain` returns SHAP values averaged over the three *base* models, weighted by the meta-learner's coefficients; the probability comes from the *meta-learner* on top of them. The weighted average is not the meta-learner's decomposition. | Demo Case C (`D-003`, `mockPatients.js`): `sigmoid(base_value + Σshap)` = **0.8225** vs meta-learner raw output **0.8498** → gap **0.0274** on the training-prior scale. After layer 4 shifted `base_value` by `log R` (R = odds(0.14)/odds(0.50)) so both sides sit on the deployment prior: reconstruction **0.4300** vs `confidence` **0.4795** → gap **0.0496**. Contrast: heart satisfies additivity to 1e-9 (Part 4) because it is one model. | Exact additivity needs SHAP computed *through* the meta-learner (e.g. KernelExplainer on the full stack, or chaining TreeSHAP with the LR coefficients in log-odds), which changes every diabetes chart and costs seconds per call. Out of scope for a scale fix. | Explain the stack end-to-end, or explain only the dominant base model and say so. Until then the gap is **reported, not hidden**: `/explain` returns `shap_reconstructed_probability_corrected` and `shap_additivity_gap` on every diabetes call. | S2 | **OPEN — disclosed in API** |
| **P-2** | **Mixed units inside the weighted `base_value`.** TreeExplainer returns log-odds for XGBoost/LightGBM but **probabilities** for sklearn RandomForest; `ensemble_loader.explain()` averages the three as if they shared a unit. | `per_model_shap` for Case C: xgboost `base_value` **−0.00133** (log-odds), lightgbm **−0.15884** (log-odds), random_forest **0.49997** (probability — ≈ the 50/50 class mean). RF weight in the average **0.1736** (17.4%). **UNVERIFIED:** whether RF's per-feature SHAP values are likewise in probability units (only the base values were inspected); if so, `chart_data` mixes units too. | Fixing it changes every diabetes SHAP chart and the P-1 gap; must be done together with P-1, not piecemeal. | Put all three explainers on one unit before averaging — log-odds, to match the chart — e.g. by explaining a logit-wrapped RF, or drop RF from the explanation and re-weight. Must be done together with P-1; re-measure the gap afterwards. | S2 | **OPEN** |
| **P-3** | **`POST /patients/{id}/visits` accepts a caller-supplied `risk_score` with no scale marker.** `PatientRiskTimeline.jsx` plots it against corrected `risk_bands`. | `patient_visits` has **0 rows** on the dev DB, so nothing is mis-plotted today. Documented in a comment at `PatientRiskTimeline.jsx`. | No current caller; adding a required field changes a public endpoint. | Add `probability_scale` to `VisitCreate` + `patient_visits` (migration), reject diabetes visits without it, and have the UI plot only same-scale points. | S3 | **OPEN** |
| **P-4** | **Prometheus histogram `omnidiag_prediction_confidence` was re-bucketed** (0.5–1.0 → 0.01–1.0, layer 3). | Old buckets left 7 of 9 unreachable for corrected diabetes values (max observed 0.653, threshold 0.0598). | Intentional. | Nothing — but **series before and after the deploy are not comparable**; any Grafana panel or alert on the old `le` labels must be rebuilt. `deploy/grafana/dashboards/omnidiag.json:94` uses `rate(..._bucket[5m])` and was not changed. | S3 | **ACCEPTED — note for ops** |
| **P-5** | **The UI was not verified in a browser.** | What *was* verified: `npm run build` (2,745 modules); `WhatIfScenarioCard` server-rendered with `react-dom/server` against a live diabetes `/counterfactuals` response for Case C — no `undefined`, no mock fallback, both reduction figures present; response shapes for all four demo patients checked through `TestClient`. **Not** verified: pixel rendering, `PDFReport` output (its code path uses the same normaliser but the PDF itself was never generated), the admin and review-queue tables. | No browser in the working environment. | Click through Clinical EMR → D-001…D-004, export the PDF for D-003, open Admin → review queue and stats. | S2 | **OPEN — manual check needed** |
| **P-6** | **The contract guard only covers Python.** `tests/test_probability_scale_contract.py` parses 10 backend modules by AST; the frontend (`*.jsx`, `*.js`) is not scanned, and the module list is maintained by hand. | Every frontend defect in the 2026-09-18 audit (wrong field name, local 1−p inversion, local band computation) was invisible to the guard. | A JS AST pass (e.g. an ESLint rule forbidding numeric literals compared against identifiers matching `/confidence|probab|risk/`) is a separate piece of tooling. | Custom ESLint rule + run it in CI (CI currently runs no tests at all — W-33). | S3 | **OPEN** |
| **P-7** | **`probability` is still accepted on `POST /api/v4/generate-report`** as a deprecated alias for `probability_corrected`. | Removed from all internal Python signatures in layer 7; kept only on the HTTP body, marked `deprecated: true` in OpenAPI. The shipped frontend sends `probability_corrected`. | Removing it breaks any HTTP client built before the change; none is known, but none can be ruled out. | Remove after one release; log any request still using it until then. | S4 | **OPEN — scheduled** |
| **P-8** | **`AdminStats.avg_confidence` is still returned** — the mean over all rows of all diseases and all releases. | Kept for client compatibility; the UI labels it "(all rows, mixed scales)" and shows the new `avg_confidence_by_scale` table (per disease × scale, `NULL` grouped as `unknown`). | Removing a response field is a breaking change. | Drop the field once no client reads it. | S4 | **OPEN** |
| **P-9** | **Rows written before the scale columns existed keep `NULL`.** `predictions.probability_scale`, `review_queue.uncertainty_scale`, `review_queue.decision_threshold`. | Dev DB: 14 prediction rows (6 diabetes at 0.90–0.93, raw-prior; 8 heart) and 1 review row. The scale of a past row cannot be recovered from the row, so it is not guessed. | Backfilling would assert a fact the data does not contain. | Nothing. Every reader must treat `NULL` explicitly — drift (`confidence` → NaN), admin (`COALESCE(…,'unknown')` as its own group), UI (explicit `== null` checks and a visible "scale not recorded" label). **2026-09-19:** the three columns were added to `omnidiag_dev.db` with plain `ALTER TABLE … ADD COLUMN` — deliberately **not** through alembic: the file has no `alembic_version` table and neither `da87946a56a6` nor `f1e2d3c4b5a6` is idempotent, while stamping `f1e2d3c4b5a6` would falsely record the `audit_logs.id` BIGINT→String(36) conversion as applied. Row counts unchanged (14 / 1 / 6 / 3 / 7,888), integrity OK, all new columns NULL; verified by loading all 14 `Prediction` rows through the ORM from the real file. A pre-migration backup was taken first (SQLite online backup, integrity OK) in the session scratchpad — **that directory was later wiped between sessions, so the backup no longer exists.** The migration is nonetheless exactly reversible without it: it only added three columns that are NULL in every row (`ALTER TABLE … DROP COLUMN`, SQLite ≥ 3.35). The dev DB therefore remains outside alembic's control — true before, still true, now stated. | — | **BY DESIGN** |
| **P-10** | **Layer commits swept in pre-existing uncommitted work, and HEAD was not importable on its own.** `git add <file>` in layers 2–5 committed the prevalence-correction integration already pending in `backend/ensemble_loader.py`, `backend/main.py`, `backend/router.py`, `frontend/src/components/PatientRiskTimeline.jsx`, while `backend/prevalence_correction.py`, `configs/diabetes.yaml`, `tests/test_diabetes_calibration.py`, `models/train_diabetes_ensemble.py` and `frontend/src/constants/thresholds.js` stayed uncommitted. | Verified on a clean clone of `90c5956`: `import backend.ensemble_loader` → `ModuleNotFoundError: No module named 'backend.prevalence_correction'`. (An earlier claim that the **frontend** build was also broken was wrong: `PatientRiskTimeline.jsx` imports exports that did not exist yet, but no component imports it, so it is never bundled and `vite build` succeeds — see P-15.) | — | **Fixed 2026-09-19 in `9a203e7`** (the pending work, committed as found; 7 evidence PNGs left out because `.gitattributes` routes `*.png` through Git LFS and `git-lfs` is not installed). **Clean-clone verification** — `git clone --no-local` of `9a203e7` into an empty directory, fresh venv, `pip install -r requirements.txt` (exit 0), fresh `npm ci` (exit 0): (A) no artifacts added: all **52/52** `backend.*` modules import; `pytest` **270 passed / 19 failed**, all 19 in the three `test_diabetes_calibration.py` classes that load real weights — the clone has **0 `.pkl` files** (W-22). (B) same clone, code untouched (0 tracked changes), only the git-ignored `models/**` weights copied in: **289 passed / 0 failed**; `npm run build` succeeds. | S1 | **FIXED** (HEAD self-contained; weights still not in git — W-22) |
| **P-11** | **The counterfactual generator proposes *worsening* some factors as part of a combined scenario.** | Case C, live call: one scenario pairs the improvements with `MentHlth 5 → 6` and `PhysHlth 12 → 20`. Cause, verified in `_sample_candidates()` (`backend/counterfactual_generator.py`): binary features pass a directional firewall (`_is_illegal_flip`), but continuous/ordinal features (`BMI`, `MentHlth`, `PhysHlth`, `GenHlth`, …) are drawn from a **symmetric** normal around the current value, clipped to `CLINICAL_BOUNDS`, with no direction constraint. Scenarios also differ between uncached calls for the same patient (see W-06). | Behaviour of the generator, not a scale defect; found while verifying layer 5. | Truncate the continuous draws to the risk-reducing side of the current value (the same firewall idea as the binary path), then re-check Case C's scenario count. | S2 | **OPEN** |
| **P-12** | **Heart rows were briefly stamped `probability_scale='corrected'`** (layer 3 → layer 7). | Heart applies no correction; the label claimed one. Fixed in layer 7: the stamp is derived from the result (`scale_of_result`) or the disease config (`scale_of_disease_config`) — heart → `raw`, diabetes → `corrected`. No heart row was written with the wrong stamp: the migration had not been run. | — | — | S3 | **FIXED** (layer 7) |
| **P-13** | **Every `pytest` run wrote to the developer's real `omnidiag_dev.db`.** `AuditMiddleware` opened `AsyncSessionLocal()` directly, bypassing the `get_db` override the suite installs; `backend/main.py` calls `load_dotenv()`, whose `DATABASE_URL` points at the dev file. Only `tests/test_audit_log.py` patched around it, and only for itself. | 664 `audit_logs` rows were appended on 2026-09-18 alone by this work's own test runs and verification scripts (endpoints `/auth/login`, `/auth/register`, `/api/v4/*/predict`, `/admin/cache/flush`, …); 7,224 older rows fit the same pattern. No other table was affected. | — | **Fixed 2026-09-19, two defences:** `backend/database.app_session()` resolves a session exactly like `Depends(get_db)` — including `app.dependency_overrides` — and the middleware uses it; `tests/conftest.py` forces `DATABASE_URL` to in-memory before any backend import (`load_dotenv` does not override an existing value). Guarded by `tests/test_test_isolation.py`. Verified: dev DB SHA-256 identical before and after a full 289-test run. The 7,888 existing audit rows were left as they are. | S2 | **FIXED** |
| **P-14** | **`run_drift` and `retrain` imported a name that does not exist.** `backend/monitoring/routes.py` and `backend/active_learning/retrain.py:237` did `from backend.database import async_session_maker`; the factory is `AsyncSessionLocal`. | Every call to `POST /api/v4/admin/drift/{disease}/run` raised `ImportError` — so the per-scale drift split described in layer 7 was **never reachable** until this fix (that earlier claim was overstated). | `retrain.py` is out of bounds for this work (no retraining, not run). | Drift: **fixed 2026-09-19** — the route takes `db: AsyncSession = Depends(get_db)`. Retrain: change the import to `AsyncSessionLocal` (or `app_session`) when W-08/W-26 are worked. Drift end-to-end still not exercised: it needs Evidently and a reference CSV before it reaches the DB. | S2 | **PARTIALLY FIXED** (drift ✅, retrain ❌) |
| **P-15** | **Two components are not rendered anywhere: `ReviewQueuePanel.jsx` and `PatientRiskTimeline.jsx`.** No component imports either; the admin dashboard has its own review table. | `git grep` finds only the files themselves. Consequence: the scale fixes made to both in this work are invisible, and errors in them do not surface — `PatientRiskTimeline.jsx` imported non-existent exports at `90c5956` and the build still passed (P-10). Its NULL-row note was corrected on 2026-09-19 (a legacy 50%-centred score is still comparable for an argmax module like heart, not for diabetes), but that text is currently invisible. | Deleting or wiring a component is a product decision. | Wire it in, or delete it. | S4 | **OPEN** |

**Status of earlier rows affected by this work** (the rows themselves are left as written):

- **W-05** (diabetes What-If renders `undefined`; PDF export throws) — **FIXED in code** (layer 5 + layer 7): field names corrected and a shared `normaliseScenario()` (`frontend/src/utils/counterfactuals.js`) turns diabetes' object-shaped `changes` into the array both consumers expect. Card verified by server-side render; PDF not generated (P-5).
- **W-23** (README API shapes contradict the live API) — **FIXED**: `README.md` clinical table rewritten from the live responses, plus a "Probability scale (diabetes)" section. `/predict` now has an explicit `response_model` (`PredictResponse`), so the shape is in OpenAPI.
- **W-28** (`ExplainResponse` strips fields) — **PARTIALLY FIXED**: the scale fields (`base_value_raw`, `shap_scale`, `shap_reconstructed_probability_corrected`, `shap_additivity_gap`) are now declared; `per_model_shap` and `shap_weights` are still stripped.
- **D-3** (implied FN:FP ≈ 15.7 : 1 at deployment prevalence) — unchanged and still the open clinical-policy decision. The threshold-relative review queue follows whatever threshold is configured.

**Part 6 counts (15 rows):** OPEN 9 (of which 1 manual check, 1 scheduled) · PARTIALLY FIXED 1 · ACCEPTED 1 · BY DESIGN 1 · FIXED 3. Not included in the Counts table above.

---

# PART 7 — HEART MODEL REPLACEMENT (heart_full_tuned.pkl): NEW FINDINGS (added 2026-09-20)

> **Context.** The heart module was moved from a 3-file manual-preprocessing loader
> (XGBoost weights + separate `label_encoders.pkl` + `standard_scaler.pkl`) to
> `models/heart_disease/heart_full_tuned.pkl`, a single self-contained sklearn
> `Pipeline` bundle (Optuna-tuned XGBoost, threshold 0.3695, 920 patients across
> 4 UCI sites). This finding surfaced while cross-checking that model's HPO
> reproducibility and is recorded here rather than fixed, per this pass's
> no-retrain rule.
>
> **ID prefix:** `HM-` (heart model). Independent of `W-`, `H-`, `D-`, `P-`.

| ID | What | Measured / evidence | Why not fixed | What would fix it | Sev | Status |
|---|---|---|---|---|---|---|
| **HM-1** | **حساسية نتائج البحث لإصدار XGBoost.** | تشغيل نفس نوتبوك ضبط المعاملات بنفس البذرة تحت xgboost 3.2.0 بدل 3.4.1 أنتج معاملات مختلفة (350/depth 7/gamma 4.85 بدل 500/depth 4/gamma 3.22) وعتبة إنتاج مختلفة (0.4581 بدل 0.3695) وحساسية أقل على كوهورت طهران بـ5.6 نقطة. السبب: Optuna تبحث تكيّفياً، فأي تغيّر في سلوك النموذج يغيّر مسار البحث كاملاً. تحقق مضاد: sklearn 1.6.1 مقابل 1.9.0 بنفس إصدار xgboost أعطى نتائج متطابقة تماماً — فالحساسية من xgboost لا sklearn. | Not a code defect — inherent to adaptive HPO (Optuna); no fix without re-running the search under whatever xgboost version ships. Retraining is out of scope for this pass. | الأثر: أي ترقية لـxgboost تستوجب إعادة تشغيل البحث واستخراج العتبة. تحسين مقترح: استخدام `Booster.save_model` (JSON) بدل pickle لأن الأخير يصدر تحذير توافق إصدار عند التحميل. | S3 | **OPEN — documented, not fixed** |
| **HM-2** | **`/batch` (`backend/main.py`) called `pipeline.predict_proba()` once per row instead of once for the whole file — not because of SHAP (`/batch` never calls `explain()`, confirmed by reading the code), simply because it reused `/predict`'s single-patient `ModelLoader.predict()` in a loop.** `IterativeImputer` inside the new Pipeline's `ColumnTransformer` (see the `feat(heart)` Pipeline-replacement commit) has a large fixed per-call cost regardless of whether the row has any missing values, so paying it 303 times dominates the whole request. | Measured on the real 303-row UCI file (`z_alizadeh_translated.csv`, all 11 columns filled) through the actual `/api/v4/heart_disease/batch` endpoint, in-process (no network): **row-by-row: 11.5s (one HTTP run) / 28.95s (raw 303x `ModelLoader.predict()` loop, system-load dependent) → vectorized: 0.027-0.041s**. Root cause isolated with `cProfile`: the nested "num" sub-pipeline (`IterativeImputer` + `StandardScaler`) inside `ColumnTransformer` costs ~30ms of every ~70ms single-row call; `pipeline.predict_proba()` on a single row vs. the same 303 rows in one call measured 319x apart at the raw-pipeline level, with **identical predictions to float equality (max abs diff = 0.0)** — vectorizing is a pure performance change, no quality trade-off. This is a real, measured, new latency source: the old (pre-Pipeline) `model_loader.py` used plain `StandardScaler.transform()` + `LabelEncoder.transform()` (no `IterativeImputer`, no `joblib.Parallel`) and did not have this cost. | — | **Fixed 2026-09-21:** `ModelLoader.predict_batch()` added (`backend/model_loader.py`) — builds one `DataFrame` from the whole validated group and calls `pipeline.predict_proba()` once; `backend/main.py::batch_predict` now validates every row individually as before (per-row isolation unchanged) but predicts in one vectorized call when the loader supports it (`hasattr(loader, "predict_batch")`), falling back to the original per-row `router.predict()` loop for any loader without it. `EnsembleModelLoader` (diabetes) has no `predict_batch` — confirmed (`hasattr` check) — so diabetes' `/batch` path is byte-for-byte the same loop it was before this change, unaffected. `IterativeImputer`'s own parameters were not touched. 290/290 tests pass after the change. | S1 | **FIXED** (heart only; diabetes untouched) |
| **HM-3** | **`HeartDiseaseInput.Oldpeak` (`backend/schemas.py:54`) was the only numeric field on the heart schema with no `ge`/`le` bounds — Age, RestingBP, Cholesterol, FastingBS and MaxHR all already had them (checked all five directly; the gap does not repeat elsewhere). This let `inf`/`-inf`/`nan` reach `ModelLoader.predict_batch()`'s single vectorized `pipeline.predict_proba()` call (HM-2), where `StandardScaler`/`check_array` reject the **entire matrix** on one such value — crashing every validated row in that batch, not just the bad one.** | Discovered by direct test, not assumption: `loader.predict_batch([good, {**good, "Oldpeak": float("inf")}, good])` → `ValueError: Input X contains infinity or a value too large for dtype('float64')` for **all 3** rows. Confirmed the value is reachable through the real HTTP path first: `HeartDiseaseInput(Oldpeak="inf", ...)` — **pydantic accepted it**, coercing the CSV string `"inf"` to `float('inf')` since the field had no bound to reject it against. Real training-data range checked (`data/heart_disease/processed/uci_heart_by_site.csv`, 920 rows, 4 sites) before picking bounds: Oldpeak observed **[-2.6, 6.2]** (not guessed). | — | **Fixed 2026-09-21:** added `ge=-3.0, le=10.0` to `Oldpeak` (comfortably covers the observed [-2.6, 6.2] range with margin). Verified after the fix: `inf`/`-inf`/`nan` all rejected by pydantic (any comparison against `nan` is `False` in Python, so it fails both bounds too, with no special-casing needed); the real observed min/max (-2.6, 6.2) still accepted. Re-ran the exact 3-row scenario end-to-end: the `inf` row is now rejected at the pydantic-validation stage (same as any other invalid-value row), and the two good rows on either side of it **succeed together in the same vectorized call** — regression guard added: `tests/test_diabetes_calibration.py::TestHeartNonRegression::test_batch_isolates_a_row_with_inf_or_nan_oldpeak` (parametrized `inf`/`-inf`/`nan`), posts a real 3-row CSV through the live `/batch` endpoint and asserts `succeeded=2, failed=1` with the bad row identified by number. This is the root fix, not a workaround: `predict_batch()` itself still provides no per-patient isolation (documented explicitly in its docstring) — the guarantee now comes entirely from upstream pydantic validation excluding every value that could reach that failure mode, for this field. 293/293 tests pass (290 + 3 new parametrized cases). | S1 | **FIXED** |

> ⚠️ **OPERATIONAL WARNING — do not use diabetes Batch Prediction for more than ~20-30 rows until HM-4 below is fixed.**
> Larger batches can make the system appear to hang, or fail silently (timeout, empty/truncated
> response — the same symptom class diagnosed for heart in HM-2, unfixed here), well before any
> clear error is shown. Anyone testing the platform — including a judge trying it live — can hit
> this with an ordinary-looking diabetes CSV upload with no warning in the UI itself. Heart is
> unaffected (fixed by HM-2/HM-3 above).

| ID | What | Measured / evidence | Why not fixed | What would fix it | Sev | Status |
|---|---|---|---|---|---|---|
| **HM-4** | **`EnsembleModelLoader` (diabetes) has no `predict_batch` equivalent to HM-2's fix.** A large diabetes `/batch` upload still runs the old per-row loop (`router.predict()` once per patient inside `backend/main.py::batch_predict`'s fallback branch), with the same shape of cost HM-2 fixed for heart — an ensemble of 3 base models + a meta-learner is if anything more expensive per row than heart's single Pipeline. | Measured directly (not the ~40ms estimate first proposed — the real number): **93.10ms/row** (50 sequential `EnsembleModelLoader.predict()` calls, same warm-loader methodology as HM-2's heart measurement). For a batch the size of HM-2's 303-row test case, that extrapolates to **~28s** — the same order of magnitude as heart's pre-fix 11.5-29s, and the same timeout exposure. | Out of scope for this pass — the task was the heart Pipeline replacement and its `/batch` fallout; touching `EnsembleModelLoader` or diabetes' `/batch` path was explicitly excluded throughout this session. | Add `EnsembleModelLoader.predict_batch()` mirroring `ModelLoader.predict_batch()`'s shape (one vectorized call per base model, then the meta-learner once on the stacked outputs, instead of the ensemble's full stack running once per row) in a dedicated diabetes session — needs its own before/after measurement and its own review, not a drive-by change riding on the heart work. | S2 | **OPEN — needs a separate diabetes session** |

**Part 7 counts (4 rows):** OPEN 2 · FIXED 2. Not included in the Counts table above.
