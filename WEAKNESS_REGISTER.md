# OmniDiag — Weakness Register

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
> means calling the GitHub/HF APIs with them, which I will not do. See Decision D-1.

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
| **`/docs` (Swagger)** | Title "OmniDiag Multi-Disease Diagnostic API", **version 4.0.0**, **description in Arabic only** | ⚠️ version 4.0.0 vs module 5.0.0 vs artifact 5.1.0; Arabic-only description — see Decision D-5 |
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
| **W-04** | **Diabetes accuracy 75.18% is measured at 0.5 while the product runs at 0.275.** `configs/diabetes.yaml:87`, applied `ensemble_loader.py:328` | AUDIT C-13/C-6 + `evaluation_evidence/diabetes_report.txt`. 73.12% @ 0.275. Arithmetic tell: (91.7+54.6)/2 = 73.17 ≠ 75.18 | **Evidence 5 / Q&A 5** | S1 | **OPEN** | docs-only | 1h |
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
| 2 | **W-03** | *"Show me where threshold 0.420 is applied in the code."* | ❌ **NO.** It is applied nowhere. The UI displays 50.0%. Requires Decision D-2 first. |
| 3 | **W-01** | *"Your repo history contains a GitHub PAT and an HF token. Were they revoked?"* | ⚠️ **PARTIAL.** We can say "found and rotated" **only after** D-1. Right now we cannot even confirm revocation. |
| 4 | **W-07** | *"How does the federated part actually work?"* (defense doc marks س31 the single most dangerous question) | ❌ **NO.** Flower `FedAvg` cannot average a pickled XGBoost model; the DP function is never called. Needs D-4: reframe as an explicit PoC. |
| 5 | **W-05 / W-06** | *"Run the diabetes What-If on your severe demo patient."* | ❌ **NO.** Case C returns 0 scenarios after 9.3 s, then the UI shows heart-disease mock values and "Coming Soon". This is a live demo failure, not a document problem. |
| 6 | **W-12** | *"Was your scaler fit before or after the train/test split?"* | ⚠️ **PARTIAL.** Diabetes: strong answer with evidence. Heart: fit on all 605 rows — needs D-3 (disclose vs re-run). |
| 7 | **W-04** | *"75.18% accuracy with 91.70% sensitivity and 54.63% specificity on a balanced set — those don't reconcile."* | ✅ **YES**, once restated. 73.1% @ 0.275 is measured and defensible as a screening trade-off. Pure docs fix. |
| 8 | **W-11** | *"Why did you remove `ca` and `thal`?"* (defense س26 — your strongest answer) | ✅ **YES** — the س26 answer is excellent (Kaufman leakage + TRIPOD). ⚠️ **But** `metadata.json` still ships both feature names and `best_model:"svm"`, which hands the judge a contradiction mid-answer. Delete the file and the answer is clean. |
| 9 | **W-09** | *"You claim hot reload with no restart, and an HPA of 2–10 replicas. Which pod gets the new model?"* | ❌ **NO.** One pod. Needs a scoping sentence (D-6). |
| 10 | **W-08** | *"What happens when I click Retrain?"* | ❌ **NO** — and worse than documented. It targets a live base model, and the README describes the opposite. |

**Answers that are already strong — lead with these:** config-driven router with zero hard-coded diseases (D21, live-demoable); `ca`/`thal` removal reasoning (س26, once W-11 is deleted); the 0.275 cost-function threshold (س20, once the 87.3% number in W-15 is corrected); the Optuna-overfitting lesson (س18); SHAP label-inversion correctness, proved by additivity to 1e-9 (`AUDIT_REPORT.md` AUDIT 7).

---

# PART 3 — DECISIONS REQUIRED FROM A HUMAN (these block tomorrow's fixes)

| ID | Decision | Why it blocks | Options |
|---|---|---|---|
| **D-1** | **Credential response.** Were `ghp_vkVLzf…` and `hf_aOYuxa…` ever actually revoked? And do we rewrite history on 7 pushed refs (incl. the public HF Space) or accept the exposure? | I will not test the tokens. History rewrite breaks every clone and is irreversible — cannot be done on my judgment. Blocks W-01. | (a) Rotate all 3 + rewrite history + force-push · (b) Rotate + leave history, document it · (c) Rotate only |
| **D-2** | **Heart metrics: restate or implement?** Publish the true 71.1%/85.5% @ argmax, or implement a real 0.420 threshold, revalidate and republish? | Determines whether W-03 is a 2h docs edit or a 1d code+revalidation task. Cascades into the proposal, DOCS, and the defense doc. | (a) Restate (honest, fast, defensible) · (b) Implement 0.420 and re-measure everything · (c) Implement a *derived* clinical threshold and justify it |
| **D-3** | **Heart leakage: disclose or re-run?** | Re-running `clean_data.py` fit-on-train-only changes every heart number in every document, including ones that currently reproduce. Nineteen days is enough, but only if started now. | (a) Disclose as a stated limitation · (b) Re-run, retrain, re-report all heart figures |
| **D-4** | **Federated learning: how do we present it?** | Cannot be made to work in 19 days (needs SecureBoost). Determines the §10.4 rewrite and the س31 answer. | (a) Explicit "architectural proof-of-concept, not functional aggregation" · (b) Remove from the proposal · (c) Demo the transport layer only, state the aggregation gap |
| **D-5** | **Demo Case C.** | W-06 is the most visible demo failure and the fix is a data/tuning judgment, not a code bug. | (a) Re-tune the patient until 3/3 generate · (b) Raise `n_samples` for the demo path · (c) Replace Case C with a patient that reliably generates · (d) Accept 0 and script the narration |
| **D-6** | **Scope claim for K8s / HPA / hot reload.** | W-09 + W-25. Either scope the claim honestly or build multi-pod invalidation (≫19d). | (a) "Single-instance deployment; multi-replica invalidation is future work" · (b) Build it |
| **D-7** | **Model weights for the judges.** | W-22. Affects whether a clone can run at all. | (a) LFS on the demo branch · (b) USB / local machine only · (c) Documented download step |
| **D-8** | **Swagger `/docs` description is Arabic-only** (`main.py:178`). | Judge-visible surface; depends on the language of the evaluation. | (a) Bilingual · (b) English · (c) Leave |
| **D-9** | **The two uncommitted files** (`backend/auth/routes.py`, `scripts/simulate_federated.py`). | W-44. Unknown whether they are wanted before the freeze. | (a) Commit · (b) Revert · (c) Leave and note |

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
| **Diabetes ROC-AUC 0.831 / sens 91.70% / spec 54.63%** | Proposal §8.1 | ✅ **REPRODUCED** (0.8305 / 91.55% / 54.68%) | `evaluation_evidence/diabetes_report.txt` |

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
