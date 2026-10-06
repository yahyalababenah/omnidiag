# Post-expo cleanup — plan and record

Rebuilt in the repository on 2026-10-06. The original lived in a session scratchpad under
`/tmp`, which was wiped at midnight. Plans, harness scripts and baselines now live here.

**Rules for every gate:**
- Yahya approves each gate: a plan first, then execution after approval.
- The diff is shown before every commit.
- Each gate is one commit, pushed to the branch and never merged by the assistant.
- A cross-layer check runs after every change.
- If something turns out bigger than expected, stop and report instead of making a partial fix.

## Branches and status

| Gate | What | Commit(s) | Branch |
|---|---|---|---|
| Phase A | Non-code clutter to `archive/post_expo_2026-10/` (`INDEX.md` there); Phase 9 research outputs to `docs/phase9/results/` | `f22b469` | `chore/post-expo-cleanup`, merged into `deploy/v2-platform` |
| B1 | Heart What-If tests decoupled from BRFSS artifacts; `brfss` pytest marker | `3557049` | `chore/retire-brfss` |
| B1b | NHANES What-If test coverage (policy, F9-32, fallbacks, determinism) | `f2b73a9` | 〃 |
| — | F9-40 recorded (NHANES cross-group non-monotonicity) | `e2dd7ff` | 〃 |
| B2 | `scripts/build_drift_reference.py` heart-only (no BRFSS CSV or config at build time) | `1b264b6` | 〃 |
| B3 | BRFSS config archived; `backend/retired_diseases.py`; 410 for new writes, records stay readable; W-08 closed | `cc655af` | 〃 |
| B4 | BRFSS loader, schemas and legacy retrain writer removed; non-candidate families get `unsupported` | `3eb4c84`, W-26 docs `4ba08c8` | 〃 |
| B5 | BRFSS CF tables, scale/correction, notes map and report bands removed; archived report for retired rows | `8186c4d` | 〃 |
| B6 | Frontend: BRFSS removed, retired items read-only, history archived note | heart Age/Sex `83b3576`; B6 `6be40b0` | 〃 |
| B7 | Docker/data (areas 0–5), tests (6), tools (7), docs (8) | 0 tools `c292333`; 1 Dockerfile `40163fb`; 2 models/diabetes JSON `9852c56`; 3 BRFSS CSV `66190c5`; 4 lightgbm `601ee57`; 5 flwr + greenlet (CL-4) `99a2bac`; 6 tests `ee78539`; 7 tools: this commit | 〃; area 8 in progress |
| Tools | Verification and maintenance scripts | — | folded into B7 area 7 |
| Docs | README and module docs | — | folded into B7 area 8 |
| Release | Frontend (Vercel) first, then the HF snapshot | — | not started |

## Decisions that stand (do not relitigate)

- **Retired disease rule.** Reading existing records of a retired disease is allowed; scoring, writing, labelling, drift and retraining answer 410 `DISEASE_RETIRED`. DB rows are never deleted. The single source of truth is `backend/retired_diseases.py`.
- **Configs are archived, not deleted**: `archive/post_expo_2026-10/configs/`.
- **`retrain_xgb` was deleted** (B4). Every family outside the candidate path returns `status: "unsupported"`, reason `retrain not yet supported for <family>`, before reading any sample. Reason: a writer whose output no live module reads is the W-08 hazard.
- **Cross-layer baseline changed at B4.** `/explain` lost two always-null keys (`ensemble_variance`, `model_agreement`). The baseline is now `baselines/crosscheck_B4.txt`.
- **No prevalence correction.** A config declaring `prevalence_train`, `prevalence_deploy` or non-null `risk_bands` fails at load with `prevalence correction is not supported (removed in B5)`. `Scale.CORRECTED` stays as the legal value of the DB column for old rows.
- **Retired rows never reach the LLM.** They render through `archived_report()`, which is rule-based and starts with the banner "ARCHIVED MODULE — replaced by NHANES dysglycaemia module". `archived_note` appears on export and history items.
- **Frontend `RETIRED_DISEASES` constant**, kept equal to the backend list by `tests/test_frontend_retired_contract.py`. It is removed after the HF release.
- **Release order.** The frontend ships first; it must work against both backends. Then the HF snapshot, with a rollback commit recorded and `docs/**/*.png` excluded. After the HF build, a real DB write + read is smoke-tested on the live Space (create a patient, predict, read the history) before the release is declared done (CL-4).

## Remaining gates

- **B7, Docker/data:**
  - remove the BRFSS weight downloads from the Dockerfile;
  - untrack the BRFSS CSV and drop its `.gitignore` exception;
  - remove `models/diabetes/*.json`;
  - remove `lightgbm` if unused, and `flwr` (separate commit, with `pip check`).
- **Tools:**
  - delete the unreachable legacy writer functions in `scripts/retrain.py`;
  - rework or archive `scripts/golden_master.py` and `scripts/verify_live.py` (heart+BRFSS only, and they read the removed `demos["diabetes"]`);
  - rework `scripts/ui_verify.py`;
  - `scripts/data/make_sample_batches.py`: switch its diabetes sample to NHANES or drop it;
  - delete the 27 remaining `brfss`-marked tests, which fail by design.
- **Docs:** README, `docs/DIABETES_AUDIT_REPORT.md`, `docs/FEATURE_VERIFICATION.md`, `evaluation_evidence/diabetes/`, `docs/ARCHITECTURE.md`.
- **Release:** as stated above.

## Gate B7 — Docker/data (PLAN, 2026-10-06, awaiting approval)

Scope, as defined under "Remaining gates" above plus backlog item 10. One commit per area.

**Measured state:**
- **Dockerfile:** downloads 5 BRFSS files (`omni_diag_xgb/lgb/rf`, `meta_learner`, `preprocessors/standard_scaler`) and creates `models/diabetes/preprocessors`. `heart_full_tuned.pkl` (heart revert path) and the NHANES joblib are not BRFSS and stay.
- **Tracked BRFSS files:** the CSV (6.3 MB, behind a `.gitignore` exception) and 6 JSON files in `models/diabetes/`.
- **Readers left:**
  - `backend/monitoring/drift.py`: `_REFERENCE_PATHS["diabetes"]` (Evidently, not live) and `_PROFILE_PATHS["diabetes"]`. Both are dead since B3, because drift for `diabetes` answers 410 before reaching them.
  - `tests/test_drift_stats.py::TestDiabetesReferenceIsReweighted` (2 tests; JSON + CSV).
  - `tests/test_diabetes_monitoring.py::test_the_rule_matches_the_other_modules`, which compares the NHANES rule with the BRFSS profile.
  - `ensemble_metrics.json`, `metrics*.json` and `preprocessors/*.json`: no reader.
- **requirements:** `lightgbm==4.6.0` and `flwr>=1.0.0`. Nothing imports either. A byte scan of every artifact the image ships finds 0 references to either; `xgboost` is in `heart_full_tuned.pkl`, so it stays.

**Commits:**

| # | Area | Change | Tests touched |
|---|---|---|---|
| 0 | tools | build-trace (`trace.py`) and `.dockerignore` matcher (`dockerignore_check.py`) into `docs/cleanup/tools/` (they were lost with `/tmp`) | — |
| 1 | Dockerfile | remove the 5 BRFSS downloads and the `models/diabetes` mkdir; reword the comments that point to them | — |
| 2 | models/diabetes JSON | `git mv` the 6 files to `archive/post_expo_2026-10/models/diabetes/` (the docs cite their numbers) + INDEX rows; remove `_PROFILE_PATHS["diabetes"]` | rule comparison → heart's profile (R); `TestDiabetesReferenceIsReweighted` deleted (D) |
| 3 | BRFSS CSV | `git rm --cached` (the file stays on disk and becomes ignored by `*.csv`) and remove its `.gitignore` exception; remove `_REFERENCE_PATHS["diabetes"]` | — |
| 4 | lightgbm | remove from requirements.txt | — |
| 5 | flwr | remove from requirements.txt. **Executed differently (approved 2026-10-06):** flwr was the only source of greenlet in the image and capped 7 packages (CL-4), so the commit also declares `sqlalchemy[asyncio]` and the 7 upper bounds | `tests/test_async_db_dependency.py` added (A) |

**Acceptance (each commit):**
- `-m "not brfss"`: 0 failures, in the normal tree and in a clean worktree with no BRFSS files.
- `docs/cleanup/tools/crosscheck.py`: identical to `baselines/crosscheck_B4.txt`.
- `-m brfss`: the set change reported.

**Acceptance (specific):**
- **Commits 1–3:** the two build steps (`train_heart_glm.py --verify`, `build_drift_reference.py --verify`) are traced in a clean worktree without BRFSS files; every file they read must be inside the Docker context. A static check confirms no Dockerfile line refers to `models/diabetes`. No full Docker build (the laptop can't).
- **Commits 4–5:** an import blocker makes `lightgbm` / `flwr` raise on import while the full suite, the cross-layer check and both build steps run, which proves nothing needs them. Plus a dependency-resolution check, `pip install --dry-run --ignore-installed --report` for the old vs the new requirements, listing which packages disappear and showing that resolution has no conflicts.

**Not verified:**
- **`pip check` on a fresh install:** a fresh environment would download several GB (transformers, spacy, mlflow…), and the disk has ~6 GB free. Resolution with `--dry-run` is the substitute.
- **The HF model repo `yahyoha/omnidiag-models`:** it still hosts the BRFSS files. It's external, so I'm leaving it.

### B7, added areas (PLAN, awaiting approval)

**BRFSS field names (definition used throughout):** HighBP, HighChol, CholCheck, GenHlth, MentHlth, PhysHlth, DiffWalk, HvyAlcoholConsump, NoDocbcCost, AnyHealthcare, HeartDiseaseorAttack, Fruits, Veggies, PhysActivity, Smoker, Education, Income, Diabetes_binary, Diabetes_Clinical_Risk. Generic names (Age, Sex, BMI, Stroke) are not counted.

**Area 6, tests:**
- **Delete the 27 `brfss`-marked tests:**
  - `test_diabetes_calibration`: 16 (`TestDiabetesPredictApi` ×6 groups, `TestDiabetesEdgeInputs`);
  - `test_clinical::TestDiabetes`: 5;
  - `test_model_family_registry`: 4;
  - `test_diabetes_gate93`: 2.
- **Also delete** the fixtures and constants that only they use (`DIABETES_LOW/HIGH/MIN/MAX`, `DIABETES_PAYLOAD`, `DIABETES_PATIENT`, `RISK_BANDS_RAW` where unused, the archived-config read).
- **Remove the `brfss` marker** from `pytest.ini`.
- **Rewrite** the BRFSS field names that remain in unmarked tests:
  - neutral lever names in the unit tests of the shared What-If helpers (`test_whatif_best_achievable`) and of the report (`test_llm_report`, `test_report_privacy_and_bands`);
  - placeholder features in the retired-row fixtures (`test_retired_disease`, `test_database`); a retired route answers 410 before reading any feature, so their names don't matter;
  - the leftovers in `test_notes_parser` and `fixtures/clinical_notes.json`.
- **From this commit on**, acceptance = the full suite, no marker filter, 0 failures.

**Area 7, tools:**

| Tool | Decision | Why |
|---|---|---|
| `scripts/retrain.py` | delete `load_reference_data`, `load_production_metrics`, `retrain_xgboost`, `promote_model`, `flush_cache` and the legacy steps after the exit-2 refusal | unreachable since B4; only the candidate path remains |
| `scripts/golden_master.py` | **rework** for heart + NHANES: NHANES edge cases and batch rows replace `DIABETES_EDGE` and the BRFSS bounds | `verify_live.py` imports its demo loader, and a full-output golden master (incl. /batch, /schema) still has value. Check: two captures in the same environment diff to 0 |
| `scripts/verify_live.py` | **keep**, docstring only (it is generic over the demo patients) | it is the release check (live Space vs local). Check: run against a local server, exit 0 |
| `scripts/ui_verify.py` | **archive** | Playwright against live URLs, 22 BRFSS references and none for NHANES, tied to the FEATURE_VERIFICATION items archived in area 8; a rework cannot be verified on this laptop |
| `scripts/data/make_sample_batches.py` | diabetes sample → **NHANES** fields (inside the schema bounds; a few deliberately invalid rows, as the heart sample has); `max_batch_rows` 100 | a demo file for the live module is useful; the BRFSS one cannot be uploaded anywhere now |
| `frontend/eslint.config.js` | `ignores: ['dist']` becomes a global ignore (its own config object) | `dist/` is linted today (~954 of the problems); new total reported; `frontend/src` must stay ≤ 389 |

**Area 8, docs:**
- **Archive:** `docs/DIABETES_AUDIT_REPORT.md` and `docs/FEATURE_VERIFICATION.md` go to `archive/post_expo_2026-10/docs/`, with INDEX rows.
- **Update the references** to them: the `README.md:216` link, the text at `docs/EVIDENTLY_COST.md:77`, the docstrings of 3 tests, and the `_about` of `tests/fixtures/clinical_notes.json`.
- **README:** everything that presents BRFSS as registered or current gets rewritten:
  - "Three modules are currently registered…";
  - the architecture mermaids (`EnsembleLoader`, `configs/diabetes.yaml`, the stacking path);
  - the retrain paragraph about `retrain_xgb`, and the `EnsembleModelLoader` predict path;
  - the project-tree rows (`ensemble_loader.py`, `stacking_ensemble.py`, `models/diabetes/`, `configs/diabetes.yaml`, "BRFSS DM" in mockPatients);
  - the API section on the correction scale and its field table;
  - the technology-table row "Model v2 (DM)".

  Historical mentions stay where they are clearly historical (e.g. "BRFSS diabetes is superseded"). Every changed README line will be listed.
- **`docs/ARCHITECTURE.md`, `docs/ADDING_A_MODEL_FAMILY.md`:** corrected; they are current guides (the `stacking_ensemble` family row, the BRFSS nodes).
- **`docs/MODEL_FAMILY_REGISTRY_DESIGN.md`, `docs/data_quality/*`:** dated records, so a dated status note goes at the top, without rewriting their history (the same pattern as the registry addendum already there).
- **`evaluation_evidence/diabetes/`:** evidence, not a doc; a short README in the folder states it belongs to the archived module.
- **`docs/phase9/*`:** a research record; BRFSS appears there as the predecessor, so it stays untouched.

**Final B7 acceptance:**
- the full suite with 0 failures;
- `crosscheck.py` identical to the baseline;
- 0 BRFSS field names in live code (backend, features, configs, frontend/src), in tests, and in the Dockerfile.

After that: the release plan (plan only).

## Phase C — remote branches (report only; nothing deleted)

The default branch is `deploy/v2-platform`, and it is the only protected branch.

| Branch | Last commit | Merged | Recommendation |
|---|---|---|---|
| fix/demo-blockers, fix/feature-verification, fix/heart-retrain-candidate, merge/heart-retrain-plus-phase9, monitoring/request-metrics | 2026-09-22 … 09-30 | yes | delete |
| main | 2026-06-28 | yes | delete |
| phase9/diabetes-nhanes-ebm | 2026-09-28 | yes | tag-then-delete (research milestone) |
| feature/diabetes-module-integration, feature/omni-platform-final | 2026-05-30 / 06-01 | no (2 non-equivalent commits each) | tag-then-delete |
| production-final-v3 | 2026-05-29 | no (7) | tag-then-delete |
| V2-advanced-model, V3-advanced-DL | 2026-05-21 / 05-26 | no (unrelated history) | tag-then-delete |
| main1 | 2026-05-22 | no (unrelated history; 4 files not in base: old model + plots) | tag-then-delete |
| hf/main | 2026-09-30 | n/a | keep (the live Space) |
| chore/post-expo-cleanup | 2026-10-05 | yes | delete after confirmation |

## Verification tools and baselines (in this folder)

- **`tools/crosscheck.py`:** predict/explain/CF/report/drift/retrain/AL for heart and NHANES against the real app. To run: from the repo root, `DATABASE_URL=sqlite+aiosqlite:///<tmp>/x.db PYTHONHASHSEED=0 backend/.venv/bin/python docs/cleanup/tools/crosscheck.py`, then diff the `OK|FAIL|DIGEST` lines with `baselines/crosscheck_B4.txt`. It reproduces the recorded B4 digests exactly.
- **`tools/screen_probe.py`:** replays what each frontend screen requests, against a running backend. Arguments: `BASE OUT demo.json patient_ids`.
- **`tools/frontend_snapshot/`:**
  - `dump_schemas.py` writes the live schemas and CF responses.
  - `snap.sh <workdir>` snapshots the frontend utils for heart/NHANES; diff the result with `baselines/frontend_utils_B6.json`.
  - `screens.mjs` applies the frontend logic to two probes. Its B6 result against both backends is `baselines/screens_B6.json`.
- **Lint baseline:** `frontend/src` = 389 problems (`npx eslint src`).

## Open verification (before release)
- Visual browser check not done in B6. Yahya will check the Vercel preview of chore/retire-brfss
  (picker, What-If, AL queue read-only items + Archived card, error banner, history archived note)
  against the production backend before release.
- PDF for history rows: not reachable from the current frontend. The archived banner reaches users
  only through report text and the backend export.

## Backlog (after Phase B, in priority order)
1. HIGH, F9-40: explain group-conditional decisions in the UI and PDF (decision wording + per-band
   note, no model change). 41% of cleared patients aged 60+ are dysglycaemic vs 4.6% at 20–39.
2. Remove the frontend RETIRED_DISEASES constant after the HF release (keep the contract test until then).
3. Run pytest in CI (CI currently runs no tests).
4. Add a JS test runner to the frontend (none exists; B6 relied on Node checks of pure functions).
   Rebuild UI verification for heart + NHANES (replaces the archived ui_verify.py).
5. ~~Fix the ESLint config~~: done in B7 area 7 (`dist` is a global ignore). `npx eslint .` went
   from 1343 to 389 problems, all of them in `frontend/src` (baseline 389, unchanged).
6. Active Learning on NHANES: wire build_candidate for ebm_platt_conformal (retrain currently returns
   `unsupported` by design).
7. Evaluate monotonic alternatives to Mondrian per-group conformal (research, no implementation yet).
8. The frontend has no export screen. Decide later whether one is needed; not part of cleanup.
9. 4 anonymous predictions in the local DB (no patient link): pre-existing; they never appear in
   history. Decide whether that's intended.
10. ~~Remove flwr/lightgbm from requirements.txt~~: done in B7 (lightgbm `601ee57`, flwr in area 5).
11. HF model repo `yahyoha/omnidiag-models`: the BRFSS files are still hosted there. Keep them for reproducibility of the archived docs, or remove them later.
12. Lift the 7 upper bounds flwr used to impose (fastapi <0.139, starlette <1.4, uvicorn <0.50,
    cryptography <47, packaging <26, rich <15, typer <0.21), with a test run on the new versions (CL-4).
13. Add a lock file (pip-compile or similar) and align the local venv with the image, so tests run on
    production versions. Today they diverge: SQLAlchemy 2.0.51 locally vs 2.1.3 in the image (CL-4).
14. LOW: `/batch` checks the file type and `chest_pain_coding` before the retired-module check, so
    `POST /api/v4/diabetes/batch?chest_pain_coding=…` answers 400 `CODING_NOT_APPLICABLE` instead of 410.
    Still a refusal, nothing is scored; found in B7 area 6 (the test that relied on it now uses NHANES).

## Findings and lessons
- W-08 (closed in B3): retrain for a retired disease would have overwritten production BRFSS weights.
- W-26 (superseded in B4): the legacy retrain writer produced files no live module read.
- F9-40: NHANES cross-group decision non-monotonicity (documented, not fixed).
- B5: the router band-correction branch was removed, so a future config declaring priors would have
  been labelled CORRECTED without any correction applied. Now refused at load.
- B6: the frontend dictionary matches fields by label case-insensitively, so heart `Sex` and NHANES
  `RIAGENDR` ("Sex") share one entry. A fix for one module silently broke the other. Caught before push.
  Cross-layer finding: shared lookups keyed by display label.
- B6: the shared `Age`/`Sex` dictionary entries showed BRFSS meanings on heart tooltips (pre-existing).
- Process: the scratchpad under /tmp was wiped at midnight and the plan was lost. Plans, harness
  scripts, and baselines now live in the repo.
- B7 (CL-4): greenlet reached the image only through the unused flwr; removing it would have broken
  the first DB query in the image while local tests stayed green (local SQLAlchemy 2.0.51 vs 2.1.3 in
  the image). Caught by the dependency-resolution diff, not by the import blocker.
- Hollow tests caught: the cached-schema test (cache disabled in tests), and the F9-32 regression
  initially caught only by the policy pin. Both fixed by testing behaviour directly.
- B7 area 6, mutation check of the renamed helper tests (`docs/cleanup/tools/mutate.py`, 8 mutations,
  all caught). Two gaps found on the way: the privacy test checked `build_prompt()`, a copy of
  `generate_report()`'s prompt assembly, so `generate_report()` forwarding the raw values went
  unnoticed (a test now captures the prompt actually sent); and dropping only the rule-based report's
  "Top contributing factors" line is not caught, because the same names appear in its drivers list.

The cross-layer items above are recorded in `docs/phase9/FINDINGS_REGISTER.md` as W-08, W-26,
F9-40, CL-1, CL-2, CL-3 and CL-4.
