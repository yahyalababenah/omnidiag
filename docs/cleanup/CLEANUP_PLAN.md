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
| B7 | Docker/data (areas 0–5), tests (6), tools (7), docs (8) | 0 tools `c292333`; 1 Dockerfile `40163fb`; 2 models/diabetes JSON `9852c56`; 3 BRFSS CSV `66190c5`; 4 lightgbm `601ee57`; 5 flwr + greenlet (CL-4) `99a2bac`; 6 tests `ee78539`; 7 tools `264e1df`; 8 docs `9421253` | 〃; **B7 done** (final acceptance below) |
| Tools | Verification and maintenance scripts | — | folded into B7 area 7 |
| Docs | README and module docs | — | folded into B7 area 8 |
| Release | Frontend (Vercel) first, then the HF snapshot, then Phase C | GitHub `8bbbfe9`; HF `c816150` | **RELEASE DONE 2026-10-06**; Phase C commands ready, not run |

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

## Gate B7 — Docker/data (approved and executed 2026-10-06)

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

### B7, added areas (approved and executed 2026-10-06)

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

## B7 final acceptance (2026-10-06, HEAD `9421253`)

- Full suite, no marker filter: **879 passed, 1 skipped, 0 failed**, in the normal tree and in a clean
  worktree with no BRFSS files (the skip: `tests/test_notes_spacy_context.py`, spaCy / en_core_web_sm not
  installed in `backend/.venv`).
- `crosscheck.py`: **IDENTICAL** to `baselines/crosscheck_B4.txt` in both trees.
- Build steps traced in the clean worktree: `train_heart_glm.py --verify` and
  `build_drift_reference.py --verify` exit 0; 25 reads, all inside the Docker context; none is a BRFSS
  file; the Dockerfile names no `models/diabetes/`.
- BRFSS field names (the 19 listed under "B7, added areas"), whole-word, in `backend`, `features`,
  `configs`, `frontend/src`, `tests` and `Dockerfile`: **0**. (A substring search finds only the English
  word "smoker" in clinical-note text.)
- Not verified here: a full Docker build (the laptop cannot). The HF build in the release is the first
  full image build of B7; that is why the rollback below is recorded before the push.

## Release plan (APPROVED 2026-10-06 with four additions; nothing below has been executed)

State recorded on 2026-10-06, before any release step:
- Live Space `yahyoha/omnidiag` (remote `hf`): `main` = `d6fb3238de7974dc7c43e8ed49420cfc291e4641`.
  It still serves `diabetes` (BRFSS): `/api/v4/diseases` lists `diabetes`, `diabetes_nhanes`, `heart_disease`.
- Live frontend `omnidiag-delta.vercel.app`: bundle `assets/index-BNeSlGQM.js`.

**Rollback targets (recorded before any step):**

| Layer | Target | How |
|---|---|---|
| Backend (HF Space) | `d6fb3238de7974dc7c43e8ed49420cfc291e4641` | `git push --force hf d6fb3238de7974dc7c43e8ed49420cfc291e4641:refs/heads/main` |
| Frontend (Vercel) | the previous production deployment, serving `assets/index-BNeSlGQM.js` (`index.html` last-modified Fri 2026-10-02 16:14:19 GMT, etag `86704c8f1bcadc129929094836ac3ef5`). Deployment ID: **not recorded** | Vercel → Overview → **Instant Rollback** (Yahya, 2026-10-06) |
| GitHub `deploy/v2-platform` | `f22b4695f14e42f9cc3c1eaa29d71c3e783bb87f` | `git revert` of the pushed range; never a force push |

The Vercel deployment **ID** is not visible from this machine: the response headers carry no deployment
identifier, and there is no Vercel CLI here. Before step 1.2, Yahya writes the ID of the current
production deployment (Deployments → Current) into this table. Per Vercel's documentation, after an
Instant Rollback new deployments are not promoted automatically until the rollback is undone in the
dashboard.

### Step 1 — frontend first (Vercel)

The B6 frontend works against both backends (verified in B6 against both, `baselines/screens_B6.json`), so
it can go live while the Space still serves BRFSS.

1. **Before merging:** Yahya checks the Vercel preview of `chore/retire-brfss` against the production
   backend (the item under "Open verification"): picker without BRFSS, What-If, the AL queue with
   read-only archived items and the Archived card, the error banner, the history archived note.
2. **Merge, as a fast-forward push.** It deploys nothing to the Space, which is a separate snapshot push.
   - **Checked 2026-10-06:** `origin/deploy/v2-platform` = `f22b469`;
     `git merge-base --is-ancestor origin/deploy/v2-platform chore/retire-brfss` is true, and the branch is
     21 commits ahead at `79e9262` (one more with this record) and 0 behind. So this is a fast-forward, and no merge commit is needed.
   - **Re-check immediately before pushing.** Both commands must succeed:
     ```
     git fetch origin
     test "$(git rev-parse origin/deploy/v2-platform)" = f22b4695f14e42f9cc3c1eaa29d71c3e783bb87f && git merge-base --is-ancestor origin/deploy/v2-platform origin/chore/retire-brfss && echo FAST-FORWARD-OK
     ```
   - **The push** (no `--force`; git refuses it if it is not a fast-forward):
     ```
     git push origin origin/chore/retire-brfss:refs/heads/deploy/v2-platform
     ```
   - **Branch protection:** `deploy/v2-platform` is marked protected. The public API shows no required
     status checks, but it cannot show whether a pull request is required. If the push is refused for
     that reason, stop. A web-UI merge would not be a fast-forward of these exact commits, so that path
     needs your decision.
3. **Confirm the deploy happened:** the live bundle name must change from `assets/index-BNeSlGQM.js`.
   Vercel deploys cannot be triggered or seen from this machine, so an unchanged name means a stale
   frontend, not a passed check.
4. **Live check against the old backend** (`d6fb323`): the same list as step 1.1, on the production URL.
5. **Frontend rollback:** in the Vercel dashboard, promote the previous production deployment (the one
   serving `index-BNeSlGQM.js`). In git: revert the merge commit on `deploy/v2-platform`.

### Step 2 — HF snapshot (backend)

Preconditions: step 1 is done and checked, and `git ls-remote hf main` still returns `d6fb323`. If it
does not, someone deployed in between: stop, and record the new hash as the rollback instead.

1. **Build the snapshot without touching the working tree.** Use a temporary index:
   - `read-tree` of the merged `deploy/v2-platform` commit;
   - `git rm --cached` of every file git treats as binary, i.e. the `-  -` rows of
     `git diff --numstat <empty-tree> <commit>`. Today that is 54 files: every `docs/**/*.png`,
     `docs/fl_report`, `reports/monitoring_local`, `archive/stale_2026-09`, the `frontend/public`
     icons, and `models/diabetes_nhanes/diabetes_nhanes_ebm.joblib`. The Dockerfile downloads that
     joblib and checks its sha256. HF rejects any binary in a push, whatever its size, and the live
     tree has none.

     Dockerfile lines 48–51, as committed:
     ```
     RUN mkdir -p models/diabetes_nhanes && \
         curl -fsSL "https://huggingface.co/yahyoha/omnidiag-models/resolve/main/diabetes_nhanes/diabetes_nhanes_ebm.joblib" \
             -o models/diabetes_nhanes/diabetes_nhanes_ebm.joblib && \
         echo "fcceeb37b53295f115e5fe41b9a8618117263ef6ddb7898f9a90ad7b50b2c408  models/diabetes_nhanes/diabetes_nhanes_ebm.joblib" | sha256sum -c -
     ```

     Checked 2026-10-06: the hash in the Dockerfile matches the git-tracked file, and the file
     downloaded from that URL passes `sha256sum -c`. A mismatch fails the build (`-c` returns
     non-zero), so a wrong file cannot ship silently;
   - `write-tree`, then `commit-tree -p d6fb323`, so the push is a fast-forward.
2. **Check before pushing:**
   - 0 binary files in the snapshot (the same numstat check);
   - `README.md` starts with the HF front matter (`sdk: docker`, `app_port: 7860`);
   - the snapshot tree equals the merged commit's tree minus exactly that file list.
3. **Push:** `git push hf <snapshot>:refs/heads/main`. This is a plain fast-forward, without `--force`.
4. **Rollback (recorded now):**
   `git push --force hf d6fb3238de7974dc7c43e8ed49420cfc291e4641:refs/heads/main`.
5. **Build log:** it must show
   - the heart bundle built and `--verify` OK, and the drift reference `--verify` OK;
   - the NHANES joblib `sha256sum -c` OK;
   - no BRFSS download;
   - `greenlet` among the installed packages (CL-4).

   If the build fails: roll back and report.
6. **Smoke test on the live Space, before the release is declared done (CL-4):**
   - **A real database write and read, with a clinician account** (credentials from Yahya, not stored
     in the repo):
     - create a patient: `POST /api/v4/patients` with `{"mrn": "RELEASE-TEST-2026-10", "full_name":
       "RELEASE TEST 2026-10 (not a patient)"}`. MRNs are unique; if a rerun is refused, use
       `RELEASE-TEST-2026-10-2`;
     - use the demo inputs P-003 (heart) and N-001 (NHANES). Both are confident decisions, so neither adds
       a row to the review queue; check that neither response carries a `review_id`. If one does,
       skip that queue item afterwards (`POST /api/v4/review/<id>/skip`), so no doctor is asked to
       label a test patient;
     - `POST /api/v4/heart_disease/predict?patient_id=<id>` and `POST /api/v4/diabetes_nhanes/predict?patient_id=<id>`;
     - read the history (`GET /api/v4/patients/<id>/predictions`): both rows present, each with its
       `decision` and `probability_scale`.

     This is the path that greenlet guards. The local tests run on SQLAlchemy 2.0.51; the image runs 2.1.3.
   - **After the check, the test patient is soft-deleted:** `DELETE /api/v4/patients/<id>` with an admin
     or super_admin account. The API has no hard delete. The patient row stays with `deleted_at` set
     and disappears from lists; its two prediction rows and audit rows stay, as for any patient. This
     follows the standing rule that records are never deleted, and the MRN marks them as a test.
   - **Database persistence on the Space (checked 2026-10-06):**
     - The Space has **no persistent storage** (HF API: `runtime.storage = null`, hardware `cpu-basic`).
     - The app uses `DATABASE_URL` if it is set, and SQLite at `./omnidiag_dev.db` inside the container
       otherwise. Whether a `DATABASE_URL` secret points to an external database is not visible from
       here; Yahya checks Space Settings → Variables and secrets.
     - **Without that secret, the database is lost on every restart and rebuild, including this
       release's build.** The test patient would then vanish at the next restart anyway, and any BRFSS
       rows in the live database will be gone after the push. The `archived_note` check below then has
       nothing to look at; the tests cover it.
   - **Retirement:**
     - `/api/v4/diseases` lists exactly `heart_disease` and `diabetes_nhanes`;
     - `POST /api/v4/diabetes/predict` answers 410 `DISEASE_RETIRED` with `replaced_by: diabetes_nhanes`;
     - if the live database holds BRFSS rows, their history items carry `archived_note`.
   - **Parity:** `scripts/verify_live.py` (default URL = the live Space) exits 0: 6 demo patients ×
     predict/explain/counterfactuals, identical to the local app within 1e-6.
   - **Frontend against the new backend:** the step 1.1 list again, plus one report generation for a
     heart result.
7. **Done:** only when all of step 6 passes. Then record the new live hash and close "Open
   verification". Next comes backlog item 2: remove the frontend `RETIRED_DISEASES` constant and its
   contract test.

**Not covered by this plan:** the HF model repo (`yahyoha/omnidiag-models`) keeps the BRFSS files
(backlog 11). The image no longer downloads them, so they are inert.

### Release log (2026-10-06)

| Step | Result |
|---|---|
| 1.1 Preview check | Done by Yahya, visually: heart + NHANES only |
| 1.2 Merge | `deploy/v2-platform` = `8bbbfe9cc08a00ea5487a35b42e3a380e04ad7e5` (= `chore/retire-brfss`); fast-forward of `f22b469` confirmed from git |
| 1.3 Bundle changed | `index-BNeSlGQM.js` → `index-CSdJeAX2.js` (`index.html` last-modified 2026-10-06 06:23:51 GMT). The bundle carries `archived_note` and `retired_pending`, no BRFSS field names, and the API base is `yahyoha-omnidiag.hf.space` |
| 1.4 New frontend vs old backend (`d6fb323`) | Backend listed `diabetes, diabetes_nhanes, heart_disease`; the frontend's `withoutRetired` gave `diabetes_nhanes, heart_disease`. 6 demo patients: schema/predict/explain/counterfactuals all 200, valid What-If states (frontend source at `8bbbfe9`, run in Node). The authenticated screens (AL queue, history) were not checked here: there is no account; Yahya checked the preview visually |
| Vercel rollback ID | **Not recorded** (twice the placeholder came through unfilled). The rollback target is identified by its bundle `index-BNeSlGQM.js` and `index.html` etag instead |
| Vercel promotion (Yahya) | The Vercel **Production Branch was `main`** (`52a1ab7`, 2026-06-28), not `deploy/v2-platform`: a push to `deploy/v2-platform` gives only a preview. Yahya promoted the `8bbbfe9` deployment to production manually; domain `omnidiag-delta.vercel.app`. Re-checked: bundle `index-CSdJeAX2.js`, `index.html` last-modified 06:44:37 GMT. Yahya is switching the Production Branch to `deploy/v2-platform`; until that is confirmed, `main` must not be deleted (Phase C 3.0) |
| 2.0 Precondition | `git ls-remote hf main` = `d6fb323` |
| 2.1–2.2 Snapshot | tree `1e9021f`, built from `8bbbfe9` minus 54 binaries. Checks: 0 binaries; HF front matter present; equals the source minus exactly the strip list; heart training CSV present; no BRFSS paths; `sqlalchemy[asyncio]`, no flwr/lightgbm |
| 2.3 Push | `c81615055d9be57672d76d5ddf90c08468d57511`, child of `d6fb323`; fast-forward `d6fb323..c816150` |
| 2.4 Rollback | `git push --force hf d6fb3238de7974dc7c43e8ed49420cfc291e4641:refs/heads/main` (unchanged) |
| 2.5 Build log | Built at 06:25:47. NHANES joblib `sha256sum -c` OK; only the heart revert file downloaded, no BRFSS file. `train_heart_glm --verify`: fingerprint OK, decisions identical for all 920. `build_drift_reference --verify`: profile matches. Installed: `greenlet-3.5.6`, `sqlalchemy-2.1.3`, fastapi 0.138.2, starlette 1.3.1, uvicorn 0.49.0; no flwr, no lightgbm. Space `RUNNING` on `c816150` at 06:29 |
| 6 Retirement | `/api/v4/diseases` and `all_registered` = `diabetes_nhanes, heart_disease`; `POST /api/v4/diabetes/predict` → 410 `DISEASE_RETIRED`, `replaced_by: diabetes_nhanes`; `/api/v4/diabetes/schema` → 410 |
| 6 Parity | `verify_live.py` against the live Space: 6 patients × 3 endpoints IDENTICAL (tol 1e-6), exit 0 |
| 6 Frontend vs new backend | The same Node check: picker = backend list (heart, NHANES); 6 patients all 200 |
| 6 Report | `POST /api/v4/generate-report` for a heart result: `source: llm`, `risk_band: null` |
| 6 **DB write + read** | **PASSED.** Run with the seeded default accounts, on Yahya's explicit approval: the doctor for the test, the admin only for the soft-delete. Results: <br>- doctor login 200 (role `doctor`); <br>- `POST /api/v4/patients/` → 201, MRN `RELEASE-TEST-2026-10`, id `836fb5ad-360d-4ae1-9338-11f5555a69f9`; <br>- patient-linked predict, heart P-003 → 200, `no_referral`, 0.0875, `ivap_calibrated_training_mix`; <br>- patient-linked predict, NHANES N-001 → 200, `no_referral`, 0.0359, `platt_calibrated_nhanes_2015_2016`; <br>- neither returned a `review_id`, and no pending review item belongs to the test patient; <br>- the history read returned exactly the two rows, with the same confidences and `probability_scale: raw`; <br>- admin soft-delete → 200; afterwards the patient → 404, and it is absent from the list. <br>The async-session write and read path, which greenlet guards (CL-4), works in the image. <br>**One check in this plan was wrong:** it expected `decision` on history items, but `PredictionOut` has never exposed it. The column is stored, the API does not return it; this was true before this release too. Recorded as backlog 16. <br>Note: the first attempt posted to `/api/v4/patients` without the trailing slash and got 307; nothing was written |
| 6 archived_note | Nothing to check: there is no `DATABASE_URL` secret (Yahya, 2026-10-06), so the rebuild wiped the SQLite DB and no BRFSS rows exist on the Space. Covered by tests |

**Declared done: 2026-10-06.** Every step-6 check passed (the `decision` expectation above was a plan error, not a product failure). Live: GitHub `deploy/v2-platform`, Vercel production `8bbbfe9` (bundle `index-CSdJeAX2.js`), HF `c816150`. Rollbacks as in the table above. `chore/retire-brfss` is now on the Phase C delete list.

### Step 3 — Phase C, branch clean-up (only after step 2 is declared done)

Commands only; Yahya runs them. They follow the approved Phase C table. Tags come first for the
tag-then-delete branches, `main1` included. `hf/main` (the live Space) and `deploy/v2-platform` are
not touched. Each tag pins the commit recorded on 2026-10-06; a branch that moved since then fails
the check in 3.1.

**3.0 Gate for `main` (added 2026-10-06).** Vercel's Production Branch was `main`. Yahya is switching it to
`deploy/v2-platform`. **`main` is not deleted until Yahya confirms the switch**, because deleting the
production branch of a Vercel project before the switch would leave it pointing at a missing branch.
`main` is therefore taken out of 3.3 and deleted on its own in 3.3b.

**3.1 Check that every branch is still where it was recorded** (prints `OK` per branch, or `MOVED`):
```
git fetch origin --prune
while read b sha; do [ "$(git rev-parse origin/$b)" = "$sha" ] && echo "OK    $b" || echo "MOVED $b"; done <<'EOF'
fix/demo-blockers c21fb50bfa1289c51f4b5fa48a855f56c541d068
fix/feature-verification 7d01d63564158bfcf3771973f4c3375cb0af94cd
fix/heart-retrain-candidate e19c8f60c3c9ef97f112ee2a4f6f5e0e52361b66
merge/heart-retrain-plus-phase9 75a79756745bf46fed2084699d2f02e06170a347
monitoring/request-metrics eb5ecb51530e8ba88f690dd11428b2b4cc0e2028
main 52a1ab7a329ac3cde5a43187e2a36820529fa7b3
chore/post-expo-cleanup f22b4695f14e42f9cc3c1eaa29d71c3e783bb87f
phase9/diabetes-nhanes-ebm b796155967ead944bdc39e429480c3eebb5e0dc7
feature/diabetes-module-integration e9aeaf2e3753cd0a53922cff5969bbd0f394d482
feature/omni-platform-final 779db957efa0af36b0c9dde4af28720dc3464d2c
production-final-v3 401cf5d54a77f6f5c35d05f1b1428bd923534353
V2-advanced-model 25c16dcd1750b94a7e888c60fc0877aaed0294b3
V3-advanced-DL 16318c72427f77da2b62f2aceedce468b44e7769
main1 8018ec56bbc1040d2277165876923ea9b2f28efc
EOF
```
Stop on any `MOVED`. `chore/retire-brfss` was added after the release; it is checked against
`deploy/v2-platform` rather than a fixed hash:
```
[ "$(git rev-parse origin/chore/retire-brfss)" = "$(git rev-parse origin/deploy/v2-platform)" ] && echo "OK    chore/retire-brfss (same commit as deploy/v2-platform)" || echo "MOVED chore/retire-brfss"
```

**3.2 Tags for the tag-then-delete branches** (annotated, pinned to the recorded commits), then push them:
```
git tag -a archive/phase9/diabetes-nhanes-ebm b796155967ead944bdc39e429480c3eebb5e0dc7 -m "Phase 9 research milestone (NHANES EBM); branch deleted in Phase C, 2026-10"
git tag -a archive/feature/diabetes-module-integration e9aeaf2e3753cd0a53922cff5969bbd0f394d482 -m "Unmerged; branch deleted in Phase C, 2026-10"
git tag -a archive/feature/omni-platform-final 779db957efa0af36b0c9dde4af28720dc3464d2c -m "Unmerged; branch deleted in Phase C, 2026-10"
git tag -a archive/production-final-v3 401cf5d54a77f6f5c35d05f1b1428bd923534353 -m "Unmerged; branch deleted in Phase C, 2026-10"
git tag -a archive/V2-advanced-model 25c16dcd1750b94a7e888c60fc0877aaed0294b3 -m "Unrelated history; branch deleted in Phase C, 2026-10"
git tag -a archive/V3-advanced-DL 16318c72427f77da2b62f2aceedce468b44e7769 -m "Unrelated history; branch deleted in Phase C, 2026-10"
git tag -a archive/main1 8018ec56bbc1040d2277165876923ea9b2f28efc -m "Unrelated history (old model + plots); branch deleted in Phase C, 2026-10"
git push origin archive/phase9/diabetes-nhanes-ebm archive/feature/diabetes-module-integration archive/feature/omni-platform-final archive/production-final-v3 archive/V2-advanced-model archive/V3-advanced-DL archive/main1
git ls-remote --tags origin 'archive/*'
```
The last command must list all 7 tags, each pointing to an annotated tag whose target is the commit above
(`git rev-parse archive/main1^{commit}` and so on). Do not run 3.4 for a branch whose tag is missing.

**3.3 Delete the merged branches.** All seven were confirmed to be ancestors of `deploy/v2-platform` on
2026-10-06. `chore/post-expo-cleanup` is the same commit as `deploy/v2-platform` (`f22b469`), which is the
confirmation its table row asked for.
```
git push origin --delete fix/demo-blockers fix/feature-verification fix/heart-retrain-candidate merge/heart-retrain-plus-phase9 monitoring/request-metrics chore/post-expo-cleanup chore/retire-brfss
```

**3.3b Delete `main`, only after Yahya confirms the Vercel Production Branch is `deploy/v2-platform`** (3.0):
```
git push origin --delete main
```

**3.4 Delete the tag-then-delete branches** (only after 3.2 is confirmed):
```
git push origin --delete phase9/diabetes-nhanes-ebm feature/diabetes-module-integration feature/omni-platform-final production-final-v3 V2-advanced-model V3-advanced-DL main1
```

**3.5 Check:** `git ls-remote --heads origin` lists only `deploy/v2-platform`, plus `main` until 3.3b has run.

**Not in the approved table, so left alone:**
- Local worktrees (`Heart_Disease_Project.worktrees/merge-p9`, `phase9`,
  `agents-mermaid-er-diagram-db-models`) and local branches: these commands delete remote branches only.

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

## Open verification (closed at release, 2026-10-06)
- ~~Visual browser check not done in B6~~: Yahya checked the Vercel preview of chore/retire-brfss
  against the production backend before the merge, and it was correct (heart + NHANES only).
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
15. LOW (accepted risk, Yahya's decision, 2026-10-06; not a finding): set the `ADMIN_PASSWORD` /
    `DOCTOR_PASSWORD` Space secrets before any external demo or sharing. Until then the live Space uses
    the seeded defaults from `backend/main.py`.
16. LOW: the history API (`PredictionOut`) does not return the stored `decision`, nor the Venn-Abers
    interval, so a conformal row reads back with `prediction`/`diagnosis` only. Found in the release
    smoke test; pre-existing.

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
