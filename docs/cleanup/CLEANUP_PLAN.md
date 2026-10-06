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
| B7 | Docker/data | — | not started |
| Tools | Verification and maintenance scripts | — | not started |
| Docs | README and module docs | — | not started |
| Release | Frontend (Vercel) first, then the HF snapshot | — | not started |

## Decisions that stand (do not relitigate)

- **Retired disease rule.** Reading existing records of a retired disease is allowed; scoring, writing, labelling, drift and retraining answer 410 `DISEASE_RETIRED`. DB rows are never deleted. The single source of truth is `backend/retired_diseases.py`.
- **Configs are archived, not deleted**: `archive/post_expo_2026-10/configs/`.
- **`retrain_xgb` was deleted** (B4). Every family outside the candidate path returns `status: "unsupported"`, reason `retrain not yet supported for <family>`, before reading any sample. Reason: a writer whose output no live module reads is the W-08 hazard.
- **Cross-layer baseline changed at B4.** `/explain` lost two always-null keys (`ensemble_variance`, `model_agreement`). The baseline is now `baselines/crosscheck_B4.txt`.
- **No prevalence correction.** A config declaring `prevalence_train`, `prevalence_deploy` or non-null `risk_bands` fails at load with `prevalence correction is not supported (removed in B5)`. `Scale.CORRECTED` stays as the legal value of the DB column for old rows.
- **Retired rows never reach the LLM.** They render through `archived_report()`, which is rule-based and starts with the banner "ARCHIVED MODULE — replaced by NHANES dysglycaemia module". `archived_note` appears on export and history items.
- **Frontend `RETIRED_DISEASES` constant**, kept equal to the backend list by `tests/test_frontend_retired_contract.py`. It is removed after the HF release.
- **Release order.** The frontend ships first; it must work against both backends. Then the HF snapshot, with a rollback commit recorded and `docs/**/*.png` excluded.

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
5. Fix the ESLint config: dist/ is linted despite `ignores: ['dist']` (~954 of the problems).
   Lint baseline for frontend/src: 389.
6. Active Learning on NHANES: wire build_candidate for ebm_platt_conformal (retrain currently returns
   `unsupported` by design).
7. Evaluate monotonic alternatives to Mondrian per-group conformal (research, no implementation yet).
8. The frontend has no export screen. Decide later whether one is needed; not part of cleanup.
9. 4 anonymous predictions in the local DB (no patient link): pre-existing; they never appear in
   history. Decide whether that's intended.
10. Remove flwr/lightgbm from requirements.txt if not done in B7.

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
- Hollow tests caught: the cached-schema test (cache disabled in tests), and the F9-32 regression
  initially caught only by the policy pin. Both fixed by testing behaviour directly.

The cross-layer items above are recorded in `docs/phase9/FINDINGS_REGISTER.md` as W-08, W-26,
F9-40, CL-1, CL-2 and CL-3.
