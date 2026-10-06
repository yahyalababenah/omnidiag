# Findings Register — NHANES/EBM diabetes module (Phase 9, Gate 9.0)

Status: **DRAFT, uncommitted, in `logs/`.** Nothing in the product repo has been
changed. Branch `phase9/diabetes-nhanes-ebm` created at `d621637`, not checked out.

Severity scale as in Phase 8: **BLOCKER** (cannot ship), **HIGH** (ships wrong or
claims something untrue), **MEDIUM** (real defect, bounded harm), **LOW** (hygiene).

Source of every number below: `research_temporal/results.md`,
`ladder_temporal/results.md`, `ablation_log.txt`, or
`omnidiag_nhanes_research (1).py` in `~/Desktop/بيانات سكر جديدة/`, plus my own
re-run of the cohort build and both leak guards (`logs/p9_gate90_cohort_verify.txt`).

---

## BLOCKERS

### F9-01 — There is no model artifact. Nothing was ever serialized.
`find` over the whole research folder for `*.pkl *.joblib *.pickle *.onnx *.pt
*.h5 *.cbm` returns **zero files**. The only persistence calls in
`omnidiag_nhanes_research (1).py` are `plt.savefig`, `json.dump(results.json)`
and `open(results.md,"w")` (lines 402–604). The fitted EBM, the Venn-Abers
isotonic fit, the Mondrian quantiles, the imputers, the feature list and the
monotonisation all live in process memory and are discarded when the script exits.

Consequence: the task as written — "verify the artifact matches its reported
numbers" — has no artifact to verify. Phase 9 cannot start at integration. It
starts at **producing** a deployable bundle, then verifying that bundle against
`results.md`.

### F9-02 — The Gate 0–7 paper trail does not exist for this module.
No gate reports, no decision log, no findings register, no claims register, no
model card, no `reviewer_protocol.md` artefacts. What exists is: two
`results.md`/`results.json` pairs, `ablation_log.txt`, 8 figures, an
`archive/README.md` (in Arabic, good and accurate), and the scripts. The
research content is strong and honest; the **governance record the heart module
was judged on is absent**. Every limitation currently lives only in prose inside
`results.md`, which no config or doc reads.

### F9-03 — No model was ever designated for deployment.
`results.md` reports seven models and never names a winner. On the temporal test
set XGBoost is highest (AUC 0.8017 [0.7882, 0.8149]) and EBM is 0.7976
[0.7843, 0.8105]; ΔAUC EBM−XGB = [-0.0089, +0.0004], i.e. **not distinguishable**.
`ladder_temporal/results.md` marks *feature sets* L3/L3b as "DEPLOYMENT
CANDIDATE", not model families. The choice of EBM is therefore an
interpretability decision that has not yet been made or recorded anywhere.
It needs to be made explicitly and written down before it enters a config.

### F9-04 — The feature set is 87 clinical/lab variables. No clinic will enter them.
Verified list (87) includes a full CBC, the full standard biochemistry panel,
HDL, urine albumin/creatinine ratio, three seated BP readings, and seven body
measurements. Compare the deployable alternative the research itself measured:
`L4_curated_20` — "only what a clinician has at a first visit" — test AUC
**0.7575** vs L3_all_clean 0.8077. That is the real product trade-off and it has
not been decided. Shipping an 87-field form is not shippable; shipping the
20-field model costs ~0.05 AUC and needs its own EBM fit, calibration and
conformal layer (none exist).

---

## HIGH

### F9-05 — The Venn-Abers implementation is an approximation, and the code says so.
`venn_abers()` (research script, lines 312–323) fits **one** isotonic regression
on the calibration set and derives `p0 = base·n/(n+1)`, `p1 = (base·n+1)/(n+1)`
from it, with the comment: *"vectorised approximation … exact per-point refit is
O(n) each and too slow"*. Inductive Venn-Abers requires refitting isotonic with
the test point labelled 0 and labelled 1. The shipped numbers are therefore **not
covered by the Vovk & Petej distribution-free validity guarantee** that the
module docstring cites. The ECE improvements are real and measured
(EBM 0.078→0.035); the *guarantee* is not. If this ships, the claim must be
"isotonic recalibration with Venn-Abers-style interval endpoints", not
"Venn-Abers predictor with a distribution-free guarantee".

### F9-06 — Mondrian conformal coverage misses its own target for the negative class.
Target is ≥0.90 per class at α=0.10. Measured on the temporal test set, EBM:
coverage y=1 **0.938**, coverage y=0 **0.857**. Soft-label EBM: 0.931 / 0.866.
Every model in the table misses on y=0 except FasterRisk (0.907). The script is
honest about the cause (temporal split breaks exchangeability by design), but the
consequence is that the module **cannot** display "90% coverage guaranteed". The
heart precedent for this is exactly why heart shows no risk badge.

### F9-07 — The conformal decision sends ~44% of patients to "uncertain".
EBM: ambiguous {0,1} = **0.436**; confident-and-correct = 0.450. Under the heart
three-way response shape, 44 of every 100 screened patients get
`decision: uncertain` → "order the HbA1c test". That may be the clinically
correct answer, but it is a product fact that must be on screen and in the
config, not discovered by a judge clicking through demo patients.

### F9-08 — HbA1c is not a neutral label, and the model's #2 and #3 features are red-cell indices.
The research measured it (`results.md`, "Label measurement bias"): in
`HbA1c ~ glucose + MCH + RDW + age`, MCH coefficient **−0.0448 (p≈4e-235)** and
RDW **−0.0052 (p=0.036)**, both independent of glucose. Microcytic patients
(MCH<27, n=1976) have mean HbA1c 5.72 vs 5.51 at **essentially identical mean
glucose** (95.3 vs 95.6). And the EBM's term importances rank
`LBXRDW` 0.1416 (#2), `LBXMCHSI` 0.1303 (#3), `LBXMC` 0.1249 (#4) — directly
behind age. So a substantial share of the model's explanation is iron status
predicting an HbA1c assay artefact, not dysglycaemia. This must be stated in the
config and in the report narrative, and it makes the SHAP/contribution panel
clinically misleading if left unannotated.

### F9-09 — Race is excluded as an input but is recoverable from creatine kinase.
Measured in the research: CK median 170 in NH Black vs 102 in others;
**AUC(CK → NH Black) = 0.706**. `LBXSCK` is the EBM's #5 term (0.1089). And
`HbA1c ~ glucose + NH Black + age` gives an NH Black coefficient of
**+0.146 (p=1.6e-68)**. The module therefore re-introduces the variable it
excluded, through a proxy, into a label that itself carries a race-associated
offset. Excluding `RIDRETH3` does not make the model race-blind and the docs must
not say it does.

### F9-10 — Sensitivity collapses in the young; AUC collapses in the old. (HF-13 pattern.)
Fairness audit, EBM: sensitivity **0.236** at age 20–39 vs 0.628 at 40–59 and
0.777 at 60+. AUC runs the other way: 0.807 / 0.757 / **0.693**. By sex:
AUC 0.825 female vs **0.763** male, sensitivity 0.671 vs 0.607. A screening tool
that finds 24% of dysglycaemic under-40s is close to useless for the group where
early detection matters most. Measured, real, and currently unstated anywhere.

### F9-11 — Subgroup calibration is badly split while pooled calibration looks fine. (HF-15 pattern.)
Pooled EBM calibration-in-the-large after recalibration is −0.181. Per race:
NH White **+0.087**, Other Hispanic +0.048, Mexican American +0.043,
NH Asian **−0.044**, NH Black −0.022. That is a ~13-point spread in mean
predicted-minus-observed across groups hidden behind an acceptable pooled number.
Per age: +0.013 / +0.038 / +0.044.

### F9-12 — Jordan's 0.237 prevalence figure does not transfer, and there is no matched substitute in the folder.
`configs/diabetes.yaml` currently sets `prevalence_deploy: 0.237` (Abu-Raddad
2020, via `archive/post_expo_2026-10/docs/OmniDiag_Proposal_Defense.md`). That is **total diabetes
prevalence including diagnosed cases**. This model's target is HbA1c ≥ 5.7
among the **undiagnosed and untreated**, measured at **32.9%** in NHANES. The
figures measure different events in different populations; carrying 0.237 across
would be a silent error of the same class as the inverted-label bug. Nothing in
the research folder proposes a Jordanian figure for this target. See my
recommendation in the Gate 9.0 reply.

### F9-13 — Environment drift between research and product, and `interpret-core` is not a dependency.
Research venv (Python 3.13): `interpret_core 0.7.8`, `numpy 2.5.3`,
`pandas 3.0.6`, `scikit_learn 1.9.1`, `xgboost 3.4.1`, `fasterrisk 0.1.10`.
Product `requirements.txt` pins `numpy==2.4.6`, `pandas==3.0.3`,
`scikit-learn==1.9.0`, `xgboost==3.3.0`, and **has no `interpret` at all**. The
header of that file records why the pins are exact: unpinned, the HF Space
resolved sklearn 1.7.2 and served different diabetes probabilities from the same
pickles (Case C 63.7% local vs 49.4% live). Adding `interpret-core` to the HF
image is a deploy risk that must be handled exactly as that incident taught, and
the bundle must be fitted and verified under **one** pinned set, not two.

---

## MEDIUM

### F9-14 — ~14 analytes appear twice, as conventional AND SI columns, in the shipped feature set.
Verified pairs in FEATS: `LBXSCH`/`LBDSCHSI`, `LBXSTR`/`LBDSTRSI`,
`LBXSUA`/`LBDSUASI`, `LBXSAL`/`LBDSALSI`, `LBXSBU`/`LBDSBUSI`,
`LBXSCA`/`LBDSCASI`, `LBXSCR`/`LBDSCRSI`, `LBXSGB`/`LBDSGBSI`,
`LBXSIR`/`LBDSIRSI`, `LBXSPH`/`LBDSPHSI`, `LBXSTB`/`LBDSTBSI`,
`LBXSTP`/`LBDSTPSI`, `LBXMC`/`LBXMCHSI`, plus `SBP` alongside
`BPXSY1/2/3` and `DBP` alongside `BPXDI1/2/3`.

The leak-guard note says SI duplicates were excluded — but only for the
*glycemic* variables. For every other analyte both columns survived. This is not
leakage (guard 2 R²=0.150, passes). It is an **explanation defect**: the EBM's
per-feature contribution is the product's explanation, and a deterministic
rescaling of the same measurement splits its attribution arbitrarily between two
rows of the SHAP panel. It also doubles the fields a form would ask for. Note
`LBXMC`/`LBXMCHSI` are the #3 and #4 terms, so this lands on the most important
part of the explanation.

### F9-15 — Blank-field behaviour is unmeasured, and EBM handles NaN natively — so blanks are not neutral.
`FEATS` admits any column with `notna().mean() > 0.5`, i.e. up to 49% missing in
training, and EBM learns a bin for missing. A clinician leaving a lab field empty
therefore selects a *learned* value, not "no information". This is precisely the
HF-11/HF-12 pattern. On the heart side this was measured and surfaced
(`blank_warning_min_decision_share` in the heart config). For diabetes it has
never been measured. With 87 optional fields the surface is much larger than
heart's.

### F9-16 — The feature list is computed at runtime, not pinned. The model is not reproducible without pinning it.
`FEATS` is derived from whichever cycle directories are passed on the command
line plus a >50%-coverage filter (research script, lines ~168–170). Pass five
cycles instead of six and the feature list changes, silently. `SAD_ht` and
`BMDAVSAD` are in the monotone `INC` list but never made the cut, and `LBDHDDSI`
is in `DEC` and never made it either — hence "+13 increasing, −3 decreasing"
against 15+4 names. Harmless today, but it means the training script alone does
not define the model: the feature list must be written to the bundle.

### F9-17 — `nhanes_diabetes_analytic.csv` in the folder root is a single-cycle EDA leftover, not the training data.
5141 rows, 57 columns. The research cohort is 26904 rows / 87 features, built
from `nhanes_raw/` at run time. Byte-identical copies sit under
`archive/01_eda/nhanes_eda_out*/`. Anyone who assumes the root CSV is "the data"
trains a different model. Given the Phase 8 precedent (`c9507de`, tracking the
BRFSS CSV so it reached the HF build context), the data-in-repo question needs an
explicit answer here: `nhanes_raw/` is **79 MB** and cannot go into the HF image
the way the BRFSS CSV did.

### F9-18 — `thresholds.js` documents a deployment prevalence of ~14%; the config says 0.237.
`frontend/src/constants/thresholds.js` header: *"stated on the deployment
prevalence (~14%)"*. `configs/diabetes.yaml` sets `prevalence_deploy: 0.237` and
its own comment calls ~14% "the earlier placeholder". Pre-existing cross-layer
drift in the current BRFSS module, and it lands in the file Phase 9 must touch.

### F9-19 — FasterRisk needs a runtime monkeypatch of a third-party library.
Research script lines ~283–290 recompile `fasterrisk.rounding` and
`fasterrisk.fasterrisk` in memory to replace `ndarray.tostring()` (removed in
NumPy 2) with `.tobytes()`. If the bedside integer score ships, that patch ships
into the product image. Recommendation: do not ship M5; if the score card is
wanted, hard-code the eight published points from `results.md` as data.

---

## LOW

### F9-20 — Old BRFSS risk bands 0.70/0.40 are defined on a 50/50-balanced raw scale.
`configs/diabetes.yaml` states them on the raw scale on purpose, and
`predict()` maps them. On a calibrated NHANES output in a 33%-prevalence cohort
they mean nothing. They must be retired (heart precedent, D-32) or recomputed —
not carried.

### F9-21 — `configs/diabetes.yaml.kaggle` and `backend/schemas.py.bak`, `backend/model_loader.py.bak` are live in the tree.
Housekeeping, but a `.kaggle` config variant next to a module being replaced is
an invitation to load the wrong one.

### F9-22 — 2017-2018 has no `apolipoproteinB.csv`; 2007–2014 have no `insulin.csv`.
Not a defect (neither analyte is in FEATS, and insulin is in the glycemic
exclusion set) — but `load()` silently skips absent files rather than recording
the gap, so a future cycle addition can shift the feature set without a warning.

---

## What I have NOT verified, and why

- **The reproducibility check has not been run.** It cannot be: F9-01 means
  there is no saved artifact to compare against. What *can* be checked — and
  what I propose for Gate 9.0b — is a one-time fit under the pinned product
  environment, saved to a bundle, then diffed against `results.md` headline
  numbers (AUC, CI, calibration intercept/slope, ECE, conformal coverage). A
  full re-run of `omnidiag_nhanes_research (1).py` also executes the ablation
  (7 configs × repeated 5-fold ×3 CV on 22805 rows) and would take hours on this
  laptop. I have not started it. Say the word and I will run a reduced script
  that fits only the deployment model + calibration + conformal.
- **Both leak guards HAVE been re-run**, on the feature set exactly as the
  research builds it, and both pass — see below.

## What I DID verify (re-run from raw NHANES, research venv)

| quantity | `results.md` claims | my re-run | match |
|---|---|---|---|
| screening cohort n | 26904 | 26904 | ✓ |
| prevalence | 0.329 | 0.3288 | ✓ |
| n features | 87 | 87 | ✓ |
| split train/val/cal/test | 18508/2148/2149/4099 | 18508/2148/2149/4099 | ✓ |
| leak guard 1 (univariate AUC < 0.85) | passes | passes (no assertion fired) | ✓ |
| leak guard 2 (glucose R² < 0.60) | 0.150 | 0.150 | ✓ |
| per-cycle positive rate | 31.7/33.9/33.0/29.6/33.6/35.9 % | identical | ✓ |

The data layer and the leakage story are sound and reproduce exactly. The gap is
entirely on the artifact and governance side.

---

# Decision D9-01 (Yahya, 2026-09-28): EBM, curated-20 feature set

Closes **F9-03** (model family chosen: EBM) and **F9-04** (feature set chosen:
`CURATED20` from `nhanes_leakage_ladder (2).py` line 130).

Verified consequences, measured on the same cohort and temporal split
(`logs/p9_curated20_characterisation.txt`):

- **F9-14 (SI-unit duplicates) — CLOSED.** No conventional/SI pair survives in
  the 20. No `SBP`+`BPXSY1/2/3` overlap either (raw readings are gone, only the
  mean remains).
- **F9-08 (red-cell indices driving the explanation) — DOWNGRADED to LOW.** No
  CBC column is in the 20, so `LBXRDW`/`LBXMCHSI`/`LBXMC` can no longer appear
  in the contribution panel at all. The *label* bias is unchanged — HbA1c still
  responds to MCH at −0.0448 — but it is now unmodelled noise rather than a
  misleading explanation. Stays in the config as a stated limitation.
- **F9-09 (race recoverable) — NOT closed. It got worse, not better.**
  Dropping `LBXSCK` (which alone reached AUC 0.706 for NH Black) does not make
  the feature set race-blind. A logistic model on the 20 features predicts
  NH Black at **AUC 0.781** and NH Asian at **0.790** — both higher than CK
  alone. Creatinine, albumin, waist, BMI and age jointly carry ancestry signal.
  The claim "race is not an input" stays true and the claim "the model is
  race-blind" stays false.
- **Leak guards re-run on the 20 alone:** guard 1 max univariate AUC **0.717**
  (`RIDAGEYR`) — PASS. Guard 2 R²(glucose | 20) = **0.115** — PASS. Also
  R²(HbA1c | 20) = 0.161.
- **F9-15 (blank fields) — still open, but much smaller.** Every field is
  ≥95.5% present; **89.5%** of the cohort has all 20. Mean 0.32 missing of 20.
  New wrinkle: because missingness is so rare, an EBM missing-bin is learned
  from 1–4.5% of rows, so the value a blank field resolves to is estimated from
  thin data and may be unstable. Must still be measured per field (Gate 9.4).
- **Monotone constraints that survive:** 8 increasing (`RIDAGEYR`, `BMXBMI`,
  `BMXWAIST`, `SBP`, `MCQ300C`, `LBXSTR`, `LBXSATSI`, `LBXSGTSI`), 3 decreasing
  (`PAQ650`, `PAQ665`, `LBDHDD`). `BPQ020`/`BPQ080` (known hypertension /
  hyperlipidaemia) are not in the 20 — measured SBP/DBP replace them.

## New finding from the decision

### F9-23 — HIGH — No measured performance number exists for this model.
`0.7575` is **XGBoost** on the curated 20, from the *ladder* script, on a
**two-way** train/test split with no calibration set, no Venn-Abers, no
conformal layer, and different hyperparameters. `0.7976` is **EBM on 87
features**. Neither describes what D9-01 selects. Every headline number — AUC,
CI, calibration intercept/slope, ECE, conformal coverage, ambiguous rate,
fairness by sex/age/race, decision curve — has to be measured fresh in Gate
9.0b before it may enter a config, a document or the UI. Until then the module
has **no** performance claim.

### F9-24 — MEDIUM — `WHtR` is absent from the 20 but was a top-2 feature.
`WHtR = BMXWAIST / BMXHT` ranked #2 by gain in L3b and #7 in the 87-feature
EBM's term importances. The curated 20 carries `BMXWAIST` and `BMXBMI` but not
`BMXHT`, so WHtR cannot be derived. Adding height is one extra collected field
in any clinic that already measures waist. Proposed for a measured side-by-side
in Gate 9.0b rather than an assumption either way.

---

# Decision D9-02 (proposed, awaiting Yahya): central adiposity as an eyeballed 3-level band

Yahya: the clinician will not tape-measure waist and height at a first visit; they judge
central adiposity relatively, by eye. Measured answer in `logs/p9_whtr_banding.txt`.

**Revises F9-24.** The answer is not "add height as a number". It is "replace the measured
waist with a band, and then neither tape measurement is needed".

Proposal: `BMXWAIST` (cm, continuous) is replaced in the input schema by
`central_adiposity` — an ordinal 3-level field, clinician-assessed:

| level | trained against | cohort share | observed prevalence |
|---|---|---|---|
| 0 normal | WHtR < 0.50 | 17.7% | 0.143 |
| 1 increased | 0.50 ≤ WHtR < 0.60 | 38.7% | 0.279 |
| 2 high | WHtR ≥ 0.60 | 39.0% | 0.456 |

Field count stays at 20. Two tape measurements become one judgement.

Supporting measurements:
- Cut-points are the **published** WHtR boundaries (0.5 / 0.6), not selected on our data. The
  train-selected FasterRisk cut-points (0.58 / 0.64) scored no better (0.6420 vs 0.6442), so
  there is no measured cost to preferring the externally defensible ones.
- **3 bands, not 4.** The <0.40 band holds 127 people (0.5%) and 4 bands buy nothing
  (0.6444 vs 0.6442).
- Prevalence is monotone across bands, so the EBM increasing-monotone constraint applies
  directly, and the contribution panel becomes three readable steps instead of a wiggly curve.
- Banding costs 0.0244 univariate AUC vs continuous WHtR; eyeball error at 20% costs 0.017 more.
  But `BMXBMI` stays in the set and correlates **0.911** with WHtR, so the model-level cost will
  be far smaller than the univariate figures suggest. It must be measured, not assumed.
- Train/deploy mismatch to state openly: the band is derived from precise measurements in
  training and supplied by eye at inference. Pre-declared robustness test in Gate 9.4.

### F9-25 — HIGH — A blank adiposity field is a risk signal in the training data.
Patients with height or waist absent (n=1240, 4.6%) have observed prevalence **0.389** — above
the "increased" band (0.279) and near the "high" band (0.456). Un-measured adiposity is not
missing-at-random; it tracks high risk, plausibly because measuring is harder in the patients who
most need it. An EBM will learn "blank → elevated risk". A clinician who skips the field out of
haste therefore receives a risk increase they did not intend, and did not state. This is the
HF-11/HF-12 pattern, measured rather than suspected.
**Consequence: the band must be a required input. No silent blank, and no "unknown" level that
quietly inherits the training-set missing bin.**

### F9-26 — LOW — `BMXBMI` also requires height and weight, and nobody has raised it.
If the tape measure is unavailable, the scale and stadiometer may be too. BMI is 98.9% present in
NHANES and is normally known in a clinic, so I am not proposing to change it — but the same
question applies and should be answered deliberately rather than by omission.

## Settled by Yahya in the same message
- Soft labels: hard-vs-soft selected on the **validation** set only. Approved.
- No post-hoc pruning of the low-univariate-signal fields (`RIAGENDR` 0.507, `BPXPLS` 0.505,
  `DBP` 0.537). Approved.

---

# Gate 9.0b — the bundle exists and has been measured

Artifact: `~/Desktop/بيانات سكر جديدة/ebm_bundle/diabetes_nhanes_ebm.joblib` (1.5 MB)
sha256 `04a89084e897c7bb112ccb8d6d80d7bbe8e18c1bc9e4bd730f22f9a5c7168668`
Script: `train_diabetes_nhanes_ebm.py` (484 lines, in the experiments folder).
Full output: `logs/p9_gate90b_results.md`, `logs/p9_gate90b_model_card.json`.

Fitted in the **pinned product stack** — numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0,
scipy 1.18.0, joblib 1.5.3, interpret-core 0.7.8, Python 3.13.5 — asserted at startup, so
**F9-13 is CLOSED**. `interpret-core==0.7.8` installed into `backend/.venv` with a dry run first
confirming it changes nothing else; every pinned version is byte-identical afterwards.

**F9-01 CLOSED** (an artifact exists, with calibrator, conformal quantiles, pinned feature list,
units, ranges and model card inside it). **F9-23 CLOSED** (the module now has measured numbers).

## Headline — held-out test set, 2017-2018, n=4099, temporal split

| quantity | value |
|---|---|
| **AUC (raw EBM score)** | **0.7523 [0.7378, 0.7662]** |
| AUC after recalibration | 0.7496 |
| AUC, MEC-weighted | 0.7655 |
| calibration intercept raw -> recal | +0.182 -> -0.096 |
| calibration slope raw -> recal | 0.965 -> 0.845 |
| ECE raw -> recal | 0.035 -> 0.033 |
| conformal referral / no_referral / uncertain | 0.237 / 0.301 / **0.462** |
| coverage y=1 / y=0 | 0.917 / **0.856** |
| sensitivity of 'flagged' | 0.917 |
| specificity of 'no_referral' | 0.423 |
| EBM additivity max deviation | **2.67e-15** (target < 1e-8) |

Pre-declared checks: AUC inside the expected 0.75-0.78 — yes (at the bottom edge). Rule 7
falsifier (AUC < 0.73) — not triggered. Stop rule (uncertain > 0.55) — not triggered at 0.462.
Selected on validation: soft labels sigma=0.12 (val 0.7796) over hard (0.7787). That 0.0009 gap is
noise and sigma=0.06 tied exactly; the selection is pre-declared and legitimate but **must not be
described as a gain**.

**F9-06 CONFIRMED on the shipped model**: coverage y=0 is 0.856, below the 0.90 target. No
coverage guarantee may be claimed.

## D9-02 is vindicated — F9-24 CLOSED

| | AUC |
|---|---|
| V1, eyeballed 3-level band | **0.7523** |
| V0, measured waist in cm | 0.7507 |

Paired bootstrap delta (V1 - V0) 95% CI **[-0.0008, +0.0040]** — no evidence of a difference. The
band is numerically higher. Replacing the tape measure cost nothing measurable.

Eyeball error is close to irrelevant:

| p(band misplaced by +/-1) | AUC | delta | decisions changed |
|---|---|---|---|
| 0.00 | 0.7523 | — | — |
| 0.10 | 0.7520 | -0.0003 | 0.006 |
| 0.20 | 0.7517 | -0.0005 | 0.013 |
| 0.35 | 0.7516 | -0.0007 | 0.023 |

Even at 35% misclassification the model loses 0.0007 AUC and 2.3% of decisions move, because
`BMXBMI` stays in the set and correlates 0.911 with WHtR. The univariate figures in
`logs/p9_whtr_banding.txt` overstated the cost by roughly fifty-fold, exactly as predicted there.

## NEW FINDINGS

### F9-27 — HIGH — The recalibration layer makes this model worse. Decision needed.
The raw EBM is already well calibrated: intercept **+0.182**, slope **0.965**, ECE **0.035**.
After isotonic recalibration: intercept -0.096, slope **0.845** (further from 1), ECE 0.033
(barely moved), and AUC drops 0.7523 -> 0.7496. On the 87-feature model recalibration earned its
place (ECE 0.078 -> 0.030). Here it does not.
Options: (a) ship the raw score with no calibration layer and state that the EBM is natively
calibrated on this cohort; (b) keep the layer for the interval endpoints only and report the raw
probability; (c) keep as-is. This changes what `probability` means in the API response, so it goes
to Yahya. Note F9-05 already forbids calling any of this a distribution-free guarantee.

### F9-28 — BLOCKER for the form design — Blank fields move risk violently. 6 of 20 flip >10% of decisions.
Measured on the shipped bundle, one field blanked at a time across all 4099 test patients:

| field | median delta p | decisions changed | direction |
|---|---|---|---|
| **PAQ650** (vigorous activity) | **+0.1079** | **0.311** | RAISES |
| **RIDAGEYR** (age) | -0.0544 | **0.385** | lowers |
| **PAQ665** (moderate activity) | -0.0614 | 0.176 | lowers |
| **LBDHDD** (HDL) | -0.0411 | 0.147 | lowers |
| **BMXBMI** | -0.0169 | 0.133 | lowers |
| **ADIPOSITY_BAND** | -0.0169 | 0.095 | lowers |
| DBP | +0.0223 | 0.075 | RAISES |
| LBXSAL | +0.0113 | 0.084 | RAISES |
| LBXSGTSI | +0.0046 | 0.077 | RAISES |
| BPXPLS | +0.0170 | 0.057 | RAISES |
| (remaining 10) | < 0.013 | < 0.07 | mixed |

Leaving the vigorous-activity checkbox empty adds **10.8 percentage points** of risk and changes
**31%** of decisions. Blanking age changes 38.5% of decisions. Twelve of twenty fields shift median
risk by more than 0.002. This is the HF-11/HF-12 pattern at a far larger scale than heart's, and it
cannot be handled by a warning banner. The form has to decide, per field, whether a blank is
allowed at all and what it means. This blocks Gate 9.2's schema work.

### F9-25 — CORRECTED. I predicted the wrong direction.
I wrote that a blank adiposity field would read as *elevated* risk, reasoning from the missing
group's observed prevalence of 0.389. Measured on the fitted model, blanking the band **lowers**
median risk by 0.0169 and flips 9.5% of decisions. The prediction was wrong; the conclusion is
unchanged and in fact stronger, because false reassurance is worse than false alarm. The field
stays required.

### F9-11 — CONFIRMED and WORSE on the shipped model.
Calibration bias by ethnic group: NH White **+0.106**, Other Hispanic +0.040, Mexican American
+0.047, NH Black **-0.076**, NH Asian **-0.097**. A **20-point spread**, against the 87-feature
model's 13. Pooled intercept is -0.096 and looks fine.

### F9-10 — CONFIRMED, and discrimination in the old is worse than before.
AUC by age band: 0.757 (20-39) / 0.691 (40-59) / **0.619** (60+). By sex: female 0.785, male
**0.713**. Sensitivity of 'flagged' is high everywhere (0.88-1.00) except age 20-39 at **0.578**
and adiposity band 0 at 0.661 — better than the 87-feature model's 0.236 for the young, because
'flagged' counts `uncertain` as flagged.

### F9-09 — CONFIRMED on the exact shipped 20.
AUC(20 shipped features -> group): NH Asian **0.785**, NH Black **0.780**, Mexican American 0.732,
Other Hispanic 0.630, Other/multi 0.629, NH White 0.615.

## Bundle verified by round-trip
Reloaded from disk in the pinned env; 20-field synthetic patient (58y male, high adiposity,
sedentary, HDL 41) scores raw 0.5959 -> recalibrated 0.6855, interval [0.6854, 0.6859],
decision `referral`, additivity deviation 1.67e-16.

## Still open before Gate 9.1
- F9-02 governance record, F9-05 calibration wording, F9-06 coverage wording, F9-07/F9-27/F9-28
  decisions, F9-12 prevalence, F9-17 data-in-repo, F9-08/F9-09/F9-10/F9-11 as stated limitations.

---

# Gate 9.0c — Yahya's nine proposals, measured on validation only

Full output: `logs/p9_gate90c_proposals.md`. Script: `gate90c_yahya_proposals.py`.
Pre-declared before measuring: the 2017-2018 test set is not touched; a variant must beat the
baseline by **more than 0.005 validation AUC** to count, because the test CI half-width is ~0.014.
Baseline (shipped config) validation AUC **0.7796**.

Result: **one of the nine is adopted, and it is a genuine improvement.** The rest are noise,
already shipped, or rest on a premise the measurement contradicts.

## ADOPT — A2, Platt scaling instead of isotonic (Yahya's diagnosis was right)

| layer | val AUC | intercept (ideal 0) | slope (ideal 1) | ECE |
|---|---|---|---|---|
| none (raw EBM) | 0.7796 | +0.308 | 1.079 | 0.054 |
| isotonic + VA endpoints (what 9.0b shipped) | 0.7783 | +0.029 | 0.960 | **0.031** |
| **Platt (logistic)** | **0.7796** | +0.038 | 1.064 | **0.023** |

Platt fit: a = **1.0139**, b = **0.2811**. The slope coefficient is within 1.4% of 1, so the raw
EBM score needs essentially no *shape* correction — only a level shift. That is exactly why a
two-parameter sigmoid is the right tool here and a non-parametric isotonic fit was overkill, which
is the mechanism Yahya proposed. Measured, it holds:
- ECE 0.023 vs isotonic 0.031 vs raw 0.054 — best of the three.
- AUC is preserved **exactly** (0.7796), because a monotone two-parameter map cannot reorder
  patients. Isotonic can tie distinct scores and it cost 0.0013 AUC.
- Intercept +0.038, against the raw +0.308.

**Consequence that needs a decision (extends F9-27):** the `p_lower`/`p_upper` interval in 9.0b came
from the Venn-Abers-style endpoints of the *isotonic* fit. Platt has no such construction. Proposed
replacement: bootstrap the Platt fit over the calibration set (B=1000) and report the resulting
percentile interval on the calibrated probability. That is an honest statement of uncertainty in the
calibration map, and unlike the current endpoints it is not an approximation of a guarantee we never
had (F9-05). The Mondrian conformal set continues to supply the decision, untouched.

## ALREADY DONE — A1 / B5, soft labels

| variant | validation AUC |
|---|---|
| hard labels | 0.7787 |
| sigma=0.06 | 0.7796 |
| **sigma=0.12 (shipped)** | **0.7796** |
| sigma=0.20 | 0.7782 |
| sigma=0.30 | 0.7776 |

The shipped model has used soft labels since 9.0b. The proposed sigma=0.30 was the **worst** of the
four on this feature set. Hard-to-best spread is 0.0009, below the bar — it is not a gain and it is
not the calibration fix. Also a mechanism correction: this is not uniform label smoothing to
95%/5%. The target is `q = Phi((HbA1c - 5.7) / sigma)`, per patient: HbA1c 7.0 gives q ~ 1.00,
HbA1c 5.71 gives q ~ 0.53. It models measurement error near the cut-point, which is why a large
sigma hurts — it blurs patients who are not actually ambiguous.

## REJECTED ON MEASUREMENT — B2, TG/HDL ratio

| variant | fields | val AUC | delta |
|---|---|---|---|
| baseline | 20 | 0.7796 | — |
| + TG_HDL (derived server-side, no new clinician input) | 20 | 0.7794 | **-0.0003** |

The reasoning was sound and worth testing: the EBM selected 10 pairwise interactions on its own and
**none** is TG x HDL, so an additive model truly cannot express the ratio unless handed it. Measured:
no gain. This is now the second independent measurement on this cohort agreeing — the 87-feature
ablation found VAI and LAP at -0.0002, CI [-0.0006, +0.0003], "no evidence".

## SETTLED PERMANENTLY — B1, height for continuous WHtR

| variant | fields | val AUC | delta |
|---|---|---|---|
| band (shipped) | 20 | 0.7796 | — |
| band -> continuous WHtR (needs measured waist **and** height) | 21 | 0.7794 | -0.0002 |
| + TG_HDL + continuous WHtR | 21 | 0.7789 | -0.0007 |

Note the premise: WHtR = waist / height, so this needs the tape measure **back** for waist as well as
height. It is not "one extra routine field" — it reverses D9-02. And it buys nothing: -0.0002.
**F9-24 stays closed. The eyeballed band is not a compromise; it is the better product at equal
accuracy.**

## REJECTED ON MEASUREMENT — A3 / B3, hyperparameters

| change | val AUC | delta | ECE |
|---|---|---|---|
| baseline (max_bins=1024, interactions=10, min_samples_leaf=4, outer_bags=8) | 0.7796 | — | 0.054 |
| interactions 10 -> 20 | 0.7798 | +0.0001 | 0.053 |
| min_samples_leaf 4 -> 20 | 0.7797 | +0.0000 | 0.054 |
| min_samples_leaf 4 -> 50 | 0.7800 | +0.0003 | 0.054 |
| outer_bags 8 -> 16 | 0.7798 | +0.0002 | 0.054 |

`max_bins` is **already 1024** — there is nothing to raise. The specific hypothesis that
`min_samples_leaf` would improve calibration was tested at 20 and 50: ECE stayed 0.054 both times.

## REJECTED — B4, upweighting patients 60+. The premise is arithmetically wrong.

Pooled AUC is not an average of subgroup AUCs. Demonstrated directly:

| set | AUC of AGE ALONE | AUC of the model |
|---|---|---|
| all validation (n=2148) | 0.716 | 0.780 |
| age 20-39 (n=826) | 0.643 | 0.764 |
| age 40-59 (n=743) | 0.604 | 0.727 |
| age 60+ (n=579) | **0.517** | 0.648 |

Age is the model's dominant term (importance 0.647, 3.7x the next). Across the whole cohort age
alone reaches 0.716; **inside the 60+ band it reaches 0.517**, because within a band everyone is
the same age. The low 60+ AUC is mostly that arithmetic, not neglect of older patients, and no
re-weighting can change it.

And the weighting fails on its own terms — it made the 60+ group **worse**:

| weight on 60+ | val AUC all | val AUC 60+ | val AUC 20-39 |
|---|---|---|---|
| 1.0 (baseline) | 0.7796 | **0.6479** | 0.7639 |
| 2.0 | 0.7782 | 0.6423 | 0.7639 |
| 4.0 | 0.7764 | 0.6432 | 0.7604 |

## A4 — the subgroup calibration gap is real, and it has a hard constraint

Agreed that the +0.106 / -0.097 spread (F9-11) is the most serious measured defect. But it cannot be
"tuned" away: correcting calibration per ethnic group requires the group variable **at inference
time**, and race is deliberately not an input. The three honest options are (a) accept and state it,
displaying no absolute probability — the heart precedent D-32; (b) make race an input, which the
research excluded on purpose; (c) replace class-conditional Mondrian conformal with
**group-conditional** conformal over a variable we legitimately have (age band, sex, adiposity
band), which restores per-group validity for the *decision* without needing race. Option (c) is a
genuine extension of what already exists and is the one worth measuring next.

## The strategic answer on AUC

Nothing in this list raises AUC, and the reason is structural rather than about tuning. Measured
in-house on the same cohort and split: old BRFSS feature set **0.733**, curated-20 **~0.752-0.758**,
100 biology features **0.802**, 105 all-clean **0.808**, 87-feature EBM **0.798**. The research's own
paired bootstrap puts the gap from 20 features to 100 at CI **[0.0345, 0.0539] — REAL**. The lever is
**measurements, not hyperparameters**. Raising AUC means adding a data source, and the cheapest one
is a CBC (a single extra tube) — which would re-open F9-08, because RDW and MCH were the 87-feature
model's #2 and #3 terms and they predict an HbA1c assay artefact rather than glycaemia. That is a
trade Yahya should make deliberately, not by tuning.

Also for the record: the ladder's XGBoost on the same curated 20 reached 0.7575 against our EBM's
0.7523 — about 0.005, within noise, and the measured price of interpretability. Consistent with the
87-feature result where EBM trailed XGBoost by an indistinguishable margin.

---

# Gate 9.0d — "can the accuracy go up?" Measured. Validation only.

Full output `logs/p9_gate90d_ceiling.md`. Script `gate90d_can_we_raise_auc.py`.
Baseline (shipped config) validation AUC **0.7796**. Bar to count: > 0.005.

## F9-29 — CRITICAL — The biggest available AUC gain is an assay artefact, not risk information.

Adding a CBC (7 columns, one extra blood tube, 99.8% coverage) gives **+0.0218** validation AUC —
four times the pre-declared bar and by far the largest gain found anywhere in Phase 9.

Then the control experiment. Refit everything against a **glucose-defined** label instead of HbA1c:

| label | 20 fields | 20 + CBC | gain from CBC |
|---|---|---|---|
| HbA1c >= 5.7 (what we ship) | 0.7796 | 0.8014 | **+0.0218** |
| glucose >= 100 (control) | 0.7268 | 0.7229 | **-0.0040** |

Surviving fraction: **-0.18**. The CBC gain does not merely shrink against a glycaemia label — it
**disappears and goes slightly negative**. The CBC is not telling the model about dysglycaemia. It
is telling the model about red-cell indices, which shift the HbA1c *reading* without shifting
glycaemia — exactly the mechanism F9-08 measured (HbA1c ~ glucose + MCH + RDW + age: MCH
coefficient -0.0448, p~4e-235, independent of glucose).

**Consequences, and they are large:**
1. Adding a CBC would raise the headline AUC by 2.2 points and add **zero** real risk information.
   It must not be done. If it ever is, the gain may not be reported as accuracy.
2. The "ceiling" figures we have been quoting are **partly inflated by this same artefact**. The
   87-feature EBM's 0.7976 and the 105-feature XGBoost's 0.8077 both include the full CBC, and
   `LBXRDW` / `LBXMCHSI` / `LBXMC` were terms #2, #3 and #4 of that 87-feature model. So a real
   share of the 0.05 gap between our 0.752 and their ~0.80 is artefact, not signal.
3. **The shipped curated-20 model is more honest than the larger models, not merely smaller.**
   D9-01's feature set excludes every CBC column, so this artefact cannot enter it at all.

Limitation of the control: `LBXSGL` is serum glucose from the standard biochemistry panel and is
not guaranteed fasting, so "glucose >= 100" is a noisier label than true impaired fasting glucose —
visible in the lower baseline (0.7268 vs 0.7796). The *direction* is still decisive: real glycaemia
signal would survive a noisy label to some degree, and this does not survive at all. Confirmatory
test worth running: repeat on the fasting subsample using `LBXGLU` from `fasting_glucose.csv`.

## TIER 1 — the three free questions: +0.0028, below the bar

| feature set | fields | val AUC | delta |
|---|---|---|---|
| baseline curated-20 | 20 | 0.7796 | — |
| + BPQ020 (ever told high BP) | 21 | 0.7801 | +0.0005 |
| + BPQ020 + BPQ080 (ever told high cholesterol) | 22 | **0.7825** | **+0.0028** |
| + all three (incl. SMQ020 smoking) | 23 | 0.7813 | +0.0016 |

No lab, no instrument, 99.9% / 88.1% / 99.9% coverage. In the research's 100-feature model BPQ020
ranked #3 by gain (31.5) behind only age and WHtR, so the small effect here is itself informative:
most of BPQ020's apparent value in that model was already captured by our measured SBP and DBP.
Adding smoking on top **reduced** AUC (0.7825 -> 0.7813).

Verdict: +0.0028 is below the 0.005 bar and cannot be called an improvement. It is, however, the
largest *non-artefact* gain found in Phase 9, and it is free. Note BPQ080 is only 88.1% covered,
which interacts with the F9-28 blank-field policy. Recommendation: offer BPQ020 and BPQ080 as
optional inputs only if F9-28 is solved first; do not add SMQ020.

## TIER 3 — F9-30 — About 0.07 AUC is locked in label noise and no model can reach it.

HbA1c >= 5.7 is a hard cut on a noisy assay. Measured on validation:

| | |
|---|---|
| patients within +/-0.10 HbA1c units of the 5.7 cut | 14.9% (319 of 2148) |
| patients within +/-0.20 units | 24.5% (527) |
| model AUC on everyone | 0.7796 |
| model AUC excluding the +/-0.10 band | **0.8304** |
| model AUC excluding the +/-0.20 band | **0.8476** |

Published within-person HbA1c variability is about 0.1-0.2 percentage points, so a patient measured
at 5.69 versus 5.71 is on the wrong side of the cut about as often as the right one. Roughly a
quarter of the cohort sits in that band, and the model scores 0.85 on the patients outside it.
(The +/-0.15 and +/-0.20 rows are identical because HbA1c is reported to 0.1 precision.)

This also reframes the 46% `uncertain` rate (F9-07): a large part of it is the model correctly
declining to call patients whose *label* is itself a coin flip. That is the conformal layer working,
not failing.

## Answer to "is it impossible to raise the accuracy?"

No — but every honest lever is small, and the one big lever is fake.

| lever | measured effect | verdict |
|---|---|---|
| soft labels, hyperparameters, TG/HDL, WHtR, age weighting (Gate 9.0c) | -0.003 to +0.0003 | noise |
| three free questions | +0.0028 | below bar, free, conditional on F9-28 |
| **CBC (one blood tube)** | **+0.0218** | **artefact — must not be used** |
| label noise at the 5.7 cut | ~ -0.07 | unattainable by any model |

The genuine remaining levers are not about AUC:
- **F9-28**, the blank-field policy. Leaving one checkbox empty moves risk by 0.108 and flips 31%
  of decisions. That destroys more real-world accuracy than the 0.05 AUC gap ever could.
- **F9-27 / Platt**, which costs nothing and fixes calibration (ECE 0.054 -> 0.023).
- **Group-conditional conformal** (A4 option c), which fixes decision validity per subgroup without
  needing race as an input.
- Changing the **target** away from HbA1c, e.g. to fasting glucose or 2h OGTT, would remove the
  artefact at its source. But HbA1c >= 5.7 is the ADA criterion a clinic actually acts on, so this
  is a clinical decision, not a modelling one.

---

# Decisions D9-03 .. D9-06 (Yahya, 2026-09-28)

### D9-03 — The AUC chase is closed.
**0.7796 validation / 0.7523 test is accepted as the honest ceiling for this label and this feature
set.** No further tuning work is authorised in pursuit of AUC.
- **CBC is categorically forbidden as a model input.** Grounds: F9-29. It buys +0.0218 against an
  HbA1c label and **-0.0040** against a glucose label, so the entire gain is the red-cell/HbA1c
  assay artefact. This prohibition is not a preference and must be encoded where it cannot be
  quietly undone: a comment in the config, a note in the model card, and a test that fails if any
  CBC column name enters the feature list.
- The three free questions (`BPQ020`, `BPQ080`, `SMQ020`) may only be added **after** D9-06 is
  implemented, and `SMQ020` is excluded outright (it reduced AUC, 0.7825 -> 0.7813).
- Closes: F9-23, F9-29 (as policy), and the "raise the AUC" line of work in its entirety.

### D9-04 — Platt scaling adopted, effective immediately.
Replaces the isotonic + Venn-Abers-endpoint layer from Gate 9.0b. Grounds: ECE 0.054 -> **0.023**
against isotonic's 0.031, and AUC preserved **exactly** (0.7796) where isotonic cost 0.0013 because
it ties distinct scores. Platt fit a=1.0139, b=0.2811 — the raw EBM needed a level shift, not a
shape correction, which is why a two-parameter sigmoid is the correct tool.
Consequences to implement: the bundle is rebuilt with the Platt layer; `p_lower`/`p_upper` come from
a B=1000 bootstrap of the Platt fit over the calibration set; the F9-05 wording problem disappears,
because a bootstrap interval on the calibration map makes no distribution-free claim in the first
place. The Mondrian conformal layer continues to supply `decision`, untouched.
Closes F9-27; supersedes F9-05.

### D9-05 — Group-conditional conformal, replacing class-conditional Mondrian.
Grounds: F9-11, a 20-point calibration spread by ethnic group (NH White +0.106, NH Asian -0.097)
behind an acceptable pooled intercept. Conditioning the conformal layer on groups we legitimately
hold — age band, sex, adiposity band — restores per-group decision validity **without making race a
model input**. The golden rule holds: race stays an audit variable only.
Note honestly: this fixes the validity of the *decision* per group. It does not fix the
*probability* calibration spread, which requires the group variable at inference time. That
limitation stays stated in the model card.

### D9-06 — Missing-field firewall. No judgement calls.
Grounds: F9-28. Fields whose blanking moves median risk materially or flips a large share of
decisions must be **hard-required in the UI and must never reach the model as a missing value**.
From the measured table, the mandatory tier is at minimum:

| field | median risk shift when blank | decisions flipped |
|---|---|---|
| RIDAGEYR | -0.0544 | 0.385 |
| PAQ650 | **+0.1079** | **0.311** |
| PAQ665 | -0.0614 | 0.176 |
| LBDHDD | -0.0411 | 0.147 |
| BMXBMI | -0.0169 | 0.133 |
| ADIPOSITY_BAND | -0.0169 | 0.095 |

Implementation requirements: the Pydantic schema rejects a null for every mandatory field; the
frontend blocks submission rather than sending a null; the batch CSV path rejects the row with a
named reason rather than scoring it; and a test asserts that no mandatory field can be scored as
NaN. Closes F9-28 once implemented; until then F9-28 stays open and blocks Gate 9.2.

## Status after these decisions

| finding | state |
|---|---|
| F9-01 no artifact | CLOSED (Gate 9.0b) |
| F9-03 no model chosen | CLOSED (D9-01) |
| F9-04 87-field form | CLOSED (D9-01) |
| F9-05 Venn-Abers wording | SUPERSEDED by D9-04 |
| F9-13 environment | CLOSED (Gate 9.0b) |
| F9-14 SI duplicates | CLOSED (D9-01) |
| F9-23 no measured numbers | CLOSED (Gate 9.0b) |
| F9-24 height / WHtR | CLOSED (Gate 9.0c) |
| F9-27 calibration layer | CLOSED (D9-04) |
| F9-29 CBC artefact | CLOSED as policy (D9-03) |
| F9-02 no governance record | open — addressed by the discovery record |
| F9-06 coverage below 0.90 | open — wording, must not claim a guarantee |
| F9-07 46% uncertain | open — reframed by F9-30, needs a UI decision |
| F9-08 HbA1c label bias | open as a stated limitation |
| F9-09 race recoverable | open as a stated limitation |
| F9-10 subgroup sensitivity | open as a stated limitation |
| F9-11 subgroup calibration | partly addressed by D9-05, rest stated |
| F9-12 Jordan prevalence | open — needs a sourced figure or the heart precedent |
| F9-15 blank fields | superseded by F9-28 / D9-06 |
| F9-17 data in repo | open — 79 MB nhanes_raw cannot enter the HF image |
| F9-25 blank adiposity | CLOSED by D9-06 (direction corrected) |
| F9-26 BMI needs height+weight | open — deliberate decision still owed |
| F9-28 blank-field firewall | open — D9-06 taken, implementation blocks Gate 9.2 |
| F9-30 label-noise ceiling | CLOSED as a stated limitation |

---

# Gate 9.0e — bundle v2 built. **STOP RULE TRIGGERED. Not approved for shipping.**

Script `train_diabetes_nhanes_ebm_v2.py`, output `logs/p9_gate90e_v2_results.md`,
alpha sweeps `ebm_bundle_v2/alpha_sweep.json` and `alpha_sweep_val.json`.
The EBM itself is byte-for-byte the 9.0b fit — same seed, same soft labels, same monotone
constraints. Only the post-hoc probability and decision layers changed.

## D9-03 enforced
`assert` against 21 complete-blood-count column names; the build fails if any reaches the feature
list. Passed. The ban is now code, not a comment.

## D9-04 confirmed on the held-out test set

| | isotonic (9.0b) | **Platt (v2)** |
|---|---|---|
| AUC | 0.7496 | **0.7523** (preserved exactly) |
| calibration intercept | -0.096 | **-0.090** |
| calibration slope | 0.845 | **0.952** |
| ECE | 0.033 | **0.025** |

Platt fit a=1.0139, b=0.2811. Bootstrap interval median width 0.0475 (B=1000 over the calibration
set). Platt wins on every axis on test, as validation predicted. **D9-04 stands.**

## F9-31 — CRITICAL — the 9.0b conformal layer was invalid per age band, and the pooled numbers hid it

Per-age-band coverage at alpha=0.10 on the test set, class-conditional — i.e. what Gate 9.0b
actually shipped:

| age band | coverage y=1 | coverage y=0 |
|---|---|---|
| 20-39 | **0.578** | 0.990 |
| 40-59 | 0.950 | 0.879 |
| 60+ | 1.000 | **0.574** |
| pooled | 0.917 | 0.856 |

The pooled figures look acceptable. Underneath, the layer **failed to flag 42% of dysglycaemic
patients aged 20-39** and failed to clear 43% of healthy patients aged 60+. It was not
conservative in a safe direction — it was confidently wrong in a structured one, and it was
confidently wrong precisely in the group where early detection matters most. This compounds F9-10
(sensitivity 0.578 in the young) rather than being a separate issue.

This is a patient-harm-grade defect in the module as built in 9.0b, found only because D9-05 forced
per-group coverage to be computed. It gets worse as alpha relaxes: at alpha=0.20 the young-positive
coverage is 0.360, at alpha=0.30 it is 0.227.

## D9-05 fixes it, and at alpha=0.10 it costs too much

Group-conditional (age band x class) at alpha=0.10: worst per-band coverage rises
**0.574 -> 0.867**, and every cell lands in 0.867-0.969. But `uncertain` rises
**0.462 -> 0.672**, and specificity of `no_referral` falls 0.423 -> 0.199.
**Pre-declared stop rule (uncertain > 0.55) triggered. Nothing ships on this setting.**

What this actually says: the class-conditional layer was buying its decisiveness by being invalid
for subgroups. The model's genuine per-group uncertainty at alpha=0.10 is two thirds of patients.

## The dial: alpha selected on VALIDATION, confirmed on test

Selecting alpha on the test set would be test-set shopping, so the sweep was repeated on validation
and the two agree closely.

Validation (selection set):

| alpha | layer | uncertain | worst per-band coverage |
|---|---|---|---|
| 0.10 | class-cond (9.0b) | 0.455 | **0.528** |
| 0.10 | group-cond | 0.655 | 0.896 |
| **0.15** | **group-cond** | **0.516** | **0.849** |
| 0.20 | group-cond | 0.356 | 0.799 |

Test (confirmation only):

| alpha | layer | uncertain | worst per-band coverage | sensitivity |
|---|---|---|---|---|
| 0.10 | class-cond (9.0b) | 0.462 | **0.574** | 0.917 |
| **0.15** | **group-cond** | **0.528** | **0.804** | 0.899 |
| 0.20 | group-cond | 0.383 | 0.717 | 0.859 |

### Recommendation: alpha = 0.15, group-conditional.

- `uncertain` 0.528 on test, under the 0.55 stop rule.
- Worst per-band coverage 0.574 -> **0.804**.
- The number that matters clinically: **dysglycaemic patients aged 20-39 go from 44% flagged to
  94% flagged** (per-band y=1 coverage at alpha=0.15: class-conditional 0.444, group-conditional
  0.942).
- Cost: specificity of `no_referral` 0.486 -> 0.278. Fewer healthy patients are actively cleared,
  and more are told "order the test".

This is Yahya's call, not mine — the stop rule exists so that a two-thirds abstention rate reaches
him rather than being tuned away quietly. What I will not do is ship alpha=0.10 class-conditional
now that 0.578 young-positive coverage has been measured.

## D9-06 in the bundle
`mandatory_fields` = RIDAGEYR, PAQ650, PAQ665, LBDHDD, BMXBMI, ADIPOSITY_BAND, with the reason
recorded in the model card. Schema, form and batch-path enforcement remain to be written in
Gate 9.2.

## Status of bundle v2
Built, measured, **withheld**. sha256 of the artifact as produced:
`407b4ec6f17a43f5e80c03122900082245234c022010123b6a489f5776c8ebcc`. It will be rebuilt once alpha
is decided, so this hash is provisional and must not be quoted as the shipped model.

---

# Gate 9.1 — backend shipped on the branch

New files only. `git diff --stat` against the branch point for every heart file, every shared
backend file, both existing configs and the whole frontend is **empty**.

| file | what it is |
|---|---|
| `backend/model_backends/diabetes_ebm_conformal.py` | the `ebm_platt_conformal` family |
| `backend/schemas_diabetes_nhanes.py` | 20-field input schema, D9-06 enforced by Pydantic |
| `configs/diabetes_nhanes.yaml` | the config, every number reproduced from the bundle |
| `models/diabetes_nhanes/` | the bundle (1.5 MB) and its model card |
| `tests/test_diabetes_ebm_backend.py` | 32 tests |
| `docs/phase9/` | the discovery record, this register, 8 figures |

Test suite: **663 pre-existing pass + 32 new = 695, zero failures, zero regressions.** The 59-test
difference against the main checkout's 722 is `tests/test_heart_candidate_retrain.py`, which is
uncommitted heart work in that checkout and does not exist on this branch.

End to end through the real app: `POST /api/v4/diabetes_nhanes/predict` returns 200 with
`decision`, `probability_lower/upper` and `output_type: conformal_decision`; omitting `PAQ650`
returns 422 naming the field, so D9-06 is enforced before the request reaches the model.

### F9-32 — MEDIUM — Raising HDL is missing from the What-If policy.
`backend/counterfactual_generator.py` understands only `decrease` and `to`. An `increase` kind
would have been needed for LBDHDD, and without it `all_improvements` sets the field to the bound
**unconditionally** — pulling a healthy HDL of 70 *down* to 40 — while `policy_violations` has no
branch that would catch it. Adding the kind means editing a module the heart module also uses, so
it is deferred to Gate 9.3 with its own regression test rather than slipped in here. The exercise
levers carry most of the same advice meanwhile.

### F9-33 — HIGH — The operating point sends most healthy patients for testing.
Measured per age band on the test cycle at the adopted alpha = 0.15:

| age band | n | prevalence | refer | no-refer | uncertain |
|---|---|---|---|---|---|
| 20-39 | 1412 | 0.159 | 0.240 | 0.253 | **0.507** |
| 40-59 | 1351 | 0.398 | 0.280 | 0.249 | **0.472** |
| 60+ | 1336 | 0.529 | 0.253 | 0.139 | **0.608** |
| all | 4099 | 0.359 | 0.257 | 0.214 | **0.528** |

Among patients who are **genuinely not dysglycaemic**, the share told to take the test anyway:
**71% of 20-39s, 66% of 40-59s, 82% of 60+**.

This is the cost side of the D9-05 + alpha=0.15 trade, and it is a product fact, not a defect: the
module is strongly biased toward ordering an HbA1c. For a cheap confirmatory test that is
defensible, and it is the direct price of lifting young-positive coverage from 0.444 to 0.942.
But it must be on screen and in the config rather than discovered by a reviewer, and Yahya may
prefer alpha = 0.20 (uncertain 0.383, worst per-band coverage 0.717) once he sees this table.

**This is the one number that should decide alpha, and it was not available when alpha was chosen.**

---

# D9-09 — D9-08 is REVERTED. Glucose is an indirect target leak, and the ban is now code.

Yahya caught this. He is right, and the error was mine.

## The error, named precisely

I tested **"is glucose sufficient on its own?"** — AUC 0.7029, below this model's own 0.7523, and
the rule `glucose >= 100` alone misses 54.2% of cases — and concluded "therefore not label-defining".

**Insufficiency is not non-leakage.** A noisy partial measurement of the target is still a
measurement of the target. An AUC of 0.70 is what a *proxy target* looks like; it is not evidence
of a legitimate covariate. I mistook a weak measurement of the outcome for a strong predictor of it.

## The distinction I missed

| kind of leak | test | glucose |
|---|---|---|
| temporal | is it available at inference time? | **passes** — which is why it fooled me |
| **construct / target** | does it measure the SAME latent quantity as the label? | **fails** |

HbA1c and glucose are two assays of glycaemia. A model predicting one from the other is doing
measurement agreement dressed up as risk prediction. For a *screening* tool the second kind is the
one that matters, because the product's entire claim is that it **finds dysglycaemia without
measuring glycaemia**. Once glycaemia is measured, the claim is void — a blood sugar test was done.

## The clinical argument, which settles it independently

A clinician holding a glucose result does not need a model to tell them to order an HbA1c: fasting
glucose 100-125 mg/dL **is itself** the ADA criterion for prediabetes, and >= 126 is diagnostic for
diabetes. So the only patients a glucose-fed model helps with are those with **normal glucose and
high HbA1c** — which is precisely the F9-08 population, whose HbA1c is shifted by red-cell indices
independently of blood sugar. The apparent +0.029 AUC may therefore be the model getting better at
predicting an **assay, artefact included**, which is the same failure mode that got the CBC banned
under F9-29. I applied that reasoning to the blood count and failed to apply it here.

## Process failure worth recording

The research excluded `LBXSGL` **deliberately and explicitly**, in a documented exclusion set. I
reversed a considered research decision on the strength of one under-specified test, and the
reviewer protocol exists to prevent exactly that. The rule that should have applied: a documented
exclusion is not overturned by showing the excluded variable is weak — only by showing the
*reasoning behind the exclusion* was wrong. I never addressed that reasoning.

## What changed

- `git revert` of 91bc6d8. The 20-feature model is the primary and only model again.
  AUC 0.7523, missed 0.1415, healthy cleared 0.3549, alpha 0.20 group-conditional.
- The 21-feature bundle and its fallback routing are gone.
- **The ban is now structural, not a comment.** `GLYCEMIC_BANNED` holds ten names — `LBXGH`,
  `LBXSGL`, `LBDSGLSI`, `LBXGLU`, `LBDGLUSI`, `LBXGLT`, `LBDGLTSI`, `LBXIN`, `LBDINSI`, `LBXSOSSI` —
  asserted when the backend loads any bundle, listed in the config, and covered by a test that
  builds a tampered bundle and requires the load to fail. This is what stops D9-08 being re-made by
  someone who reads only the AUC.
- The cascade (Gate 9.4 / 9.5) is closed **on principle**, not merely on measurement. Stage 2 was
  the 21-feature model; the objection applies to it identically.

## What glucose keeps

Its clinical role, which was already built and is unaffected: the `uncertain` tier of the clinical
action plan sends the patient for a cheap glucose or FBG, and the **clinician** acts on the number
directly. That is a clinician reading a test result, not a model consuming a proxy for its own
label. The distinction is the whole point.

## The honest position on Yahya's objective

The objective — spare healthy patients the false alarm without missing one more real patient —
has **no remaining lever** that I can find and defend.

- Tuning, hyperparameters, TG/HDL, WHtR, age weighting: measured, all noise (Gate 9.0c).
- The cascade: cannot meet the constraint (Gate 9.5), and is now barred on principle anyway.
- Glucose: barred.
- CBC: barred (F9-29).
- The three free questions: +0.0028, below the bar, and needing the blank-field firewall first.

What is left is the frontier the 20-feature model actually supports: at alpha 0.20, 0.1415 of
dysglycaemic patients missed and 0.3549 of healthy patients cleared. Moving along it trades one for
the other; nothing available moves it outward. Saying otherwise would require a measurement I do
not have.

Raising it honestly needs **non-glycaemic** information the cohort does not contain — and that is a
data-collection question, not a modelling one.

---

# Gate 9.6 — Active Learning: a candidate path that mostly refuses

Answers open question 3 from the Phase 9 brief. The recommendation then was "document
it as broken and defer"; that was before the two hazards below were measured, and both
are specific to this module rather than to active learning in general.

New file `backend/active_learning/diabetes_nhanes_candidate.py`. Nothing shared is
edited: `retrain.py` hard-codes `models/{disease}/omni_diag_xgb_optimized.pkl` and an
XGBoost refit, neither of which exists here, and its own docstring already records that
the heart cycle fails on that same line.

## F9-35 — HIGH — The review queue stores an opinion; this model's target is an assay

`ReviewQueue.label` is documented as "expert annotation (0 or 1)". For the heart module
a clinician can meaningfully annotate *should this patient have been referred*. Here the
target is **HbA1c >= 5.7%** — a laboratory value nobody can judge from waist, lipids or
family history.

An annotation that is a clinician's guess teaches the model to imitate clinicians. An
annotation that is a transcribed HbA1c teaches it the target. Retraining on the first
while believing it is the second is a silent change of what the model predicts.

**Guard:** a row is admitted only when the reviewer's note carries a lab value in the
HbA1c range. "confirmed", "agree with the model" and "looks high" are refused. A number
outside 3–20 is refused rather than coerced, because it is a glucose value, a date or a
typo and guessing which is the failure this module exists to prevent.

## F9-36 — HIGH — Partial verification bias, measured on this cohort

Only referred and uncertain patients plausibly get an HbA1c ordered. Cleared patients are
never verified, so they never enter the annotated set. On the training cycle:

| stage-1 decision | n | share | true prevalence |
|---|---|---|---|
| referral | 5181 | 0.280 | 0.523 |
| uncertain | 6908 | 0.373 | 0.324 |
| **no_referral** | **6419** | **0.347** | **0.154** — never verified |

The annotated set an uncorrected loop would see has prevalence **0.409** against a true
cohort prevalence of **0.321** — an enrichment of **+8.8 points** — and it discards
**990 genuinely dysglycaemic patients, 16.7% of all positives**, who are precisely the
cases the model already gets wrong and most needs to learn from.

The enrichment is also **not uniform by age band** (+6.0 / +10.9 / +5.5 points), so it
would distort the group-conditional conformal layer as well as the prevalence.

**Guard:** `verification_bias()` measures the decision mix and the prevalence enrichment
on the actual annotations; `build_candidate()` refuses past **5 percentage points**. The
tolerance sits deliberately below the 8.8 it exists to catch — a tolerance at or above
the failure would be decoration, and a test asserts that relationship.
A second guard requires at least 10% of annotations to be patients the model **cleared**:
a loop that never sees its own clearances cannot learn from the decision it is most
likely getting wrong.

## Sampling needs no threshold

`backend/active_learning/sampler.py` centres entropy on a decision threshold. This module
configures none (`inference_threshold: null`, D-32), and inventing one to compute entropy
would reintroduce exactly the number the module exists without. The conformal layer
already states which patients it could not separate — that is the uncertainty signal, it
carries the per-group coverage property, and it needs no cut-point. So
`should_queue_for_review(decision)` queues on `decision == "uncertain"` and nothing else.

## The promise, asserted rather than stated

Never writes `model.weights_path`. Never calls `invalidate()`. Never hot-reloads a loader.
Produces a candidate card recording the parent bundle's sha256, how many annotations were
supplied, how many were used, how many were dropped for lacking a lab value, and the full
verification-bias report. Promotion is a human act performed elsewhere. A test reads the
module's own source to confirm the live path is absent.

A nine-cell transition matrix is provided so that a reviewer can see a referral flipping
straight to no_referral as its own number rather than averaged into a summary.

## Status

35 tests, all passing; suite now **770 passed, zero failures**. The honest summary is
that this path will refuse far more often than it fires, and that is the correct
behaviour: on the data as it is collected today, every candidate would be refused for
F9-36. Making it fire requires collecting outcomes for cleared patients — a
data-collection change, not a modelling one.

---

# Gate 9.8 — the frontend, and one defect it exposed

The form turned out to be **fully schema-driven**: `useDiseaseSchema` fetches the JSON
Schema from the API, `schemaFieldParser` picks a widget from it, and
`featureCategorizer` groups the fields. So the 20-field form needed no new React. What it
needed was for the schema to say the right things.

## What changed

- **`featureCategorizer.js`** — a `Laboratory` category, and the NHANES codes added to
  the existing groups. All twenty fields now land somewhere meaningful; none falls into
  `General`. The Laboratory keywords are **NHANES column codes only** — never a generic
  word like "cholesterol", which would also match the heart module's `Cholesterol` field
  through its description and silently move it out of Vitals & Signs. Verified by
  simulating the categorizer before and after against all 30 heart and BRFSS fields:
  **every one keeps its category.**
- **`ADIPOSITY_BAND` is now a labelled dropdown.** As `int` with `ge=0, le=2` it hit the
  parser's small-range rule and rendered an **unlabelled 0-2 slider** — asking a clinician
  to express a visual judgement by dragging a number. It is now
  `Literal["normal", "increased", "high"]`, which the parser renders as a select. The
  model still reads the ordinal; the mapping lives in exactly one place and integers are
  still accepted for API clients and the batch CSV path. A stored record now says
  `"high"` rather than `2`.

## F9-38 — HIGH — The form pre-fills every field from the schema example, which defeats D9-06

`extractDefaultValues` takes the schema's `example` block first and uses it as the form's
initial values. Our schema carries a full example patient, so **the form opens with all
twenty fields already filled in with someone else's values.**

The D9-06 firewall rejects a *null* mandatory field. It cannot reject a *pre-filled* one,
because the form never sends null — it sends the example. So a clinician who skips the
adiposity question submits `"high"` because that is what the example patient had, and the
backend has no way to know the question was never answered. That is worse than the blank
it was built to prevent: a blank is refused, a wrong value is scored.

This is pre-existing behaviour shared with the heart and BRFSS forms, so it is not a
regression introduced here — but D9-06 is the reason it now matters. The fix belongs in
`extractDefaultValues` (do not pre-fill required fields; block submit until they are
answered) which is shared frontend logic, so it is recorded rather than changed in this
pass.

**Until it is fixed, the D9-06 firewall protects the API but not the form.**

## F9-37 — HIGH — The monotone clinical priors are enforced on main effects only

`monotonize()` does not constrain pairwise interaction terms. Every main effect is
correctly monotone; the interactions reverse the net direction for real patients:

| feature | step | patients whose risk moves the wrong way | worst reversal | decisions flipped |
|---|---|---|---|---|
| GGT | +5 U/L | **668 / 4099 (16.3%)** | 0.0100 | 10 |
| BMI | +1 | **371 / 4099 (9.1%)** | 0.0117 | 7 |
| HDL | +5 | 165 / 4099 (4.0%) | 0.0275 | 4 |
| age | +5 years | 50 / 4099 (1.2%) | **0.0403** | 4 |
| SBP, triglycerides | — | 0 | — | 0 |

The research describes this constraint as *"a hard mathematical guarantee that the model
can never say older implies lower risk"*. **For the shipped model that is false.**

Refitting with `interactions=0` makes every prior hold and costs **0.0028 validation AUC
and 0.0010 on test** — inside the CI. That refit was measured and **deliberately not
taken** (Yahya's call: the AUC difference does not justify the rebuild). So this is the
standing state of the module, not a pending fix, and the claim has been corrected in the
config accordingly. It is recorded as a limitation with its finding ID so that no
document or screen repeats the guarantee.

## Status

**789 tests passing, zero failures.** Heart and BRFSS byte-identical to the branch point
apart from the two additive shared-file edits recorded in Gate 9.3.

Remaining before the module is fully live: demo patients for the new disease, a drift
reference on the NHANES distribution, a component to display `clinical_action_plan`, and
F9-38.

---

# Gate 9.9 — F9-38 closed, and monitoring is live

## F9-38 CLOSED — required fields are no longer pre-filled

The form seeds its initial values from the schema `example`, so the six D9-06 fields
arrived already carrying the example patient's values and the firewall was defeated: it
rejects a null, and the form never sent one.

Fixed with the mechanism the codebase already had for exactly this shape of problem —
`x_unused_by_model` (Gate 8.9). A new schema-level key `x_requires_deliberate_entry`
lists the six fields; `schemaFieldParser` reads it, never pre-fills them, and exposes
`requiresDeliberateEntry` on the field so the form can mark them.

Every other module is unaffected **by construction**: a schema that does not declare the
key gets an empty set. Verified by simulating the default logic before and after across
all three modules:

| module | fields | defaults changed |
|---|---|---|
| heart_disease | 11 | **0** |
| diabetes (BRFSS) | 21 | **0** |
| diabetes_nhanes | 20 | **6** — exactly the mandatory six |

## Monitoring is live, and it fires

A reference profile now sits at `models/diabetes_nhanes/drift_reference.json`.
`get_monitor()` already fell back to `models/{disease}/drift_reference.json`, so **no
edit to `drift.py` was needed** — and `scripts/build_drift_reference.py` was left alone,
because adding this disease there would break every run for heart and BRFSS, whose CI
rebuilds and diffs them from committed CSVs. Our source is 79 MB of raw survey files that
are not committed (F9-17), so the profile is built by
`build_nhanes_drift_reference.py` in the experiments folder — reusing the repo's own
profile functions so the JSON shape cannot drift — and it records
`rebuildable_in_repo: false` rather than looking checkable and silently never being
checked.

The reference is the **training split** (2007-2014, n=18508), not the whole cohort: drift
means incoming patients differ from the ones the model learned on. Unlike the BRFSS
profile there is **no reweighting**, because this cohort is a natural-prevalence screening
population rather than a 50/50 balanced sample.

Run against five scenarios:

| scenario | rows | flagged | which |
|---|---|---|---|
| training years vs themselves | 2000 | **0/20** | — |
| held-out 2017-2018 vs 2007-2014 | 4099 | 2/20 | `LBXSAL`, `LBXSATSI` |
| corrupted: BMI +4, HDL ×0.85 | 4099 | 4/20 | `BMXBMI`, `LBDHDD`, + the two above |
| older clinic: age +12 years | 4099 | 3/20 | `RIDAGEYR` + the two above |
| 30 rows (below the 50-row floor) | 30 | — | refused: `insufficient_data` |

It stays quiet where it should and fires where it should, including on injected faults it
had never seen.

## F9-39 — MEDIUM — Two model inputs had already drifted before the model shipped

The held-out cycle is not clean against the training years:

    LBXSAL   (serum albumin)   PSI 0.382   p 1.1e-185
    LBXSATSI (ALT)             PSI 0.247   p 1.6e-109

PSI above 0.2 is major drift by the module's own rule. So the headline **AUC 0.7523 was
measured under real input drift on two of its twenty features** — which is a robustness
result and mildly reassuring, but it has two consequences that must be stated rather than
discovered:

1. A clinic whose albumin or ALT assay is calibrated differently again from NHANES
   2017-2018 is a third distribution, further from the training one than the test set was.
2. The drift monitor will flag these two on day one in production. That is correct
   behaviour, not a fault, and whoever reads the first report needs to know it was already
   true at ship time — otherwise the first alert looks like a deployment problem.

The cohort's own positive rate also rose across these cycles (31.7% to 35.9%), so part of
the movement is the population changing rather than an assay fault.

## Status

**802 tests passing, zero failures.**

# 2026-10-05 — post-expo observation

## F9-40 — HIGH — The decision is not monotone in probability across age bands

**Recorded as an observation. The model and the decision logic are deliberately unchanged.**

### What happens

A patient with a higher estimated probability can receive a *less* conservative decision
than a patient with a lower one, when the two are in different age bands. Within one band
this never happens. Across bands it always runs the same way: the **older** patient, with
the **higher** probability, is the one cleared.

### Cause

The conformal layer is group-conditional (D9-05, `alpha = 0.2`):
`DiabetesEbmConformalBackend._decide` compares the raw EBM score with a separate
calibration quantile for each age band x class (`bundle["conformal"]["q_group"]`). Each
band therefore gets its own cut-points. Older patients without dysglycaemia score higher
than younger ones, so the score at which an older patient is cleared is higher too. This is
what Mondrian (per-group) conformal prediction is for: ~80% coverage per class *within
each band*. A single ranking by probability across bands is not something it promises.

### Numbers — held-out 2017-2018 cycle, shipped bundle (sha256 `fcceeb37…`)

n = 4,099 (3,832 with all six mandatory fields; the figures for that subset agree to
within 0.002). Probabilities are the displayed Platt `confidence`. A sample of 300 test
patients was re-scored through the production backend: 300/300 identical decisions,
probability difference 0.0.

Probability ranges that map to each decision:

| Age band | n | `no_referral` | `uncertain` | `referral` |
|---|---|---|---|---|
| 20-39 | 1,412 | p < 0.091 | 0.091 – 0.204 | p > 0.204 |
| 40-59 | 1,351 | p < 0.310 | 0.310 – 0.469 | p > 0.469 |
| 60+ | 1,336 | p < 0.440 | 0.440 – 0.608 | p > 0.608 |

Pairs of patients where the higher-p patient gets the *less* conservative decision, among
all pairs with different p (8,398,851). The 3-level order is referral > uncertain >
no_referral. "Binary" counts only tested (referral or uncertain) vs cleared, which is what
changes what happens to the patient:

| Higher-p patient in … vs lower-p patient in … | 3-level | Binary |
|---|---|---|
| **All pairs** | **10.9%** | **5.4%** |
| 40-59 vs 20-39 | 21.3% | 12.5% |
| 60+ vs 20-39 | 23.2% | 10.8% |
| 60+ vs 40-59 | 9.4% | 3.4% |
| Same band (each of the three) | 0% | 0% |
| Younger band vs older band (each) | 0% | 0% |

### Clinical implication

Equal sensitivity per band is bought with very different risk among the patients who are
cleared:

| Age band | Prevalence | Share cleared | Dysglycaemic among the cleared | Share of the band's dysglycaemic patients cleared |
|---|---|---|---|---|
| 20-39 | 15.9% | 34.2% | **4.6%** | 9.8% |
| 40-59 | 39.8% | 30.4% | **20.7%** | 15.8% |
| 60+ | 52.9% | 18.5% | **40.9%** | 14.3% |

Among patients aged 60+ shown `no_referral`, 4 in 10 have an HbA1c ≥ 5.7 in this cohort,
and a 60+ patient can be cleared at p = 0.43. A 30-year-old with p = 0.10 is sent for a test. Each is correct under the
module's stated guarantee, but a clinician who reads `no_referral` as "low risk", or who
compares two patients by their probability, is misled. Severity is HIGH because of that
reading risk, not because of a coverage failure: the per-band miss rate stays below the
20% the module was built for.

### Follow-up (backlog, not implemented)

- Explain group-conditional decisions in the UI: show the patient's age band and that
  band's cut-points next to the decision, and stop presenting `no_referral` as "low risk"
  for the 60+ band.
- Evaluate monotone alternatives against the current per-band coverage: a marginal
  (single-quantile) conformal layer, cut-points constrained to be non-decreasing across
  bands, or an explicit risk cap on clearing. Each has to be weighed against equal
  sensitivity per age band, which is the property the current design keeps.

Reproduce: `docs/phase9/results/f9_40_cross_group_monotonicity.py`. It needs the research
data directory, rebuilds the cohort exactly as the training script does, and uses the
backend's own `_platt` and `_decide`. Outputs: `docs/phase9/results/f9_40_cross_group_monotonicity.json`
and `f9_40_clinical_readout.json`.

## W-08 — CLOSED by gate B3 — retraining a retired disease could overwrite the old weights

**Originally** `WEAKNESS_REGISTER.md` W-08 (now `archive/post_expo_2026-10/`): `/admin/retrain`
for `diabetes` wrote `models/diabetes/omni_diag_xgb_optimized.pkl` through the legacy
`retrain_xgb` path.

**Why it got worse before it was closed.** Archiving `configs/diabetes.yaml` (gate B3) means
`_model_family("diabetes")` returns None, so the pipeline treats the disease as "not a
candidate family" and falls through to `retrain_xgb`. Measured on a copy of the local
database with the config removed: with annotated `diabetes` rows present the route answered
200 `failed` only because that worktree held no BRFSS weights. The production image still
downloads them (Dockerfile, until the Docker/data gate), so there the same request would have
found the file and overwritten it.

**Closed.** `backend/retired_diseases.py` is the single list of retired modules. The
`/admin/retrain` route refuses a retired disease with 410 **before** its catch-all `try`
(which would otherwise turn the refusal into a 200 "error"), and `run_retrain_pipeline`
refuses it again before reading a single sample, so a CLI call is covered too.

**Proof:** `tests/test_retired_disease.py::test_retrain_never_reaches_the_legacy_xgboost_writer`
replaces `get_annotated_samples` and `retrain_xgb` with functions that fail if called, and
asserts a 410; `test_retrain_answers_410` covers the route. With the guard disabled in memory,
both fail.

## W-26 — SUPERSEDED by gate B4 — the active-learning XGBoost retrain path was removed

**Originally** `WEAKNESS_REGISTER.md` W-26 (now `archive/post_expo_2026-10/`): the retrain
path built its design matrix from dict key order and raw, un-engineered predict-time values,
so heart crashed on its text inputs and BRFSS diabetes trained on the wrong columns.

**Status: superseded, not fixed.** Commit `3eb4c84` (gate B4, branch `chore/retire-brfss`)
deleted `retrain_xgb`, the function W-26 describes. No live module read what it wrote: BRFSS
diabetes is retired (gate B3), the NHANES module has no XGBoost file, and the heart revert
path (`sklearn_pipeline`) loads a different file. Every family outside the candidate path
(`glm_ivap_conformal`) now gets `status: "unsupported"`, reason `retrain not yet supported for
<family>`, before a single review row is read. Heart's live retrain is the candidate builder
(Gate 8.10), which never had this defect.

**Proof:** `tests/test_heart_candidate_retrain.py::TestRoutingIsByModelFamily::test_every_other_family_is_refused_before_reading`
runs NHANES and the heart `sklearn_pipeline` revert with sample reading and candidate building
replaced by functions that fail if called; `test_the_legacy_xgboost_writer_is_gone` asserts the
function is absent. Routing those families back to a builder makes the first test fail.

# 2026-10-06 — cross-layer findings from the BRFSS retirement (gates B5-B6)

## CL-1 — CLOSED in B5 (8186c4d) — a config declaring priors would have been mislabelled CORRECTED

B5 removed the router branch that mapped a config's risk bands onto a deployment prior, and
nothing corrects a probability any more. `scale_of_disease_config` still answered CORRECTED for
any config that declared both priors. So a future config with priors would have been served
and stored as "corrected" while no correction was applied: a silent mislabel across the API, the
database and the report.

**Fix:** such a config is refused when the router loads it. This covers `prevalence_train`,
`prevalence_deploy` and non-null `risk_bands`. The error is `PrevalenceCorrectionUnsupported`:
"prevalence correction is not supported (removed in B5)". A result that claims a correction is
refused too. `Scale.CORRECTED` stays a legal stored value for rows written by the retired module.

**Proof:** `tests/test_probability_scale_contract.py::TestConfigsDeclaringACorrectionAreRefused`,
including `test_the_router_fails_at_load`.

## CL-2 — FIXED in B6 (83b3576) — shared lookups keyed by display label

The frontend dictionary (`frontend/src/utils/medicalDictionary.js`) matches a field by its name
**or its display label**, case-insensitively. Heart's field `Sex` and NHANES's field `RIAGENDR`,
whose label is "Sex", therefore resolve to one entry. The two modules code it differently: heart
uses M/F; NHANES uses 1 = male, 0 = female. A first fix written for heart silently made the NHANES
tooltip wrong. It was caught before the push, by a before/after snapshot of every heart and NHANES
lookup by name and by label (`docs/cleanup/tools/frontend_snapshot/`). The entry now states both
codings.

**The general hazard:** any lookup keyed by display text can join two modules' fields that merely
share a label. A change to such an entry must be checked against every module's labels, not
only the module being fixed.

## CL-3 — FIXED in B6 (83b3576) — heart tooltips showed BRFSS meanings (pre-existing)

The shared `Age` and `Sex` dictionary entries described the BRFSS coding: 5-year age bands, and
0/1 sex. Heart's `Age` is in years and its `Sex` is M/F, so the SHAP tooltip and the Variable
Scales view gave heart patients the retired module's meanings. Fixed in its own commit, because
it changes heart-visible text.
