# Phase 9 — Discovery record: what was found, and the method that found it

Purpose: this is the governance record F9-02 said did not exist. It is not a summary of results —
the results live in `p9_gate90b_results.md`, `p9_gate90c_proposals.md`, `p9_gate90d_ceiling.md`.
This file records **how each finding was reached**, so that a reviewer can attack the method rather
than take the number on trust, and so that Rule 7 ("what would prove this wrong") has an answer per
conclusion.

Scope: OmniDiag Phase 9, the NHANES/EBM dysglycaemia module. Dates 2026-09-28.
Branch `phase9/diabetes-nhanes-ebm` @ `d621637`, never checked out. No product-repo source was
modified during any of this work.

---

## The method, in general

Five rules were applied to every question, in this order.

**1. Look before believing.** Every premise handed to me was checked against the filesystem before
being planned around. This is how F9-01 was found within the first ten minutes: the task described a
trained model to integrate, and `find` for `*.pkl *.joblib *.pickle *.onnx *.pt *.h5 *.cbm` over the
entire research folder returned zero files. No amount of reading the reports would have revealed
that, because the reports are correct — they simply describe a model that was never saved.

**2. Reproduce the data layer before touching the model layer.** Before any fitting, the cohort was
rebuilt from the raw NHANES CSVs by executing lines 59-175 of the research script verbatim, and its
outputs compared to the published `results.md`: n, prevalence, feature count, split sizes, per-cycle
positive rate, and both leak guards. All seven matched exactly. That established the research as
trustworthy at the data layer, which is what made it safe to disagree with it elsewhere.

**3. Pre-declare, then measure.** For every experiment, the expected result, the bar for calling
something an improvement, and the falsifier were written into the script's docstring **before** the
script was run. The bar was derived from the test-set CI half-width (~0.014), giving 0.005 — so a
+0.002 result could not later be narrated as a win.

**4. Change one thing.** Every variant reuses the shipped configuration (soft labels sigma=0.12, the
same monotone constraints, the same split) so a delta is attributable.

**5. Never spend the test set.** The 2017-2018 held-out cycle was scored exactly once, for the
shipped model in Gate 9.0b. Gates 9.0c and 9.0d ran entirely on validation. This is why the
comparisons can be trusted and why the headline number is not the product of shopping.

---

## Finding by finding: the method that produced it

### F9-01 — No artifact existed. Nothing was ever serialized.
**Method:** filesystem search for every common model-serialization extension across the research
folder, excluding the venv; then `grep` of the research script for every persistence call
(`joblib`, `pickle`, `.dump`, `savez`, `save(`, `json.dump`, `savefig`). The only writes are three
figures, `results.json` and `results.md`.
**Why it was missed before:** the reports are complete and accurate. A reader checking whether the
*research* was done would find nothing wrong. The gap is only visible if you ask a different
question: "which file would the product load?"
**What would prove it wrong:** a bundle anywhere on the machine whose contents reproduce
`research_temporal/results.md`. None was found.

### F9-29 — The largest available AUC gain is an assay artefact. (The most important finding.)
**Method — a control experiment, not an opinion.** Adding a CBC raised validation AUC by +0.0218,
four times the pre-declared bar. Rather than accept it, the entire pipeline was refitted against a
**different label** — `LBXSGL >= 100`, a glucose-defined definition of dysglycaemia — on the same
patients, same split, same features. The logic: if the CBC carries real glycaemia information, it
must help against a glycaemia label too. If it only helps against HbA1c, it is predicting the
*measurement*, not the disease.

| label | 20 fields | 20 + CBC | gain |
|---|---|---|---|
| HbA1c >= 5.7 | 0.7796 | 0.8014 | **+0.0218** |
| glucose >= 100 | 0.7268 | 0.7229 | **-0.0040** |

The gain does not shrink — it inverts. Surviving fraction -0.18.
**Where the idea came from:** the research had already measured the mechanism (F9-08: HbA1c ~
glucose + MCH + RDW + age, MCH coefficient -0.0448 at p~4e-235, independent of glucose) but had
treated it as a caveat in prose. Turning a caveat into a control experiment is what made it
actionable — and it flipped the conclusion from "CBC would improve the model" to "CBC must be
banned".
**Consequence nobody had noticed:** the 87-feature (0.7976) and 105-feature (0.8077) reference
numbers both include the full CBC, so the "ceiling" the project had been measuring itself against
is partly inflated by the same artefact.
**What would prove it wrong:** the same control on the fasting subsample using `LBXGLU` from
`fasting_glucose.csv` showing the CBC gain surviving. `LBXSGL` is serum glucose and not guaranteed
fasting, which is the known weakness of this test and the reason the confirmatory run is listed as
outstanding.

### F9-28 — Blank fields move risk violently.
**Method:** counterfactual ablation on the fitted bundle. For each of the 20 fields in turn, that
column was set to NaN for all 4099 test patients, the model re-scored, and two quantities recorded:
the median shift in probability, and the fraction of conformal decisions that changed. This is a
*measurement on the shipped artifact*, not a property of the training data.
**Why it matters more than the AUC:** blanking `PAQ650` — one yes/no checkbox — moves median risk by
+0.108 and changes 31% of decisions. That is a larger effect on real output than the entire 0.05 AUC
gap the project had been chasing.
**Where the idea came from:** the heart module's HF-11/HF-12. Porting an audit that caught something
once is cheaper than inventing a new one.
**What would prove it wrong:** nothing measured here is inferential — it is a direct evaluation of
the shipped model. The open question is which of these shifts are *correct* inference from absence
rather than defects, and that is a clinical judgement, not a statistical one.

### F9-25 — I predicted the wrong direction, and the measurement corrected me.
**Method and the error:** from the training data I observed that patients with a missing adiposity
measurement have prevalence 0.389, above the "increased" band's 0.279, and predicted that the EBM
would therefore read a blank as *elevated* risk. When the counterfactual ablation was run on the
fitted model, blanking the band **lowered** median risk by 0.0169.
**Why the reasoning failed:** the observed prevalence of the missing group is a marginal quantity
confounded with age and BMI. What the EBM learns for its missing bin is conditional on the other 19
features, and is estimated from only ~4.6% of rows. Marginal prevalence does not predict a
conditional shape function.
**Recorded deliberately:** the conclusion (make the field required) was unchanged, but the reasoning
behind it was wrong, and false reassurance is worse than false alarm. A register that only lists
confirmed predictions is not a register.

### F9-30 — About 0.07 AUC is locked in label noise.
**Method:** HbA1c >= 5.7 is a hard cut on a continuous assay with within-person variability of
roughly 0.1-0.2 percentage points. The fraction of validation patients inside that band was counted
(14.9% within +/-0.10, 24.5% within +/-0.20), then the model's AUC was recomputed on the patients
*outside* the band: 0.8304 and 0.8476 against 0.7796 on everyone.
**What it reframes:** the 46% `uncertain` rate (F9-07) is partly the conformal layer correctly
declining to call patients whose label is itself a coin flip. That is the layer working.
**What would prove it wrong:** a published within-person HbA1c CV materially tighter than 0.1
percentage points would shrink the band and the claim with it. The +/-0.15 and +/-0.20 rows are
identical because HbA1c is reported to 0.1 precision — a detail worth stating so the table is not
read as an error.

### D9-02 / F9-24 — The eyeballed band costs nothing, and the univariate figures lied by 50x.
**Method, in two stages, and the first stage was misleading.** Stage one, on the data layer only:
banding continuous WHtR cost 0.0244 univariate AUC, and simulated clinician misclassification at
20% cost 0.017 more. Reported to Yahya as "-0.041 on this feature alone, univariately" with the
explicit warning that BMI correlates 0.911 with WHtR so the model-level cost would be far smaller
and could not be read off those numbers. Stage two, after fitting: band 0.7523 vs measured waist
0.7507 on test, paired bootstrap CI **[-0.0008, +0.0040]** — no evidence of a difference. Eyeball
error at 35% cost 0.0007 AUC and moved 2.3% of decisions.
**The lesson recorded:** a univariate screen overstated the cost by roughly fifty-fold because the
information was redundant with a feature already in the model. Univariate feature screens answer a
different question from the one that matters, and the gap was flagged before it was measured rather
than explained afterwards.

### B4 — A proposal rejected on arithmetic, then confirmed by measurement.
**Method:** the proposal was to upweight patients 60+ so their AUC would rise and "lift the overall
average". The premise is false — pooled AUC is a ranking statistic over all pairs, not an average of
subgroup AUCs. This was demonstrated rather than asserted, by measuring the AUC of **age alone**:
0.716 across the whole validation set, **0.517** within the 60+ band. Age is the model's dominant
term (importance 0.647, 3.7x the next), so within a single age band the strongest feature carries
nothing and the subgroup AUC must collapse. Then the weighting was actually run at 2x and 4x: overall
AUC fell (0.7796 -> 0.7782 -> 0.7764) and the 60+ AUC **also** fell (0.6479 -> 0.6423 -> 0.6432).
**Recorded because:** the proposal failed on its own terms. Showing that is more useful than
explaining the statistics, and cost one extra fit.

### D9-04 / F9-27 — The calibration layer was making the model worse.
**Method:** noticed, not sought. The 9.0b results table printed raw and recalibrated calibration
side by side, and the recalibrated slope (0.845) was *further* from 1 than the raw (0.965), while AUC
fell 0.7523 -> 0.7496. Isotonic regression was solving a problem the EBM did not have.
Yahya then proposed Platt on the mechanism that isotonic is non-parametric and breaks smooth
probabilities into steps. Measured on validation, calibrators fitted on the calibration half:
ECE raw 0.054, isotonic 0.031, **Platt 0.023**; AUC raw 0.7796, isotonic 0.7783, **Platt 0.7796**.
**The mechanism, confirmed numerically:** the Platt fit is a=1.0139, b=0.2811 — the slope needed a
1.4% correction and the level needed a real shift. A two-parameter sigmoid is the right tool for
that, and a free-form monotone fit is not. Platt also preserves AUC *exactly*, because a monotone
two-parameter map cannot reorder patients, whereas isotonic can tie distinct scores.
**Credit:** the diagnosis was Yahya's. The measurement confirmed it.

### F9-09 — The finding that got worse when the fix was applied.
**Method:** after D9-01 removed creatine kinase (which alone reached AUC 0.706 for NH Black), the
expectation was that race would become less recoverable. Measured instead: a logistic model on the
20 shipped features recovers NH Asian at **0.785** and NH Black at **0.780** — higher than CK alone.
Creatinine, albumin, waist, BMI and age jointly carry more ancestry signal than CK did by itself.
**Recorded because:** the intuitive fix made the audited quantity worse, and measuring rather than
assuming is the only reason we know. "Race is not an input" stays true; "the model is race-blind"
was never true and must not be written.

### F9-13 — Closed by refusing the convenient path.
**Method:** the research venv (numpy 2.5.3 / pandas 3.0.6 / sklearn 1.9.1) differs from the product
pins (2.4.6 / 3.0.3 / 1.9.0), and `interpret-core` was absent from the product entirely. Fitting in
the research venv and shipping the pickle would have been one command — and would have repeated the
incident recorded at the top of `requirements.txt`, where an unpinned HF Space resolved sklearn 1.7.2
and served different diabetes probabilities from the same files (Case C 63.7% local vs 49.4% live).
Instead: `pip install --dry-run` first, which showed `Would install interpret-core-0.7.8` and nothing
else, every dependency already satisfied at the pinned version. Then installed, then every pin
re-verified. The training script now asserts the pinned versions at startup and exits if they differ,
so the shortcut cannot be taken later by accident.

---

## What this record does not claim

- The reproducibility check demanded by the non-negotiables was **not** performed in the form
  specified, because F9-01 means there was no saved artifact to reproduce. What was done instead:
  the data layer was reproduced exactly, and a bundle was created from scratch in the pinned
  environment with its numbers measured rather than carried.
- No number in this record has entered a config, a document or the UI. The product repo is untouched.
- The confirmatory F9-29 control on the fasting subsample (`LBXGLU`) has not been run.
- Gate 9.1 onward has not started.

## Files

| file | what it holds |
|---|---|
| `p9_repo_status_start.txt` | branch point, sha256 baseline of 45 files, heart files to stay identical |
| `p9_gate90_cohort_verify.txt` | data-layer reproduction and both leak guards |
| `p9_curated20_characterisation.txt` | the 20 fields, units, ranges, coverage, monotone constraints |
| `p9_whtr_banding.txt` | the banding analysis, stage one (univariate, later shown to overstate) |
| `p9_gate90b_results.md` | the shipped bundle's measured numbers |
| `p9_gate90b_model_card.json` | model card and metrics as serialized into the bundle |
| `p9_gate90c_proposals.md` | the nine proposals, measured |
| `p9_gate90d_ceiling.md` | the ceiling question and the CBC control experiment |
| `findings_register_diabetes.md` | 30 findings and 6 decisions, with severities |
| `p9_discovery_method.md` | this file |

Scripts, in the experiments folder `~/Desktop/بيانات سكر جديدة/`:
`train_diabetes_nhanes_ebm.py`, `gate90c_yahya_proposals.py`, `gate90d_can_we_raise_auc.py`.
Bundle: `ebm_bundle/diabetes_nhanes_ebm.joblib`,
sha256 `04a89084e897c7bb112ccb8d6d80d7bbe8e18c1bc9e4bd730f22f9a5c7168668`.
