# Did we stay faithful to the research?

An honest audit, written because the question was asked and it deserves an answer with a
table rather than a reassurance. Sources: `ladder_temporal/results.md`,
`research_temporal/results.md`, `ablation_log.txt` and the two research scripts in the
experiments folder; against `configs/diabetes_nhanes.yaml` and the shipped bundle's card.

**Short answer: faithful on the data layer and the method, deviating on five decisions.
Four of those deviations are backed by a new measurement. One is not — it is driven by a
product constraint, and it is the honest weak point of this module.**

---

## Where we are exactly faithful

| what the research established | what we ship | how it is enforced |
|---|---|---|
| Cohort: adults >= 20, not pregnant, HbA1c present, NOT diagnosed (DIQ010) and NOT treated (DIQ050/DIQ070) | identical | `assert len(D) == 26904` in the training script — the build fails if the cohort drifts |
| Prevalence 0.3288 | identical | reproduced, and the assert above covers it |
| Temporal split: train 2007-2014, val+cal 2015-2016, test 2017-2018 | identical | `assert (len(tr), len(va), len(ca), len(te)) == (18508, 2148, 2149, 4099)` |
| Headline is the temporal split; the random split never appears | held | the random split is not fitted by our script at all |
| Leak guard 1: no single feature reaches AUC 0.85 | run on the set as it ships | `assert worst[0] < 0.85` |
| Leak guard 2: the feature set cannot reconstruct the label | run on the set as it ships | `assert _r2 < 0.60` |
| Glycaemic variables excluded as label-defining | held, and **strengthened** | the research had this as a set inside a script; we made it ten banned names checked on every bundle load, listed in the config, with a test that builds a tampered bundle and requires the load to fail (D9-09) |
| Monotone clinical priors (INC/DEC lists) | the subset applicable to 20 features: 8 increasing, 3 decreasing | applied per feature, recorded in the model card |
| Fairness audit by sex, age band and race, with race as an audit variable only | reproduced on the shipped model | in the model card; race is never an input and never a conditioning variable |
| Decision-curve net benefit | reproduced | in the model card |
| Label measurement bias: HbA1c responds to MCH at -0.0448 (p~4e-235) independently of glucose | carried as a stated limitation, F9-08 | in the config's `limitations`, with its finding ID |
| Soft labels chosen on validation, never on test | held | the sigma sweep runs on validation only |

Every number in the shipped config is re-derived from the artifact we built, not copied
from a research report. A test asserts it: `test_config_headline_auc_matches_the_bundle`.

---

## Where we deviated, with evidence

### 1. Model family — EBM, which the research never designated
The research compared seven models and **named no winner**. On the temporal test set
XGBoost led (0.8017) and EBM followed (0.7976), with ΔAUC CI [-0.0089, +0.0004] — not
distinguishable. We chose EBM for interpretability, which the research's own docstring
argues for (Rudin 2019) but never decided.
**Deviation type:** filling a gap the research left open. Recorded as D9-01.

### 2. Calibration — Platt instead of the research's isotonic layer
Measured on our cal set of 2149: Platt ECE 0.025 against isotonic 0.033, and Platt
preserves AUC exactly where isotonic cost 0.0013. The Platt fit is a=1.0139 — the sigmoid
assumption holds almost perfectly. The literature agrees that below roughly 2000
calibration cases the non-parametric fit overfits; ours sits on that boundary.
We also found the research's own layer was **not** Venn-Abers: it produced an interval of
width exactly 1/(n+1) = 0.00047 for every patient, carrying no per-patient information
(F9-05).
**Deviation type:** corrected on measurement. Recorded as D9-04.

### 3. Conformal — group-conditional by age band, not class-conditional; alpha 0.20 not 0.10
The research's class-conditional layer, applied to our feature set, covered only **0.444**
of dysglycaemic 20-39 year olds and 0.391 of healthy 60+ while the pooled figures
(0.917 / 0.856) looked acceptable. Two failures in opposite directions cancelling (F9-31).
**Deviation type:** corrected on measurement. Recorded as D9-05 and D9-07.

### 4. FasterRisk bedside score — not shipped
The research produced an integer point score. It requires a runtime monkeypatch of a
third-party library (`ndarray.tostring()` removed in NumPy 2), which would ship into the
product image.
**Deviation type:** a deliberate omission on engineering grounds. Recorded as F9-19.

### 5. **The feature set — and this one is not backed by a measurement**

This is the deviation to be honest about.

> `ladder_temporal/results.md`, the research's own words:
> `L3_all_clean | 105 features | test AUC 0.808 | **DEPLOYMENT CANDIDATE**`
> `L3b_bio_only | 100 features | test AUC 0.802 | portable, **DEPLOYMENT CANDIDATE**`
> `L4_curated_20 | 20 features | test AUC 0.758 | only what a clinician has at a first visit`

**The research designated the 100- and 105-feature sets for deployment. It did not
designate the curated 20.** We ship the curated 20.

The reason is a product constraint, not a finding: an 87-to-105 field form requiring a
full blood count and a complete biochemistry panel is not something a clinic fills in at a
first visit. That is a real constraint and it was Yahya's explicit decision (D9-01), taken
with the cost in front of him.

**The cost is measured and it is not small:** the research's own paired bootstrap puts the
gap from 20 features to 100 at CI **[0.0345, 0.0539] — REAL**. We ship 0.7523 where the
research's deployment candidate reached 0.802-0.808.

Two things make this defensible rather than merely convenient, and both came later:

- **F9-29.** The 87- and 105-feature numbers both include a complete blood count, and
  we measured that the CBC gain is an HbA1c **assay artefact**: +0.0218 against an HbA1c
  label, **-0.0040** against a glucose label. So a real share of the gap we gave up was
  never signal. The research could not have known this; it ran no such control.
- **F9-30.** About 0.07 AUC is locked in label noise at the 5.7 cut-point regardless of
  feature set.

But neither of those was known when the decision was made, and neither fully closes the
gap. **We ship a less accurate model than the research recommended, for a product reason,
and the difference is partly real.** That sentence belongs in the defence, stated by us
before anyone else says it.

---

## Where we went beyond the research

Not deviations — things it did not do, which a shipped product needs:

- The artifact itself. The research serialized no model (F9-01).
- A blank-field audit on the shipped bundle (F9-28), and a firewall for the six fields
  that move risk materially.
- Per-group conformal coverage, which is what exposed F9-31.
- The CBC control experiment (F9-29) and the glycaemic ban as enforced code (D9-09).
- The clinician-assessed adiposity band, replacing a tape measurement at no measured cost
  (D9-02).
- Partial-verification-bias measurement for the active-learning loop (F9-36).
- The governance record the research lacked (F9-02).

---

## The verdict

**Method: faithful.** Cohort, split, guards, monotone priors, audit design and the
"never spend the test set" discipline are the research's, reproduced exactly and asserted
in code.

**Modelling choices: four documented deviations, each re-measured rather than assumed.**

**Deployment recommendation: one deviation, driven by product constraints and not by a
finding.** It costs a measured 0.05 AUC, part of which we later showed was artefact. This
is the module's honest weak point and it should be stated first in any defence, not
discovered by a reviewer.

And one process failure worth keeping in the record: D9-08 reversed the research's
deliberate exclusion of glucose on the strength of a single under-specified test, and was
reverted as D9-09. The rule learned — *a documented exclusion is overturned by showing
the reasoning behind it was wrong, not by showing the excluded variable is weak* — is now
written into the method.
