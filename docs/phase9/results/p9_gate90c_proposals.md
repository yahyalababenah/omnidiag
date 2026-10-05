# Gate 9.0c — Yahya's proposals, measured
generated 2026-09-28T06:42:13  |  validation-only, test untouched
cohort 26904  train 18508  val 2148  cal 2149  (test 4099 DELIBERATELY UNUSED)

## A1 / B5 — soft labels: ALREADY IN THE SHIPPED MODEL. Re-stating what 9.0b measured.
| variant | validation AUC |
|---|---|
| hard labels | 0.7787 |
| sigma=0.06 | 0.7796 |
| sigma=0.12 | 0.7796 |
| sigma=0.20 | 0.7782 |
| sigma=0.30 | 0.7776 |
The shipped model already uses soft labels at sigma=0.12, selected on validation in Gate 9.0b.
sigma=0.30 -- the value proposed -- was the WORST of the four on this feature set.
Spread hard->best is 0.0009: below the 0.005 pre-declared bar. Not a gain, and not the fix for calibration.

baseline refit reproduces: validation AUC 0.7796  intercept +0.308  slope 1.079  ECE 0.054

## A2 — Platt scaling instead of isotonic. Calibrators fitted on cal (n=2149), judged on validation.
| layer | val AUC | intercept (ideal 0) | slope (ideal 1) | ECE |
|---|---|---|---|---|
| none (raw EBM) | 0.7796 | +0.308 | 1.079 | 0.054 |
| isotonic + VA endpoints (9.0b) | 0.7783 | +0.029 | 0.960 | 0.031 |
| Platt (logistic) | 0.7796 | +0.038 | 1.064 | 0.023 |
Platt coefficient a=1.0139  b=0.2811  (a=1,b=0 would mean the raw score is already perfectly calibrated)

## B2 (TG/HDL) and B1 (continuous WHtR) — one change at a time, validation AUC
Note: the shipped EBM selected 10 pairwise interactions on its own and NONE of them is TG x HDL,
so an additive model genuinely cannot express the ratio unless it is handed the ratio. B2 is sound in principle.

| variant | fields the clinician fills | validation AUC | delta vs baseline |
|---|---|---|---|
| A baseline (shipped) | 20 | 0.7796 | +0.0000 |
| B + TG_HDL (derived, no new input) | 20 | 0.7794 | -0.0003 |
| C band -> continuous WHtR (needs measured waist AND height) | 21 | 0.7794 | -0.0002 |
| D + TG_HDL + continuous WHtR | 21 | 0.7789 | -0.0007 |

## A3 / B3 — EBM hyperparameters. Current: max_bins=1024, interactions=10 (cap is binding), min_samples_leaf=4, outer_bags=8.
max_bins is ALREADY 1024 -- there is nothing to raise there; raising it further on 18508 rows buys resolution the data cannot support.

| change | validation AUC | delta | intercept | slope | ECE |
|---|---|---|---|---|---|
| baseline | 0.7796 | — | +0.308 | 1.079 | 0.054 |
| interactions 10 -> 20 | 0.7798 | +0.0001 | +0.306 | 1.069 | 0.053 |
| min_samples_leaf 4 -> 20 | 0.7797 | +0.0000 | +0.309 | 1.078 | 0.054 |
| min_samples_leaf 4 -> 50 | 0.7800 | +0.0003 | +0.311 | 1.079 | 0.054 |
| outer_bags 8 -> 16 | 0.7798 | +0.0002 | +0.310 | 1.077 | 0.054 |

## B4 — upweighting patients 60+ to 'raise the overall AUC'

### First, the arithmetic. Pooled AUC is NOT an average of subgroup AUCs.
| set | AUC of AGE ALONE | AUC of the model |
|---|---|---|
| all validation (n=2148) | 0.716 | 0.780 |
| age 20-39 (n=826) | 0.643 | 0.764 |
| age 40-59 (n=743) | 0.604 | 0.727 |
| age 60+ (n=579) | 0.517 | 0.648 |

Age alone reaches AUC ~0.72 across the whole set and collapses to ~0.5 inside any single age band,
because inside a band everyone is the same age. The model's dominant feature (age, term importance
0.647, 3.7x the next) therefore carries almost no signal within the 60+ group. The low 60+ AUC is
mostly this arithmetic, not neglect of older patients -- and no re-weighting can change it.

### Second, what the weighting actually does. Measured.
| weight on 60+ | val AUC (all) | val AUC 60+ | val AUC 20-39 |
|---|---|---|---|
| 1.0 (baseline) | 0.7796 | 0.6479 | 0.7639 |
| 2.0 | 0.7782 | 0.6423 | 0.7639 |
| 4.0 | 0.7764 | 0.6432 | 0.7604 |

## Verdict against the pre-declared 0.005 bar
| proposal | measured effect on validation AUC | verdict |
|---|---|---|
| A1/B5 soft labels | already shipped; spread 0.0009 | ALREADY DONE, not a gain |
| A2 Platt vs isotonic | see calibration table | worth adopting |
| B + TG_HDL (derived, no new input) | -0.0003 | below the 0.005 bar — noise |
| C band -> continuous WHtR (needs measured waist AND height) | -0.0002 | below the 0.005 bar — noise |
| D + TG_HDL + continuous WHtR | -0.0007 | below the 0.005 bar — noise |
| interactions 10 -> 20 | +0.0001 | below the 0.005 bar — noise |
| min_samples_leaf 4 -> 20 | +0.0000 | below the 0.005 bar — noise |
| min_samples_leaf 4 -> 50 | +0.0003 | below the 0.005 bar — noise |
| outer_bags 8 -> 16 | +0.0002 | below the 0.005 bar — noise |
| B4 weight 60+ x2 | -0.0014 overall | see arithmetic above |
| B4 weight 60+ x4 | -0.0032 overall | see arithmetic above |