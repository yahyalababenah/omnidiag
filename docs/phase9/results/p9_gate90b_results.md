# OmniDiag — NHANES/EBM dysglycaemia module (Gate 9.0b)
generated: 2026-09-28T06:13:51
environment (pinned product stack, verified): {"numpy": "2.4.6", "pandas": "3.0.3", "scikit-learn": "1.9.0", "scipy": "1.18.0", "joblib": "1.5.3", "interpret-core": "0.7.8", "python": "3.13.5"}

cohort n=26904  prevalence=0.3288
leak guard 1 (shipped set): max univariate AUC 0.7168 (RIDAGEYR) -- PASS
leak guard 2 (shipped set): R2(glucose | 20) = 0.1148 -- PASS
temporal split: train=18508 (2007-2014)  val=2148 cal=2149 (2015-2016)  test=4099 (2017-2018)

## Model selection -- sigma chosen on VALIDATION only
| variant | validation AUC |
|---|---|
| V1 band, hard labels | 0.7787 |
| V1 band, soft sigma=0.06 | 0.7796 |
| V1 band, soft sigma=0.12 | 0.7796 |
| V1 band, soft sigma=0.20 | 0.7782 |
| V1 band, soft sigma=0.30 | 0.7776 |

selected on validation: **V1_soft_0.12**   monotonised terms: ['RIDAGEYR', 'BMXBMI', 'ADIPOSITY_BAND', 'SBP', 'MCQ300C', 'PAQ650', 'PAQ665', 'LBDHDD', 'LBXSTR', 'LBXSATSI', 'LBXSGTSI']

fitting V0 (BMXWAIST in cm, hard labels) purely as the comparison baseline for D9-02

## Held-out test set (2017-2018), n=4099 -- the ONLY headline numbers
test prevalence: 0.3586

| quantity | value |
|---|---|
| AUC (raw EBM score) | **0.7523** [0.7378, 0.7662] |
| AUC (after recalibration) | 0.7496 |
| AUC, MEC-weighted (US-representative) | 0.7655 |
| calibration intercept raw -> recal | +0.182 -> -0.096 (ideal 0) |
| calibration slope raw -> recal | 0.965 -> 0.845 (ideal 1) |
| ECE raw -> recal | 0.035 -> 0.033 |
| scaled Brier raw -> recal | 0.171 -> 0.170 |

### What D9-02 (eyeballed band instead of measured waist) costs, paired bootstrap
V1 band AUC 0.7523  vs  V0 measured-waist AUC 0.7507
delta AUC (V1 - V0) 95% CI: [-0.0008, 0.004]  ->  no evidence of a difference

## Mondrian conformal decision (alpha=0.1 per class)
quantiles from the calibration set: q0=0.4865  q1=0.8226

| quantity | value | target |
|---|---|---|
| coverage y=1 | 0.917 | >= 0.90 |
| coverage y=0 | 0.856 | >= 0.90 |
| decision = referral | 0.237 | — |
| decision = no_referral | 0.301 | — |
| decision = uncertain | **0.462** | pre-declared stop rule at 0.55 |
| sensitivity of 'flagged' (referral or uncertain) | 0.917 | — |
| specificity of 'no_referral' | 0.423 | — |
NOTE (F9-06): class coverage below the 0.90 target under the temporal split -- exchangeability is violated by design. The module must NOT claim a 90% coverage guarantee.

## EBM additivity check (replaces any post-hoc SHAP)
max | intercept + sum(term contributions) - decision_function | over all 4099 test rows = 2.665e-15
PASS at < 1e-8. shap_scale = log_odds_raw_score; contributions explain the RAW score, not the recalibrated probability.

## Term importances (mean |contribution|, log-odds) -- top 15
- RIDAGEYR: 0.6474
- BMXBMI: 0.1745
- LBXSGTSI: 0.1338
- LBXSAL: 0.1315
- LBDHDD: 0.1212
- ADIPOSITY_BAND: 0.1165
- MCQ300C: 0.1108
- SBP: 0.0956
- PAQ650: 0.0781
- LBXSCH: 0.0545
- PAQ665: 0.0471
- LBXSUA: 0.0463
- LBXSTR: 0.0453
- LBXSATSI: 0.0400
- RIDAGEYR & RIAGENDR: 0.0371

## Net benefit (recalibrated probability)
| threshold | model | screen everyone |
|---|---|---|
| 0.10 | 0.2957 | 0.2874 |
| 0.20 | 0.2369 | 0.1983 |
| 0.30 | 0.1745 | 0.0837 |

## Fairness audit -- race is NOT a model input, used here only to audit
| group | n | prevalence | AUC | mean predicted - observed | sensitivity of 'flagged' |
|---|---|---|---|---|---|
| sex=female | 2155 | 0.344 | 0.785 | +0.021 | 0.919 |
| sex=male | 1944 | 0.374 | 0.713 | +0.014 | 0.915 |
| age=20-39 | 1412 | 0.159 | 0.757 | -0.001 | 0.578 |
| age=40-59 | 1351 | 0.398 | 0.691 | +0.016 | 0.950 |
| age=60+ | 1336 | 0.529 | 0.619 | +0.038 | 1.000 |
| race=Mexican American | 537 | 0.318 | 0.750 | +0.047 | 0.912 |
| race=Other Hispanic | 396 | 0.343 | 0.792 | +0.040 | 0.934 |
| race=NH White | 1438 | 0.293 | 0.760 | +0.106 | 0.964 |
| race=NH Black | 946 | 0.468 | 0.746 | -0.076 | 0.883 |
| race=NH Asian | 579 | 0.406 | 0.788 | -0.097 | 0.889 |
| race=Other/multi | 203 | 0.315 | 0.803 | +0.030 | 0.922 |
| adiposity_band=0.0 | 676 | 0.166 | 0.806 | -0.007 | 0.661 |
| adiposity_band=1.0 | 1452 | 0.306 | 0.715 | +0.018 | 0.885 |
| adiposity_band=2.0 | 1775 | 0.468 | 0.700 | +0.029 | 0.964 |
| adiposity_band=MISSING | 196 | 0.423 | 0.718 | -0.001 | 0.964 |

## F9-09 re-check on the shipped feature set (race recoverability, in-sample logistic)
- AUC(20 shipped features -> Mexican American) = 0.732
- AUC(20 shipped features -> Other Hispanic) = 0.630
- AUC(20 shipped features -> NH White) = 0.615
- AUC(20 shipped features -> NH Black) = 0.780
- AUC(20 shipped features -> NH Asian) = 0.785
- AUC(20 shipped features -> Other/multi) = 0.629
Excluding RIDRETH3 does not make the model race-blind. The docs must not say it does.

## D9-02 robustness: clinician misplaces the band by +/-1
| p(misclassified) | AUC | delta | decisions changed |
|---|---|---|---|
| 0.00 | 0.7523 | — | — |
| 0.10 | 0.7520 +/- 0.0004 | -0.0003 | 0.006 |
| 0.20 | 0.7517 +/- 0.0004 | -0.0005 | 0.013 |
| 0.35 | 0.7516 +/- 0.0004 | -0.0007 | 0.023 |

## F9-15 blank-field audit -- risk shift when one field is left empty
| field | median delta p | mean delta p | decisions changed | direction |
|---|---|---|---|---|
| RIDAGEYR | -0.0544 | -0.0413 | 0.385 | lowers risk |
| RIAGENDR | -0.0010 | -0.0011 | 0.019 | neutral |
| BMXBMI | -0.0169 | -0.0377 | 0.133 | lowers risk |
| ADIPOSITY_BAND | -0.0169 | -0.0274 | 0.095 | lowers risk |
| SBP | -0.0038 | -0.0076 | 0.050 | lowers risk |
| DBP | +0.0223 | +0.0214 | 0.075 | RAISES risk |
| BPXPLS | +0.0170 | +0.0168 | 0.057 | RAISES risk |
| MCQ300C | +0.0034 | -0.0074 | 0.062 | RAISES risk |
| CVD_ANY | +0.0034 | -0.0011 | 0.021 | RAISES risk |
| PAQ650 | +0.1079 | +0.0983 | 0.311 | RAISES risk |
| PAQ665 | -0.0614 | -0.0564 | 0.176 | lowers risk |
| LBDHDD | -0.0411 | -0.0419 | 0.147 | lowers risk |
| LBXSCH | +0.0046 | +0.0021 | 0.033 | RAISES risk |
| LBXSTR | -0.0015 | -0.0053 | 0.027 | neutral |
| LBXSATSI | +0.0074 | +0.0086 | 0.033 | RAISES risk |
| LBXSGTSI | +0.0046 | +0.0075 | 0.077 | RAISES risk |
| LBXSCR | +0.0120 | +0.0106 | 0.041 | RAISES risk |
| LBXSBU | +0.0011 | +0.0031 | 0.028 | neutral |
| LBXSAL | +0.0113 | +0.0205 | 0.084 | RAISES risk |
| LBXSUA | -0.0032 | -0.0063 | 0.036 | lowers risk |

written: ebm_bundle/diabetes_nhanes_ebm.joblib (1.5 MB), model_card.json, results.md
bundle sha256: 04a89084e897c7bb112ccb8d6d80d7bbe8e18c1bc9e4bd730f22f9a5c7168668

## Pre-declared checks
- AUC in expected 0.75-0.78: YES
- Rule 7 falsifier (AUC < 0.73 means the 20-field choice does not beat the old BRFSS 0.733): not triggered
- stop rule (uncertain > 0.55): not triggered