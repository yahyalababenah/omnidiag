# Gate 9.0d — can the accuracy go up?
generated 2026-09-28T06:48:39  |  validation only, test untouched
cohort 26904  train 18508  val 2148  (test 4099 UNUSED)
tier-1 free questions available: ['BPQ020', 'BPQ080', 'SMQ020']   coverage: {'BPQ020': 0.999, 'BPQ080': 0.881, 'SMQ020': 0.999}
tier-2 CBC columns available: ['LBXRDW', 'LBXMCHSI', 'LBXMCVSI', 'LBXWBCSI', 'LBXPLTSI', 'LBDLYMNO', 'LBXHGB']   coverage: {'LBXRDW': 0.998, 'LBXMCHSI': 0.998, 'LBXMCVSI': 0.998, 'LBXWBCSI': 0.998, 'LBXPLTSI': 0.998, 'LBDLYMNO': 0.997, 'LBXHGB': 0.998}

## TIER 1 — three free questions. No lab, no instrument, no cost.
| feature set | fields | validation AUC | delta vs baseline |
|---|---|---|---|
| baseline curated-20 | 20 | 0.7796 | — |
| + BPQ020 (known high BP) | 21 | 0.7801 | +0.0005 (below 0.005 bar) |
| + BPQ020 + BPQ080 (known high chol) | 22 | 0.7825 | +0.0028 (below 0.005 bar) |
| + all three free questions | 23 | 0.7813 | +0.0016 (below 0.005 bar) |

## TIER 2 — one extra blood tube (CBC)
| feature set | fields | validation AUC | delta |
|---|---|---|---|
| baseline curated-20 | 20 | 0.7796 | — |
| + CBC | 27 | 0.8014 | +0.0218 |
| + free questions + CBC | 30 | 0.8018 | +0.0222 |

### The artefact control (F9-08): does CBC also help against a GLUCOSE-defined label?
glucose label available for 0.986 of the cohort; validation n=2130
prevalence of glucose >= 100: 0.270   (vs HbA1c >= 5.7: 0.329)

| label | 20 fields | 20 + CBC | delta from CBC |
|---|---|---|---|
| HbA1c >= 5.7 (what we ship) | 0.7796 | 0.8014 | **+0.0218** |
| glucose >= 100 (control) | 0.7268 | 0.7229 | **-0.0040** |

Fraction of the CBC gain that survives against a glucose label: -0.18
If this is far below 1, most of the CBC gain is the HbA1c assay artefact (red-cell indices shifting HbA1c without shifting glycaemia), not additional risk information.

## TIER 3 — how much of the missing AUC is unattainable in principle?
- validation patients within +/-0.10 HbA1c units of the 5.7 cut: 0.149 (319 of 2148)
- validation patients within +/-0.15 HbA1c units of the 5.7 cut: 0.245 (527 of 2148)
- validation patients within +/-0.20 HbA1c units of the 5.7 cut: 0.245 (527 of 2148)

Published within-person HbA1c variability is roughly 0.1-0.2 percentage points, so patients in that band are on the wrong side of the cut about as often as the right side. No model can separate them. Measured directly: the model's AUC restricted to patients OUTSIDE the band vs on everyone:
- AUC excluding the +/-0.10 ambiguous band: 0.8304   (vs 0.7796 on everyone)
- AUC excluding the +/-0.20 ambiguous band: 0.8476   (vs 0.7796 on everyone)

## Where the ceiling actually is (in-house measurements, same cohort and split)
| feature set | model | test AUC | source |
|---|---|---|---|
| old BRFSS-style 15 | XGBoost | 0.7334 | ladder_temporal |
| curated 20 | XGBoost | 0.7575 | ladder_temporal |
| curated 20 | EBM (shipped) | 0.7523 | Gate 9.0b |
| biology only 100 | XGBoost | 0.8017 | ladder_temporal |
| all clean 105 | XGBoost | 0.8077 | ladder_temporal |
| all clean 87 | EBM | 0.7976 | research_temporal |
| + HbA1c/glucose as inputs | XGBoost | 1.0000 | circular, not a ceiling |