# Gate 9.4 — two-stage cascade
stage-1 uncertain: train 6908  cal 790  test 1570
prevalence inside uncertain: train 0.324  test 0.344

## Discrimination INSIDE the uncertain group (test cycle, n=1570)
| candidate | trained on | AUC |
|---|---|---|
| R1  glucose alone | — | 0.6634 |
| R3  stage-1 model, unchanged | all train, 20 feat | 0.6929 |
| R2  20 features, refit on the uncertain subgroup | uncertain train, 20 feat | 0.6875 |
| **S2  21 features (+ glucose), refit on the uncertain subgroup** | uncertain train, 21 feat | **0.7367** |
| S2b 21 features, trained on ALL train | all train, 21 feat | 0.7396 |

best rival 0.6929 -> best stage-2 0.7396  =  **+0.0467**
pre-declared bar 0.02 -> **CLEARS IT**

glucose term importance in stage 2: 0.2648
stage-2 top terms: RIDAGEYR 0.565, LBXSGL 0.265, SBP 0.051, MCQ300C 0.044, BPXPLS 0.038, ADIPOSITY_BAND 0.032

## The cascade end to end, over ALL 4099 test patients
| design | tests ordered | MISSED positives | sensitivity | cleared correctly | still uncertain |
|---|---|---|---|---|---|
| single stage, alpha 0.20 | 0.722 | **0.141** | 0.859 | 0.355 | 0.383 |
| cascade, stage-2 alpha 0.10 | 0.383 glucose + 0.583 HbA1c | **0.194** | 0.806 | 0.541 | 0.215 |
| cascade, stage-2 alpha 0.20 | 0.383 glucose + 0.568 HbA1c | **0.209** | 0.791 | 0.556 | 0.162 |
| cascade, stage-2 alpha 0.30 | 0.383 glucose + 0.552 HbA1c | **0.227** | 0.773 | 0.572 | 0.114 |
| cascade, stage-2 alpha 0.50 | 0.383 glucose + 0.574 HbA1c | **0.233** | 0.767 | 0.534 | 0.124 |