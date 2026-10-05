# Gate 9.5 — choosing the deployable design
objective: maximise healthy cleared, subject to missed <= 0.141

## Real inductive Venn-Abers vs the Platt layer (stage 1, test cycle)
| layer | AUC | ECE | Brier | interval width: median | p10 | p90 |
|---|---|---|---|---|---|---|
| Platt (shipped) | 0.7523 | 0.0249 | 0.1896 | 0.0475 | ~0.04 | ~0.06 |
| **real IVAP** | 0.7497 | 0.0322 | 0.1907 | **0.0087** | 0.0041 | 0.0244 |
| old 'isotonic+endpoints' (F9-05) | — | — | — | 0.00047 | 0.00047 | 0.00047 (constant — carries no information) |

IVAP width varies 6.0x between the 10th and 90th percentile,
so it does carry per-patient uncertainty. Correlation |width| vs |p-0.5|: -0.001

## Stage 2 (20 features + serum glucose), trained on the full training cycle
AUC over everyone: 0.7812   (stage 1: 0.7523)
AUC inside stage-1 uncertain (n=1570): stage1 0.6929 -> stage2 0.7396
AUC inside stage-1 referral (n=1388): stage1 0.6727 -> stage2 0.7203

## The frontier. Baseline P0: missed 0.1415, healthy cleared 0.3549, glucose 0.000, HbA1c 0.722

| policy | stage-2 α+ / α− | glucose | HbA1c | missed | healthy CLEARED | still uncertain | meets constraint |
|---|---|---|---|---|---|---|---|
| P0 single stage | — | 0.000 | 0.722 | **0.1415** | **0.3549** | 0.383 | baseline |
| P1 uncertain only | 0.005 / 0.20 | 0.383 | 0.711 | 0.1435 | 0.3701 | 0.283 | no |
| P1 uncertain only | 0.005 / 0.50 | 0.383 | 0.711 | 0.1435 | 0.3701 | 0.137 | no |
| P1 uncertain only | 0.010 / 0.20 | 0.383 | 0.711 | 0.1435 | 0.3701 | 0.283 | no |
| P1 uncertain only | 0.010 / 0.50 | 0.383 | 0.711 | 0.1435 | 0.3701 | 0.137 | no |
| P1 uncertain only | 0.020 / 0.20 | 0.383 | 0.681 | 0.1497 | 0.4142 | 0.252 | no |
| P1 uncertain only | 0.020 / 0.50 | 0.383 | 0.681 | 0.1497 | 0.4142 | 0.107 | no |
| P2 everyone not cleared | 0.005 / 0.20 | 0.722 | 0.711 | 0.1435 | 0.3701 | 0.523 | no |
| P2 everyone not cleared | 0.005 / 0.50 | 0.722 | 0.711 | 0.1435 | 0.3701 | 0.261 | no |
| P2 everyone not cleared | 0.010 / 0.20 | 0.722 | 0.711 | 0.1435 | 0.3701 | 0.523 | no |
| P2 everyone not cleared | 0.010 / 0.50 | 0.722 | 0.711 | 0.1435 | 0.3701 | 0.261 | no |
| P2 everyone not cleared | 0.020 / 0.20 | 0.722 | 0.707 | 0.1476 | 0.3739 | 0.519 | no |
| P2 everyone not cleared | 0.020 / 0.50 | 0.722 | 0.707 | 0.1476 | 0.3739 | 0.257 | no |

### NO candidate meets the constraint. The single-stage module ships unchanged.