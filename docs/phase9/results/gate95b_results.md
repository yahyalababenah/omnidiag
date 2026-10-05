# Gate 9.5b — 20 features vs 21 features, full frontier

AUC: 20 features 0.7523  ->  21 features 0.7812

## At each missed-positive level the 20-feature model can reach, who clears more healthy patients?
| missed <= | 20 features: best clear | 21 features: best clear | gain |
|---|---|---|---|
| 0.080 | 0.1989 | **0.2438** | +0.0449 |
| 0.100 | 0.1989 | **0.2438** | +0.0449 |
| 0.120 | 0.2777 | **0.3294** | +0.0517 |
| 0.141 | 0.3480 | **0.4283** | +0.0803 |
| 0.160 | 0.3549 | **0.4283** | +0.0734 |
| 0.180 | 0.3998 | **0.4911** | +0.0913 |
| 0.200 | 0.4382 | **0.4911** | +0.0529 |

21 features clears more at 7/7 levels.

## Candidates meeting BOTH constraints (missed <= 0.1415, worst per-band coverage >= 0.70)
| α+ / α− | missed | healthy cleared | HbA1c ordered | uncertain | worst band coverage |
|---|---|---|---|---|---|
| 0.200 / 0.05 | 0.1306 | **0.4283** | 0.678 | 0.596 | 0.844 |
| 0.200 / 0.10 | 0.1306 | **0.4283** | 0.678 | 0.504 | 0.844 |
| 0.200 / 0.20 | 0.1306 | **0.4283** | 0.678 | 0.364 | 0.775 |
| 0.150 / 0.05 | 0.1007 | **0.3294** | 0.753 | 0.670 | 0.883 |
| 0.150 / 0.10 | 0.1007 | **0.3294** | 0.753 | 0.578 | 0.883 |
| 0.150 / 0.20 | 0.1007 | **0.3294** | 0.753 | 0.438 | 0.775 |

### RECOMMENDED: 21-feature primary model, α+=0.2, α−=0.05
| | shipped (20 feat, α=0.20) | recommended (21 feat) |
|---|---|---|
| missed dysglycaemic patients | 0.1415 | **0.1306** |
| healthy patients CLEARED | 0.3549 | **0.4283** |
| HbA1c tests ordered | 0.722 | **0.678** |
| left uncertain | 0.383 | **0.596** |
| worst per-band coverage | 0.717 | **0.844** |
| extra cost | — | one glucose test per patient |

per age band — healthy cleared, then missed positives:
  20-39: cleared 0.501   missed 0.120
  40-59: cleared 0.438   missed 0.156
  60+: cleared 0.278   missed 0.115