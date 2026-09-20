# Superseded artefacts — archived 2026-09-17

These files describe models, data and operating points that no longer
apply. Kept for provenance only. Do NOT cite any number from them.

## Heart

- metrics.json / metadata.json
  Report best_model "svm" with ROC-AUC 0.957, and a 13-column schema
  including `ca` and `thal`, target `target`. Neither matches the
  production XGBoost model or its 11-feature schema.

- heart_disease_report.txt, heart_disease_confusion_matrix*.png
  Computed on data/heart_disease/processed/merged_heart_data.csv
  (605 rows). That file merged two sources with OPPOSITE label
  conventions — Cleveland (1 = no disease) and Z-Alizadeh Sani
  (1 = CAD) — and carried two fabricated constant columns for the
  Z-Alizadeh rows: Oldpeak = 0.0 and ST_Slope = 'Flat'.

  Replaced by data/heart_disease/processed/uci_heart_by_site.csv
  (920 patients, four original UCI sites, per-site labels).

## Diabetes

- diabetes_report.txt, diabetes_confusion_matrix.png
  Computed at inference threshold 0.275, which was selected on the
  TEST set (train_diabetes_ensemble.py:933), and reported on the
  uncorrected probability scale. The model was trained on the
  artificially balanced 50/50 BRFSS file while real prevalence is
  ~14%, so the probabilities are inflated by roughly 2.7x and PPV
  falls from 67% to 24.7% at the real base rate.

  See docs/DIABETES_AUDIT_REPORT.md and
  evaluation_evidence/diabetes/ for the measured replacements.

## omnidiag_diabetes_artifacts (archived 2026-09-20)

Snapshot from 30 May / 1 June 2026, superseded by this session's work.
Its configs/diabetes.yaml carries inference_threshold: 0.2750 — the value
selected on the TEST set — and has no prevalence_train, prevalence_deploy
or risk_bands keys. ensemble_metrics.json reports the same 0.275 as
clinical_threshold.

The live configuration is configs/diabetes.yaml at the repo root:
inference_threshold 0.059776 (raw equivalent 0.280854, selected on training
out-of-fold predictions), with Bayes prior-shift correction applied in
backend/prevalence_correction.py.

Model weights here are also pre-correction. Do not restore any file from
this folder without re-running the evidence scripts.
