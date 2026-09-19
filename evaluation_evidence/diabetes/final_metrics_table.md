# Diabetes module — final numbers (single source)

Held-out test n=14139; bootstrap 2000 resamples (seed 42), percentile 95% CI. `*_at_14` metrics use the test set re-weighted to prevalence 0.14.

| metric | threshold_set | value | ci95_low | ci95_high | threshold | evidence_file |
|---|---|---|---|---|---|---|
| roc_auc | - | 0.8305 | 0.8238 | 0.8371 | threshold-free | evaluation_evidence/diabetes/roc_curve.png |
| roc_auc_dedup | - | 0.8256 | 0.8187 | 0.8324 | threshold-free | evaluation_evidence/diabetes/final_metrics_table.json |
| pr_auc_test_50 | - | 0.8041 | 0.7937 | 0.8143 | threshold-free | evaluation_evidence/diabetes/pr_curve.png |
| pr_auc_at_14 | - | 0.4281 | 0.4105 | 0.4484 | threshold-free | evaluation_evidence/diabetes/pr_curve.png |
| brier_raw_at_14 | - | 0.1753 | 0.1707 | 0.1798 | threshold-free | evaluation_evidence/diabetes/calibration.json |
| brier_corrected_at_14 | - | 0.0974 | 0.0959 | 0.0989 | threshold-free | evaluation_evidence/diabetes/calibration.json |
| brier_raw_at_test_50 | - | 0.1676 | 0.1641 | 0.1712 | threshold-free | evaluation_evidence/diabetes/calibration.json |
| ece_corrected_at_14 | - | 0.0082 | 0.0059 | 0.0129 | threshold-free | evaluation_evidence/diabetes/calibration.json |
| sensitivity | old | 0.9155 | 0.9089 | 0.9221 | 0.2750 raw | evaluation_evidence/diabetes/confusion_matrix_old_0.2750.txt |
| specificity | old | 0.5468 | 0.5353 | 0.5588 | 0.2750 raw | evaluation_evidence/diabetes/confusion_matrix_old_0.2750.txt |
| accuracy | old | 0.7312 | 0.7239 | 0.7386 | 0.2750 raw | evaluation_evidence/diabetes/classification_report_old_0.2750.txt |
| f1 | old | 0.7730 | 0.7660 | 0.7798 | 0.2750 raw | evaluation_evidence/diabetes/classification_report_old_0.2750.txt |
| ppv_test_50 | old | 0.6689 | 0.6597 | 0.6781 | 0.2750 raw | evaluation_evidence/diabetes/classification_report_old_0.2750.txt |
| npv_test_50 | old | 0.8662 | 0.8564 | 0.8761 | 0.2750 raw | evaluation_evidence/diabetes/classification_report_old_0.2750.txt |
| ppv_at_14 | old | 0.2475 | 0.2427 | 0.2527 | 0.2750 raw | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| npv_at_14 | old | 0.9755 | 0.9736 | 0.9773 | 0.2750 raw | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| flagged_at_deploy | old | 0.5179 | 0.5078 | 0.5278 | 0.2750 raw | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| cost_per_1000_test | old | 311.0545 | 301.9308 | 320.3197 | 0.2750 raw | evaluation_evidence/diabetes/confusion_matrix_old_0.2750.txt |
| sensitivity | new | 0.9131 | 0.9064 | 0.9196 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/confusion_matrix_new_0.2809.txt |
| specificity | new | 0.5542 | 0.5426 | 0.5661 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/confusion_matrix_new_0.2809.txt |
| accuracy | new | 0.7336 | 0.7264 | 0.7409 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/classification_report_new_0.2809.txt |
| f1 | new | 0.7742 | 0.7674 | 0.7809 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/classification_report_new_0.2809.txt |
| ppv_test_50 | new | 0.6719 | 0.6629 | 0.6812 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/classification_report_new_0.2809.txt |
| npv_test_50 | new | 0.8645 | 0.8546 | 0.8743 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/classification_report_new_0.2809.txt |
| ppv_at_14 | new | 0.2501 | 0.2452 | 0.2554 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| npv_at_14 | new | 0.9751 | 0.9732 | 0.9770 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| flagged_at_deploy | new | 0.5113 | 0.5009 | 0.5210 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/ppv_collapse_table.csv |
| cost_per_1000_test | new | 309.7815 | 300.6578 | 318.8344 | 0.2809 raw / 0.0598 deployed | evaluation_evidence/diabetes/confusion_matrix_new_0.2809.txt |
| threshold_raw_old (from y_test) | - | 0.2750 |  |  | - | evaluation_evidence/diabetes/oof_threshold.json |
| threshold_raw_new (from train OOF) | - | 0.2809 |  |  | - | evaluation_evidence/diabetes/oof_threshold.json |
| threshold_deployed (pi=0.14) | - | 0.0598 |  |  | - | evaluation_evidence/diabetes/oof_threshold.json |
| oof_auc_stacking_train | - | 0.8306 |  |  | - | evaluation_evidence/diabetes/oof_threshold.json |
| test_rows_duplicated_in_train_pct | - | 3.9890 |  |  | - | evaluation_evidence/diabetes/final_metrics_table.json |
| roc_auc_drop_from_duplicates | - | 0.0049 |  |  | - | evaluation_evidence/diabetes/final_metrics_table.json |
| test_cost_change_pct_new_vs_old | - | -0.4093 |  |  | - | evaluation_evidence/diabetes/oof_threshold.json |
| decisions_changed_by_prevalence_correction | - | 0.0000 |  |  | - | evaluation_evidence/diabetes/final_metrics_table.json |
| n_test | - | 14139.0000 |  |  | - | evaluation_evidence/diabetes/final_metrics_table.json |
| n_train | - | 56553.0000 |  |  | - | evaluation_evidence/diabetes/final_metrics_table.json |
