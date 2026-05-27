# OmniDiag: Feature Engineering A/B Testing Results

## 1. Baseline Performance (Original Data)
* **TabNet Accuracy:** 87.35%
* **XGBoost Accuracy:** 88.16%
* **Conclusion on Baseline:** Reached the data ceiling. Models are making identical errors, necessitating Feature Engineering.

---

## 2. A/B Testing: Heuristic vs. Clinical Approaches

### A. Statistical Heuristic Approach
* **Dataset:** `data_heuristic.csv`
* **Features Added:** Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio
* **Optuna Best Accuracy:** 88.98% 🏆 (Winner)
* **Best Hyperparameters:** `{'n_estimators': 898, 'max_depth': 5, 'learning_rate': 0.0136, 'subsample': 0.963, 'colsample_bytree': 0.545}`

### B. Clinical Approach (Medical Risk Approximation)
* **Dataset:** `data_clinical.csv`
* **Features Added:** Clinical_Risk_Score (Based on medical weights for Age, BP, and Cholesterol)
* **Optuna Best Accuracy:** 88.57%
* **Best Hyperparameters:** `{'n_estimators': 424, 'max_depth': 4, 'learning_rate': 0.0236, 'subsample': 0.884, 'colsample_bytree': 0.626}`

---

## 3. Final Conclusion & Chosen Path
* **Winning Approach:** Statistical Heuristic Approach (88.98%)
* **Impact:** Feature engineering successfully broke the 88.16% baseline ceiling. The model performed better when given raw mathematical interactions to find its own patterns, rather than being forced into a generalized pre-calculated medical formula.