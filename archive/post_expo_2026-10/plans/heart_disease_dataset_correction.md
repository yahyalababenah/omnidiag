| Attribute | Detail |
|-----------|--------|
| Source | Merged: UCI Cleveland + Z-Alizadeh Sani dataset |
| Sample Size | 606 records |
| Base Features | 11 (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope) |
| Engineered Features | 5 (Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio, RPP, Exercise_Risk_Index) |
| Total Model Features | 16 |
| Label | HeartDisease — binary (0 = No, 1 = Yes) |
| Split | 80/20 train-test stratified |
| Preprocessing | Label encoding: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope |
| | StandardScaler: Age, RestingBP, Cholesterol, MaxHR, Oldpeak |
| | FastingBS kept as-is (binary) |
| Imputation | MissForest-style for Cholesterol and RestingBP (zeros → NaN → imputed) |
