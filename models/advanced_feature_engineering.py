import pandas as pd
import numpy as np
import os

print("🧬 جاري بناء مسارات هندسة الميزات (الإحصائية والطبية)...")

# 1. قراءة البيانات المنظفة (الأساسية)
df = pd.read_csv("data/processed/final_ready_data.csv")

# ==========================================
# المسار الأول: الميزات الإحصائية (Heuristic)
# ==========================================
df_heuristic = df.copy()
if 'Age' in df_heuristic.columns and 'RestingBP' in df_heuristic.columns:
    df_heuristic['Age_BP_Interaction'] = df_heuristic['Age'] * df_heuristic['RestingBP']
if 'Age' in df_heuristic.columns and 'MaxHR' in df_heuristic.columns:
    df_heuristic['HR_Age_Ratio'] = df_heuristic['MaxHR'] / df_heuristic['Age']
if 'Cholesterol' in df_heuristic.columns and 'Age' in df_heuristic.columns:
    df_heuristic['Chol_Age_Ratio'] = df_heuristic['Cholesterol'] / df_heuristic['Age']

df_heuristic.to_csv("data/processed/data_heuristic.csv", index=False)


# ==========================================
# المسار الثاني: الميزات السريرية (Clinical)
# ==========================================
df_clinical = df.copy()
# نستخدم تقريب لوغاريتمي بأوزان طبية متعارف عليها في نماذج تقييم الخطر
# (العمر يملك الوزن الأكبر، ثم الضغط، ثم الكوليسترول)
if all(col in df_clinical.columns for col in ['Age', 'RestingBP', 'Cholesterol']):
    # حساب Clinical Risk Score (مؤشر الخطر السريري)
    # ملاحظة: خوارزمية الأشجار لا تحتاج الدالة اللوجستية الكاملة، يكفيها الأس الداخلي لتصنيف الخطر
    df_clinical['Clinical_Risk_Score'] = np.exp((df_clinical['Age'] * 0.048) + (df_clinical['RestingBP'] * 0.015) + (df_clinical['Cholesterol'] * 0.002))

df_clinical.to_csv("data/processed/data_clinical.csv", index=False)


print("✅ اكتملت العملية! تم حفظ نسختين من البيانات:")
print("1. data_heuristic.csv (يحتوي على التفاعلات الرياضية)")
print("2. data_clinical.csv (يحتوي على مؤشر الخطر الطبي)")