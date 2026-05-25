import pandas as pd
import numpy as np
import os
import sys

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path


# ==========================================
# المسار الأول: الميزات الإحصائية (Heuristic)
# ==========================================
def engineer_heuristic_features(df):
    """
    Generate statistical heuristic features.
    Math is 100% identical to the original script.
    """
    df = df.copy()
    if 'Age' in df.columns and 'RestingBP' in df.columns:
        df['Age_BP_Interaction'] = df['Age'] * df['RestingBP']
    if 'Age' in df.columns and 'MaxHR' in df.columns:
        df['HR_Age_Ratio'] = df['MaxHR'] / df['Age']
    if 'Cholesterol' in df.columns and 'Age' in df.columns:
        df['Chol_Age_Ratio'] = df['Cholesterol'] / df['Age']
    return df


# ==========================================
# المسار الثاني: الميزات السريرية (Clinical)
# ==========================================
def engineer_clinical_features(df):
    """
    Generate clinical risk score features.
    Math is 100% identical to the original script.
    """
    df = df.copy()
    # نستخدم تقريب لوغاريتمي بأوزان طبية متعارف عليها في نماذج تقييم الخطر
    # (العمر يملك الوزن الأكبر، ثم الضغط، ثم الكوليسترول)
    if all(col in df.columns for col in ['Age', 'RestingBP', 'Cholesterol']):
        # حساب Clinical Risk Score (مؤشر الخطر السريري)
        # ملاحظة: خوارزمية الأشجار لا تحتاج الدالة اللوجستية الكاملة، يكفيها الأس الداخلي لتصنيف الخطر
        df['Clinical_Risk_Score'] = np.exp(
            (df['Age'] * 0.048) + (df['RestingBP'] * 0.015) + (df['Cholesterol'] * 0.002)
        )
    return df


# ==========================================
# Script execution (preserved original behavior)
# ==========================================
if __name__ == "__main__":
    print("🧬 جاري بناء مسارات هندسة الميزات (الإحصائية والطبية)...")

    # Load config for config-driven paths
    cfg = load_config("heart_disease")
    processed_path = resolve_path(cfg, "data", "processed_path")

    # 1. قراءة البيانات المنظفة (الأساسية)
    final_clean_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
    print(f"📂 قراءة البيانات من: {final_clean_path}")
    df = pd.read_csv(final_clean_path)

    # المسار الأول: الميزات الإحصائية (Heuristic)
    df_heuristic = engineer_heuristic_features(df)
    heuristic_path = os.path.join(processed_path, cfg["data"]["heuristic_file"])
    df_heuristic.to_csv(heuristic_path, index=False)
    print(f"💾 تم حفظ الميزات الإحصائية في: {heuristic_path}")

    # المسار الثاني: الميزات السريرية (Clinical)
    df_clinical = engineer_clinical_features(df)
    clinical_path = os.path.join(processed_path, cfg["data"]["clinical_file"])
    df_clinical.to_csv(clinical_path, index=False)
    print(f"💾 تم حفظ الميزات السريرية في: {clinical_path}")

    print("✅ اكتملت العملية! تم حفظ نسختين من البيانات:")
    print(f"1. {heuristic_path} (يحتوي على التفاعلات الرياضية)")
    print(f"2. {clinical_path} (يحتوي على مؤشر الخطر الطبي)")