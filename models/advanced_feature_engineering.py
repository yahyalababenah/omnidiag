import pandas as pd
import numpy as np
import os
import sys
import logging

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)


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
# المسار الرابع: تصنيف العمر العمري (Age Binning)
# ==========================================
def engineer_age_bins(df):
    """
    Transform 'Age' into ordinal categorical bins to capture non-linear risk jumps.

    Bins (raw age scale):
        0: Young (0–40)
        1: Middle Age (41–60)
        2: Senior (60+)

    If Age appears to be standardized (z-scores, typical range -3 to 3), uses
    quantile-based binning (tertiles) instead, since the bin edges on the raw
    age scale do not apply to scaled data.

    The ordinal encoding lets XGBoost learn non-linear risk jumps between bins
    rather than forcing a linear age-risk relationship.

    Args:
        df: DataFrame with 'Age' column.

    Returns:
        DataFrame with 'Age_Bins' column appended.
    """
    df = df.copy()
    if 'Age' not in df.columns:
        logger.warning("⚠️ Age_Bins: 'Age' column not found — skipping.")
        return df

    age_min = df['Age'].min()
    age_max = df['Age'].max()

    # Detect if Age is raw (e.g., 28–77) or scaled (z-scores like -2.7 to 2.9)
    if age_min > 10 and age_max > 50:
        # Raw age scale — use clinical age bins
        bins = [0, 40, 60, 200]
        labels = [0, 1, 2]
        df['Age_Bins'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True).astype(int)
        logger.info(f"   ✅ Age_Bins from raw scale: 0=Young(≤40), 1=Middle(41-60), 2=Senior(>60)")
    else:
        # Standardized/scaled Age — use quantile-based tertiles
        # This works for z-scores, normalized values, or any unknown scale
        try:
            df['Age_Bins'] = pd.qcut(df['Age'], q=3, labels=[0, 1, 2], duplicates='drop').astype(int)
            logger.info(f"   ✅ Age_Bins from quantiles (Age is scaled/standardized)")
        except ValueError:
            # Fallback: if too many duplicate values for qcut, use manual bins
            logger.warning("   ⚠️ Age_Bins: qcut failed — using median split fallback")
            median_val = df['Age'].median()
            df['Age_Bins'] = (df['Age'] > median_val).astype(int)

    logger.info(f"      Distribution: {df['Age_Bins'].value_counts().to_dict()}")
    return df


# ==========================================
# المسار الخامس: مؤشر الخطر الكلي (Global Risk Score)
# ==========================================
def engineer_global_risk(df):
    """
    Create a composite Global_Risk_Score from existing engineered features.

    Formula:
        Global_Risk_Score = (Age_BP_Interaction * 0.4)
                          + (Chol_Age_Ratio * 0.3)
                          + (RPP * 0.3)

    Weighting rationale:
        - Age_BP_Interaction (40%): Age × BP is the strongest single composite
          risk factor in cardiology literature.
        - Chol_Age_Ratio (30%): Cholesterol burden normalized by age captures
          lifetime lipid exposure.
        - RPP (30%): Rate-Pressure Product reflects myocardial oxygen demand.

    This feature requires that the prerequisite features have already been
    computed (via engineer_heuristic_features and engineer_medical_features).

    Args:
        df: DataFrame with 'Age_BP_Interaction', 'Chol_Age_Ratio', 'RPP' columns.

    Returns:
        DataFrame with 'Global_Risk_Score' column appended.
    """
    df = df.copy()
    required = ['Age_BP_Interaction', 'Chol_Age_Ratio', 'RPP']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"⚠️ Global_Risk_Score: missing columns {missing} — skipping. "
                       f"Run engineer_heuristic_features and engineer_medical_features first.")
        return df

    df['Global_Risk_Score'] = (
        df['Age_BP_Interaction'] * 0.4
        + df['Chol_Age_Ratio'] * 0.3
        + df['RPP'] * 0.3
    )
    logger.info(f"   ✅ Global_Risk_Score created — "
                f"range: [{df['Global_Risk_Score'].min():.2f}, {df['Global_Risk_Score'].max():.2f}]")
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
# المسار الثالث: الميزات الطبية (Medical)
# ==========================================
def engineer_medical_features(df):
    """
    Generate medical diagnostic features based on cardiology domain knowledge.

    Features created:
        - RPP (Rate-Pressure Product): RestingBP × MaxHR
          A global cardiology standard for myocardial oxygen consumption.
          Higher values indicate greater cardiac workload and risk.

        - Exercise_Risk_Index: Oldpeak × ExerciseAngina (encoded)
          Composite marker: if a patient has both exercise-induced angina (Y)
          AND significant ST depression (Oldpeak), CAD probability is very high.
          Zero if ExerciseAngina is 'N'.

    These features are computed from existing columns only, so they create
    ZERO new NaN values — safe for production inference.
    """
    df = df.copy()

    if 'RestingBP' in df.columns and 'MaxHR' in df.columns:
        df['RPP'] = df['RestingBP'] * df['MaxHR']
    else:
        print("   ⚠️ RPP: الأعمدة المطلوبة (RestingBP, MaxHR) غير موجودة")

    if 'Oldpeak' in df.columns and 'ExerciseAngina' in df.columns:
        # Encode ExerciseAngina: handles BOTH string ('Y'/'N') and numeric (0/1) formats
        # This makes the function safe for pre-encoded data (after clean_data.py) and raw data
        # Note: must use is_string_dtype() not dtype == 'object' because newer Pandas
        # uses StringDtype (not object dtype) for string columns.
        if pd.api.types.is_string_dtype(df['ExerciseAngina']):
            exang_num = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
        else:
            # Already numeric (0 or 1 after label encoding)
            exang_num = df['ExerciseAngina'].astype(float)
        df['Exercise_Risk_Index'] = df['Oldpeak'] * exang_num
    else:
        print("   ⚠️ Exercise_Risk_Index: الأعمدة المطلوبة (Oldpeak, ExerciseAngina) غير موجودة")

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

    # المسار الثالث: الميزات الطبية (Medical) — جديد في v5.0
    df_medical = engineer_medical_features(df)
    medical_path = os.path.join(processed_path, "data_medical.csv")
    df_medical.to_csv(medical_path, index=False)
    print(f"💾 تم حفظ الميزات الطبية في: {medical_path}")

    print("✅ اكتملت العملية! تم حفظ 3 نسخ من البيانات:")
    print(f"1. {heuristic_path} (يحتوي على التفاعلات الرياضية)")
    print(f"2. {clinical_path} (يحتوي على مؤشر الخطر الطبي)")
    print(f"3. {medical_path} (يحتوي على RPP ومؤشر خطورة التمرين)")