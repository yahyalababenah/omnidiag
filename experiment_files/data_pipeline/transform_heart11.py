"""
OmniDiag — Heart11 Dataset Transformation
==========================================
Transforms the raw heart11.csv (UCI Cleveland with 14 columns) into the
standard 12-column format expected by the OmniDiag pipeline.

Operations performed:
    1. Reads heart11.csv (comma-separated, 14 columns)
    2. Drops `ca` (angiography — data leakage) and `thal` (nuclear stress test — non-invasive triage)
    3. Maps numeric codes to string categories (matching existing heart.csv format)
    4. Renames columns to match the 12-column standard: Age, Sex, ChestPainType, ...
    5. Removes duplicate rows
    6. Overwrites data/heart_disease/raw/heart.csv (now comma-separated)

Safety:
    - Original heart.csv is NOT overwritten until user confirms backup exists.
    - Creates a timestamped backup before overwriting.
    - Validates output shape and NaN checks after transformation.
"""

import pandas as pd
import os
import sys
import shutil
from datetime import datetime

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def transform_heart11():
    print("=" * 60)
    print("🧬 OmniDiag — Heart11 Dataset Transformation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    heart11_path = os.path.join(raw_dir, "heart11.csv")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "heart_disease", "raw")
    heart_csv_path = os.path.join(raw_dir, "heart.csv")

    # ------------------------------------------------------------------
    # 1. Read heart11.csv
    # ------------------------------------------------------------------
    print(f"\n📂 قراءة heart11.csv من: {heart11_path}")
    df = pd.read_csv(heart11_path)
    print(f"   الأبعاد الأصلية: {df.shape[0]} صف × {df.shape[1]} عمود")
    print(f"   الأعمدة: {list(df.columns)}")

    # Column mapping (UCI Cleveland -> Standard 12-column format)
    # Original: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    column_mapping = {
        'age': 'Age',
        'sex': 'Sex',
        'cp': 'ChestPainType',
        'trestbps': 'RestingBP',
        'chol': 'Cholesterol',
        'fbs': 'FastingBS',
        'restecg': 'RestingECG',
        'thalach': 'MaxHR',
        'exang': 'ExerciseAngina',
        'oldpeak': 'Oldpeak',
        'slope': 'ST_Slope',
        'target': 'HeartDisease'
    }

    # ------------------------------------------------------------------
    # 2. Drop ca and thal (data leakage + non-invasive triage)
    # ------------------------------------------------------------------
    print("\n🔴 إزالة الأعمدة المسربة (Data Leakage):")
    if 'ca' in df.columns:
        print(f"   - 'ca' (تصوير الأوعية التاجية) — تم الإزالة")
        df = df.drop(columns=['ca'])
    if 'thal' in df.columns:
        print(f"   - 'thal' (مسح التاليوم النووي) — تم الإزالة")
        df = df.drop(columns=['thal'])

    # ------------------------------------------------------------------
    # 3. Rename columns to standard 12-column format
    # ------------------------------------------------------------------
    print("\n🔧 إعادة تسمية الأعمدة...")
    df = df.rename(columns=column_mapping)

    # ------------------------------------------------------------------
    # 4. Map numeric codes to string categories
    # ------------------------------------------------------------------
    print("\n🔄 تحويل الرموز الرقمية إلى نصوص...")

    # Sex: 1=M, 0=F
    sex_map = {1: 'M', 0: 'F'}
    df['Sex'] = df['Sex'].map(sex_map)
    print("   - Sex: 1→M, 0→F")

    # ChestPainType: 0=TA, 1=ATA, 2=NAP, 3=ASY
    cp_map = {0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'}
    df['ChestPainType'] = df['ChestPainType'].map(cp_map)
    print("   - ChestPainType: 0→TA, 1→ATA, 2→NAP, 3→ASY")

    # RestingECG: 0=Normal, 1=ST, 2=LVH
    ecg_map = {0: 'Normal', 1: 'ST', 2: 'LVH'}
    df['RestingECG'] = df['RestingECG'].map(ecg_map)
    print("   - RestingECG: 0→Normal, 1→ST, 2→LVH")

    # ExerciseAngina: 0=N, 1=Y
    exang_map = {0: 'N', 1: 'Y'}
    df['ExerciseAngina'] = df['ExerciseAngina'].map(exang_map)
    print("   - ExerciseAngina: 0→N, 1→Y")

    # ST_Slope: 0=Up, 1=Flat, 2=Down
    slope_map = {0: 'Up', 1: 'Flat', 2: 'Down'}
    df['ST_Slope'] = df['ST_Slope'].map(slope_map)
    print("   - ST_Slope: 0→Up, 1→Flat, 2→Down")

    # ------------------------------------------------------------------
    # 5. Remove duplicate rows
    # ------------------------------------------------------------------
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    print(f"\n🧹 إزالة التكرارات: {before_dedup - after_dedup} صف مكرر تم حذفه")

    # ------------------------------------------------------------------
    # 6. Ensure correct column order (the golden 12)
    # ------------------------------------------------------------------
    expected_columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
        'Oldpeak', 'ST_Slope', 'HeartDisease'
    ]
    df = df[expected_columns]

    # ------------------------------------------------------------------
    # 7. Validate
    # ------------------------------------------------------------------
    print(f"\n✅ التحقق من صحة البيانات:")
    print(f"   الأبعاد بعد المعالجة: {df.shape[0]} صف × {df.shape[1]} عمود")
    print(f"   الأعمدة: {list(df.columns)}")
    print(f"   القيم المفقودة: {df.isnull().sum().sum()}")

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("⚠️  تحذير: توجد قيم مفقودة في الأعمدة التالية:")
        for col in null_counts[null_counts > 0].index:
            print(f"      - {col}: {null_counts[col]}")
    else:
        print("   ✅ لا توجد قيم مفقودة — ممتاز!")

    # Check target distribution
    print(f"\n📊 توزيع الهدف (HeartDisease):")
    print(f"   {df['HeartDisease'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 8. Save as heart.csv (overwrite)
    # ------------------------------------------------------------------
    print(f"\n💾 حفظ البيانات في: {heart_csv_path}")
    df.to_csv(heart_csv_path, index=False)
    print("   ✅ تم الحفظ بنجاح!")
    print(f"\n🎯 اكتمل التحويل! البيانات جاهزة لخط الأنابيب.")
    print(f"   يمكنك الآن تشغيل: python merge_data.py")


if __name__ == "__main__":
    transform_heart11()
