import pandas as pd
import numpy as np

def harmonize_and_merge():
    print("🔄 جاري بدء عملية التوحيد المعيارية...")
    
    # 1. قراءة البيانات
    old_data = pd.read_csv("data/raw/heart.csv",sep='\t')
    new_data = pd.read_excel("data/raw/Z-Alizadeh sani dataset.xlsx", engine='openpyxl')
    
    # القالب الذهبي (الأعمدة الـ 12 بالترتيب الدقيق)
    expected_columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
        'Oldpeak', 'ST_Slope', 'HeartDisease'
    ]
    
    # 2. ترجمة الكنز الإيراني ليتطابق مع القالب القديم
    df_new = pd.DataFrame()
    
    df_new['Age'] = new_data['Age']
    df_new['Sex'] = new_data['Sex'].map({'Male': 'M', 'Female': 'F'})
    
    # دالة ذكية لترجمة نوع ألم الصدر
    def map_cp(row):
        if row['Typical Chest Pain'] == 1: return 'TA'
        if row['Atypical'] == 'Y': return 'ATA'
        if row['Nonanginal'] == 'Y': return 'NAP'
        return 'ASY'
    df_new['ChestPainType'] = new_data.apply(map_cp, axis=1)
    
    df_new['RestingBP'] = new_data['BP']
    df_new['Cholesterol'] = np.round(new_data['LDL'] + (new_data['TG'] / 5.0))
    df_new['FastingBS'] = new_data['DM']
    
    # دالة ذكية لترجمة تخطيط القلب
    def map_ecg(row):
        if row['LVH'] == 'Y': return 'LVH'
        if row['St Elevation'] == 1 or row['St Depression'] == 1: return 'ST'
        return 'Normal'
    df_new['RestingECG'] = new_data.apply(map_ecg, axis=1)
    
    df_new['MaxHR'] = new_data['PR']
    df_new['ExerciseAngina'] = new_data['Exertional CP'] # هي أصلاً Y أو N في الإيرانية
    df_new['Oldpeak'] = 0.0 # غير متوفرة بشكل مباشر
    df_new['ST_Slope'] = 'Flat' # قيمة افتراضية محايدة
    df_new['HeartDisease'] = new_data['Cath'].map({'Cad': 1, 'Normal': 0})
    
    # 3. توحيد الأعمدة والتأكد من الترتيب
    old_data = old_data[expected_columns]
    df_new = df_new[expected_columns]
    
    # 4. الدمج النهائي (إلغاء الفهرس القديم لتجنب التكرار)
    final_df = pd.concat([old_data, df_new], axis=0, ignore_index=True)
    
    # 5. الحفظ (سيقوم بالكتابة فوق الملف المشوه القديم)
    final_df.to_csv("data/processed/merged_heart_data.csv", index=False)
    
    print(f"✅ تم الدمج الاحترافي بنجاح!")
    print(f"📊 حجم البيانات القديمة: {old_data.shape[0]} مريض.")
    print(f"📊 حجم البيانات الجديدة: {df_new.shape[0]} مريض.")
    print(f"🚀 الحجم الكلي العظيم: {final_df.shape[0]} مريض جاهز للذكاء الاصطناعي!")

if __name__ == "__main__":
    harmonize_and_merge()