import pandas as pd

def harmonize_and_merge():
    # 1. تحميل البيانات القديمة
    old_data = pd.read_csv("data/raw/heart.csv")
    
    # 2. تحميل البيانات الإيرانية
    new_data = pd.read_excel("data/raw/Z-Alizadeh sani dataset.xlsx")
    
    # 3. خطة الدمج (تبسيط البيانات الإيرانية لتطابق أعمدة مشروعك)
    df_merged = pd.DataFrame()
    
    df_merged['Age'] = new_data['Age']
    df_merged['Sex'] = new_data['Sex'].map({'Male': 1, 'Female': 0})
    # المنطق لدمج أنواع ألم الصدر
    df_merged['ChestPainType'] = new_data['Typical Chest Pain'] # تبسيط مبدئي
    df_merged['RestingBP'] = new_data['BP']
    df_merged['Cholesterol'] = new_data['LDL'] + (new_data['TG'] / 5) # تقدير تقريبي
    df_merged['FastingBS'] = new_data['DM']
    df_merged['MaxHR'] = new_data['PR']
    df_merged['ExerciseAngina'] = new_data['Exertional CP'].map({'Y': 1, 'N': 0})
    df_merged['Oldpeak'] = 0.0 # غير متوفر في الداتا الجديدة
    df_merged['ST_Slope'] = 0 # غير متوفر
    df_merged['HeartDisease'] = new_data['Cath'].map({'Cad': 1, 'Normal': 0})
    
    # 4. دمج الـ DataFrame القديم مع الجديد
    final_df = pd.concat([old_data, df_merged], axis=0)
    
    # 5. حفظ النتيجة
    final_df.to_csv("data/processed/merged_heart_data.csv", index=False)
    print(f"✅ تم الدمج بنجاح! حجم البيانات الجديدة: {final_df.shape[0]} مريض.")

if __name__ == "__main__":
    harmonize_and_merge()