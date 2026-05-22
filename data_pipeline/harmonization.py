import pandas as pd
import os

def explore_new_data():
    # المسار المباشر من المجلد الرئيسي
    file_path = "data/raw/Z-Alizadeh sani dataset.xlsx"
    
    print("🔍 جاري البحث عن ملف Z-Alizadeh Sani بصيغة Excel...")
    
    if not os.path.exists(file_path):
        print(f"❌ خطأ: لم يتم العثور على الملف في المسار: {file_path}")
        return

    df_new = pd.read_excel(file_path, engine='openpyxl')
    
    print("\n✅ تم العثور على الملف وقراءته بنجاح!")
    print(f"📊 حجم البيانات: {df_new.shape[0]} مريض، و {df_new.shape[1]} ميزة (عمود)")
    
    print("\n📋 قائمة الأعمدة المتاحة:")
    columns = df_new.columns.tolist()
    for i in range(0, len(columns), 5):
        print(", ".join(columns[i:i+5]))
        
    print("\n🩺 عينة من بيانات أول مريض:")
    print(df_new.head(1).to_dict(orient='records')[0])

if __name__ == "__main__":
    explore_new_data()