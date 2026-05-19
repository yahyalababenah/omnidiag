import joblib
import pandas as pd
from pathlib import Path
import logging

# إعداد الـ Logger
logger = logging.getLogger(__name__)

def predict_patient(patient_data: dict):
    """
    يقوم بتحميل النموذج الجاهز (final_model.pkl) وعمل تنبؤ لمريض جديد.
    
    Args:
        patient_data (dict): قاموس يحتوي على بيانات المريض (مثل {'Age': 40, 'Sex': 'M', ...})
        
    Returns:
        prediction (int): 0 أو 1 (سليم أو مصاب)
        probability (float): نسبة ثقة النموذج في التوقع
    """
    # 1. تحديد مسار الموديل (نصعد من هذا الملف إلى الجذر ثم models)
    root_dir = Path(__file__).resolve().parents[3]
    model_path = root_dir / "models" / "final_model.pkl"
    
    if not model_path.exists():
        logger.error(f"الموديل غير موجود في: {model_path}")
        raise FileNotFoundError("لم يتم العثور على الموديل. هل قمت بتشغيل train.py؟")
        
    # 2. تحميل الموديل
    model = joblib.load(model_path)
    
    # 3. تحويل بيانات المريض إلى DataFrame (ضروري لكي تعمل الـ Pipeline)
    df = pd.DataFrame([patient_data])
    
    # 4. التوقع
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    logger.info(f"تم التوقع بنجاح: {prediction[0]} بنسبة ثقة {probability[0][1]:.2f}")
    
    return prediction[0], probability[0][1]

if __name__ == "__main__":
    # تجربة سريعة للتأكد أن الموديل يعمل
    sample_patient = {
        'Age': 40, 'Sex': 'M', 'ChestPainType': 'ATA', 'RestingBP': 140, 
        'Cholesterol': 289, 'FastingBS': 0, 'RestingECG': 'Normal', 
        'MaxHR': 172, 'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
    }
    
    try:
        pred, prob = predict_patient(sample_patient)
        print(f"التوقع: {'مصاب' if pred == 1 else 'سليم'} (نسبة الثقة: {prob:.2%})")
    except Exception as e:
        print(f"حدث خطأ أثناء التوقع: {e}")