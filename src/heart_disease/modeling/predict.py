import joblib
import pandas as pd
from pathlib import Path
import logging

# إعداد الـ Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_patient(patient_data: dict):
    """
    يتلقى بيانات المريض الخام (نصوص وأرقام)، يمررها عبر الـ Pipeline الجاهز، ويعيد التوقع.
    
    Args:
        patient_data (dict): قاموس ببيانات المريض مثل:
                             {'Age': 50, 'Sex': 'M', 'ChestPainType': 'ATA', ...}
                             يجب أن تكون القيم النصية (Categories) مطابقة للقيم الأصلية في CSV.
    """
    # 1. تحديد مسار الموديل ديناميكياً
    root_dir = Path(__file__).resolve().parents[3]
    model_path = root_dir / "models" / "final_model.pkl"
    
    if not model_path.exists():
        logger.error(f"الموديل غير موجود في: {model_path}")
        raise FileNotFoundError(f"لم يتم العثور على ملف final_model.pkl في مجلد models.")
        
    # 2. تحميل الموديل المجمع (بما في ذلك Pipeline المعالجة)
    logger.info("جاري تحميل الموديل والـ Pipeline...")
    final_pipeline = joblib.load(model_path)
    
    # 3. تحويل القاموس إلى DataFrame (خطوة إجبارية لعمل الـ Scikit-Learn Pipeline)
    df = pd.DataFrame([patient_data])
    
    # 4. التوقع عبر خط الإنتاج الشامل (التحويل والتصنيف يحدث هنا تلقائياً)
    logger.info("جاري المعالجة وعمل التنبؤ...")
    prediction = final_pipeline.predict(df)
    probability = final_pipeline.predict_proba(df)
    
    # النتيجة النهائية: 1 (مصاب) أو 0 (سليم)
    is_risk = int(prediction[0])
    risk_prob = float(probability[0][1])
    
    logger.info(f"النتيجة: {is_risk}، احتمالية: {risk_prob:.2f}")
    
    return is_risk, risk_prob

if __name__ == "__main__":
    # حالة تجريبية لمريض مصاب (بيانات خام نصية مطابقة لـ CSV)
    test_patient = {
        'Age': 49, 'Sex': 'F', 'ChestPainType': 'NAP', 'RestingBP': 160, 
        'Cholesterol': 180, 'FastingBS': 0, 'RestingECG': 'Normal', 
        'MaxHR': 156, 'ExerciseAngina': 'N', 'Oldpeak': 1.0, 'ST_Slope': 'Flat'
    }
    
    try:
        # تشغيل التنبؤ (Inference)
        res, prob = predict_patient(test_patient)
        print(f"\n--- نتيجة فحص V2 ---\nالتشخيص: {res}\nالاحتمالية: {prob:.2%}")
    except Exception as e:
        print(f"حدث خطأ: {e}")