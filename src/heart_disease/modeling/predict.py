import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple

# إضافة المسار الجذري لضمان التعرف على الحزم الداخلية
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.heart_disease.dataset import load_raw_data, get_X_y
from src.heart_disease.modeling.train import build_classic_stacking_model

# --- ميزة التسريع (Memory Caching) ---
# متغير عام (Global) لتخزين النموذج بعد تدريبه لأول مرة، لمنع إعادة التدريب
_trained_pipeline = None

def get_trained_model():
    """
    يقوم بتدريب النموذج مرة واحدة فقط عند أول استدعاء، ثم يحتفظ به في الذاكرة.
    هذا يجعل استجابة واجهة Streamlit سريعة جداً (فورية).
    """
    global _trained_pipeline
    if _trained_pipeline is None:
        print("⚙️ جاري تدريب النموذج لأول مرة وحفظه في الذاكرة...")
        df = load_raw_data()
        X, y = get_X_y(df)
        pipeline = build_classic_stacking_model()
        pipeline.fit(X, y)
        _trained_pipeline = pipeline
    return _trained_pipeline

def predict_patient(patient_data: Union[dict, list, pd.DataFrame]) -> Tuple[Union[int, np.ndarray], Union[float, np.ndarray]]:
    """
    يستقبل بيانات مريض واحد (من الواجهة الرئيسية) أو دفعة مرضى (من تبويب الإحصائيات).
    """
    # 1. جلب النموذج السريع من الذاكرة
    pipeline = get_trained_model()
    
    # 2. تحويل البيانات إلى DataFrame بذكاء
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, list):
        patient_df = pd.DataFrame(patient_data)
    elif isinstance(patient_data, pd.DataFrame):
        patient_df = patient_data
    else:
        raise ValueError("صيغة البيانات غير مدعومة. يرجى إرسال dict أو list أو DataFrame.")
    
    # 3. حساب الاحتماليات
    risk_probabilities = pipeline.predict_proba(patient_df)[:, 1]
    
    # 4. إرجاع النتيجة بناءً على عدد المرضى
    if len(patient_df) == 1:
        # إذا كان مريضاً واحداً -> نرجع رقم مفرد (Float) لتناسب شاشة الواجهة
        prob = float(risk_probabilities[0])
        base_prediction = 1 if prob >= 0.50 else 0
        return base_prediction, prob
    else:
        # إذا كانت مجموعة مرضى -> نرجع مصفوفة كاملة (Array) لرسم الإحصائيات
        preds = (risk_probabilities >= 0.50).astype(int)
        return preds, risk_probabilities

if __name__ == "__main__":
    # اختبار سريع للتأكد من المزامنة
    mock_patient = {
        "Age": 50, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 120,
        "Cholesterol": 200, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Flat"
    }
    
    print("--- الاختبار الأول (سيطول قليلاً للتدريب) ---")
    pred, prob = predict_patient(mock_patient)
    print(f"Test Probability: {prob:.2%}")
    
    print("\n--- الاختبار الثاني (سيكون لحظياً بفضل الكاش) ---")
    pred, prob = predict_patient(mock_patient)
    print(f"Test Probability: {prob:.2%}")