import sys
import pandas as pd
from pathlib import Path
import streamlit as st

# 1. إعداد المسارات للربط مع المحرك الخلفي (Backend)
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

# استدعاء دالة الاستنتاج من ملف predict.py
from src.heart_disease.modeling.predict import predict_patient

# 2. إعدادات الصفحة
st.set_page_config(
    page_title="نظام التشخيص الطبي لأمراض القلب",
    page_icon="🩺",
    layout="wide"
)

# 3. حقن CSS (تصميمك السريري الاحترافي)
st.markdown("""
    <style>
    /* تكبير الخطوط في كامل التطبيق */
    html, body, [class*="css"] { font-size: 1.15rem !important; }
    
    /* تغيير لون الأزرار إلى الأزرق الطبي */
    .stButton>button {
        background-color: #005b96;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover { background-color: #004370; }
    
    /* تخصيص صناديق المقاييس */
    [data-testid="metric-container"] {
        background-color: #eaf4fc !important;
        border-right: 5px solid #005b96 !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }
    
    /* إجبار كافة النصوص داخل الصندوق على اللون الكحلي الداكن */
    [data-testid="metric-container"] * { color: #002244 !important; }
    
    /* تلوين العناوين */
    h1, h2, h3 { color: #005b96 !important; }
    </style>
    """, unsafe_allow_html=True)

# 4. الواجهة الرئيسية
st.title("🩺 السجل الطبي: تقييم مخاطر أمراض القلب (V2)")
st.markdown("يرجى إدخال القيم المخبرية والسريرية بدقة في الحقول المخصصة أدناه.")

# الحاوية الرئيسية للنموذج
with st.container():
    st.subheader("البيانات السريرية للمريض")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        age = st.number_input("العمر (سنوات)", min_value=20, max_value=100, value=50, step=1)
        sex_ar = st.selectbox("الجنس", ["أنثى", "ذكر"], index=1)
        cp_ar = st.selectbox("نوع ألم الصدر", ["نموذجي", "غير نمطي", "بدون ألم", "إقفاري"])
        trestbps = st.number_input("ضغط الدم الانقباضي (الراحة)", min_value=80, max_value=220, value=120, step=1)

    with c2:
        chol = st.number_input("الكولسترول الكلي (mg/dl)", min_value=100, max_value=600, value=200, step=1)
        fbs_ar = st.selectbox("سكر صائم ≥ 120 mg/dl", ["لا", "نعم"], index=0)
        restecg_ar = st.selectbox("نتائج تخطيط القلب", ["طبيعي", "شذوذ ST-T", "تضخم البطين الأيسر"])
        thalach = st.number_input("أقصى نبض تم الوصول إليه", min_value=60, max_value=220, value=150, step=1)

    with c3:
        exang_ar = st.selectbox("ألم صدري ناتج عن مجهود", ["لا", "نعم"], index=0)
        oldpeak = st.number_input("انخفاض ST (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f")
        slope_ar = st.selectbox("ميل مقطع ST", ["هابط", "مسطّح", "صاعد"], index=1)

st.markdown("---")

col_action1, col_action2 = st.columns([1, 3])
with col_action1:
    analyze_btn = st.button("استخراج التقرير الطبي", use_container_width=True)
with col_action2:
    thr = st.number_input("عتبة القرار السريري (Threshold)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

# 5. معالجة البيانات والتشخيص
if analyze_btn:
    
    # تحويل الاختيارات العربية إلى النصوص الدقيقة التي يفهمها الـ Pipeline
    # لا نستخدم أرقام (0, 1) للـ Categories، لأن OneHotEncoder يتوقع النصوص الأصلية
    ui_to_model_map = {
        'sex': {"ذكر": "M", "أنثى": "F"},
        'cp': {"نموذجي": "TA", "غير نمطي": "ATA", "بدون ألم": "ASY", "إقفاري": "NAP"},
        'fbs': {"لا": 0, "نعم": 1}, # هذا رقمي في البيانات الأصلية
        'restecg': {"طبيعي": "Normal", "شذوذ ST-T": "ST", "تضخم البطين الأيسر": "LVH"},
        'exang': {"لا": "N", "نعم": "Y"},
        'slope': {"صاعد": "Up", "مسطّح": "Flat", "هابط": "Down"}
    }
    
    # بناء قاموس متطابق 100% مع أسماء أعمدة V2
    patient_data = {
        "Age": age,
        "Sex": ui_to_model_map['sex'][sex_ar],
        "ChestPainType": ui_to_model_map['cp'][cp_ar],
        "RestingBP": trestbps,
        "Cholesterol": chol,
        "FastingBS": ui_to_model_map['fbs'][fbs_ar],
        "RestingECG": ui_to_model_map['restecg'][restecg_ar],
        "MaxHR": thalach,
        "ExerciseAngina": ui_to_model_map['exang'][exang_ar],
        "Oldpeak": oldpeak,
        "ST_Slope": ui_to_model_map['slope'][slope_ar]
    }
    
    try:
        # إرسال البيانات للمحرك الخلفي واستقبال النتائج
        _, risk_probability = predict_patient(patient_data)
        
        # تطبيق عتبة القرار السريري (Threshold) الخاصة بك
        is_risk = risk_probability >= thr
        
        st.markdown("### 📋 التقرير التشخيصي")
        r1, r2 = st.columns([1, 2])
        
        with r1:
            st.metric("احتمالية الخطر المئوية", f"{risk_probability:.1%}")
            
        with r2:
            if is_risk:
                st.error(f"تنبيه: احتمالية الإصابة تجاوزت العتبة السريرية المحددة ({thr:.0%}). يُنصح بتحويل المريض لتقييم قلبي شامل.")
            else:
                st.success(f"تقييم النظام: المؤشرات دون عتبة الخطر ({thr:.0%}). لا توجد علامات خطورة حادة حالياً.")
                
    except Exception as e:
        st.error(f"حدث خطأ في النظام الخلفي: {str(e)}")