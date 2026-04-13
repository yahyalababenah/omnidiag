import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import streamlit as st

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="نظام التشخيص الطبي لأمراض القلب",
    page_icon="",
    layout="wide"
)

# 2. حقن CSS لتكبير الخطوط وإعطاء طابع طبي (أزرق سريري)
# 2. حقن CSS لتكبير الخطوط وإعطاء طابع طبي (مع إصلاح الألوان)
st.markdown("""
    <style>
    /* تكبير الخطوط في كامل التطبيق */
    html, body, [class*="css"] {
        font-size: 1.15rem !important; 
    }
    /* تغيير لون الأزرار إلى الأزرق الطبي */
    .stButton>button {
        background-color: #005b96;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #004370;
    }
    /* تخصيص صناديق المقاييس وإجبار لون النص على التباين */
    [data-testid="metric-container"] {
        background-color: #eaf4fc !important;
        border-right: 5px solid #005b96 !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }
    /* إجبار كافة النصوص داخل الصندوق على اللون الكحلي الداكن */
    [data-testid="metric-container"] * {
        color: #002244 !important; 
    }
    /* تلوين العناوين */
    h1, h2, h3 {
        color: #005b96 !important;
    }
    </style>
    """, unsafe_allow_html=True)
@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parents[1]
    meta_path = base / "models" / "metadata.json"
    model_path = base / "models" / "final_model.pkl"
    if not meta_path.exists() or not model_path.exists():
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load(model_path)
    return meta, model

meta, model = load_model()

if model is None:
    st.error(" النموذج غير موجود. يرجى تشغيل التدريب أولاً.")
    st.stop()

# 3. الواجهة الرئيسية
st.title("🩺 السجل الطبي: تقييم مخاطر أمراض القلب")
st.markdown("يرجى إدخال القيم المخبرية والسريرية بدقة في الحقول المخصصة أدناه.")

# الحاوية الرئيسية للنموذج
with st.container():
    st.subheader(" البيانات السريرية للمريض")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # إدخال يدوي دقيق بدلاً من Sliders
        age = st.number_input("العمر (سنوات)", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("الجنس", ["أنثى", "ذكر"], index=1)
        cp = st.selectbox("نوع ألم الصدر", ["نموذجي", "غير نمطي", "بدون ألم", "إقفاري"])
        trestbps = st.number_input("ضغط الدم الانقباضي (الراحة)", min_value=80, max_value=220, value=120, step=1)

    with c2:
        chol = st.number_input("الكولسترول الكلي (mg/dl)", min_value=100, max_value=600, value=200, step=1)
        fbs = st.selectbox("سكر صائم ≥ 120 mg/dl", ["لا", "نعم"], index=0)
        restecg = st.selectbox("نتائج تخطيط القلب", ["طبيعي", "شذوذ ST-T", "تضخم البطين الأيسر"])
        thalach = st.number_input("أقصى نبض تم الوصول إليه", min_value=60, max_value=220, value=150, step=1)

    with c3:
        exang = st.selectbox("ألم صدري ناتج عن مجهود", ["لا", "نعم"], index=0)
        oldpeak = st.number_input("انخفاض ST (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f")
        slope = st.selectbox("ميل مقطع ST", ["هابط", "مسطّح", "صاعد"], index=1)
        ca = st.number_input("عدد الأوعية المصبوغة", min_value=0, max_value=3, value=0, step=1)
        thal = st.selectbox("نتائج فحص الثلاسيميا", ["طبيعي", "عيب ثابت", "عيب قابل للعكس"], index=2)

st.markdown("---")

col_action1, col_action2 = st.columns([1, 3])
with col_action1:
    analyze_btn = st.button("استخراج التقرير الطبي", use_container_width=True)
with col_action2:
    thr = st.number_input("عتبة القرار السريري (Threshold)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

# 4. معالجة البيانات
if analyze_btn:
    mapping = {"أنثى": 0, "ذكر": 1, "لا": 0, "نعم": 1, 
               "نموذجي": 0, "غير نمطي": 1, "بدون ألم": 2, "إقفاري": 3,
               "طبيعي": 0, "شذوذ ST-T": 1, "تضخم البطين الأيسر": 2,
               "هابط": 0, "مسطّح": 1, "صاعد": 2}
    
    # تصحيح قيم الثلاسيميا حسب الكود الخاص بك
    thal_mapping = {"طبيعي": 1, "عيب ثابت": 2, "عيب قابل للعكس": 3}
    # في بعض النسخ 'طبيعي' للـ ecg تكون 0 أو 1، نعتمد المابينج هنا:
    restecg_val = mapping.get(restecg, 0)
    
    input_df = pd.DataFrame([{
        "age": age,
        "sex": mapping[sex],
        "cp": mapping[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": mapping[fbs],
        "restecg": restecg_val,
        "thalach": thalach,
        "exang": mapping[exang],
        "oldpeak": oldpeak,
        "slope": mapping[slope],
        "ca": ca,
        "thal": thal_mapping[thal]
    }])

    p = model.predict_proba(input_df)[0, 1]
    is_risk = p >= thr
    
    st.markdown("###  التقرير التشخيصي")
    r1, r2 = st.columns([1, 2])
    
    with r1:
        st.metric("احتمالية الخطر المئوية", f"{p:.1%}")
        
    with r2:
        if is_risk:
            st.error(" تقييم النظام: مؤشرات عالية الخطورة. يُنصح بتحويل المريض لتقييم قلبي شامل.")
        else:
            st.success("تقييم النظام: مؤشرات مستقرة. لا توجد علامات خطورة حادة حالياً.")
