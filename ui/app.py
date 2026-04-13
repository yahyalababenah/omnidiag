import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import streamlit as st

# 1. إعدادات الصفحة والسمة العامة
st.set_page_config(
    page_title="نظام التشخيص الذكي لأمراض القلب",
    page_icon="❤️",
    layout="wide" # استخدام العرض الكامل للشاشة
)

# 2. تحميل النموذج والبيانات الوصفية مع التخزين المؤقت
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

# 3. واجهة الشريط الجانبي (Sidebar) للمعلومات الفنية
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png")
    st.title("معلومات النظام")
    if meta:
        st.info(f"🏆 أفضل نموذج: {meta['best_model'].upper()}")
        st.metric("دقة النموذج (AUC)", f"{meta['best_cv_roc_auc']:.1%}")
    st.markdown("---")
    st.write("🔧 **إعدادات التحليل**")
    thr = st.slider("عتبة القرار (Sensitivity)", 0.1, 0.9, 0.5, 0.05, 
                    help="رفع العتبة يجعل النموذج أكثر صرامة في تشخيص الإصابة.")

# 4. العنوان الرئيسي والتبويبات
st.title("🏥 نظام التحليل التنبؤي لمخاطر القلب")
tab1, tab2 = st.tabs(["🔍 إدخال البيانات والتحليل", "📊 حول المشروع"])

with tab1:
    # تنسيق المدخلات في حاوية منظمة
    with st.container():
        st.subheader("📋 البيانات الحيوية للمريض")
        
        # توزيع المدخلات على 3 أعمدة بدلاً من 2 لتقليل الفراغات
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("👤 العمر", 20, 100, 50)
            sex = st.selectbox("🚻 الجنس", ["أنثى", "ذكر"], index=1)
            cp = st.selectbox("💔 نوع ألم الصدر", ["نموذجي", "غير نمطي", "بدون ألم", "إقفاري"])
            trestbps = st.slider("🩸 ضغط الدم (الراحة)", 80, 200, 120)

        with c2:
            chol = st.slider("🧪 الكولسترول", 100, 600, 200)
            fbs = st.radio("🍬 سكر صائم ≥ 120", ["لا", "نعم"], horizontal=True)
            restecg = st.selectbox("📈 تخطيط القلب", ["طبيعي", "شذوذ ST-T", "تضخم البطين"])
            thalach = st.slider("💓 أقصى نبض", 60, 220, 150)

        with c3:
            exang = st.radio("🏃 ألم مجهودي؟", ["لا", "نعم"], horizontal=True)
            oldpeak = st.number_input("📉 انخفاض ST", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("📏 ميل ST", ["هابط", "مسطّح", "صاعد"])
            ca = st.selectbox("🚢 الأوعية المصبوغة", [0, 1, 2, 3])
            thal = st.selectbox("🧬 الثلاسيميا", ["طبيعي", "عيب ثابت", "عيب قابل للعكس"])

    st.markdown("---")
    
    # أزرار الإجراءات بتنسيق جذاب
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        go = st.button("🚀 بدء التحليل الذكي", use_container_width=True)
    with col_btn2:
        demo_low = st.button("✅ مثال حالة مستقرة", use_container_width=True)
    with col_btn3:
        demo_high = st.button("⚠️ مثال حالة حرجة", use_container_width=True)

# 5. منطق المعالجة وعرض النتائج
def process_and_show(input_row):
    p = model.predict_proba(input_row)[0, 1]
    is_risk = p >= thr
    
    st.markdown("### 🧬 مخرجات التحليل الرقمي")
    
    res_c1, res_c2 = st.columns([1, 2])
    
    with res_c1:
        st.metric("احتمال الإصابة", f"{p:.1%}", delta=f"{p-thr:.1%}", delta_color="inverse")
    
    with res_c2:
        if is_risk:
            st.error(f"⚠️ النتيجة: خطر مرتفع ({p:.1%})")
            st.write("💡 **توصية:** يرجى مراجعة طبيب المختص لإجراء فحوصات معمقة.")
        else:
            st.success(f"✅ النتيجة: خطر منخفض ({p:.1%})")
            st.write("💡 **توصية:** المؤشرات الحيوية ضمن النطاق الآمن حالياً.")
    
    st.progress(p)

# استدعاء الدوال بناءً على الأزرار
if go:
    # تحويل المدخلات لـ DataFrame (استخدم دالة make_row الأصلية هنا)
    pass 
elif demo_low:
    # محاكاة بيانات الحالة المنخفضة
    pass
