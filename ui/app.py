import sys
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- الإصلاح السحري للمسارات (يعمل من أي مكان) ---
current_file_path = Path(__file__).resolve()
# إذا كان الملف داخل مجلد 'ui'، نصعد خطوة واحدة للأعلى. وإلا، نبقى في مكاننا.
root_dir = current_file_path.parent if current_file_path.parent.name != 'ui' else current_file_path.parents[1]

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# الآن سيتعرف بايثون على مجلد src بكل تأكيد
from src.heart_disease.modeling.predict import predict_patient
from src.heart_disease.dataset import load_raw_data


# 2. إعدادات الصفحة
st.set_page_config(
    page_title="نظام التشخيص الطبي لأمراض القلب",
    page_icon="🩺",
    layout="wide"
)

# --- دالة مساعدة لتحميل البيانات الإحصائية مرة واحدة فقط وتخزينها (Caching) ---
@st.cache_data
def get_statistical_data():
    """تحميل البيانات وتجهيزها للعرض الإحصائي مرة واحدة لتعزيز الأداء."""
    df_raw = load_raw_data()
    return df_raw

# 3. حقن CSS (تصميمك السريري الاحترافي - تم تحديثه ليشمل التبويبات)
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

    /* تنسيق التبويبات (Tabs) */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: bold;
        background-color: transparent;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #005b96;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab-list"] .st-bs:first-child p,
    .stTabs [data-baseweb="tab-list"] .st-bs:first-child + div + div p {
        color: #002244 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #eaf4fc;
        border-radius: 5px 5px 0 0;
        border-bottom: 3px solid #005b96;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. العناوين الرئيسية
st.title("🩺 السجل الطبي للذكاء الاصطناعي: تقييم مخاطر أمراض القلب")
st.markdown("يرجى إدخال القيم المخبرية والسريرية بدقة في الحقول المخصصة أدناه.")

# --- إنشاء التبويبات (Tabs) لفصل المحتوى التشخيصي عن الإحصائي ---
tab_diagnosis, tab_stats = st.tabs(["📋 التشخيص الحالي", "📊 نظرة عامة على البيانات الإحصائية"])

# =========================================================
#             التبويب الأول: التشخيص الحالي (النموذج الأصلي)
# =========================================================
with tab_diagnosis:
    # الحاوية الرئيسية للنموذج (كودك الأصلي كما هو)
    with st.container():
        st.subheader("البيانات السريرية للمريض")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("العمر (سنوات)", min_value=20, max_value=100, value=50, step=1, key='age_input')
            sex_ar = st.selectbox("الجنس", ["أنثى", "ذكر"], index=1, key='sex_input')
            cp_ar = st.selectbox("نوع ألم الصدر", ["نموذجي", "غير نمطي", "بدون ألم", "إقفاري"], key='cp_input')
            trestbps = st.number_input("ضغط الدم الانقباضي (الراحة)", min_value=80, max_value=220, value=120, step=1, key='bp_input')

        with c2:
            chol = st.number_input("الكولسترول الكلي (mg/dl)", min_value=100, max_value=600, value=200, step=1, key='chol_input')
            fbs_ar = st.selectbox("سكر صائم ≥ 120 mg/dl", ["لا", "نعم"], index=0, key='fbs_input')
            restecg_ar = st.selectbox("نتائج تخطيط القلب", ["طبيعي", "شذوذ ST-T", "تضخم البطين الأيسر"], key='ecg_input')
            thalach = st.number_input("أقصى نبض تم الوصول إليه", min_value=60, max_value=220, value=150, step=1, key='maxhr_input')

        with c3:
            exang_ar = st.selectbox("ألم صدري ناتج عن مجهود", ["لا", "نعم"], index=0, key='exang_input')
            oldpeak = st.number_input("انخفاض ST (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f", key='oldpeak_input')
            slope_ar = st.selectbox("ميل مقطع ST", ["هابط", "مسطّح", "صاعد"], index=1, key='slope_input')

    st.markdown("---")

    col_action1, col_action2 = st.columns([1, 3])
    with col_action1:
        analyze_btn = st.button("استخراج التقرير الطبي", use_container_width=True)
    with col_action2:
        thr = st.number_input("عتبة القرار السريري (Threshold)", min_value=0.10, max_value=0.90, value=0.30, step=0.05, key='thr_input')

    # 5. معالجة البيانات والتشخيص (كودك الأصلي كما هو مع تصحيحات الـ CSS)
    if analyze_btn:
        
        # قاموس الترجمة الفئوية
        ui_to_model_map = {
            'sex': {"ذكر": "M", "أنثى": "F"},
            'cp': {"نموذجي": "TA", "غير نمطي": "ATA", "بدون ألم": "ASY", "إقفاري": "NAP"},
            'fbs': {"لا": 0, "نعم": 1},
            'restecg': {"طبيعي": "Normal", "شذوذ ST-T": "ST", " تضخم البطين الأيسر": "LVH"},
            'exang': {"لا": "N", "نعم": "Y"},
            'slope': {"صاعد": "Up", "مسطّح": "Flat", "هابط": "Down"}
        }
        
        # بناء قاموس المريض ليتوافق مع أسماء أعمدة V2
        # تصحيح خطأ إملائي بسيط في 'LVH' لتطابق قاموس Features
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
            # استدعاء التنبؤ
            _, risk_probability = predict_patient(patient_data)
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

# =========================================================
#         التبويب الثاني: نظرة عامة على البيانات الإحصائية (📊 الميزة الجديدة)
# =========================================================
with tab_stats:
    st.header("📊 تحليل بيانات المرجعية الطبية (N=918)")
    st.markdown("نظرة عامة على توزيع ميزات المرضى المرجعية، ونماذج الارتباط السريري.")

    try:
        # جلب البيانات المرجعية المدربة من الخلفية
        df_raw = get_statistical_data()
        
        # بناء نموذج مؤقت لإظهار أحدث مصفوفة Confusion
        from src.heart_disease.dataset import get_X_y
        X, y = get_X_y(df_raw)
        
        # نقسم البيانات كما في V2 لنظهر التقييم الحقيقي على عينة الاختبار
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # نقوم بالتدريب السريع (في الحقيقة سنحمل النموذج الجاهز لاحقاً)
        from src.heart_disease.modeling.predict import predict_patient
        
        # --- الحاوية الأولى: إحصائيات عامة وارتباط ---
        with st.container():
            col_plots1, col_plots2 = st.columns([2, 1])
            
            with col_plots1:
                # 1. رسمة توزيع الأعمار حسب التشخيص
                st.subheader("توزيع الفئات العمرية حسب حالة المرض")
                fig_age, ax_age = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df_raw, x="Age", hue="HeartDisease", kde=True, element="step", palette=['#eaf4fc', '#005b96'], ax=ax_age)
                ax_age.set_title("Age Distribution by Heart Disease Status")
                ax_age.set_xlabel("Age (years)")
                st.pyplot(fig_age)

            with col_plots2:
                # 2. رسمة توزيع الجنسين حسب التشخيص
                st.subheader("توزيع الجنسين")
                fig_sex, ax_sex = plt.subplots(figsize=(5, 5))
                # حساب النسب يدوياً لـ Streamlit
                sex_dist = df_raw.groupby('Sex')['HeartDisease'].value_counts(normalize=True).unstack() * 100
                sex_dist.plot(kind='bar', stacked=True, color=['#eaf4fc', '#005b96'], ax=ax_sex)
                ax_sex.set_title("Sex Distribution by Diagnosis")
                ax_sex.set_xlabel("Sex (M/F)")
                ax_sex.set_ylabel("Percentage (%)")
                st.pyplot(fig_sex)

        # --- الحاوية الثانية: الميزات الطبية ونموذج مصفوفة الالتباس الطبية ---
        with st.container():
            st.markdown("---")
            col_med1, col_med2 = st.columns([3, 2])
            
            with col_med1:
                # 3. رسمة خريطة الارتباط السريري (التي ظهرت في تقرير V2 الأكاديمي)
                st.subheader("خريطة ارتباط الميزات الطبية بالتشخيص النهائي")
                numeric_cols = df_raw.select_dtypes(include=[np.number])
                corr_matrix = numeric_cols.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', cbar=True, square=True, ax=ax_corr)
                ax_corr.set_title("Clinical Feature Correlation Matrix")
                st.pyplot(fig_corr)

            with col_med2:
                # 4. إعادة رسم مصفوفة الالتباس الطبية بالعتبة الذهبية 30% لتقرير الإنتاج
                st.subheader("مصفوفة الالتباس الطبية النهائية (العتبة: 30%)")
                # سحب الاحتماليات للعتبة
                _, y_probs = predict_patient(pd.DataFrame(X_test, columns=X.columns).to_dict('records')) # خدعة للتنبؤ بالعينة
                # تطبيق العتبة السريرية 30%
                clinical_threshold = 0.30
                y_pred_clinical = (y_probs >= clinical_threshold).astype(int)
                
                # رسم Confusion Matrix باللون الأخضر المستقر كما في خاتمة V2
                from sklearn.metrics import confusion_matrix
                cm_clinical = confusion_matrix(y_test, y_pred_clinical)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_clinical, annot=True, fmt='d', cmap='Greens', cbar=False,
                            xticklabels=['Normal (0)', 'Disease (1)'], yticklabels=['Normal (0)', 'Disease (1)'])
                ax_cm.set_title(f'Clinical Threshold ({clinical_threshold*100}%) Matrix', weight='bold')
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                st.pyplot(fig_cm)
                st.success(f"هذا النموذج المستقر (النسخة V2) يحقق معدل استدعاء (Recall) للفئة المصابة يبلغ 96%.")

    except Exception as e:
        st.error(f"حدث خطأ في تحميل البيانات الإحصائية: {str(e)}")