import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import streamlit as st

st.set_page_config(page_title="توقع خطر أمراض القلب", layout="centered")
base = Path(__file__).resolve().parents[1]
meta_path = base / "models" / "metadata.json"
model_path = base / "models" / "final_model.pkl"

if not meta_path.exists() or not model_path.exists():
    st.error("النموذج غير موجود. شغّل التدريب أولًا: python src/train.py")
    st.stop()

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
model = load(model_path)

labels = {
    "age": "العمر",
    "sex": "الجنس",
    "cp": "نوع ألم الصدر",
    "trestbps": "ضغط الدم الراحة",
    "chol": "الكولسترول",
    "fbs": "سكر صائم ≥ 120 mg/dl",
    "restecg": "تخطيط القلب الراحة",
    "thalach": "أقصى نبض",
    "exang": "ألم صدري بالمجهود",
    "oldpeak": "انخفاض ST",
    "slope": "ميل مقطع ST",
    "ca": "عدد الأوعية المصبوغة",
    "thal": "ثلاسيميا",
}

sex_opts = {"أنثى": 0, "ذكر": 1}
cp_opts = {"نموذجي": 0, "غير نمطي": 1, "بدون ألم": 2, "إقفاري": 3}
fbs_opts = {"< 120": 0, "≥ 120": 1}
restecg_opts = {"طبيعي": 0, "شذوذ موجة ST-T": 1, "تضخم بطين أيسر": 2}
exang_opts = {"لا": 0, "نعم": 1}
slope_opts = {"هابط": 0, "مسطّح": 1, "صاعد": 2}
ca_opts = {"0": 0, "1": 1, "2": 2, "3": 3}
thal_opts = {"طبيعي": 1, "عيب ثابت": 2, "عيب قابل للعكس": 3}

def make_row(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
):
    return pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex_opts[sex],
                "cp": cp_opts[cp],
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs_opts[fbs],
                "restecg": restecg_opts[restecg],
                "thalach": thalach,
                "exang": exang_opts[exang],
                "oldpeak": oldpeak,
                "slope": slope_opts[slope],
                "ca": ca_opts[ca],
                "thal": thal_opts[thal],
            }
        ]
    )

st.title("توقع خطر أمراض القلب")
st.caption("أدخِل القيم أو جرّب أمثلة جاهزة ثم اضغط قيّم.")

with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider(labels["age"], 20, 90, 50, 1)
        trestbps = st.slider(labels["trestbps"], 80, 200, 130, 1)
        chol = st.slider(labels["chol"], 100, 600, 220, 1)
        thalach = st.slider(labels["thalach"], 60, 220, 150, 1)
        oldpeak = st.number_input(labels["oldpeak"], min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    with c2:
        sex = st.selectbox(labels["sex"], list(sex_opts.keys()), index=1)
        cp = st.selectbox(labels["cp"], list(cp_opts.keys()), index=1)
        fbs = st.selectbox(labels["fbs"], list(fbs_opts.keys()), index=0)
        restecg = st.selectbox(labels["restecg"], list(restecg_opts.keys()), index=0)
        exang = st.selectbox(labels["exang"], list(exang_opts.keys()), index=0)
        slope = st.selectbox(labels["slope"], list(slope_opts.keys()), index=1)
        ca = st.selectbox(labels["ca"], list(ca_opts.keys()), index=0)
        thal = st.selectbox(labels["thal"], list(thal_opts.keys()), index=2)

    thr = st.slider("عتبة القرار", 0.05, 0.95, 0.50, 0.01)
    col_a, col_b, col_c = st.columns([1,1,2])
    go = col_a.form_submit_button("قيّم")
    demo_low = col_b.form_submit_button("مثال منخفض")
    demo_high = col_c.form_submit_button("مثال مرتفع")

if demo_low:
    X = make_row(45, "أنثى", "بدون ألم", 120, 200, "< 120", "طبيعي", 170, "لا", 0.0, "صاعد", "0", "طبيعي")
    p = model.predict_proba(X)[0, 1]
    pred = int(p >= thr)
    st.metric("احتمال الخطر", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.write("النتيجة:", "معرّض للخطر" if pred == 1 else "منخفض الخطر")

elif demo_high:
    X = make_row(62, "ذكر", "إقفاري", 160, 300, "≥ 120", "شذوذ موجة ST-T", 110, "نعم", 2.5, "مسطّح", "2", "عيب قابل للعكس")
    p = model.predict_proba(X)[0, 1]
    pred = int(p >= thr)
    st.metric("احتمال الخطر", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.write("النتيجة:", "معرّض للخطر" if pred == 1 else "منخفض الخطر")

elif go:
    X = make_row(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    p = model.predict_proba(X)[0, 1]
    pred = int(p >= thr)
    st.metric("احتمال الخطر", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.write("النتيجة:", "معرّض للخطر" if pred == 1 else "منخفض الخطر")
