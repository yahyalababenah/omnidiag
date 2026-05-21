```markdown
# ⚕️ Heart Disease Risk Assessment: A Clinical AI Approach (V2)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Stacking%20Ensemble-success.svg)

## 📌 Executive Summary
This project is a production-ready Machine Learning system designed for the early detection of cardiovascular diseases. Unlike conventional ML projects that strictly optimize for generic `Accuracy`, this system is engineered with a **Clinical-First mindset**. 

By applying **Clinical Threshold Tuning (30%)** on a highly stable **Stacking Ensemble**, the system successfully maximizes **Recall (Sensitivity) to 96%** for the disease class, ensuring that life-threatening false negatives are minimized to the absolute limit.

## 🎯 The Engineering & Clinical Problem
* **The Data Constraint:** The dataset comprises 918 patient records with 11 tabular features. 
* **The Engineering Challenge:** State-of-the-Art (SOTA) algorithms like XGBoost and CatBoost exhibited severe overfitting (memorization) on this limited dataset, struggling to capture meaningful medical generalizations.
* **The Clinical Goal:** In medical diagnostics, a False Negative (sending a sick patient home) is fatal, whereas a False Positive (requesting further tests for a healthy patient) is an acceptable clinical precaution. 

## 🧠 Architectural Decisions & Technical Achievements
1. **Model Selection (Occam's Razor):** Reverted from data-hungry boosting frameworks to a stable, classic **Stacking Ensemble** (`RandomForest`, `GradientBoosting`, `SVC` → `LogisticRegression`). This prevented overfitting and captured the feature space robustly.
2. **Clinical Decision Threshold:** Adjusted the probability decision boundary from the default `50%` to a clinically aggressive `30%`. 
    * *Result:* Recall skyrocketed to **96%**, effectively turning the model into a highly sensitive triage tool.
3. **Robust Data Pipeline:** Implemented `KNNImputer` to handle physiological missing values logically, rather than relying on blunt statistical means.
4. **Production-Ready Modularity:** Transitioned from exploratory Jupyter Notebooks to a clean, modular Python architecture (`src/` backend) with decoupled data processing, training, and prediction modules.
5. **Lazy Loading UI:** Built an interactive Streamlit UI utilizing memory caching (`@st.cache_data` and global variables) for instantaneous patient inference without re-initializing the model.

## 📊 Final Clinical Performance (V2)
Based on the testing set evaluation:
* **Recall (Sensitivity) for Disease Class:** `96%` 🔥 (Primary clinical metric)
* **Overall Accuracy:** `89.67%`
* **F1-Score (Disease):** `91%`

## 🏗️ Project Structure
```text
HEART_DISEASE_PROJECT/
│
├── data/                  # Raw and interim datasets
├── notebooks/             # Exploratory Data Analysis (EDA) & architectural drafts
├── src/heart_disease/     # ⚙️ Production Backend (Modular Code)
│   ├── dataset.py         # Data loading and target splitting
│   ├── features.py        # Preprocessing pipelines (KNN Imputation, Scaling, OHE)
│   ├── modeling/
│   │   ├── train.py       # Orchestrates training and outputs evaluation metrics
│   │   └── predict.py     # Inference engine with Lazy Loading caching
│
├── app.py                 # 🖥️ Interactive Streamlit Clinical UI
└── README.md              # Project documentation

```

## 🚀 How to Run the System

**1. Clone and Setup Environment**

```bash
# Ensure you are in your virtual environment (.venv)
pip install -r requirements.txt

```

**2. Run the Clinical UI (Streamlit)**

```bash
streamlit run app.py

```

*This will launch the interactive web application featuring the Diagnostic Form and the Statistical Data Overview tabs.*

**3. Retrain the Model (Optional)**

```bash
python src/heart_disease/modeling/train.py

```

## 🔮 Future Work (V3)

The next phase of this project will focus on **Explainable AI (XAI)**. While the current system is highly sensitive, doctors need to know *why* a prediction was made. V3 will integrate Deep Learning (TabNet) and `SHAP` values to provide feature-level transparency for every diagnostic decision.

---

*Developed by **Yahya Mohammad Ali Lababneh** | Data Science and Artificial Intelligence*

```

---

### 💡 لماذا هذا الـ README يعتبر "ضربة معلم" في المقابلات؟
1. **القسم الأول (The Engineering & Clinical Problem):** يظهر للمقيّم أنك تفهم "البزنس" أو "المجال الطبي"، وأنك لا تكتب أكواداً عمياء، بل تحل مشكلة حقيقية (تقليل الـ False Negatives).
2. **قسم (Architectural Decisions):** هذا هو الكنز! هنا أنت تشرح بصراحة أنك جربت XGBoost واكتشفت أنه يفعل Overfitting على البيانات الصغيرة، فعدت للـ Stacking وعدلت العتبة. هذا يدل على أنك مهندس براغماتي وواقعي.
3. **توقيعك في النهاية:** يربط المشروع بهويتك وتخصصك بشكل احترافي وأنيق.

بعد أن تحفظ هذا الملف، تأكد من رفعه إلى مستودعك عبر Git (`git add README.md` ثم `commit` و `push`). هل أنت مستعد لنغلق فرع V2 نهائياً وننتقل للمستوى القادم؟

```
