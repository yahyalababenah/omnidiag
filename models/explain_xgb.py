import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. تحميل البيانات والموديل
df = pd.read_csv("data/processed/data_heuristic.csv")
target_col = 'HeartDisease'
X = df.drop(columns=[target_col])
model = joblib.load("models/omni_diag_xgb_optimized.pkl")

# 2. إنشاء المفسر (Explainer)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 3. عرض الرسوم البيانية
# أ. أهم الميزات (Feature Importance)
plt.title("أهم الميزات التي تؤثر على قرار OmniDiag")
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.show()

# ب. تأثير الميزات (Bee Swarm Plot)
# يوضح كيف تزيد أو تنقص كل ميزة من احتمالية الإصابة
shap.summary_plot(shap_values, X)