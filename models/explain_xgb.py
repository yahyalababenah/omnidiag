import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

# Load config for config-driven paths
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_path = resolve_path(cfg, "model", "weights_path")
target_col = cfg["disease"]["target_column"]

# 1. تحميل البيانات والموديل
data_path = os.path.join(processed_path, cfg["data"]["heuristic_file"])
print(f"📂 قراءة البيانات من: {data_path}")
print(f"📂 تحميل الموديل من: {model_path}")
df = pd.read_csv(data_path)
X = df.drop(columns=[target_col])
model = joblib.load(model_path)

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