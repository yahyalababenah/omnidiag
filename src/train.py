import json, numpy as np, pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

base = Path(__file__).resolve().parents[1]
df = pd.read_csv(base / "data" / "heart.csv")
y = df["target"]
X = df.drop(columns=["target"])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
numeric = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
categorical = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

models = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "dt": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(class_weight="balanced", random_state=42),
    "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "accuracy": "accuracy"
}
results = []
best_name, best_score, best_pipe = None, -1, None

for name, est in models.items():
    pipe = Pipeline([("pre", pre), ("clf", est)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    row = {"model": name}
    for k,v in scores.items():
        if k.startswith("test_"):
            row[k.replace("test_","mean_")] = float(np.mean(v))
    results.append(row)
    if name != "dummy":
        if row["mean_roc_auc"] > best_score:
            best_score, best_name, best_pipe = row["mean_roc_auc"], name, pipe

if best_name == "rf":
    param_dist = {"clf__n_estimators":[100,200], "clf__max_depth":[None,10,20], "clf__min_samples_split":[2,5]}
elif best_name == "svm":
    param_dist = {"clf__C":[0.1,1,10], "clf__gamma":["scale","auto"]}
elif best_name == "logreg":
    param_dist = {"clf__C":[0.1,1,10], "clf__penalty":["l2"], "clf__solver":["lbfgs"]}
elif best_name == "dt":
    param_dist = {"clf__max_depth":[None,5,10,20], "clf__min_samples_split":[2,5,10]}
else:
    param_dist = {}

if param_dist:
    # استخدام GridSearchCV أأمن هنا لأن بعض النماذج احتمالاتها أقل من 8
    search = GridSearchCV(best_pipe, param_grid=param_dist, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y)
    final_model = search.best_estimator_
    best_cv = float(search.best_score_)
else:
    best_pipe.fit(X, y)
    final_model = best_pipe
    best_cv = float(best_score)

models_dir = base / "models"
results_dir = base / "results"
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

dump(final_model, models_dir / "final_model.pkl")

# تحويل النتائج إلى DataFrame وتصديرها
df_results = pd.DataFrame(results)
df_results.to_csv(results_dir / "cv_scores.csv", index=False)

meta = {
    "numeric_columns": num_cols, 
    "categorical_columns": cat_cols, 
    "all_columns": X.columns.tolist(), 
    "target": "target", 
    "best_model": best_name, 
    "best_cv_roc_auc": best_cv
}

with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

proba = final_model.predict_proba(X)[:, 1]
roc = roc_auc_score(y, proba)
fpr, tpr, thr = roc_curve(y, proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "roc_curve.png", dpi=160)

with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"roc_auc_full_fit": float(roc), "best_model": best_name, "best_cv_roc_auc": best_cv}, f, ensure_ascii=False, indent=2)

# إصلاح تقرير المقاييس ليأخذ أرقام أفضل نموذج فقط بدلاً من متوسط كل النماذج
report_path = results_dir / "evaluation_metrics.txt"
try:
    best_model_row = df_results[df_results["model"] == best_name].iloc[0]
    cols = [c for c in df_results.columns if c.startswith("mean_")]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write("-" * 20 + "\n")
        for c in cols:
            f.write(f"{c}: {best_model_row[c]:.4f}\n")
except Exception as e:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Metrics summary unavailable. Error: {str(e)}\n")
