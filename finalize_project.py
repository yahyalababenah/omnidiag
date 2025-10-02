import json, os, subprocess, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

base = Path(__file__).resolve().parent
data_path = base/"data"/"heart.csv"
models_dir = base/"models"
results_dir = base/"results"
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

if not data_path.exists():
    print("missing data/heart.csv")
    sys.exit(1)

def ensure_trained():
    fm = models_dir/"final_model.pkl"
    if fm.exists():
        return
    cmd = [sys.executable, str(base/"src"/"train.py")]
    subprocess.run(cmd, check=True)

def supervised_metrics():
    df = pd.read_csv(data_path)
    y = df["target"]
    X = df.drop(columns=["target"])
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    models = {"logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
              "dt": DecisionTreeClassifier(class_weight="balanced", random_state=42),
              "rf": RandomForestClassifier(class_weight="balanced", random_state=42),
              "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc":"roc_auc","f1": make_scorer(f1_score, zero_division=0),"precision": make_scorer(precision_score, zero_division=0),
               "recall": make_scorer(recall_score, zero_division=0),"accuracy":"accuracy"}
    rows = []
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        s = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        rows.append({"model": name,"roc_auc": float(np.mean(s["test_roc_auc"])),"f1": float(np.mean(s["test_f1"])),
                     "precision": float(np.mean(s["test_precision"])),"recall": float(np.mean(s["test_recall"])),
                     "accuracy": float(np.mean(s["test_accuracy"]))})
    pd.DataFrame(rows).to_csv(results_dir/"supervised_metrics.csv", index=False)

def unsupervised_outputs():
    df = pd.read_csv(data_path)
    y = df["target"]
    Xn = df.drop(columns=["target"]).select_dtypes(include=[int,float]).fillna(df.median(numeric_only=True))
    Z = StandardScaler().fit_transform(Xn.values)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
    labs = km.labels_
    Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
    pd.DataFrame({"pc1":Z2[:,0],"pc2":Z2[:,1],"cluster":labs,"target":y.values}).to_csv(results_dir/"kmeans_summary.csv", index=False)
    plt.figure(); plt.scatter(Z2[:,0], Z2[:,1], c=labs); plt.tight_layout(); plt.savefig(results_dir/"pca_scatter.png", dpi=140)
    L = linkage(Z, method="ward")
    plt.figure(figsize=(10,4)); dendrogram(L, truncate_mode="level", p=5, no_labels=True); plt.tight_layout(); plt.savefig(results_dir/"hierarchical_dendrogram.png", dpi=140)

def gridsearch_rf():
    df = pd.read_csv(data_path)
    y = df["target"]
    X = df.drop(columns=["target"])
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
    param_grid = {"clf__n_estimators":[200,400], "clf__max_depth":[None,10,20], "clf__min_samples_split":[2,5]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    g = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X, y)
    pd.DataFrame(g.cv_results_).to_csv(results_dir/"gridsearch_rf.csv", index=False)
    with open(models_dir/"grid_best.json","w",encoding="utf-8") as f:
        json.dump({"best_score": float(g.best_score_), "best_params": g.best_params_}, f, ensure_ascii=False, indent=2)

def checklist():
    needed = [
        models_dir/"final_model.pkl",
        models_dir/"metadata.json",
        models_dir/"grid_best.json",
        results_dir/"cv_scores.csv",
        results_dir/"metrics.json",
        results_dir/"evaluation_metrics.txt",
        results_dir/"roc_curve.png",
        results_dir/"supervised_metrics.csv",
        results_dir/"kmeans_summary.csv",
        results_dir/"pca_scatter.png",
        results_dir/"hierarchical_dendrogram.png",
        results_dir/"gridsearch_rf.csv",
    ]
    missing = [str(p) for p in needed if not p.exists() or (p.is_file() and p.stat().st_size==0)]
    if missing:
        print("missing:", *missing, sep="\n")
    else:
        print("all_good")

def main():
    ensure_trained()
    supervised_metrics()
    unsupervised_outputs()
    gridsearch_rf()
    checklist()

if __name__ == "__main__":
    main()
