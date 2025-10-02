import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

base = Path(__file__).resolve().parents[1]
data_path = base / "data" / "heart.csv"
df = pd.read_csv(data_path)
y = df["target"]
X = df.drop(columns=["target"])
X_num = X.select_dtypes(include=[np.number]).copy()
X_num = X_num.fillna(X_num.median())
sc = StandardScaler()
Z = sc.fit_transform(X_num.values)
pca = PCA(n_components=2, random_state=42)
Z2 = pca.fit_transform(Z)
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(Z)
res = pd.DataFrame({"pc1":Z2[:,0],"pc2":Z2[:,1],"cluster":labels,"target":y.values})
results_dir = base / "results"
results_dir.mkdir(exist_ok=True)
res.to_csv(results_dir / "kmeans_summary.csv", index=False)
plt.figure()
plt.scatter(res["pc1"], res["pc2"], c=res["cluster"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(results_dir / "pca_scatter.png", dpi=160)