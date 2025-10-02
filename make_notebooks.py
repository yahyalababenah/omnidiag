import os, json, nbformat as nbf
from pathlib import Path

root = Path(__file__).resolve().parent
data_csv = root / "data" / "heart.csv"
notebooks = root / "notebooks"
results = root / "results"
notebooks.mkdir(exist_ok=True)
results.mkdir(exist_ok=True)

def nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(c) for c in cells]
    return nb

nb1 = nb([
f"import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\ndf=pd.read_csv(r'{data_csv.as_posix()}')\ndf.head()",
"df.describe(include='all')",
"plt.figure(); df.hist(figsize=(12,10)); plt.tight_layout(); plt.savefig('../results/eda_hist.png', dpi=140)",
"plt.figure(figsize=(9,7)); sns.heatmap(df.corr(numeric_only=True), annot=False); plt.tight_layout(); plt.savefig('../results/eda_corr.png', dpi=140)"
])
nbf.write(nb1, open(notebooks/'01_data_preprocessing.ipynb','w',encoding='utf-8'))

nb2 = nb([
"import pandas as pd, numpy as np, matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\ndf=pd.read_csv('../data/heart.csv')\nX=df.drop(columns=['target']).select_dtypes(include=[int,float]).fillna(df.median(numeric_only=True))\nZ=StandardScaler().fit_transform(X)\npca=PCA().fit(Z)\nimport numpy as np\ncum=np.cumsum(pca.explained_variance_ratio_)\nplt.figure(); plt.plot(range(1,len(cum)+1), cum, marker='o'); plt.xlabel('n_components'); plt.ylabel('cumulative explained variance'); plt.grid(True); plt.tight_layout(); plt.savefig('../results/pca_cumulative.png', dpi=140)"
])
nbf.write(nb2, open(notebooks/'02_pca_analysis.ipynb','w',encoding='utf-8'))

nb3 = nb([
"import pandas as pd, numpy as np\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.feature_selection import RFE, chi2, SelectKBest\nfrom sklearn.linear_model import LogisticRegression\ndf=pd.read_csv('../data/heart.csv')\ny=df['target']\nx=df.drop(columns=['target'])\nnum=x.select_dtypes(include=[int,float]).columns.tolist()\ncat=[c for c in x.columns if c not in num]\npre_num=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())])\npre_cat=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))])\npre=ColumnTransformer([('num',pre_num,num),('cat',pre_cat,cat)])\nest=LogisticRegression(max_iter=1000,class_weight='balanced')\npipe=Pipeline([('pre',pre),('clf',est)]).fit(x,y)\nrfe=RFE(estimator=LogisticRegression(max_iter=1000,class_weight='balanced'),n_features_to_select=8)\nrfe=rfe.fit(pre.fit_transform(x),y)\noh=pipe.named_steps['pre'].named_transformers_['cat'].named_steps['oh'] if len(cat)>0 else None\noh_names=list(oh.get_feature_names_out(cat)) if oh is not None else []\nfeat=num+oh_names\nimport pandas as pd\npd.DataFrame({'feature':feat,'selected':rfe.get_support()}).to_csv('../results/feature_rfe.csv',index=False)\nfrom sklearn.preprocessing import MinMaxScaler\nnumX=df[num].fillna(df[num].median())\nchiX=MinMaxScaler().fit_transform(numX)\nfrom sklearn.feature_selection import SelectKBest\nsel=SelectKBest(chi2,k=min(8,chiX.shape[1])).fit(chiX,y)\npd.DataFrame({'feature':num,'selected':sel.get_support()}).to_csv('../results/feature_chi2.csv',index=False)"
])
nbf.write(nb3, open(notebooks/'03_feature_selection.ipynb','w',encoding='utf-8'))

nb4 = nb([
"import json, numpy as np, pandas as pd\nfrom sklearn.model_selection import StratifiedKFold, cross_validate\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import make_scorer, f1_score, precision_score, recall_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\ndf=pd.read_csv('../data/heart.csv')\ny=df['target']\nx=df.drop(columns=['target'])\nnum=x.select_dtypes(include=[int,float]).columns.tolist()\ncat=[c for c in x.columns if c not in num]\npre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])\nmodels={'logreg':LogisticRegression(max_iter=1000,class_weight='balanced'),'dt':DecisionTreeClassifier(class_weight='balanced',random_state=42),'rf':RandomForestClassifier(class_weight='balanced',random_state=42),'svm':SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=42)}\ncv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)\nscoring={'roc_auc':'roc_auc','f1':make_scorer(f1_score,zero_division=0),'precision':make_scorer(precision_score,zero_division=0),'recall':make_scorer(recall_score,zero_division=0),'accuracy':'accuracy'}\nrows=[]\nfor name,est in models.items():\n    pipe=Pipeline([('pre',pre),('clf',est)])\n    s=cross_validate(pipe,x,y,cv=cv,scoring=scoring)\n    rows.append({'model':name,'roc_auc':float(np.mean(s['test_roc_auc'])),'f1':float(np.mean(s['test_f1'])),'precision':float(np.mean(s['test_precision'])),'recall':float(np.mean(s['test_recall'])),'accuracy':float(np.mean(s['test_accuracy']))})\npd.DataFrame(rows).to_csv('../results/supervised_metrics.csv',index=False)\nprint(pd.read_csv('../results/supervised_metrics.csv'))"
])
nbf.write(nb4, open(notebooks/'04_supervised_learning.ipynb','w',encoding='utf-8'))

nb5 = nb([
"import pandas as pd, numpy as np, matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.cluster import KMeans\nfrom scipy.cluster.hierarchy import linkage, dendrogram\ndf=pd.read_csv('../data/heart.csv')\ny=df['target']\nx=df.drop(columns=['target']).select_dtypes(include=[int,float]).fillna(df.median(numeric_only=True))\nZ=StandardScaler().fit_transform(x)\nkm=KMeans(n_clusters=3,n_init=10,random_state=42).fit(Z)\nlabels=km.labels_\npca=PCA(n_components=2,random_state=42).fit_transform(Z)\nimport pandas as pd\npd.DataFrame({'pc1':pca[:,0],'pc2':pca[:,1],'cluster':labels,'target':y}).to_csv('../results/kmeans_summary.csv',index=False)\nplt.figure(); plt.scatter(pca[:,0],pca[:,1],c=labels); plt.tight_layout(); plt.savefig('../results/pca_scatter.png',dpi=140)\nL=linkage(Z,method='ward')\nplt.figure(figsize=(10,4)); dendrogram(L,truncate_mode='level',p=5,no_labels=True); plt.tight_layout(); plt.savefig('../results/hierarchical_dendrogram.png',dpi=140)"
])
nbf.write(nb5, open(notebooks/'05_unsupervised_learning.ipynb','w',encoding='utf-8'))

nb6 = nb([
"import numpy as np, pandas as pd, json\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.ensemble import RandomForestClassifier\ndf=pd.read_csv('../data/heart.csv')\ny=df['target']\nx=df.drop(columns=['target'])\nnum=x.select_dtypes(include=[int,float]).columns.tolist()\ncat=[c for c in x.columns if c not in num]\npre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])\nest=RandomForestClassifier(class_weight='balanced',random_state=42)\npipe=Pipeline([('pre',pre),('clf',est)])\nparam_grid={'clf__n_estimators':[200,400],'clf__max_depth':[None,10,20],'clf__min_samples_split':[2,5]}\ncv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)\ng=GridSearchCV(pipe,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1).fit(x,y)\npd.DataFrame(g.cv_results_).to_csv('../results/gridsearch_rf.csv',index=False)\nopen('../models/grid_best.json','w',encoding='utf-8').write(json.dumps({'best_score':float(g.best_score_),'best_params':g.best_params_},ensure_ascii=False,indent=2))"
])
nbf.write(nb6, open(notebooks/'06_hyperparameter_tuning.ipynb','w',encoding='utf-8'))

print("Notebooks created in:", notebooks.as_posix())
