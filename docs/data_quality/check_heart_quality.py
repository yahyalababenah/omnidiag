import pandas as pd, numpy as np
R="data/heart_disease/raw/"; P="data/heart_disease/processed/"
def prof(n,d):
    print(f"\n### {n}: shape={d.shape} dups={d.duplicated().sum()} NaN_total={int(d.isna().sum().sum())}")
    na=d.isna().sum(); print("NaN:",na[na>0].to_dict())
    for c in ["RestingBP","Cholesterol","trestbps","chol"]:
        if c in d: print(c,"zeros:",int((d[c]==0).sum()))
    for c in ["Oldpeak","oldpeak"]:
        if c in d: print(c,"neg:",int((d[c]<0).sum()))
    if "HeartDisease" in d: print("prev:",round(d.HeartDisease.mean(),3))
    if "target" in d: print("prev:",round(d.target.mean(),3))
for n,f,kw in [("raw heart.csv",R+"heart.csv",{}),("raw heart11.csv",R+"heart11.csv",{}),("raw heart.csv.bak",R+"heart.csv.bak",{"sep":"\t"}),("processed UCI by site",P+"uci_heart_by_site.csv",{}),("Z-Alizadeh translated",P+"z_alizadeh_translated.csv",{}),("data_clinical",P+"data_clinical.csv",{}),("data_heuristic",P+"data_heuristic.csv",{})]:
    d=pd.read_csv(f,**kw); prof(n,d)
u=pd.read_csv(P+"uci_heart_by_site.csv")
print("\n--- UCI by site")
g=u.groupby("site")
print(pd.DataFrame({"n":g.size(),"prev":g.HeartDisease.mean().round(3),"chol_NaN%":g.Cholesterol.apply(lambda s:s.isna().mean()*100).round(1),"fbs_NaN%":g.FastingBS.apply(lambda s:s.isna().mean()*100).round(1),"slope_NaN%":g.ST_Slope.apply(lambda s:s.isna().mean()*100).round(1),"bp_NaN%":g.RestingBP.apply(lambda s:s.isna().mean()*100).round(1)}))
print(u.describe().round(1).T[["min","50%","max"]])
for c in ["Sex","ChestPainType","RestingECG","ST_Slope","ExerciseAngina"]: print(c,u[c].value_counts(dropna=False).to_dict())
print("female diseased",((u.Sex=="F")&(u.HeartDisease==1)).sum(),"female total",(u.Sex=="F").sum())
print("dup rows:");print(u[u.duplicated(keep=False)])
print("Age range",u.Age.min(),u.Age.max(),"RestingBP==0",(u.RestingBP==0).sum(),"MaxHR<60",(u.MaxHR<60).sum(),"Chol>500",(u.Cholesterol>500).sum(),"BP>200",(u.RestingBP>200).sum())
# L3 features
L3=["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG"]
print("\nL3 missing cells %:",round(u[L3].isna().mean().mean()*100,2),"; rows with >=1 missing in L3:",int(u[L3].isna().any(axis=1).sum()))
# informative missingness: label rate by missingness
for c in ["Cholesterol","FastingBS"]:
    m=u[c].isna(); print(c,"prev missing vs observed:",round(u[m].HeartDisease.mean(),3),round(u[~m].HeartDisease.mean(),3))
# AFTER: apply the shipped pipeline on the data
import sys; sys.path.insert(0,".")
from backend.heart_glm import stack
enc=stack.encode_for_training(u); print("\nencoded cols:",list(enc.columns)); print("encoded NaN:",enc.isna().sum()[lambda s:s>0].to_dict())
pipe=stack.build_pipeline(); X=enc[stack._NUMERIC+stack._BINARY+stack._CATEGORICAL] if hasattr(stack,'_NUMERIC') else enc
pipe.fit(X,u.HeartDisease)
Xt=pipe.named_steps["prep"].transform(X)
print("AFTER transform shape",Xt.shape,"NaN",int(np.isnan(Xt).sum()),"inf",int(np.isinf(Xt).sum()))
print("imputed numeric ranges: Chol",float(X.Cholesterol.min()),float(X.Cholesterol.max()))
num=pipe.named_steps["prep"].named_transformers_["num"]
sc=num.named_steps["imp"].transform(X[stack._NUMERIC]); print("post-imputation min/max per numeric:",sc.min(0).round(1),sc.max(0).round(1))
fb=pipe.named_steps["prep"].named_transformers_["bin"]
print("Sex_m values:",np.unique(fb.transform(X[stack._BINARY])))
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
p=np.zeros(len(u))
for tr,te in LeaveOneGroupOut().split(X,u.HeartDisease,u.site):
    m=clone(pipe).fit(X.iloc[tr],u.HeartDisease.iloc[tr]); p[te]=m.predict_proba(X.iloc[te])[:,1]
print("LOSO pooled AUC",round(roc_auc_score(u.HeartDisease,p),3))
for s in u.site.unique():
    i=(u.site==s).values; print(s,round(roc_auc_score(u.HeartDisease[i],p[i]),3))
