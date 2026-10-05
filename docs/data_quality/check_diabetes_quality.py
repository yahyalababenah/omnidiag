import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
CY=sorted(os.listdir("nhanes_raw")); CY=[c for c in CY if c[0]=="2"]
WANT={"demographics/demographics.csv":["SEQN","RIDSTATR","RIAGENDR","RIDAGEYR","RIDEXPRG","RIDRETH3","WTMEC2YR"],
"examination/body_measures.csv":["SEQN","BMXBMI","BMXWAIST","BMXHT","BMXWT"],
"examination/blood_pressure.csv":["SEQN","BPXPLS"]+[f"BPXSY{i}" for i in (1,2,3)]+[f"BPXDI{i}" for i in (1,2,3)],
"laboratory/glycohemoglobin.csv":None,"laboratory/standard_biochem_profile.csv":None,
"laboratory/cholestrol_hdl.csv":["SEQN","LBDHDD"],
"questionnaire/diabetes.csv":["SEQN","DIQ010","DIQ050","DIQ070"],
"questionnaire/medical_conditions.csv":["SEQN","MCQ300C","MCQ160B","MCQ160C","MCQ160E","MCQ160F"],
"questionnaire/physical_activity.csv":["SEQN","PAQ650","PAQ665"]}
print("## per-file integrity (all 6 cycles)")
rows=[]; parts={}
for p,cols in WANT.items():
    dfs=[]
    for c in CY:
        fp=f"nhanes_raw/{c}/{p}"
        if not os.path.exists(fp): rows.append((p,c,"MISSING FILE"));continue
        d=pd.read_csv(fp,encoding="latin1"); 
        if cols: d=d[[x for x in cols if x in d.columns]]
        d["cycle"]=c; dfs.append(d)
        if cols and len(d.columns)-1<len(cols): rows.append((p,c,"missing cols "+str(set(cols)-set(d.columns))))
    parts[p]=pd.concat(dfs)
    a=parts[p]; print(f"{p:45s} rows={len(a):6d} dupSEQN={a.duplicated(['SEQN']).sum()} exactdups={a.drop(columns='cycle').duplicated().sum()}")
print("anomalies:",rows)
df=parts["demographics/demographics.csv"]
for p in list(WANT)[1:]:
    d=parts[p].drop(columns="cycle"); df=df.merge(d.loc[:,~d.columns.duplicated()],on="SEQN",how="left",suffixes=("","_dup"))
df=df.loc[:,~df.columns.str.endswith("_dup")]
print("\n## BEFORE: merged raw",df.shape,"dup SEQN",df.SEQN.duplicated().sum())
print("participants per cycle",df.cycle.value_counts().sort_index().to_dict())
na=(df.isna().mean()*100).round(1); print("missing% (raw merged, all participants):");print(na[na>0].sort_values(ascending=False).to_string())
# cohort flow
f=[("all",len(df))]
d=df[df.RIDSTATR==2]; f.append(("MEC examined",len(d)))
d=d[d.RIDAGEYR>=20]; f.append(("age>=20",len(d)))
d=d[d.RIDEXPRG.fillna(0)!=1]; f.append(("not pregnant",len(d)))
d=d[d.LBXGH.notna()]; f.append(("HbA1c present",len(d)))
diag=d.DIQ010.eq(1); meds=d.DIQ050.eq(1)|d.DIQ070.eq(1)
f.append(("diagnosed or treated (excluded)",int((diag|meds).sum())))
D=d[~(diag|meds)].copy(); f.append(("FINAL cohort",len(D)))
print("\ncohort flow:");[print(" ",a,b) for a,b in f]
D["y"]=(D.LBXGH>=5.7).astype(int)
print("prevalence",round(D.y.mean(),4),"per cycle",D.groupby("cycle").y.mean().round(3).to_dict(),"n/cycle",D.cycle.value_counts().sort_index().to_dict())
# raw plausibility
for i in (1,2,3): 
    z=(D[f"BPXDI{i}"]==0).sum(); D.loc[D[f"BPXDI{i}"]==0,f"BPXDI{i}"]=np.nan; print(f"BPXDI{i}==0 -> NaN: {z}")
D["SBP"]=D[[f"BPXSY{i}" for i in (1,2,3)]].mean(axis=1); D["DBP"]=D[[f"BPXDI{i}" for i in (1,2,3)]].mean(axis=1)
D["CVD_ANY"]=D[["MCQ160B","MCQ160C","MCQ160E","MCQ160F"]].eq(1).any(axis=1).astype(float)
for c in ["MCQ300C","PAQ650","PAQ665"]: D[c]=D[c].map({1:1.0,2:0.0})
D["RIAGENDR"]=D.RIAGENDR.map({1:1.0,2:0.0})
print("MCQ300C etc raw codes 7/9 (refused/don't know) were mapped to NaN: yes via map")
F=["RIDAGEYR","RIAGENDR","BMXBMI","BMXWAIST","SBP","DBP","BPXPLS","MCQ300C","CVD_ANY","PAQ650","PAQ665","LBDHDD","LBXSCH","LBXSTR","LBXSATSI","LBXSGTSI","LBXSCR","LBXSBU","LBXSAL","LBXSUA"]
print("\n## final-cohort feature quality (20 features)")
t=pd.DataFrame({"missing%":(D[F].isna().mean()*100).round(2),"min":D[F].min(),"p1":D[F].quantile(.01),"median":D[F].median(),"p99":D[F].quantile(.99),"max":D[F].max(),"skew":D[F].skew().round(1),"uniAUC":[round(max(roc_auc_score(D.y[D[c].notna()],D[c].dropna()),1-roc_auc_score(D.y[D[c].notna()],D[c].dropna())),3) for c in F]})
print(t.to_string())
print("rows complete-case:",int(D[F].notna().all(axis=1).sum()),"of",len(D),f"({D[F].notna().all(axis=1).mean()*100:.1f}%)")
print("rows with >=1 missing:",int(D[F].isna().any(axis=1).sum()),"; missing cells %:",round(D[F].isna().mean().mean()*100,2))
# missingness by cycle for key features
print("\nmissing% by cycle:");print((D.groupby("cycle")[["BMXWAIST","SBP","MCQ300C","LBXSTR","PAQ650","LBDHDD"]].apply(lambda g:g.isna().mean()*100)).round(1).to_string())
# informative missingness
print("\nprevalence when missing vs observed:")
for c in ["BMXWAIST","SBP","LBXSTR","MCQ300C","LBDHDD"]:
    m=D[c].isna(); print(f"  {c}: n_miss={m.sum()} prev_missing={D.y[m].mean():.3f} prev_obs={D.y[~m].mean():.3f}")
# outliers
print("\noutliers (1.5*IQR*3 extreme counts) & clinical impossible:")
lim={"BMXBMI":(12,80),"BMXWAIST":(45,200),"SBP":(70,260),"DBP":(30,150),"BPXPLS":(30,200),"LBDHDD":(5,200),"LBXSCH":(50,700),"LBXSTR":(10,3000),"LBXSCR":(0.2,15),"LBXSAL":(2,6)}
for c,(lo,hi) in lim.items(): print(f"  {c}: outside [{lo},{hi}] = {int(((D[c]<lo)|(D[c]>hi)).sum())}")
for c in ["LBXSTR","LBXSATSI","LBXSGTSI"]:
    q1,q3=D[c].quantile([.25,.75]); print(f"  {c} extreme (>Q3+3IQR): {(D[c]>q3+3*(q3-q1)).sum()} ; max {D[c].max()}")
# drift across cycles
print("\nmean by cycle (harmonisation check):");print(D.groupby("cycle")[["RIDAGEYR","BMXBMI","SBP","LBDHDD","LBXSCH","LBXSTR","LBXSCR","LBXSAL","LBXSUA","LBXSGTSI"]].mean().round(2).to_string())
# correlations
cm=D[F].corr(method="spearman").abs(); pairs=[(a,b,round(cm.loc[a,b],2)) for i,a in enumerate(F) for b in F[i+1:] if cm.loc[a,b]>=0.6]; print("\n|rho|>=0.6:",pairs)
# label ambiguity
print("\nHbA1c within 5.5-5.9 (label-ambiguity band):",f"{((D.LBXGH>=5.5)&(D.LBXGH<=5.9)).mean()*100:.1f}%", "| 5.6-5.8:",f"{((D.LBXGH>=5.6)&(D.LBXGH<=5.8)).mean()*100:.1f}%")
# AFTER pipeline: median imputation + scale on TRAIN (2007-2014) vs test
from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
tr=D[D.cycle<"2015-2016"]; te=D[D.cycle=="2017-2018"]
imp=SimpleImputer(strategy="median").fit(tr[F]); sc=StandardScaler().fit(imp.transform(tr[F]))
Xtr=sc.transform(imp.transform(tr[F])); Xte=sc.transform(imp.transform(te[F]))
print("\nAFTER: train",Xtr.shape,"NaN",int(np.isnan(Xtr).sum()),"| test",Xte.shape,"NaN",int(np.isnan(Xte).sum()))
print("train z-score mean/std max dev:",np.abs(Xtr.mean(0)).max().round(3),np.abs(Xtr.std(0)-1).max().round(3))
print("test shift: max |mean z| =",np.abs(Xte.mean(0)).max().round(2),"on",F[int(np.abs(Xte.mean(0)).argmax())])
print("test z max abs",np.abs(Xte).max().round(1), "train z max abs",np.abs(Xtr).max().round(1))
print("share of imputed cells train:",round(tr[F].isna().mean().mean()*100,2))
# leak after
print("\nleak check: max uniAUC among features =",t.uniAUC.max(), t.uniAUC.idxmax(), "| HbA1c, DIQ010 excluded")
from sklearn.linear_model import LinearRegression
g=D.dropna(subset=["LBXGH"]); Xg=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(g[F])); print("R2 HbA1c from features:",round(LinearRegression().fit(Xg,g.LBXGH).score(Xg,g.LBXGH),3))
D.to_pickle("/tmp/claude-1000/-home-yahia-Desktop-Projects-Heart-Disease-Project/07193457-349a-4a70-8f7c-19043cfa99cd/scratchpad/D.pkl")
