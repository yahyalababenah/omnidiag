"""
Build a site-labelled heart-disease dataset from the four original UCI files.

Source files (14-attribute 'processed' versions, mirrored on GitHub from the
UCI Machine Learning Repository, Heart Disease, 1988):
    processed.cleveland.data    Cleveland Clinic Foundation      (Robert Detrano, M.D., Ph.D.)
    processed.hungarian.data    Hungarian Institute of Cardiology (Andras Janosi, M.D.)
    processed.switzerland.data  University Hospital, Zurich       (William Steinbrunn, M.D.)
    processed.va.data           V.A. Medical Center, Long Beach

Mapping to the OmniDiag 11-feature production schema. `ca` and `thal` are
dropped: both require invasive angiography / nuclear imaging and are absent
from every production layer.

Codes (UCI 1-indexed, per heart-disease.names):
    sex     1=male 0=female
    cp      1=typical angina 2=atypical angina 3=non-anginal 4=asymptomatic
    restecg 0=normal 1=ST-T abnormality 2=LV hypertrophy
    exang   1=yes 0=no
    slope   1=upsloping 2=flat 3=downsloping
    num     0=no disease, 1-4=disease  ->  binarised
Missing values are '?' in the source; chol==0 is a documented sentinel for
"not measured" (pervasive at Zurich).
"""
import numpy as np
import pandas as pd

RAW = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
       "exang", "oldpeak", "slope", "ca", "thal", "num"]

FILES = {
    "cleveland": "uci/processed.cleveland.data",
    "hungarian": "uci/processed.hungarian.data",
    "switzerland": "uci/processed.switzerland.data",
    "long_beach_va": "uci/processed.va.data",
}

CP = {1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}
ECG = {0: "Normal", 1: "ST", 2: "LVH"}
SLOPE = {1: "Up", 2: "Flat", 3: "Down"}

frames = []
for site, path in FILES.items():
    # the Hungarian file ships a header line; sniff for it
    first = open(path).readline()
    skip = 1 if "(age)" in first else 0
    d = pd.read_csv(path, header=None, names=RAW, skiprows=skip,
                    na_values=["?", "-9", "-9.0"])
    d = d.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "Age": d["age"],
        "Sex": d["sex"].map({1: "M", 0: "F"}),
        "ChestPainType": d["cp"].map(CP),
        "RestingBP": d["trestbps"].replace(0, np.nan),
        "Cholesterol": d["chol"].replace(0, np.nan),   # 0 = not measured
        "FastingBS": d["fbs"],
        "RestingECG": d["restecg"].map(ECG),
        "MaxHR": d["thalach"],
        "ExerciseAngina": d["exang"].map({1: "Y", 0: "N"}),
        "Oldpeak": d["oldpeak"],
        "ST_Slope": d["slope"].map(SLOPE),
        "HeartDisease": (d["num"] > 0).astype(int),
        "site": site,
    })
    frames.append(out)

df = pd.concat(frames, ignore_index=True)
df.to_csv("uci_heart_by_site.csv", index=False)

print("=" * 78)
print("PER-SITE SUMMARY")
print("=" * 78)
rows = []
for site, g in df.groupby("site", sort=False):
    rows.append(dict(
        site=site, n=len(g),
        disease=int(g.HeartDisease.sum()),
        prevalence=f"{g.HeartDisease.mean()*100:.1f}%",
        male=f"{(g.Sex=='M').mean()*100:.0f}%",
        age=f"{g.Age.mean():.1f}",
    ))
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  TOTAL {len(df)} patients, {int(df.HeartDisease.sum())} with disease "
      f"({df.HeartDisease.mean()*100:.1f}%)")
print(f"  exact duplicate rows: {df.drop(columns='site').duplicated().sum()}")

print("\n" + "=" * 78)
print("MISSINGNESS BY SITE (%)")
print("=" * 78)
feat = [c for c in df.columns if c not in ("HeartDisease", "site")]
miss = df.groupby("site", sort=False)[feat].apply(
    lambda g: (g.isna().mean() * 100).round(1))
print(miss.T.to_string())

print("\n" + "=" * 78)
print("CLINICAL DIRECTION CHECK (label 1 must look like disease everywhere)")
print("=" * 78)
for site, g in df.groupby("site", sort=False):
    a = g[g.HeartDisease == 1]
    b = g[g.HeartDisease == 0]
    print(f"  {site:<14} age {b.Age.mean():5.1f}->{a.Age.mean():5.1f}  "
          f"MaxHR {b.MaxHR.mean():6.1f}->{a.MaxHR.mean():6.1f}  "
          f"Oldpeak {b.Oldpeak.mean():4.2f}->{a.Oldpeak.mean():4.2f}  "
          f"exang-Y {(b.ExerciseAngina=='Y').mean()*100:4.1f}%->"
          f"{(a.ExerciseAngina=='Y').mean()*100:4.1f}%")
print("  (each arrow: no-disease -> disease. Expect age up, MaxHR down,")
print("   Oldpeak up, exercise angina up. Any site breaking this is suspect.)")
