"""
Z-Alizadeh Sani -> OmniDiag schema translator.
==============================================
Purpose: produce an EXTERNAL VALIDATION cohort only. These 303 patients are
never used for training, threshold selection, or preprocessing fitting.

Source: Z-Alizadeh Sani dataset, Tehran (Rajaie Cardiovascular Center),
303 patients, 56 attributes, label `Cath` in {Cad, Normal} from coronary
angiography.

Every mapping below is one of three kinds, and the kind is recorded per column
so a reviewer can audit it:
    DIRECT      same measurement, same units, rename only
    DERIVED     computed from source columns by a stated, published rule
    UNAVAILABLE no acceptable source column; the feature is dropped, never
                filled with a constant

The last category is the whole point. The previous pipeline wrote
`Oldpeak = 0.0` and `ST_Slope = 'Flat'` for all 303 patients, which invents
data. Here those features are dropped instead, and the external validation is
run with a reduced-feature model trained on the same reduced feature set.
"""
import json

import numpy as np
import pandas as pd

MAPPING = {
    "Age": dict(kind="DIRECT", source="Age", rule="years, identical definition"),
    "Sex": dict(kind="DIRECT", source="Sex",
                rule="'Male'->M, 'Fmale'->F (source misspells 'Female')"),
    "RestingBP": dict(kind="DIRECT", source="BP",
                      rule="systolic blood pressure at rest, mm Hg"),
    "FastingBS": dict(kind="DERIVED", source="FBS",
                      rule="1 if FBS > 120 mg/dl else 0 — the UCI definition. "
                           "NOT the DM column: DM is a recorded diabetes "
                           "diagnosis, which includes treated patients whose "
                           "fasting glucose is currently normal. The two agree "
                           "on only 88.8% of patients."),
    "Cholesterol": dict(kind="DERIVED", source="LDL, HDL, TG",
                        rule="total cholesterol = LDL + HDL + TG/5, the "
                             "Friedewald equation rearranged. Valid only for "
                             "TG < 400 mg/dl; patients above that are set "
                             "missing rather than estimated."),
    "ChestPainType": dict(kind="DERIVED",
                          source="Typical Chest Pain, Atypical, Nonanginal",
                          rule="TA if Typical Chest Pain==1; else ATA if "
                               "Atypical=='Y'; else NAP if Nonanginal=='Y'; "
                               "else ASY (asymptomatic by exclusion)."),
    "RestingECG": dict(kind="DERIVED",
                       source="LVH, St Elevation, St Depression, Tinversion",
                       rule="LVH if LVH=='Y'; else ST if any ST-T abnormality "
                            "(St Elevation, St Depression or T-wave inversion); "
                            "else Normal. Matches UCI restecg 0/1/2."),
    "MaxHR": dict(kind="UNAVAILABLE", source="—",
                  rule="UCI thalach is peak heart rate during an EXERCISE "
                       "stress test. The source only records PR, resting pulse "
                       "(range 50-110 vs 71-202 for thalach). Different "
                       "measurement; not substitutable."),
    "Oldpeak": dict(kind="UNAVAILABLE", source="—",
                    rule="UCI oldpeak is ST depression in mm, a magnitude. The "
                         "source records St Depression as binary present/absent "
                         "only. No magnitude exists to map."),
    "ST_Slope": dict(kind="UNAVAILABLE", source="—",
                     rule="No exercise-ECG ST-segment slope is recorded."),
    "ExerciseAngina": dict(kind="UNAVAILABLE", source="Exertional CP",
                           rule="The column exists but is 'N' for all 303 "
                                "patients — zero variance in the source itself. "
                                "Carries no information."),
}

SHARED = [k for k, v in MAPPING.items() if v["kind"] != "UNAVAILABLE"]
DROPPED = [k for k, v in MAPPING.items() if v["kind"] == "UNAVAILABLE"]


def translate(path="Z-Alizadeh_sani_dataset.xlsx"):
    z = pd.read_excel(path)
    out = pd.DataFrame(index=z.index)

    out["Age"] = z["Age"]
    out["Sex"] = z["Sex"].map({"Male": "M", "Fmale": "F", "Female": "F"})
    out["RestingBP"] = z["BP"]
    out["FastingBS"] = (z["FBS"] > 120).astype(int)

    tc = z["LDL"] + z["HDL"] + z["TG"] / 5.0
    out["Cholesterol"] = np.where(z["TG"] < 400, tc.round(), np.nan)

    out["ChestPainType"] = np.select(
        [z["Typical Chest Pain"] == 1, z["Atypical"] == "Y",
         z["Nonanginal"] == "Y"], ["TA", "ATA", "NAP"], default="ASY")

    st_abnormal = ((z["St Elevation"] == 1) | (z["St Depression"] == 1)
                   | (z["Tinversion"] == 1))
    out["RestingECG"] = np.select([z["LVH"] == "Y", st_abnormal],
                                  ["LVH", "ST"], default="Normal")

    out["HeartDisease"] = z["Cath"].map({"Cad": 1, "Normal": 0})
    out["site"] = "z_alizadeh_tehran"
    return out, z


if __name__ == "__main__":
    df, z = translate()
    df.to_csv("z_alizadeh_translated.csv", index=False)
    json.dump(MAPPING, open("z_alizadeh_mapping.json", "w"), indent=2)

    print("=" * 78)
    print("TRANSLATION MAP")
    print("=" * 78)
    for k, v in MAPPING.items():
        print(f"  {k:<16}{v['kind']:<13}<- {v['source']}")
    print(f"\n  shared with UCI ({len(SHARED)}): {', '.join(SHARED)}")
    print(f"  dropped ({len(DROPPED)}): {', '.join(DROPPED)}")

    print("\n" + "=" * 78)
    print("OUTPUT CHECKS")
    print("=" * 78)
    print(f"  rows: {len(df)}   label: Cad={int(df.HeartDisease.sum())} "
          f"Normal={int((df.HeartDisease == 0).sum())}  "
          f"(published: 216 / 87)")
    print(f"  missing per column:")
    for c in df.columns.drop(["site"]):
        n = df[c].isna().sum()
        print(f"     {c:<16}{n:>4}" + ("   <- TG>=400, Friedewald invalid"
                                       if c == "Cholesterol" and n else ""))
    print(f"\n  constant columns (must be none): "
          f"{[c for c in df.columns.drop('site') if df[c].nunique() <= 1]}")

    print("\n  ChestPainType:", dict(df.ChestPainType.value_counts()))
    print("  RestingECG   :", dict(df.RestingECG.value_counts()))
    print("  FastingBS=1  :", int(df.FastingBS.sum()),
          f"({df.FastingBS.mean()*100:.1f}%)   [DM column would give "
          f"{int(z['DM'].sum())}]")
    overlap = ((z["Typical Chest Pain"] == 1).astype(int)
               + (z["Atypical"] == "Y").astype(int)
               + (z["Nonanginal"] == "Y").astype(int))
    print(f"  chest-pain flags set simultaneously >1: {(overlap > 1).sum()} "
          f"patients (priority order resolves these)")

    print("\n" + "=" * 78)
    print("CLINICAL DIRECTION (label 1 must look like disease)")
    print("=" * 78)
    a, b = df[df.HeartDisease == 1], df[df.HeartDisease == 0]
    for c in ["Age", "RestingBP", "Cholesterol", "FastingBS"]:
        print(f"  {c:<14} no-disease {b[c].mean():7.2f}  ->  "
              f"disease {a[c].mean():7.2f}")
    print("  ChestPainType by class (%):")
    print((pd.crosstab(df.ChestPainType, df.HeartDisease,
                       normalize="columns") * 100).round(1).to_string())
