"""Generate demo CSVs for the batch-prediction page (synthetic, seeded)."""
import csv, random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "sample_batch"
rng = random.Random(42)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def heart_row():
    age = int(clip(rng.gauss(55, 10), 29, 80))
    sex = rng.choices(["M", "F"], [0.7, 0.3])[0]
    sick = rng.random() < 0.45
    cp = rng.choices(["ASY", "NAP", "ATA", "TA"], [0.6, 0.15, 0.15, 0.1] if sick else [0.2, 0.3, 0.4, 0.1])[0]
    return {
        "Age": age, "Sex": sex, "ChestPainType": cp,
        "RestingBP": int(clip(rng.gauss(132, 17), 90, 200)),
        "Cholesterol": int(clip(rng.gauss(245, 50), 130, 500)),
        "FastingBS": int(rng.random() < (0.25 if sick else 0.1)),
        "RestingECG": rng.choices(["Normal", "ST", "LVH"], [0.6, 0.2, 0.2])[0],
        "MaxHR": int(clip(rng.gauss(130 if sick else 152, 20), 70, 200)),
        "ExerciseAngina": "Y" if rng.random() < (0.55 if sick else 0.12) else "N",
        "Oldpeak": round(clip(rng.gauss(1.6 if sick else 0.5, 1.0), 0, 6), 1),
        "ST_Slope": rng.choices(["Flat", "Down", "Up"], [0.6, 0.2, 0.2] if sick else [0.2, 0.05, 0.75])[0],
    }


def nhanes_row():
    """NHANES dysglycaemia module (configs/diabetes_nhanes.yaml): values inside
    the schema bounds, loosely correlated so both decisions occur."""
    hi = rng.random() < 0.45
    age = int(clip(rng.gauss(58 if hi else 40, 12), 20, 80))
    bmi = round(clip(rng.gauss(32 if hi else 25, 5), 16, 60), 1)
    band = "high" if bmi >= 32 else "increased" if bmi >= 26 else "normal"
    return {
        "RIDAGEYR": age, "RIAGENDR": rng.randint(0, 1), "BMXBMI": bmi, "ADIPOSITY_BAND": band,
        "SBP": int(clip(rng.gauss(138 if hi else 118, 15), 90, 200)),
        "DBP": int(clip(rng.gauss(82 if hi else 72, 10), 50, 120)),
        "BPXPLS": int(clip(rng.gauss(74, 10), 45, 120)),
        "MCQ300C": int(rng.random() < (0.45 if hi else 0.2)),
        "CVD_ANY": int(rng.random() < (0.15 if hi else 0.03)),
        "PAQ650": int(rng.random() < (0.15 if hi else 0.35)),
        "PAQ665": int(rng.random() < (0.35 if hi else 0.6)),
        "LBDHDD": int(clip(rng.gauss(42 if hi else 56, 12), 20, 110)),
        "LBXSCH": int(clip(rng.gauss(200, 38), 110, 340)),
        "LBXSTR": int(clip(rng.gauss(190 if hi else 110, 60), 40, 600)),
        "LBXSATSI": int(clip(rng.gauss(30 if hi else 22, 10), 8, 120)),
        "LBXSGTSI": int(clip(rng.gauss(40 if hi else 24, 18), 8, 200)),
        "LBXSCR": round(clip(rng.gauss(0.9, 0.2), 0.4, 2.0), 2),
        "LBXSBU": int(clip(rng.gauss(14, 4), 5, 40)),
        "LBXSAL": round(clip(rng.gauss(4.3, 0.3), 3.2, 5.2), 1),
        "LBXSUA": round(clip(rng.gauss(6.2 if hi else 5.2, 1.2), 2.5, 10.0), 1),
    }


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# Heart: 100 rows; 6 with missing optional values (valid), 3 invalid (error rows)
heart = [heart_row() for _ in range(100)]
for i in (4, 17, 33, 58, 71, 90):
    heart[i]["Cholesterol"] = ""
    if i % 2:
        heart[i]["Oldpeak"] = ""
heart[10]["Age"] = 150          # out of range
heart[45]["ChestPainType"] = "XYZ"  # invalid category
heart[80]["Sex"] = ""           # required field missing
write("heart_disease_batch_sample.csv", heart)

# NHANES dysglycaemia: the module cap (max_batch_rows) is 100 rows. 6 with
# missing optional values (valid), 3 invalid (error rows). The BRFSS sample
# (diabetes_batch_sample.csv) went with that module in gate B7.
nhanes = [nhanes_row() for _ in range(100)]
for i in (3, 21, 39, 52, 68, 84):
    nhanes[i]["LBXSTR"] = ""
    if i % 2:
        nhanes[i]["SBP"] = nhanes[i]["DBP"] = ""
nhanes[12]["RIDAGEYR"] = 15          # below the adult cohort (20-80)
nhanes[47]["ADIPOSITY_BAND"] = "XL"  # invalid category
nhanes[77]["LBDHDD"] = ""            # required field missing
write("diabetes_nhanes_batch_sample.csv", nhanes)
print("ok")
