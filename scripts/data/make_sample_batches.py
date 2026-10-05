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


def diabetes_row():
    hi = rng.random() < 0.45
    p = lambda a, b: int(rng.random() < (a if hi else b))
    return {
        "HighBP": p(.7, .25), "HighChol": p(.6, .3), "CholCheck": 1, "BMI": round(clip(rng.gauss(33 if hi else 25, 5), 16, 60), 1),
        "Smoker": p(.5, .4), "Stroke": p(.1, .02), "HeartDiseaseorAttack": p(.25, .05),
        "PhysActivity": p(.5, .8), "Fruits": p(.5, .65), "Veggies": p(.7, .85),
        "HvyAlcoholConsump": p(.03, .06), "AnyHealthcare": 1, "NoDocbcCost": p(.15, .07),
        "GenHlth": rng.choice([3, 4, 5] if hi else [1, 2, 3]), "MentHlth": rng.choice([0, 0, 2, 5, 10, 20]),
        "PhysHlth": rng.choice([0, 3, 10, 20, 30] if hi else [0, 0, 0, 2, 5]), "DiffWalk": p(.4, .08),
        "Sex": rng.randint(0, 1), "Age": rng.randint(8, 13) if hi else rng.randint(3, 9),
        "Education": rng.randint(3, 6), "Income": rng.randint(2, 8),
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

# Diabetes: module cap is 20 rows; 18 valid + 2 invalid
diab = [diabetes_row() for _ in range(20)]
diab[6]["BMI"] = 5       # below allowed range
diab[14]["GenHlth"] = 9  # out of range
write("diabetes_batch_sample.csv", diab)
print("ok")
