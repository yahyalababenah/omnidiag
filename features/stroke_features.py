import pandas as pd
import numpy as np
from features.base_features import BaseFeatureEngineer


class StrokeFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for the Stroke Risk module.

    Input: Kaggle Stroke Prediction Dataset
    (healthcare-dataset-stroke-data.csv — 5110 rows, 12 columns)
    """

    def engineer_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(df["bmi"].median() if "bmi" in df else 25.0)

        df["Age_Glucose_Interaction"] = df["age"] * df["avg_glucose_level"]
        df["BMI_Age_Risk"] = df["bmi"] * df["age"]
        df["Vascular_Burden"] = df["hypertension"].astype(int) + df["heart_disease"].astype(int)
        return df

    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(df["bmi"].median() if "bmi" in df else 25.0)

        exponent = (
            df["age"] * 0.05
            + df["avg_glucose_level"] * 0.003
            + df["bmi"] * 0.02
            + df["hypertension"].astype(int) * 0.8
            + df["heart_disease"].astype(int) * 0.6
        )
        df["Stroke_Clinical_Risk"] = np.exp(exponent.clip(upper=20))
        return df
