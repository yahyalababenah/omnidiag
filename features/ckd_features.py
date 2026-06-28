import pandas as pd
import numpy as np
from features.base_features import BaseFeatureEngineer


class CKDFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for the Chronic Kidney Disease module.

    Input: UCI CKD Dataset (kidney_disease.csv — 400 rows, 25 columns).
    Target column 'classification' has values 'ckd' / 'notckd'; caller
    should encode to 1/0 before training.
    """

    _NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]

    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def engineer_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._coerce_numerics(df.copy())

        sc = df["sc"].replace(0, np.nan).fillna(1.0)
        df["BUN_Creatinine_Ratio"] = df["bu"] / sc

        hemo = df["hemo"].fillna(df["hemo"].median())
        rc = df["rc"].fillna(df["rc"].median())
        df["Anemia_Index"] = hemo * rc

        sod = df["sod"].fillna(df["sod"].median())
        pot = df["pot"].fillna(df["pot"].median())
        df["Electrolyte_Balance"] = sod - pot

        al = df["al"].fillna(0)
        su = df["su"].fillna(0)
        df["Protein_Sugar_Load"] = al + su

        return df

    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._coerce_numerics(df.copy())

        sc = df["sc"].fillna(1.0)
        bu = df["bu"].fillna(df["bu"].median())
        bp = df["bp"].fillna(df["bp"].median())
        al = df["al"].fillna(0)
        hemo = df["hemo"].fillna(df["hemo"].median())

        exponent = sc * 0.3 + bu * 0.01 + bp * 0.015 + al * 0.2 - hemo * 0.05
        df["CKD_Clinical_Risk"] = np.exp(exponent.clip(upper=20))
        return df
