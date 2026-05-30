"""
OmniDiag — Diabetes Feature Engineer
======================================
Implements feature engineering paths for diabetes risk assessment:

    1. Heuristic (Statistical) Path:
        - BMI_Age_Interaction: BMI × Age
        - Health_Index: GenHlth × (MentHlth + PhysHlth)
        - Lifestyle_Score: PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump
        - SES_Composite: Education × Income

    2. Clinical (Medical) Path:
        - Diabetes_Clinical_Risk: exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)

All features are computed from existing columns only — zero new NaN values.
"""

import pandas as pd
import numpy as np
from features.base_features import BaseFeatureEngineer


class DiabetesFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for the Diabetes module.

    Generates heuristic and clinical features for diabetes risk prediction
    from the CDC BRFSS 2015 Health Indicators dataset.

    Usage:
        config = load_yaml_config("configs/diabetes.yaml")
        engineer = DiabetesFeatureEngineer(config)
        df_heuristic = engineer.engineer_heuristic(df)
        df_clinical = engineer.engineer_clinical(df)
    """

    def engineer_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical heuristic features for diabetes risk.

        Features created:
            - BMI_Age_Interaction: BMI × Age
              Captures compounding effect of obesity and age on diabetes risk.
              Older individuals with high BMI face exponentially higher risk.

            - Health_Index: GenHlth × (MentHlth + PhysHlth)
              Composite morbidity score: self-rated health amplified by total
              number of poor-health days (mental + physical). Higher = worse.

            - Lifestyle_Score: PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump
              Aggregate healthy-lifestyle indicator. Positive values indicate
              protective behaviours; negative values indicate risk behaviours.

            - SES_Composite: Education × Income
              Socioeconomic status proxy. Lower SES is strongly associated
              with higher diabetes prevalence in epidemiological literature.

        Args:
            df: DataFrame with at least 'BMI', 'Age', 'GenHlth', 'MentHlth',
                'PhysHlth', 'PhysActivity', 'Fruits', 'Veggies', 'Smoker',
                'HvyAlcoholConsump', 'Education', 'Income'.

        Returns:
            DataFrame with heuristic features appended.
        """
        df = df.copy()
        required_base = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
                         'PhysActivity', 'Fruits', 'Veggies', 'Smoker',
                         'HvyAlcoholConsump', 'Education', 'Income']
        missing = [c for c in required_base if c not in df.columns]
        if missing:
            raise KeyError(
                f"DiabetesFeatureEngineer.engineer_heuristic: missing columns {missing}. "
                f"Required: {required_base}"
            )

        # BMI_Age_Interaction
        df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
        self._log_feature("BMI_Age_Interaction", df)

        # Health_Index
        df['Health_Index'] = df['GenHlth'] * (df['MentHlth'] + df['PhysHlth'])
        self._log_feature("Health_Index", df)

        # Lifestyle_Score
        df['Lifestyle_Score'] = (
            df['PhysActivity']
            + df['Fruits']
            + df['Veggies']
            - df['Smoker']
            - df['HvyAlcoholConsump']
        )
        self._log_feature("Lifestyle_Score", df)

        # SES_Composite
        df['SES_Composite'] = df['Education'] * df['Income']
        self._log_feature("SES_Composite", df)

        return df

    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate clinical risk score features for diabetes.

        Note: The primary clinical risk score (Diabetes_Clinical_Risk) is
        computed in engineer_medical() because ModelLoader._engineer_features()
        calls heuristic → medical during inference but skips clinical for
        backward compatibility with the heart disease module.

        This method is reserved for future clinically-derived features
        that are not needed during real-time inference (e.g., for
        offline A/B testing or research).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame unchanged (identity transform).
        """
        return df.copy()

    def engineer_medical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Medical feature engineering for diabetes risk.

        Creates the Diabetes_Clinical_Risk score — a logarithmic risk index
        based on epidemiological weights. This is placed here rather than in
        engineer_clinical() because ModelLoader._engineer_features() calls
        heuristic → medical → (skips clinical for backward compatibility).

        Features created:
            - Diabetes_Clinical_Risk: exp(BMI × 0.05 + Age × 0.03 + GenHlth × 0.2 + HighBP × 0.5)
              Weights based on epidemiological literature:
                  * BMI (5%): Each unit increase raises risk ~5%
                  * Age (3%): Each age category raises risk ~3%
                  * GenHlth (20%): Self-rated health captures unmeasured comorbidities
                  * HighBP (50%): Hypertension is the single strongest modifiable risk factor

        Args:
            df: DataFrame with at least 'BMI', 'Age', 'GenHlth', 'HighBP'.

        Returns:
            DataFrame with Diabetes_Clinical_Risk appended.
        """
        df = df.copy()
        required = ['BMI', 'Age', 'GenHlth', 'HighBP']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"DiabetesFeatureEngineer.engineer_medical: missing columns {missing}. "
                f"Required: {required}"
            )

        df['Diabetes_Clinical_Risk'] = np.exp(
            (df['BMI'] * 0.05)
            + (df['Age'] * 0.03)
            + (df['GenHlth'] * 0.2)
            + (df['HighBP'] * 0.5)
        )
        self._log_feature("Diabetes_Clinical_Risk", df)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_feature(name: str, df: pd.DataFrame) -> None:
        """Log a brief summary of a newly created feature."""
        import logging
        log = logging.getLogger("omnidiag.diabetes_features")
        log.debug(
            f"Created '{name}': "
            f"range=[{df[name].min():.4f}, {df[name].max():.4f}], "
            f"mean={df[name].mean():.4f}, nulls={df[name].isnull().sum()}"
        )
