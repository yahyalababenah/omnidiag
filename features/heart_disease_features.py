"""
OmniDiag — Heart Disease Feature Engineer
==========================================
Implements two feature engineering paths for heart disease diagnosis:

    1. Heuristic (Statistical) Path:
        - Age_BP_Interaction: Age × RestingBP
        - HR_Age_Ratio: MaxHR / Age
        - Chol_Age_Ratio: Cholesterol / Age
    
    2. Clinical (Medical) Path:
        - Clinical_Risk_Score: exp(Age × 0.048 + RestingBP × 0.015 + Cholesterol × 0.002)

These features were validated through Optuna A/B testing (100 trials),
with the heuristic path achieving 88.98% accuracy (winner).
"""

import pandas as pd
import numpy as np
from features.base_features import BaseFeatureEngineer


class HeartDiseaseFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for the Heart Disease module.
    
    Generates both heuristic (statistical interaction) and clinical
    (medical risk score) features for the heart disease dataset.
    
    Usage:
        config = load_yaml_config("configs/heart_disease.yaml")
        engineer = HeartDiseaseFeatureEngineer(config)
        df_heuristic = engineer.engineer_heuristic(df)
        df_clinical = engineer.engineer_clinical(df)
    """
    
    def engineer_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical heuristic features.
        
        Features created:
            - Age_BP_Interaction: Captures combined risk of age × blood pressure.
            - HR_Age_Ratio: Heart rate relative to age (lower = fitter).
            - Chol_Age_Ratio: Cholesterol burden normalized by age.
        
        Args:
            df: DataFrame with at least 'Age', 'RestingBP', 'MaxHR', 'Cholesterol'.
        
        Returns:
            DataFrame with heuristic features appended.
        """
        if 'Age' in df.columns and 'RestingBP' in df.columns:
            df['Age_BP_Interaction'] = df['Age'] * df['RestingBP']
        
        if 'Age' in df.columns and 'MaxHR' in df.columns:
            df['HR_Age_Ratio'] = df['MaxHR'] / df['Age']
        
        if 'Cholesterol' in df.columns and 'Age' in df.columns:
            df['Chol_Age_Ratio'] = df['Cholesterol'] / df['Age']
        
        return df
    
    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate clinical risk score features.
        
        Features created:
            - Clinical_Risk_Score: A logarithmic risk index using fixed medical weights.
              Formula: exp(Age × 0.048 + RestingBP × 0.015 + Cholesterol × 0.002)
              These weights approximate established cardiovascular risk models.
        
        Args:
            df: DataFrame with at least 'Age', 'RestingBP', 'Cholesterol'.
        
        Returns:
            DataFrame with clinical features appended.
        """
        if all(col in df.columns for col in ['Age', 'RestingBP', 'Cholesterol']):
            df['Clinical_Risk_Score'] = np.exp(
                (df['Age'] * 0.048) + 
                (df['RestingBP'] * 0.015) + 
                (df['Cholesterol'] * 0.002)
            )
        
        return df
