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

Integration Note:
    The actual math is delegated to models/advanced_feature_engineering.py
    to avoid code duplication. This class provides the OOP wrapper for the
    config-driven architecture while the core logic lives in the original script.
"""

import pandas as pd
from features.base_features import BaseFeatureEngineer
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_clinical_features,
)


class HeartDiseaseFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for the Heart Disease module.
    
    Delegates to models/advanced_feature_engineering.py for the actual math.
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
        
        Delegates to models.advanced_feature_engineering.engineer_heuristic_features().
        
        Features created:
            - Age_BP_Interaction: Captures combined risk of age × blood pressure.
            - HR_Age_Ratio: Heart rate relative to age (lower = fitter).
            - Chol_Age_Ratio: Cholesterol burden normalized by age.
        
        Args:
            df: DataFrame with at least 'Age', 'RestingBP', 'MaxHR', 'Cholesterol'.
        
        Returns:
            DataFrame with heuristic features appended.
        """
        return engineer_heuristic_features(df)
    
    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate clinical risk score features.
        
        Delegates to models.advanced_feature_engineering.engineer_clinical_features().
        
        Features created:
            - Clinical_Risk_Score: A logarithmic risk index using fixed medical weights.
              Formula: exp(Age × 0.048 + RestingBP × 0.015 + Cholesterol × 0.002)
              These weights approximate established cardiovascular risk models.
        
        Args:
            df: DataFrame with at least 'Age', 'RestingBP', 'Cholesterol'.
        
        Returns:
            DataFrame with clinical features appended.
        """
        return engineer_clinical_features(df)
