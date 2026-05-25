"""
OmniDiag — Base Feature Engineer
=================================
Abstract base class for disease-specific feature engineering modules.
All disease feature engineers must inherit from this class and implement
the engineer_heuristic() and engineer_clinical() methods.

This ensures a consistent interface across all diseases, making the
system predictable, testable, and extensible.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class BaseFeatureEngineer(ABC):
    """
    Abstract base class for disease-specific feature engineering.
    
    Each disease defines two feature engineering paths:
        1. Heuristic (Statistical): Mathematical interactions derived from raw features.
        2. Clinical (Medical): Risk scores based on domain-specific medical formulas.
    
    Usage:
        class MyDiseaseFeatures(BaseFeatureEngineer):
            def engineer_heuristic(self, df):
                df['my_feature'] = df['A'] * df['B']
                return df
            
            def engineer_clinical(self, df):
                df['risk_score'] = exp(df['A'] * 0.1)
                return df
    """
    
    def __init__(self, config: dict):
        """
        Initialize with the disease-specific configuration dictionary.
        
        Args:
            config: The parsed YAML config dict for this disease.
                    Expected to have keys: disease, data, features, model, etc.
        """
        self.config = config
        self.disease_name = config.get("disease", {}).get("name", "unknown")
        self.target_col = config.get("disease", {}).get("target_column", "target")
    
    @abstractmethod
    def engineer_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical/heuristic features from raw data.
        
        These are mathematical interactions (multiplication, ratios, etc.)
        that let the model discover non-linear patterns on its own.
        
        Args:
            df: Input DataFrame with base features.
        
        Returns:
            DataFrame with heuristic features added.
        """
        pass
    
    @abstractmethod
    def engineer_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate clinically-derived features from raw data.
        
        These are risk scores or indices based on established medical
        formulas with fixed coefficients.
        
        Args:
            df: Input DataFrame with base features.
        
        Returns:
            DataFrame with clinical features added.
        """
        pass
    
    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run both feature engineering paths and return results as a dict.
        
        This is the main entry point for generating both datasets
        for A/B testing (heuristic vs clinical).
        
        Args:
            df: Input DataFrame with base features.
        
        Returns:
            Dictionary with keys 'heuristic' and 'clinical', each
            containing the respective DataFrame.
        """
        return {
            "heuristic": self.engineer_heuristic(df.copy()),
            "clinical": self.engineer_clinical(df.copy())
        }
