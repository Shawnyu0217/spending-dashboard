"""
Base classes and interfaces for data transformers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Protocol


class DataTransformer(Protocol):
    """Protocol for data transformation steps."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        ...


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that required columns exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns exist, False otherwise
        """
        return all(col in df.columns for col in required_columns) 