"""
Date-related data transformations.
"""

import pandas as pd
from .base import BaseTransformer


class DateTransformer(BaseTransformer):
    """Adds date-derived columns to the DataFrame."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add date-derived columns to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional date-derived columns
        """
        if not self._validate_columns(df, ["date"]):
            return df
        
        df_result = df.copy()
        
        # Date-based derived columns
        df_result["year"] = df_result["date"].dt.year
        df_result["month"] = df_result["date"].dt.month
        df_result["day"] = df_result["date"].dt.day
        df_result["weekday"] = df_result["date"].dt.day_name()
        df_result["month_name"] = df_result["date"].dt.month_name()
        
        # Year-month period for grouping
        df_result["ym"] = df_result["date"].dt.to_period("M")
        df_result["ym_str"] = df_result["ym"].astype(str)
        
        # Quarter
        df_result["quarter"] = df_result["date"].dt.quarter
        df_result["quarter_str"] = df_result["year"].astype(str) + "-Q" + df_result["quarter"].astype(str)
        
        return df_result 