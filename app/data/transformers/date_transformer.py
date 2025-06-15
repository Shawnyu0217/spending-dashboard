"""
Date-related data transformations.
"""

import pandas as pd
from typing import List
from .base import BaseTransformer


class DateTransformer(BaseTransformer):
    """Adds date-derived columns to the DataFrame."""
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for this transformer."""
        return ["date"]
    
    def get_output_columns(self) -> List[str]:
        """Get list of columns that this transformer adds."""
        return [
            "year", "month", "day", "weekday", "month_name",
            "ym", "ym_str", "quarter", "quarter_str"
        ]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add date-derived columns to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional date-derived columns
        """
        if not self._validate_columns(df, self.get_required_columns()):
            self.logger.warning("Required date column not found, returning DataFrame unchanged")
            return df
        
        df_result = df.copy()
        
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_result["date"]):
                self.logger.info("Converting date column to datetime")
                df_result["date"] = pd.to_datetime(df_result["date"])
            
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
            
            self.logger.info(f"Successfully added {len(self.get_output_columns())} date-derived columns")
            
        except Exception as e:
            self.logger.error(f"Error processing date columns: {str(e)}")
            raise
        
        return df_result 