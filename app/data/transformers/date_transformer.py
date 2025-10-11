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
            "ym", "ym_str", "quarter", "quarter_str",
            "hour", "minute", "time_of_day", "is_weekend", "hour_minute_str"
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

            # Time-based columns (hour, minute, time of day)
            df_result["hour"] = df_result["date"].dt.hour
            df_result["minute"] = df_result["date"].dt.minute
            df_result["hour_minute_str"] = df_result["date"].dt.strftime("%H:%M")

            # Time of day categorization
            # Night: 0-6, Morning: 6-12, Afternoon: 12-18, Evening: 18-24
            df_result["time_of_day"] = pd.cut(
                df_result["hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["Night", "Morning", "Afternoon", "Evening"],
                include_lowest=True,
                right=False
            )

            # Weekend indicator
            df_result["is_weekend"] = df_result["date"].dt.dayofweek >= 5  # Saturday=5, Sunday=6

            self.logger.info(f"Successfully added {len(self.get_output_columns())} date-derived columns (including time components)")
            
        except Exception as e:
            self.logger.error(f"Error processing date columns: {str(e)}")
            raise
        
        return df_result 