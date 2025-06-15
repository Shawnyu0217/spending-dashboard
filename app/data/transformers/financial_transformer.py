"""
Financial calculation transformations.
"""

import pandas as pd
from .base import BaseTransformer


class FinancialTransformer(BaseTransformer):
    """Adds financial analysis columns to the DataFrame."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add financial analysis columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with financial columns
        """
        if not self._validate_columns(df, ["amount", "transaction_type"]):
            return df
        
        df_result = df.copy()
        
        # Create net_amount column (positive for income, negative for expenses)
        df_result["net_amount"] = df_result.apply(
            self._calculate_net_amount, axis=1
        )
        
        # Create separate amount columns for easier filtering
        df_result["income_amount"] = df_result.apply(
            lambda row: row["amount"] if row["transaction_type"] == "income" else 0, axis=1
        )
        
        df_result["expense_amount"] = df_result.apply(
            lambda row: row["amount"] if row["transaction_type"] == "expense" else 0, axis=1
        )
        
        # Sort by date for cumulative calculations
        df_result = df_result.sort_values("date")
        
        # Running balance (cumulative net amount)
        df_result["running_balance"] = df_result["net_amount"].cumsum()
        
        # Monthly cumulative amounts (reset each month)
        if "ym_str" in df_result.columns:
            df_result["monthly_cumulative"] = df_result.groupby("ym_str")["net_amount"].cumsum()
        
        return df_result
    
    def _calculate_net_amount(self, row):
        """
        Calculate net amount based on transaction type.
        
        Args:
            row: DataFrame row
            
        Returns:
            Net amount (positive for income, negative for expenses)
        """
        if row["transaction_type"] == "income":
            return row["amount"]
        elif row["transaction_type"] == "expense":
            return -row["amount"]
        else:
            return 0  # transfers are neutral 