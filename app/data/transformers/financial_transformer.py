"""
Financial calculation transformations.
"""

import pandas as pd
from typing import List
from .base import BaseTransformer


class FinancialTransformer(BaseTransformer):
    """Adds financial analysis columns to the DataFrame."""
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for this transformer."""
        return ["amount", "transaction_type"]
    
    def get_output_columns(self) -> List[str]:
        """Get list of columns that this transformer adds."""
        return [
            "net_amount", "income_amount", "expense_amount",
            "running_balance", "monthly_cumulative"
        ]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add financial analysis columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with financial columns
        """
        if not self._validate_columns(df, self.get_required_columns()):
            self.logger.warning("Required financial columns not found, returning DataFrame unchanged")
            return df
        
        df_result = df.copy()
        
        try:
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
            
            # Sort by date for cumulative calculations if date column exists
            if "date" in df_result.columns:
                df_result = df_result.sort_values("date")
            else:
                self.logger.warning("No date column found, cumulative calculations may not be chronologically ordered")
            
            # Running balance (cumulative net amount)
            df_result["running_balance"] = df_result["net_amount"].cumsum()
            
            # Monthly cumulative amounts (reset each month)
            if "ym_str" in df_result.columns:
                df_result["monthly_cumulative"] = df_result.groupby("ym_str")["net_amount"].cumsum()
            else:
                self.logger.warning("No ym_str column found, monthly cumulative calculation skipped")
                df_result["monthly_cumulative"] = df_result["net_amount"].cumsum()
            
            self.logger.info(f"Successfully added {len(self.get_output_columns())} financial columns")
            
        except Exception as e:
            self.logger.error(f"Error processing financial columns: {str(e)}")
            raise
        
        return df_result
    
    def _calculate_net_amount(self, row) -> float:
        """
        Calculate net amount based on transaction type.
        
        Args:
            row: DataFrame row
            
        Returns:
            Net amount (positive for income, negative for expenses)
        """
        try:
            if row["transaction_type"] == "income":
                return float(row["amount"])
            elif row["transaction_type"] == "expense":
                return -float(row["amount"])
            else:
                # transfers or other types are neutral
                return 0.0
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error calculating net amount for row: {e}")
            return 0.0 