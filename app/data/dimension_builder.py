"""
Dimension table creation for filters and analytics.
"""

import pandas as pd
from typing import Dict


class DimensionBuilder:
    """Creates dimension tables for filters and dropdowns."""
    
    def create_dimension_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create dimension tables for filters and dropdowns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of dimension tables
        """
        dim_tables = {}
        
        # Years available
        if "year" in df.columns:
            dim_tables["years"] = pd.DataFrame({
                "year": sorted(df["year"].dropna().unique())
            })
        
        # Months available  
        if "month" in df.columns and "month_name" in df.columns:
            months_df = df[["month", "month_name"]].drop_duplicates().sort_values("month")
            dim_tables["months"] = months_df
        
        # Categories
        if "category_display" in df.columns:
            categories_df = df.groupby(["category_display", "top_level_category"]).agg({
                "amount": "sum",
                "transaction_type": "count"
            }).reset_index()
            categories_df.columns = ["category", "top_level_category", "total_amount", "transaction_count"]
            dim_tables["categories"] = categories_df.sort_values("total_amount", ascending=False)
        
        # Members
        if "member" in df.columns:
            members_df = df.groupby("member").agg({
                "amount": "sum", 
                "transaction_type": "count"
            }).reset_index()
            members_df.columns = ["member", "total_amount", "transaction_count"]
            dim_tables["members"] = members_df.sort_values("total_amount", ascending=False)
        
        # Accounts
        if "account" in df.columns:
            accounts_df = df.groupby("account").agg({
                "amount": "sum",
                "transaction_type": "count"
            }).reset_index()
            accounts_df.columns = ["account", "total_amount", "transaction_count"]
            dim_tables["accounts"] = accounts_df.sort_values("total_amount", ascending=False)
        
        return dim_tables 