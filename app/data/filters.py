"""
Data filtering utilities for the dashboard.
"""

import pandas as pd
from typing import List, Tuple, Optional


def filter_data_by_selections(
    df: pd.DataFrame,
    selected_years: List[int] = None,
    selected_months: List[int] = None,
    selected_categories: List[str] = None,
    selected_members: List[str] = None,
    selected_accounts: List[str] = None,
    date_range: Tuple[pd.Timestamp, pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Filter DataFrame based on user selections.
    
    Args:
        df: DataFrame to filter
        selected_years: List of years to include
        selected_months: List of months to include
        selected_categories: List of categories to include
        selected_members: List of members to include
        selected_accounts: List of accounts to include
        date_range: Tuple of (start_date, end_date)
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df_filtered = df.copy()
    
    # Filter by years
    if selected_years and "year" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]
    
    # Filter by months
    if selected_months and "month" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["month"].isin(selected_months)]
    
    # Filter by categories
    if selected_categories and "category_display" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["category_display"].isin(selected_categories)]
    
    # Filter by members
    if selected_members and "member" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["member"].isin(selected_members)]
    
    # Filter by accounts
    if selected_accounts and "account" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["account"].isin(selected_accounts)]
    
    # Filter by date range
    if date_range and "date" in df_filtered.columns:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered["date"] >= start_date) & 
            (df_filtered["date"] <= end_date)
        ]
    
    return df_filtered


def get_filter_summary(df_filtered: pd.DataFrame, df_total: pd.DataFrame) -> dict:
    """
    Get summary statistics about applied filters.
    
    Args:
        df_filtered: Filtered DataFrame
        df_total: Total (unfiltered) DataFrame
        
    Returns:
        Dictionary with filter summary statistics
    """
    if df_total.empty:
        return {}
    
    total_records = len(df_total)
    filtered_records = len(df_filtered)
    filter_ratio = (filtered_records / total_records * 100) if total_records > 0 else 0
    
    return {
        "total_records": total_records,
        "filtered_records": filtered_records,
        "filter_ratio": filter_ratio,
        "records_excluded": total_records - filtered_records
    } 