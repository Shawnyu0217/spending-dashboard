"""
Data preprocessing and transformation functions.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
import numpy as np

from ..config import CATEGORY_MAPPINGS

@st.cache_data
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional derived columns
    """
    df_derived = df.copy()
    
    # Ensure we have a date column
    if "date" not in df_derived.columns:
        st.warning("No date column found, skipping date-based derived columns")
        return df_derived
    
    # Date-based derived columns
    df_derived["year"] = df_derived["date"].dt.year
    df_derived["month"] = df_derived["date"].dt.month
    df_derived["day"] = df_derived["date"].dt.day
    df_derived["weekday"] = df_derived["date"].dt.day_name()
    df_derived["month_name"] = df_derived["date"].dt.month_name()
    
    # Year-month period for grouping
    df_derived["ym"] = df_derived["date"].dt.to_period("M")
    df_derived["ym_str"] = df_derived["ym"].astype(str)
    
    # Quarter
    df_derived["quarter"] = df_derived["date"].dt.quarter
    df_derived["quarter_str"] = df_derived["year"].astype(str) + "-Q" + df_derived["quarter"].astype(str)
    
    return df_derived

@st.cache_data  
def add_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add financial analysis columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with financial columns
    """
    df_financial = df.copy()
    
    if "amount" not in df_financial.columns or "transaction_type" not in df_financial.columns:
        st.warning("Missing amount or transaction_type columns for financial calculations")
        return df_financial
    
    # Create net_amount column (positive for income, negative for expenses)
    df_financial["net_amount"] = df_financial.apply(
        lambda row: row["amount"] if row["transaction_type"] == "income" 
        else -row["amount"] if row["transaction_type"] == "expense"
        else 0,  # transfers are neutral
        axis=1
    )
    
    # Create separate amount columns for easier filtering
    df_financial["income_amount"] = df_financial.apply(
        lambda row: row["amount"] if row["transaction_type"] == "income" else 0, axis=1
    )
    
    df_financial["expense_amount"] = df_financial.apply(
        lambda row: row["amount"] if row["transaction_type"] == "expense" else 0, axis=1
    )
    
    # Sort by date for cumulative calculations
    df_financial = df_financial.sort_values("date")
    
    # Running balance (cumulative net amount)
    df_financial["running_balance"] = df_financial["net_amount"].cumsum()
    
    # Monthly cumulative amounts (reset each month)
    df_financial["monthly_cumulative"] = df_financial.groupby("ym_str")["net_amount"].cumsum()
    
    return df_financial

@st.cache_data
def add_category_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add category mappings and top-level category groupings.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with category mappings
    """
    df_categories = df.copy()
    
    if "category" not in df_categories.columns:
        st.warning("No category column found for category mappings")
        return df_categories
    
    # Map Chinese categories to English display names
    df_categories["category_display"] = df_categories["category"].map(
        CATEGORY_MAPPINGS
    ).fillna(df_categories["category"])
    
    # Enhanced category mapping - map original categories directly to top-level if not in display mapping
    def get_top_level_category(category_display, original_category):
        """Get top-level category with fallback logic."""
        # First try mapping from display category
        top_level_mapping = {
            # Core living expenses
            "Food & Dining": "Living Expenses",
            "Shopping": "Living Expenses",
            "Transportation": "Living Expenses",
            "Personal Care": "Living Expenses",
            "Pet Care": "Living Expenses",
            "Maintenance": "Living Expenses",
            "Communication": "Living Expenses",
            
            # Essential fixed costs
            "Utilities": "Essential",
            "Insurance & Financial": "Essential",
            "Healthcare": "Essential",
            "Healthcare & Education": "Essential",
            "Education": "Essential",
            "Living Services": "Essential",

            # Property & Housing
            "Housing Loan": "Property & Financial",
            "Investment Property": "Property & Financial",
            "Rent": "Property & Financial",
            "Renovation": "Property & Financial",
            "Property Tax": "Property & Financial",
            "Government Fees": "Property & Financial",

            # Lifestyle / Discretionary
            "Entertainment": "Discretionary",
            "Travel": "Discretionary",
            "Social": "Discretionary",
            "Other": "Discretionary",
            "Bad Debt": "Discretionary",
            
            # Income Types
            "Income": "Income",
            "Salary": "Income",
            "Bonus": "Income",
            "Investment Income": "Income",
            "Other Income": "Income",
            "Allowance": "Income",
            "Subsidy": "Income"
        }

        
        # If display category is mapped, use it
        if category_display in top_level_mapping:
            return top_level_mapping[category_display]
        
        # Direct mapping from common Chinese categories to top-level
        chinese_to_top_level = {
            "餐饮": "Living Expenses",
            "购物": "Living Expenses",
            "交通": "Living Expenses", 
            "生活服务": "Living Expenses",
            "日用品": "Living Expenses",
            "超市": "Living Expenses",
            "服饰": "Living Expenses",
            "家居": "Living Expenses",
            "医疗健康": "Essential",
            "教育": "Essential",
            "医疗": "Essential",
            "娱乐": "Discretionary",
            "旅游": "Discretionary",
            "人情往来": "Discretionary",
            "通讯": "Essential",
            "运动": "Discretionary",
            "美容": "Discretionary",
            "宠物": "Living Expenses",
            "其他": "Other",
            "收入": "Income",
            "工资": "Income",
            "奖金": "Income",
            "投资收益": "Income",
            "理财": "Income"
        }
        
        # Try direct Chinese mapping
        if original_category in chinese_to_top_level:
            return chinese_to_top_level[original_category]
        
        # If it's an income transaction, classify as Income
        if any(income_word in str(original_category) for income_word in ['收入', '工资', '奖金', '薪', '津贴']):
            return "Income"
        
        # Default fallback based on common patterns
        category_str = str(original_category).lower()
        if any(food_word in category_str for food_word in ['餐', '食', '饮', '吃', '喝']):
            return "Living Expenses"
        elif any(shop_word in category_str for shop_word in ['购', '买', '商', '市']):
            return "Living Expenses"
        elif any(transport_word in category_str for transport_word in ['交通', '车', '油', '停车', '地铁', '公交']):
            return "Living Expenses"
        elif any(health_word in category_str for health_word in ['医', '药', '健康', '保健']):
            return "Essential"
        elif any(fun_word in category_str for fun_word in ['娱乐', '游戏', '电影', '旅游', '运动']):
            return "Discretionary"
        else:
            return "Other"
    
    # Apply the enhanced mapping
    df_categories["top_level_category"] = df_categories.apply(
        lambda row: get_top_level_category(row["category_display"], row["category"]), 
        axis=1
    )
    
    return df_categories

def create_dimension_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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

def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Main preprocessing function that applies all transformations.
    
    Args:
        df_raw: Raw DataFrame from loader
        
    Returns:
        Tuple of (processed_df, dimension_tables)
    """
    if df_raw.empty:
        return df_raw, {}
    
    # Apply all preprocessing steps
    df_processed = df_raw.copy()
    
    # Add derived columns
    df_processed = add_derived_columns(df_processed)
    
    # Add financial columns
    df_processed = add_financial_columns(df_processed)
    
    # Add category mappings
    df_processed = add_category_mappings(df_processed)
    
    # Create dimension tables
    dim_tables = create_dimension_tables(df_processed)
    
    st.success(f"Preprocessing complete: {df_processed.shape[0]} rows processed")
    
    return df_processed, dim_tables

def get_date_range_options(df: pd.DataFrame) -> List[str]:
    """
    Get available date range options for filtering.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        List of date range option strings
    """
    if "ym_str" not in df.columns:
        return []
    
    return sorted(df["ym_str"].dropna().unique())

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