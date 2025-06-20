"""
Data schema definitions and validation helpers for the spending dashboard.
"""

import pandas as pd
from typing import Dict, List, Optional
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from ..config import COLUMN_MAPPINGS, TRANSACTION_TYPE_MAPPINGS

# Expected data types for columns
COLUMN_DTYPES = {
    "date": "datetime64[ns]",
    "transaction_type": "category", 
    "category": "category",
    "subcategory": "category",
    "amount": "float64",
    "notes": "object",
    "member": "category"
}

# Pandera schema for data validation
SPENDING_SCHEMA = DataFrameSchema({
    "date": Column(pd.Timestamp, nullable=False),
    "transaction_type": Column(str, checks=[
        Check.isin(["income", "expense", "transfer"])
    ], nullable=False),
    "category": Column(str, nullable=False),
    "subcategory": Column(str, nullable=True),
    "amount": Column(float, checks=[
        Check.greater_than_or_equal_to(0)
    ], nullable=False),
    "notes": Column(str, nullable=True),
    "member": Column(str, nullable=True)
})

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from Chinese to English using the mapping.
    
    Args:
        df: DataFrame with original column names
        
    Returns:
        DataFrame with normalized column names
    """
    # Create a copy to avoid modifying original
    df_normalized = df.copy()
    
    # Rename columns using the mapping
    df_normalized = df_normalized.rename(columns=COLUMN_MAPPINGS)
    
    # Handle any columns not in the mapping by converting to lowercase snake_case
    def to_snake_case(name: str) -> str:
        import re
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name.lower()
    
    # Apply snake_case to any remaining columns
    final_columns = {}
    for col in df_normalized.columns:
        if col not in COLUMN_MAPPINGS.values():
            final_columns[col] = to_snake_case(col)
    
    df_normalized = df_normalized.rename(columns=final_columns)
    
    return df_normalized

def normalize_transaction_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize transaction types from Chinese to English.
    
    Args:
        df: DataFrame with transaction_type column
        
    Returns:
        DataFrame with normalized transaction types
    """
    df_normalized = df.copy()
    
    if "transaction_type" in df_normalized.columns:
        df_normalized["transaction_type"] = df_normalized["transaction_type"].map(
            TRANSACTION_TYPE_MAPPINGS
        ).fillna(df_normalized["transaction_type"])
    
    return df_normalized

def apply_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply proper data types to DataFrame columns.
    
    Args:
        df: DataFrame to type-cast
        
    Returns:
        DataFrame with proper dtypes
    """
    df_typed = df.copy()
    
    for col, dtype in COLUMN_DTYPES.items():
        if col in df_typed.columns:
            try:
                if dtype == "datetime64[ns]":
                    df_typed[col] = pd.to_datetime(df_typed[col])
                elif dtype == "category":
                    df_typed[col] = df_typed[col].astype("category")
                elif dtype == "float64":
                    # Handle potential string numbers and remove currency symbols
                    if df_typed[col].dtype == "object":
                        df_typed[col] = df_typed[col].astype(str).str.replace(
                            r'[$¥,\s]', '', regex=True
                        ).str.replace(',', '')
                    df_typed[col] = pd.to_numeric(df_typed[col], errors="coerce")
                else:
                    df_typed[col] = df_typed[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    
    return df_typed

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame against the expected schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        SPENDING_SCHEMA.validate(df)
        return True
    except pa.errors.SchemaError as e:
        print(f"Schema validation failed: {e}")
        return False

def get_required_columns() -> List[str]:
    """Get list of required column names."""
    return list(COLUMN_DTYPES.keys())

def get_missing_columns(df: pd.DataFrame) -> List[str]:
    """Get list of missing required columns."""
    required = set(get_required_columns())
    present = set(df.columns)
    return list(required - present) 