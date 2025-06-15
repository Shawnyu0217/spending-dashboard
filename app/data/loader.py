"""
Data loading functionality for Excel files.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional, Union, List
import os
from pathlib import Path

from ..config import SAMPLE_FILE_PATH
from .schema import (
    normalize_column_names,
    normalize_transaction_types, 
    apply_column_dtypes,
    get_missing_columns
)

@st.cache_data
def load_excel_file(file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load Excel file and return all sheets as DataFrames.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Dictionary mapping sheet names to DataFrames
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file can't be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Read all sheets from Excel file
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        
        # Log sheet information
        sheet_info = {name: df.shape for name, df in sheets_dict.items()}
        st.sidebar.success(f"Loaded {len(sheets_dict)} sheets: {sheet_info}")
        
        return sheets_dict
        
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        raise

@st.cache_data  
def process_uploaded_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    """
    Process an uploaded Streamlit file into DataFrames.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    try:
        # Read all sheets from uploaded file
        sheets_dict = pd.read_excel(uploaded_file, sheet_name=None)
        
        # Log sheet information  
        sheet_info = {name: df.shape for name, df in sheets_dict.items()}
        st.sidebar.success(f"Loaded {len(sheets_dict)} sheets: {sheet_info}")
        
        return sheets_dict
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        raise

def clean_and_normalize_data(sheets_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Clean and normalize data from multiple sheets into a single DataFrame.
    
    Args:
        sheets_dict: Dictionary of sheet name -> DataFrame
        
    Returns:
        Single cleaned and normalized DataFrame
    """
    processed_sheets = []
    
    for sheet_name, df in sheets_dict.items():
        if df.empty:
            continue
            
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Normalize column names
        df_clean = normalize_column_names(df_clean)
        
        # Add sheet source information
        df_clean['sheet_source'] = sheet_name
        
        processed_sheets.append(df_clean)
    
    if not processed_sheets:
        raise ValueError("No valid data found in any sheets")
    
    # Combine all sheets
    combined_df = pd.concat(processed_sheets, ignore_index=True)
    
    # Normalize transaction types
    combined_df = normalize_transaction_types(combined_df)
    
    # Apply proper data types
    combined_df = apply_column_dtypes(combined_df)
    
    # Check for missing required columns
    missing_cols = get_missing_columns(combined_df)
    if missing_cols:
        st.warning(f"Missing required columns: {missing_cols}")
    
    return combined_df

def get_data_for_dashboard(uploaded_file=None) -> pd.DataFrame:
    """
    Main function to get data for the dashboard.
    Uses uploaded file if provided, otherwise falls back to sample file.
    
    Args:
        uploaded_file: Optional Streamlit UploadedFile
        
    Returns:
        Cleaned and processed DataFrame ready for analysis
    """
    try:
        if uploaded_file is not None:
            # Use uploaded file
            st.info("Using uploaded file...")
            sheets_dict = process_uploaded_file(uploaded_file)
        else:
            # Fall back to sample file
            if not os.path.exists(SAMPLE_FILE_PATH):
                st.error(f"Sample file not found: {SAMPLE_FILE_PATH}")
                st.info("Please upload an Excel file to continue.")
                return pd.DataFrame()
            
            st.info("Using sample file...")
            sheets_dict = load_excel_file(SAMPLE_FILE_PATH)
        
        # Clean and normalize the data
        df = clean_and_normalize_data(sheets_dict)
        
        # Store in session state for persistence
        st.session_state["df_raw"] = df
        
        st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics about the loaded data.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {}
    
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "date_range": None,
        "transaction_types": [],
        "categories": [],
        "memory_usage": df.memory_usage(deep=True).sum()
    }
    
    # Date range
    if "date" in df.columns and not df["date"].isna().all():
        summary["date_range"] = {
            "start": df["date"].min(),
            "end": df["date"].max()
        }
    
    # Unique values for categorical columns
    for col in ["transaction_type", "category"]:
        if col in df.columns:
            summary[f"{col}s"] = df[col].dropna().unique().tolist()
    
    return summary 