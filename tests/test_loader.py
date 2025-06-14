"""
Unit tests for the data loader module.
"""

import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.data.loader import (
    clean_and_normalize_data,
    get_data_summary,
)
from app.data.schema import normalize_column_names, get_missing_columns

class TestDataLoader:
    """Test cases for data loader functionality."""
    
    def test_normalize_column_names(self):
        """Test column name normalization."""
        # Create test DataFrame with Chinese column names
        test_data = {
            "日期": ["2024-01-01", "2024-01-02"],
            "金额": [100, 200],
            "分类": ["食物", "交通"]
        }
        df = pd.DataFrame(test_data)
        
        # Normalize column names
        df_normalized = normalize_column_names(df)
        
        # Check that columns are renamed correctly
        expected_columns = ["date", "amount", "category"]
        assert list(df_normalized.columns) == expected_columns
    
    def test_clean_and_normalize_data_empty_input(self):
        """Test handling of empty input data."""
        empty_sheets = {"sheet1": pd.DataFrame()}
        
        # Should raise ValueError for no valid data
        with pytest.raises(ValueError, match="No valid data found"):
            clean_and_normalize_data(empty_sheets)
    
    def test_clean_and_normalize_data_valid_input(self):
        """Test processing of valid input data."""
        # Create test data
        test_data = pd.DataFrame({
            "日期": ["2024-01-01", "2024-01-02"],
            "交易类型": ["支出", "收入"],
            "金额": [100, 200],
            "分类": ["食物", "工资"]
        })
        
        sheets_dict = {"transactions": test_data}
        
        # Process the data
        result_df = clean_and_normalize_data(sheets_dict)
        
        # Check basic structure
        assert not result_df.empty
        assert "date" in result_df.columns
        assert "transaction_type" in result_df.columns
        assert "amount" in result_df.columns
        assert "category" in result_df.columns
        assert "sheet_source" in result_df.columns
        assert len(result_df) == 2
    
    def test_get_missing_columns(self):
        """Test missing column detection."""
        # Create DataFrame missing some required columns
        test_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "amount": [100]
            # Missing: transaction_type, category, account, etc.
        })
        
        missing = get_missing_columns(test_df)
        
        # Should have several missing columns
        assert len(missing) > 0
        assert "transaction_type" in missing
        assert "category" in missing
    
    def test_get_data_summary_empty(self):
        """Test data summary for empty DataFrame."""
        empty_df = pd.DataFrame()
        summary = get_data_summary(empty_df)
        
        assert summary == {}
    
    def test_get_data_summary_valid(self):
        """Test data summary for valid DataFrame."""
        test_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "transaction_type": ["expense", "income"],
            "category": ["food", "salary"],
            "account": ["cash", "bank"],
            "amount": [100.5, 200.0]
        })
        
        summary = get_data_summary(test_df)
        
        assert summary["total_rows"] == 2
        assert summary["total_columns"] == 5
        assert "date_range" in summary
        assert summary["date_range"]["start"] == pd.Timestamp("2024-01-01")
        assert summary["date_range"]["end"] == pd.Timestamp("2024-01-02")
        assert "expense" in summary["transaction_types"]
        assert "income" in summary["transaction_types"]

class TestDataValidation:
    """Test cases for data validation."""
    
    def test_required_columns_present(self):
        """Test that required columns are identified correctly."""
        from app.data.schema import get_required_columns
        
        required = get_required_columns()
        
        # Should include basic required columns
        expected_required = ["date", "transaction_type", "category", "amount"]
        for col in expected_required:
            assert col in required
    
    def test_column_type_conversion(self):
        """Test that column types are applied correctly."""
        from app.data.schema import apply_column_dtypes
        
        # Create test data with string amounts
        test_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "amount": ["¥100.50", "200"],
            "transaction_type": ["expense", "income"]
        })
        
        # Apply type conversion
        result_df = apply_column_dtypes(test_df)
        
        # Check that date is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
        
        # Check that amount is converted to float (currency symbols removed)
        assert pd.api.types.is_numeric_dtype(result_df["amount"])
        assert result_df["amount"].iloc[0] == 100.50
        assert result_df["amount"].iloc[1] == 200.0

if __name__ == "__main__":
    pytest.main([__file__]) 