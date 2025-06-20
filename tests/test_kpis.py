"""
Unit tests for KPI calculation functions.
"""

import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.features.kpis import (
    total_income,
    total_expense,
    net_savings,
    savings_rate,
    largest_expense_category,
    average_daily_spending,
    format_currency,
    format_percentage,
    get_kpi_metrics,
    identify_best_worst_months
)

class TestKPIFunctions:
    """Test cases for KPI calculation functions."""
    
    def setup_method(self):
        """Set up test data for each test."""
        # Create sample data for testing
        self.test_data = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03", 
                "2024-01-04", "2024-01-05"
            ]),
            "transaction_type": ["expense", "income", "expense", "expense", "income"],
            "category_display": ["Food", "Salary", "Transport", "Shopping", "Bonus"],
            "amount": [50.0, 1000.0, 30.0, 100.0, 500.0],
            "income_amount": [0.0, 1000.0, 0.0, 0.0, 500.0],
            "expense_amount": [50.0, 0.0, 30.0, 100.0, 0.0],
            "net_amount": [-50.0, 1000.0, -30.0, -100.0, 500.0]
        })
        
        self.empty_data = pd.DataFrame()
    
    def test_total_income(self):
        """Test total income calculation."""
        income = total_income(self.test_data)
        assert income == 1500.0  # 1000 + 500
        
        # Test with empty data
        assert total_income(self.empty_data) == 0.0
    
    def test_total_expense(self):
        """Test total expense calculation."""
        expense = total_expense(self.test_data)
        assert expense == 180.0  # 50 + 30 + 100
        
        # Test with empty data
        assert total_expense(self.empty_data) == 0.0
    
    def test_net_savings(self):
        """Test net savings calculation."""
        savings = net_savings(self.test_data)
        assert savings == 1320.0  # 1500 - 180
        
        # Test with empty data
        assert net_savings(self.empty_data) == 0.0
    
    def test_savings_rate(self):
        """Test savings rate calculation."""
        rate = savings_rate(self.test_data)
        expected_rate = (1320.0 / 1500.0) * 100  # 88%
        assert abs(rate - expected_rate) < 0.01
        
        # Test with zero income
        zero_income_data = pd.DataFrame({
            "income_amount": [0.0],
            "expense_amount": [100.0]
        })
        assert savings_rate(zero_income_data) == 0.0
    
    def test_largest_expense_category(self):
        """Test largest expense category identification."""
        category, amount = largest_expense_category(self.test_data)
        assert category == "Shopping"
        assert amount == 100.0
        
        # Test with empty data
        category_empty, amount_empty = largest_expense_category(self.empty_data)
        assert category_empty == "N/A"
        assert amount_empty == 0.0
    
    def test_average_daily_spending(self):
        """Test average daily spending calculation."""
        avg_spending = average_daily_spending(self.test_data)
        
        # Total expenses: 180, Date range: 4 days (01-01 to 01-05)
        expected_avg = 180.0 / 4
        assert abs(avg_spending - expected_avg) < 0.01
        
        # Test with empty data
        assert average_daily_spending(self.empty_data) == 0.0
    
    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(0) == "$0.00"
        assert format_currency(1000000) == "$1,000,000.00"
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(25.5) == "25.5%"
        assert format_percentage(0) == "0.0%"
        assert format_percentage(100) == "100.0%"
    
    def test_get_kpi_metrics(self):
        """Test comprehensive KPI metrics calculation."""
        metrics = get_kpi_metrics(self.test_data)
        
        # Check all expected keys are present
        expected_keys = [
            "total_income", "total_expense", "net_savings", "savings_rate",
            "largest_expense_category", "largest_expense_amount",
            "average_daily_spending", "total_transactions",
            "income_transactions", "expense_transactions"
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Check some specific values
        assert metrics["total_income"] == 1500.0
        assert metrics["total_expense"] == 180.0
        assert metrics["net_savings"] == 1320.0
        assert metrics["largest_expense_category"] == "Shopping"
        assert metrics["total_transactions"] == 5
        assert metrics["income_transactions"] == 2
        assert metrics["expense_transactions"] == 3
        
        # Test with empty data
        empty_metrics = get_kpi_metrics(self.empty_data)
        assert empty_metrics["total_transactions"] == 0
        assert empty_metrics["total_income"] == 0.0

class TestKPIEdgeCases:
    """Test edge cases for KPI functions."""
    
    def test_missing_columns(self):
        """Test behavior when required columns are missing."""
        incomplete_data = pd.DataFrame({
            "date": ["2024-01-01"],
            "amount": [100.0]
            # Missing: income_amount, expense_amount, etc.
        })
        
        # Functions should handle missing columns gracefully
        assert total_income(incomplete_data) == 0.0
        assert total_expense(incomplete_data) == 0.0
        assert net_savings(incomplete_data) == 0.0
    
    def test_negative_amounts(self):
        """Test handling of negative amounts."""
        negative_data = pd.DataFrame({
            "income_amount": [-100.0],  # Unusual but possible
            "expense_amount": [-50.0]
        })
        
        # Should handle negative values
        income = total_income(negative_data)
        expense = total_expense(negative_data)
        
        assert income == -100.0
        assert expense == -50.0
    
    def test_single_transaction(self):
        """Test calculations with single transaction."""
        single_data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "transaction_type": ["expense"],
            "category_display": ["Food"],
            "amount": [50.0],
            "income_amount": [0.0],
            "expense_amount": [50.0],
            "net_amount": [-50.0]
        })
        
        assert total_expense(single_data) == 50.0
        assert total_income(single_data) == 0.0
        assert net_savings(single_data) == -50.0
        
        category, amount = largest_expense_category(single_data)
        assert category == "Food"
        assert amount == 50.0

    def test_identify_best_worst_months_improvement_potential(self):
        """Test improvement potential calculation in identify_best_worst_months."""
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-15", "2024-01-20",
                "2024-02-10", "2024-02-15",
                "2024-03-05", "2024-03-20"
            ]),
            "transaction_type": [
                "income", "expense",
                "income", "expense",
                "income", "expense"
            ],
            "category_display": [
                "Salary", "Food",
                "Salary", "Rent",
                "Salary", "Misc"
            ],
            "amount": [1000, 500, 800, 700, 1200, 1000]
        })

        df["ym_str"] = df["date"].dt.to_period("M").astype(str)
        df["net_amount"] = df.apply(lambda r: r["amount"] if r["transaction_type"] == "income" else -r["amount"], axis=1)
        df["income_amount"] = df.apply(lambda r: r["amount"] if r["transaction_type"] == "income" else 0, axis=1)
        df["expense_amount"] = df.apply(lambda r: r["amount"] if r["transaction_type"] == "expense" else 0, axis=1)

        analysis = identify_best_worst_months(df)

        assert analysis["best_month"]["month"] == "2024-01"
        assert analysis["worst_month"]["month"] == "2024-02"

        imp = analysis["improvement_potential"]
        assert imp["months_below_average"] == 2
        assert imp["average_shortfall"] == pytest.approx(11.8055, rel=1e-4)
        assert imp["potential_additional_savings"] == pytest.approx(100.3472, rel=1e-4)

if __name__ == "__main__":
    pytest.main([__file__]) 