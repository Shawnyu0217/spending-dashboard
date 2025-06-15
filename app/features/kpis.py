"""
KPI calculation functions for the spending dashboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

from ..config import CURRENCY_FORMAT, PERCENTAGE_FORMAT

def total_income(df: pd.DataFrame) -> float:
    """Calculate total income from the DataFrame."""
    if df.empty or "income_amount" not in df.columns:
        return 0.0
    return df["income_amount"].sum()

def total_expense(df: pd.DataFrame) -> float:
    """Calculate total expenses from the DataFrame."""
    if df.empty or "expense_amount" not in df.columns:
        return 0.0
    return df["expense_amount"].sum()

def net_savings(df: pd.DataFrame) -> float:
    """Calculate net savings (income - expenses)."""
    return total_income(df) - total_expense(df)

def savings_rate(df: pd.DataFrame) -> float:
    """Calculate savings rate as percentage of income."""
    income = total_income(df)
    if income == 0:
        return 0.0
    return (net_savings(df) / income) * 100

def largest_expense_category(df: pd.DataFrame) -> Tuple[str, float]:
    """Find the category with largest total expenses."""
    if df.empty or "category_display" not in df.columns or "expense_amount" not in df.columns:
        return "N/A", 0.0
    
    expense_by_category = df[df["transaction_type"] == "expense"].groupby("category_display")["amount"].sum()
    if expense_by_category.empty:
        return "N/A", 0.0
    
    largest_category = expense_by_category.idxmax()
    largest_amount = expense_by_category.max()
    
    return largest_category, largest_amount

def average_daily_spending(df: pd.DataFrame) -> float:
    """Calculate average daily spending."""
    if df.empty or "date" not in df.columns or "expense_amount" not in df.columns:
        return 0.0
    
    expenses_df = df[df["transaction_type"] == "expense"]
    if expenses_df.empty:
        return 0.0
    
    date_range = (df["date"].max() - df["date"].min()).days
    if date_range == 0:
        date_range = 1
    
    return total_expense(df) / date_range

def monthly_trend(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate month-over-month trends."""
    if df.empty or "ym_str" not in df.columns:
        return {"income_trend": 0.0, "expense_trend": 0.0, "savings_trend": 0.0}
    
    monthly_summary = df.groupby("ym_str").agg({
        "income_amount": "sum",
        "expense_amount": "sum",
        "net_amount": "sum"
    }).reset_index()
    
    if len(monthly_summary) < 2:
        return {"income_trend": 0.0, "expense_trend": 0.0, "savings_trend": 0.0}
    
    # Calculate percentage change from previous month
    monthly_summary = monthly_summary.sort_values("ym_str")
    
    def calc_trend(series):
        if len(series) < 2 or series.iloc[-2] == 0:
            return 0.0
        return ((series.iloc[-1] - series.iloc[-2]) / abs(series.iloc[-2])) * 100
    
    return {
        "income_trend": calc_trend(monthly_summary["income_amount"]),
        "expense_trend": calc_trend(monthly_summary["expense_amount"]), 
        "savings_trend": calc_trend(monthly_summary["net_amount"])
    }

def top_expense_categories(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N expense categories."""
    if df.empty or "category_display" not in df.columns:
        return pd.DataFrame()
    
    expenses_df = df[df["transaction_type"] == "expense"]
    if expenses_df.empty:
        return pd.DataFrame()
    
    top_categories = expenses_df.groupby("category_display").agg({
        "amount": ["sum", "count", "mean"]
    }).reset_index()
    
    # Flatten column names
    top_categories.columns = ["category", "total_amount", "transaction_count", "avg_amount"]
    
    # Calculate percentage of total expenses
    total_expenses = total_expense(df)
    if total_expenses > 0:
        top_categories["percentage"] = (top_categories["total_amount"] / total_expenses) * 100
    else:
        top_categories["percentage"] = 0
    
    return top_categories.sort_values("total_amount", ascending=False).head(n)

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly summary statistics."""
    if df.empty or "ym_str" not in df.columns:
        return pd.DataFrame()
    
    monthly_stats = df.groupby("ym_str").agg({
        "income_amount": "sum",
        "expense_amount": "sum", 
        "net_amount": "sum",
        "transaction_type": "count"
    }).reset_index()
    
    monthly_stats.columns = ["month", "income", "expenses", "net_savings", "transaction_count"]
    
    # Calculate savings rate
    monthly_stats["savings_rate"] = np.where(
        monthly_stats["income"] > 0,
        (monthly_stats["net_savings"] / monthly_stats["income"]) * 100,
        0
    )
    
    return monthly_stats.sort_values("month")

def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create account-wise summary."""
    if df.empty or "account" not in df.columns:
        return pd.DataFrame()
    
    account_stats = df.groupby("account").agg({
        "income_amount": "sum",
        "expense_amount": "sum",
        "net_amount": "sum",
        "transaction_type": "count"
    }).reset_index()
    
    account_stats.columns = ["account", "income", "expenses", "net", "transactions"]
    
    return account_stats.sort_values("net", ascending=False)

def get_kpi_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get all key KPI metrics in a single function call.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary with all KPI metrics
    """
    metrics = {}
    
    # Basic financial metrics
    metrics["total_income"] = total_income(df)
    metrics["total_expense"] = total_expense(df)
    metrics["net_savings"] = net_savings(df)
    metrics["savings_rate"] = savings_rate(df)
    
    # Largest expense category
    largest_cat, largest_amount = largest_expense_category(df)
    metrics["largest_expense_category"] = largest_cat
    metrics["largest_expense_amount"] = largest_amount
    
    # Other metrics
    metrics["average_daily_spending"] = average_daily_spending(df)
    
    # Trends
    trends = monthly_trend(df)
    metrics.update(trends)
    
    # Transaction counts
    if not df.empty:
        metrics["total_transactions"] = len(df)
        metrics["income_transactions"] = len(df[df["transaction_type"] == "income"])
        metrics["expense_transactions"] = len(df[df["transaction_type"] == "expense"])
    else:
        metrics["total_transactions"] = 0
        metrics["income_transactions"] = 0
        metrics["expense_transactions"] = 0
    
    return metrics

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return CURRENCY_FORMAT.format(amount)

def format_percentage(percentage: float) -> str:
    """Format percentage with proper symbol."""
    return PERCENTAGE_FORMAT.format(percentage)

def get_kpi_cards_data(df: pd.DataFrame) -> List[Dict]:
    """
    Get data for KPI cards display.
    
    Returns:
        List of dictionaries with card data
    """
    metrics = get_kpi_metrics(df)
    
    cards_data = [
        {
            "title": "Total Income",
            "value": format_currency(metrics["total_income"]),
            "delta": f"{format_percentage(metrics['income_trend'])} vs last month",
            "delta_color": "normal" if metrics["income_trend"] >= 0 else "inverse"
        },
        {
            "title": "Total Expenses", 
            "value": format_currency(metrics["total_expense"]),
            "delta": f"{format_percentage(metrics['expense_trend'])} vs last month",
            "delta_color": "inverse" if metrics["expense_trend"] >= 0 else "normal"
        },
        {
            "title": "Net Savings",
            "value": format_currency(metrics["net_savings"]),
            "delta": f"{format_percentage(metrics['savings_trend'])} vs last month",
            "delta_color": "normal" if metrics["savings_trend"] >= 0 else "inverse"
        },
        {
            "title": "Savings Rate",
            "value": format_percentage(metrics["savings_rate"]),
            "delta": f"Target: 20%",
            "delta_color": "normal" if metrics["savings_rate"] >= 20 else "off"
        },
        {
            "title": "Largest Expense Category",
            "value": metrics["largest_expense_category"],
            "delta": format_currency(metrics["largest_expense_amount"]),
            "delta_color": "off"
        },
        {
            "title": "Daily Average Spending",
            "value": format_currency(metrics["average_daily_spending"]),
            "delta": f"{metrics['expense_transactions']} expense transactions",
            "delta_color": "off"
        }
    ]
    
    return cards_data

def monthly_savings_rate_with_targets(df: pd.DataFrame, target_rate: float = 10.0) -> pd.DataFrame:
    """
    Get monthly savings rate data with target achievement indicators.
    
    Args:
        df: Processed DataFrame
        target_rate: Target savings rate percentage
        
    Returns:
        DataFrame with monthly data and target achievement flags
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        return pd.DataFrame()
    
    # Add target achievement indicators
    monthly_data["target_achieved"] = monthly_data["savings_rate"] >= target_rate
    monthly_data["target_diff"] = monthly_data["savings_rate"] - target_rate
    
    # Add performance categories
    monthly_data["performance"] = pd.cut(
        monthly_data["savings_rate"],
        bins=[-float('inf'), 0, target_rate/2, target_rate, target_rate*1.5, float('inf')],
        labels=['Negative', 'Poor', 'Below Target', 'Good', 'Excellent']
    )
    
    return monthly_data

def savings_rate_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive savings rate statistics.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary with savings rate statistics
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        return {}
    
    savings_rates = monthly_data["savings_rate"]
    
    stats = {
        "average_savings_rate": savings_rates.mean(),
        "median_savings_rate": savings_rates.median(),
        "min_savings_rate": savings_rates.min(),
        "max_savings_rate": savings_rates.max(),
        "std_savings_rate": savings_rates.std(),
        "months_above_target_10": (savings_rates >= 10).sum(),
        "months_above_target_20": (savings_rates >= 20).sum(),
        "months_negative": (savings_rates < 0).sum(),
        "total_months": len(savings_rates),
        "consistency_score": 100 - (savings_rates.std() / savings_rates.mean() * 100) if savings_rates.mean() != 0 else 0
    }
    
    # Add percentages
    if stats["total_months"] > 0:
        stats["pct_months_above_target_10"] = (stats["months_above_target_10"] / stats["total_months"]) * 100
        stats["pct_months_above_target_20"] = (stats["months_above_target_20"] / stats["total_months"]) * 100
        stats["pct_months_negative"] = (stats["months_negative"] / stats["total_months"]) * 100
    
    return stats

def identify_best_worst_months(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify the best and worst performing months for savings rate.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary with best/worst month analysis
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        return {}
    
    # Sort by savings rate
    sorted_data = monthly_data.sort_values("savings_rate", ascending=False)
    
    analysis = {
        "best_month": {
            "month": sorted_data.iloc[0]["month"],
            "savings_rate": sorted_data.iloc[0]["savings_rate"],
            "net_savings": sorted_data.iloc[0]["net_savings"],
            "income": sorted_data.iloc[0]["income"],
            "expenses": sorted_data.iloc[0]["expenses"]
        },
        "worst_month": {
            "month": sorted_data.iloc[-1]["month"],
            "savings_rate": sorted_data.iloc[-1]["savings_rate"],
            "net_savings": sorted_data.iloc[-1]["net_savings"],
            "income": sorted_data.iloc[-1]["income"],
            "expenses": sorted_data.iloc[-1]["expenses"]
        }
    }
    
    # Add top 3 and bottom 3 if we have enough data
    if len(sorted_data) >= 3:
        analysis["top_3_months"] = []
        analysis["bottom_3_months"] = []
        
        for i in range(min(3, len(sorted_data))):
            analysis["top_3_months"].append({
                "month": sorted_data.iloc[i]["month"],
                "savings_rate": sorted_data.iloc[i]["savings_rate"],
                "net_savings": sorted_data.iloc[i]["net_savings"]
            })
            
            analysis["bottom_3_months"].append({
                "month": sorted_data.iloc[-(i+1)]["month"],
                "savings_rate": sorted_data.iloc[-(i+1)]["savings_rate"],
                "net_savings": sorted_data.iloc[-(i+1)]["net_savings"]
            })
    
    # Calculate improvement opportunities
    avg_rate = monthly_data["savings_rate"].mean()
    below_avg_months = monthly_data[monthly_data["savings_rate"] < avg_rate]
    
    if not below_avg_months.empty:
        analysis["improvement_potential"] = {
            "months_below_average": len(below_avg_months),
            "average_shortfall": avg_rate - below_avg_months["savings_rate"].mean(),
            "potential_additional_savings": below_avg_months["expenses"].mean() * (analysis["improvement_potential"]["average_shortfall"] / 100) if "improvement_potential" in analysis else 0
        }
    
    return analysis

def get_enhanced_kpi_metrics(df: pd.DataFrame, target_rate: float = 10.0) -> Dict[str, Any]:
    """
    Get enhanced KPI metrics including savings rate analysis.
    
    Args:
        df: Processed DataFrame
        target_rate: Target savings rate percentage
        
    Returns:
        Dictionary with enhanced KPI metrics
    """
    # Get basic metrics
    basic_metrics = get_kpi_metrics(df)
    
    # Get enhanced savings rate analysis
    savings_stats = savings_rate_statistics(df)
    best_worst = identify_best_worst_months(df)
    
    # Combine all metrics
    enhanced_metrics = {
        **basic_metrics,
        "savings_rate_stats": savings_stats,
        "best_worst_analysis": best_worst,
        "target_rate": target_rate
    }
    
    return enhanced_metrics 