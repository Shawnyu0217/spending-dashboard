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

def monthly_summary(df: pd.DataFrame, include_comparisons: bool = False) -> pd.DataFrame:
    """
    Create monthly summary statistics with optional YoY and MoM comparisons.

    Args:
        df: Processed DataFrame
        include_comparisons: Whether to include YoY and MoM comparison metrics

    Returns:
        DataFrame with monthly statistics and optional comparison metrics
    """
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

    monthly_stats = monthly_stats.sort_values("month")

    # Add comparison metrics if requested
    if include_comparisons and len(monthly_stats) > 1:
        # Extract year and month for comparisons
        monthly_stats["year"] = monthly_stats["month"].str[:4].astype(int)
        monthly_stats["month_num"] = monthly_stats["month"].str[5:7].astype(int)

        # Month-over-Month comparisons
        for col in ["income", "expenses", "net_savings", "savings_rate"]:
            monthly_stats[f"{col}_mom_change"] = monthly_stats[col].diff()
            monthly_stats[f"{col}_mom_pct"] = (
                monthly_stats[f"{col}_mom_change"] / monthly_stats[col].shift(1) * 100
            ).fillna(0)

        # Year-over-Year comparisons
        for col in ["income", "expenses", "net_savings", "savings_rate"]:
            # Create a mapping of (year, month) -> value for quick lookup
            yoy_map = monthly_stats.set_index(["year", "month_num"])[col].to_dict()

            def get_yoy_value(row):
                prev_year_key = (row["year"] - 1, row["month_num"])
                return yoy_map.get(prev_year_key, None)

            monthly_stats[f"{col}_yoy_prev"] = monthly_stats.apply(get_yoy_value, axis=1)
            monthly_stats[f"{col}_yoy_change"] = monthly_stats[col] - monthly_stats[f"{col}_yoy_prev"]
            monthly_stats[f"{col}_yoy_pct"] = np.where(
                monthly_stats[f"{col}_yoy_prev"].notna() & (monthly_stats[f"{col}_yoy_prev"] != 0),
                (monthly_stats[f"{col}_yoy_change"] / monthly_stats[f"{col}_yoy_prev"]) * 100,
                np.nan
            )

    return monthly_stats

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
        average_shortfall = avg_rate - below_avg_months["savings_rate"].mean()
        analysis["improvement_potential"] = {
            "months_below_average": len(below_avg_months),
            "average_shortfall": average_shortfall,
            "potential_additional_savings": below_avg_months["expenses"].mean() * (average_shortfall / 100)
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

def calculate_time_pattern_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate spending pattern statistics based on time of day and day of week.

    Args:
        df: Processed DataFrame with time components (hour, time_of_day, is_weekend)

    Returns:
        Dictionary with time-based spending pattern statistics
    """
    if df.empty:
        return {}

    # Check for required columns
    required_cols = ["hour", "time_of_day", "is_weekend", "weekday"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}

    expenses_df = df[df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        return {"error": "No expense data available"}

    stats = {}

    # 1. Hour-based statistics
    hourly_spending = expenses_df.groupby("hour").agg({
        "amount": ["sum", "count", "mean"]
    }).reset_index()
    hourly_spending.columns = ["hour", "total_amount", "transaction_count", "avg_amount"]

    peak_hour = hourly_spending.loc[hourly_spending["total_amount"].idxmax()]
    stats["peak_spending_hour"] = {
        "hour": int(peak_hour["hour"]),
        "hour_display": f"{int(peak_hour['hour']):02d}:00",
        "total_amount": float(peak_hour["total_amount"]),
        "transaction_count": int(peak_hour["transaction_count"]),
        "avg_amount": float(peak_hour["avg_amount"])
    }

    # 2. Time of day statistics
    time_of_day_spending = expenses_df.groupby("time_of_day").agg({
        "amount": ["sum", "count", "mean"]
    }).reset_index()
    time_of_day_spending.columns = ["time_of_day", "total_amount", "transaction_count", "avg_amount"]

    stats["spending_by_time_of_day"] = {}
    for _, row in time_of_day_spending.iterrows():
        stats["spending_by_time_of_day"][str(row["time_of_day"])] = {
            "total_amount": float(row["total_amount"]),
            "transaction_count": int(row["transaction_count"]),
            "avg_amount": float(row["avg_amount"]),
            "percentage": float((row["total_amount"] / expenses_df["amount"].sum()) * 100)
        }

    # 3. Day of week statistics
    weekday_spending = expenses_df.groupby("weekday").agg({
        "amount": ["sum", "count", "mean"]
    }).reset_index()
    weekday_spending.columns = ["weekday", "total_amount", "transaction_count", "avg_amount"]

    # Reorder weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_spending["weekday"] = pd.Categorical(weekday_spending["weekday"], categories=weekday_order, ordered=True)
    weekday_spending = weekday_spending.sort_values("weekday")

    peak_weekday = weekday_spending.loc[weekday_spending["total_amount"].idxmax()]
    stats["peak_spending_weekday"] = {
        "weekday": str(peak_weekday["weekday"]),
        "total_amount": float(peak_weekday["total_amount"]),
        "transaction_count": int(peak_weekday["transaction_count"]),
        "avg_amount": float(peak_weekday["avg_amount"])
    }

    stats["spending_by_weekday"] = weekday_spending.to_dict('records')

    # 4. Weekend vs Weekday comparison
    weekend_expenses = expenses_df[expenses_df["is_weekend"]]
    weekday_expenses = expenses_df[~expenses_df["is_weekend"]]

    weekend_total = weekend_expenses["amount"].sum() if not weekend_expenses.empty else 0
    weekday_total = weekday_expenses["amount"].sum() if not weekday_expenses.empty else 0
    weekend_count = len(weekend_expenses)
    weekday_count = len(weekday_expenses)

    # Calculate per-day averages
    weekend_days = 2  # Saturday and Sunday
    weekday_days = 5  # Monday to Friday

    # Get actual number of unique weekend and weekday dates
    if not weekend_expenses.empty and "date" in weekend_expenses.columns:
        actual_weekend_days = weekend_expenses["date"].dt.date.nunique()
        weekend_avg_per_day = weekend_total / actual_weekend_days if actual_weekend_days > 0 else 0
    else:
        weekend_avg_per_day = 0

    if not weekday_expenses.empty and "date" in weekday_expenses.columns:
        actual_weekday_days = weekday_expenses["date"].dt.date.nunique()
        weekday_avg_per_day = weekday_total / actual_weekday_days if actual_weekday_days > 0 else 0
    else:
        weekday_avg_per_day = 0

    stats["weekend_vs_weekday"] = {
        "weekend": {
            "total_amount": float(weekend_total),
            "transaction_count": int(weekend_count),
            "avg_per_day": float(weekend_avg_per_day),
            "avg_per_transaction": float(weekend_expenses["amount"].mean()) if not weekend_expenses.empty else 0
        },
        "weekday": {
            "total_amount": float(weekday_total),
            "transaction_count": int(weekday_count),
            "avg_per_day": float(weekday_avg_per_day),
            "avg_per_transaction": float(weekday_expenses["amount"].mean()) if not weekday_expenses.empty else 0
        },
        "comparison": {
            "weekend_spends_more": weekend_avg_per_day > weekday_avg_per_day,
            "difference_pct": float(((weekend_avg_per_day - weekday_avg_per_day) / weekday_avg_per_day * 100) if weekday_avg_per_day > 0 else 0)
        }
    }

    # 5. Peak spending time combination (day + hour)
    day_hour_spending = expenses_df.groupby(["weekday", "hour"])["amount"].sum().reset_index()
    if not day_hour_spending.empty:
        peak_combo = day_hour_spending.loc[day_hour_spending["amount"].idxmax()]
        stats["peak_day_hour_combo"] = {
            "weekday": str(peak_combo["weekday"]),
            "hour": int(peak_combo["hour"]),
            "hour_display": f"{int(peak_combo['hour']):02d}:00",
            "total_amount": float(peak_combo["amount"])
        }

    # 6. Most active hours (top 5)
    top_hours = hourly_spending.nlargest(5, "transaction_count")[["hour", "transaction_count", "total_amount"]]
    stats["most_active_hours"] = top_hours.to_dict('records')

    return stats


def calculate_yoy_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Year-over-Year comparison metrics for the latest available month.

    Args:
        df: Processed DataFrame with ym_str column

    Returns:
        Dictionary with YoY metrics for latest month vs same month last year
    """
    monthly_data = monthly_summary(df, include_comparisons=True)

    if monthly_data.empty or len(monthly_data) < 2:
        return {}

    # Get the latest month with YoY data available
    latest_with_yoy = monthly_data[monthly_data["income_yoy_prev"].notna()]

    if latest_with_yoy.empty:
        return {"error": "Insufficient data for year-over-year comparison"}

    latest_row = latest_with_yoy.iloc[-1]

    yoy_metrics = {
        "latest_month": latest_row["month"],
        "income": {
            "current": latest_row["income"],
            "previous_year": latest_row["income_yoy_prev"],
            "change": latest_row["income_yoy_change"],
            "change_pct": latest_row["income_yoy_pct"]
        },
        "expenses": {
            "current": latest_row["expenses"],
            "previous_year": latest_row["expenses_yoy_prev"],
            "change": latest_row["expenses_yoy_change"],
            "change_pct": latest_row["expenses_yoy_pct"]
        },
        "net_savings": {
            "current": latest_row["net_savings"],
            "previous_year": latest_row["net_savings_yoy_prev"],
            "change": latest_row["net_savings_yoy_change"],
            "change_pct": latest_row["net_savings_yoy_pct"]
        },
        "savings_rate": {
            "current": latest_row["savings_rate"],
            "previous_year": latest_row["savings_rate_yoy_prev"],
            "change": latest_row["savings_rate_yoy_change"],
            "change_pct": latest_row["savings_rate_yoy_pct"]
        }
    }

    return yoy_metrics


def calculate_mom_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Month-over-Month comparison metrics for the latest available month.

    Args:
        df: Processed DataFrame with ym_str column

    Returns:
        Dictionary with MoM metrics for latest month vs previous month
    """
    monthly_data = monthly_summary(df, include_comparisons=True)

    if monthly_data.empty or len(monthly_data) < 2:
        return {}

    # Get the latest month (MoM always available for 2nd month onward)
    latest_row = monthly_data.iloc[-1]
    previous_row = monthly_data.iloc[-2]

    mom_metrics = {
        "latest_month": latest_row["month"],
        "previous_month": previous_row["month"],
        "income": {
            "current": latest_row["income"],
            "previous": previous_row["income"],
            "change": latest_row["income_mom_change"],
            "change_pct": latest_row["income_mom_pct"]
        },
        "expenses": {
            "current": latest_row["expenses"],
            "previous": previous_row["expenses"],
            "change": latest_row["expenses_mom_change"],
            "change_pct": latest_row["expenses_mom_pct"]
        },
        "net_savings": {
            "current": latest_row["net_savings"],
            "previous": previous_row["net_savings"],
            "change": latest_row["net_savings_mom_change"],
            "change_pct": latest_row["net_savings_mom_pct"]
        },
        "savings_rate": {
            "current": latest_row["savings_rate"],
            "previous": previous_row["savings_rate"],
            "change": latest_row["savings_rate_mom_change"],
            "change_pct": latest_row["savings_rate_mom_pct"]
        }
    }

    return mom_metrics


def analyze_original_categories(df: pd.DataFrame, transaction_type: str = "expense") -> pd.DataFrame:
    """
    Analyze spending by original Chinese category and subcategory names.

    Args:
        df: Processed DataFrame with original 'category' and 'subcategory' columns
        transaction_type: Type of transaction to analyze ('expense', 'income', or 'all')

    Returns:
        DataFrame with category, subcategory, amount, count, avg, and percentage
    """
    if df.empty:
        return pd.DataFrame()

    # Filter by transaction type if specified
    if transaction_type != "all":
        filtered_df = df[df["transaction_type"] == transaction_type].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        return pd.DataFrame()

    # Check for required columns
    if "category" not in filtered_df.columns:
        return pd.DataFrame()

    # Handle cases where subcategory might not exist
    has_subcategory = "subcategory" in filtered_df.columns

    if has_subcategory:
        # Group by both category and subcategory
        grouped = filtered_df.groupby(["category", "subcategory"], dropna=False).agg({
            "amount": ["sum", "count", "mean"]
        }).reset_index()
        grouped.columns = ["category", "subcategory", "total_amount", "transaction_count", "avg_amount"]
    else:
        # Group by category only
        grouped = filtered_df.groupby("category", dropna=False).agg({
            "amount": ["sum", "count", "mean"]
        }).reset_index()
        grouped.columns = ["category", "total_amount", "transaction_count", "avg_amount"]
        grouped["subcategory"] = ""

    # Calculate percentage of total
    total_amount = grouped["total_amount"].sum()
    if total_amount > 0:
        grouped["percentage"] = (grouped["total_amount"] / total_amount) * 100
    else:
        grouped["percentage"] = 0

    # Sort by total amount descending
    grouped = grouped.sort_values("total_amount", ascending=False)

    return grouped


def top_original_subcategories(df: pd.DataFrame, n: int = 20, transaction_type: str = "expense") -> pd.DataFrame:
    """
    Get top N original subcategories by spending amount (Chinese names).

    Args:
        df: Processed DataFrame
        n: Number of top subcategories to return
        transaction_type: Type of transaction to analyze

    Returns:
        DataFrame with top subcategories and their statistics
    """
    category_analysis = analyze_original_categories(df, transaction_type)

    if category_analysis.empty:
        return pd.DataFrame()

    # Filter out entries without subcategory if needed
    if "subcategory" in category_analysis.columns:
        category_analysis = category_analysis[
            category_analysis["subcategory"].notna() &
            (category_analysis["subcategory"] != "")
        ]

    # Return top N
    return category_analysis.head(n) 