"""
Reusable Streamlit UI components for the spending dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date

from ..features.kpis import get_kpi_cards_data, get_kpi_metrics, get_enhanced_kpi_metrics
from ..data.filters import filter_data_by_selections

def create_sidebar_filters(dim_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create sidebar filters and return selected values.
    
    Args:
        dim_tables: Dictionary of dimension tables
        
    Returns:
        Dictionary with selected filter values
    """
    st.sidebar.header("📊 Dashboard Filters")
    
    filters = {}
    
    st.sidebar.divider()
    
    # Year filter
    if "years" in dim_tables and not dim_tables["years"].empty:
        st.sidebar.subheader("📅 Time Period")
        available_years = dim_tables["years"]["year"].tolist()
        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=available_years,
            default=available_years,  # Select all years by default
            help="Choose which years to include in analysis"
        )
        filters["selected_years"] = selected_years
    else:
        filters["selected_years"] = []
    
    # Month filter
    if "months" in dim_tables and not dim_tables["months"].empty:
        available_months = dim_tables["months"]["month"].tolist()
        month_names = dim_tables["months"]["month_name"].tolist()
        
        selected_months = st.sidebar.multiselect(
            "Select Months",
            options=available_months,
            format_func=lambda x: month_names[available_months.index(x)] if x in available_months else str(x),
            default=available_months,
            help="Choose which months to include"
        )
        filters["selected_months"] = selected_months
    else:
        filters["selected_months"] = []
    
    st.sidebar.divider()
    
    # Category filter
    if "categories" in dim_tables and not dim_tables["categories"].empty:
        st.sidebar.subheader("🏷️ Categories")
        available_categories = dim_tables["categories"]["category"].tolist()
        
        # Select all categories by default
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=available_categories,
            default=available_categories,  # Select all categories by default
            help="Choose expense categories to analyze"
        )
        filters["selected_categories"] = selected_categories
    else:
        filters["selected_categories"] = []
    
    # Account filter
    if "accounts" in dim_tables and not dim_tables["accounts"].empty:
        st.sidebar.subheader("🏦 Accounts")
        available_accounts = dim_tables["accounts"]["account"].tolist()
        selected_accounts = st.sidebar.multiselect(
            "Select Accounts",
            options=available_accounts,
            default=available_accounts,
            help="Choose accounts to include"
        )
        filters["selected_accounts"] = selected_accounts
    else:
        filters["selected_accounts"] = []
    
    # Member filter
    if "members" in dim_tables and not dim_tables["members"].empty:
        st.sidebar.subheader("👥 Family Members")
        available_members = dim_tables["members"]["member"].tolist()
        selected_members = st.sidebar.multiselect(
            "Select Members",
            options=available_members,
            default=available_members,
            help="Choose family members to include"
        )
        filters["selected_members"] = selected_members
    else:
        filters["selected_members"] = []
    
    return filters

def display_kpi_cards(df: pd.DataFrame):
    """
    Display KPI cards in the main area.
    
    Args:
        df: Filtered DataFrame
    """
    st.subheader("📈 Key Performance Indicators")
    
    cards_data = get_kpi_cards_data(df)
    
    # Create columns for KPI cards
    cols = st.columns(3)
    
    for i, card in enumerate(cards_data):
        col_idx = i % 3
        
        with cols[col_idx]:
            # Determine delta color
            delta_color = "normal"
            if card["delta_color"] == "inverse":
                delta_color = "inverse"
            elif card["delta_color"] == "off":
                delta_color = "off"
            
            st.metric(
                label=card["title"],
                value=card["value"],
                delta=card["delta"] if delta_color else None,
                delta_color=delta_color
            )

def display_data_summary(df: pd.DataFrame):
    """
    Display a summary of the loaded data.
    
    Args:
        df: DataFrame to summarize
    """
    if df.empty:
        st.warning("⚠️ No data available")
        return
    
    with st.expander("📋 Data Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if "date" in df.columns:
                date_range = df["date"].max() - df["date"].min()
                st.metric("Date Range", f"{date_range.days} days")
            else:
                st.metric("Date Range", "N/A")
        
        with col3:
            if "transaction_type" in df.columns:
                unique_types = df["transaction_type"].nunique()
                st.metric("Transaction Types", unique_types)
            else:
                st.metric("Transaction Types", "N/A")
        
        with col4:
            if "category_display" in df.columns:
                unique_categories = df["category_display"].nunique()
                st.metric("Categories", unique_categories)
            else:
                st.metric("Categories", "N/A")

def display_filter_summary(filters: Dict[str, Any], df_filtered: pd.DataFrame, df_total: pd.DataFrame):
    """
    Display a summary of applied filters.
    
    Args:
        filters: Dictionary of filter selections
        df_filtered: Filtered DataFrame
        df_total: Total (unfiltered) DataFrame
    """
    if not df_total.empty:
        filter_ratio = len(df_filtered) / len(df_total) * 100
        
        st.info(
            f"📊 Showing {len(df_filtered):,} of {len(df_total):,} records "
            f"({filter_ratio:.1f}%) based on current filters"
        )

def create_data_table_section(df: pd.DataFrame):
    """
    Create an expandable section with detailed data table.
    
    Args:
        df: DataFrame to display
    """
    with st.expander("🗂️ Detailed Transaction Data", expanded=False):
        if df.empty:
            st.warning("No data to display")
            return
        
        # Select relevant columns for display
        display_columns = [
            "date", "transaction_type", "category_display", "amount", 
            "account", "notes", "running_balance"
        ]
        
        # Filter to only columns that exist
        available_columns = [col for col in display_columns if col in df.columns]
        
        if not available_columns:
            st.warning("No suitable columns found for display")
            return
        
        # Format the data for better display
        df_display = df[available_columns].copy()
        
        # Format currency columns
        currency_columns = ["amount", "running_balance"]
        for col in currency_columns:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"¥{x:,.2f}" if pd.notnull(x) else "")
        
        # Format date column
        if "date" in df_display.columns:
            df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            show_rows = st.selectbox(
                "Rows to show:",
                options=[25, 50, 100, 500],
                index=1
            )
        
        with col2:
            search_term = st.text_input(
                "Search in notes/categories:",
                placeholder="Enter search term..."
            )
        
        # Apply search filter
        if search_term:
            search_mask = (
                df_display.get("notes", pd.Series(dtype=str)).str.contains(search_term, case=False, na=False) |
                df_display.get("category_display", pd.Series(dtype=str)).str.contains(search_term, case=False, na=False)
            )
            df_display = df_display[search_mask]
        
        # Display the table
        st.dataframe(
            df_display.head(show_rows),
            use_container_width=True,
            hide_index=True
        )
        
        if len(df_display) > show_rows:
            st.info(f"Showing first {show_rows} of {len(df_display)} matching records")

def create_export_section(df: pd.DataFrame):
    """
    Create export functionality section.
    
    Args:
        df: Filtered DataFrame to export
    """
    st.header("📤 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"spending_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export Summary"):
            metrics = get_kpi_metrics(df)
            
            summary_text = f"""
            Spending Dashboard Summary - {datetime.now().strftime('%Y-%m-%d')}
            
            📊 Financial Overview:
            • Total Income: ¥{metrics['total_income']:,.2f}
            • Total Expenses: ¥{metrics['total_expense']:,.2f}
            • Net Savings: ¥{metrics['net_savings']:,.2f}
            • Savings Rate: {metrics['savings_rate']:.1f}%
            
            📋 Transaction Details:
            • Total Transactions: {metrics['total_transactions']}
            • Income Transactions: {metrics['income_transactions']}
            • Expense Transactions: {metrics['expense_transactions']}
            
            🏆 Top Expense Category: {metrics['largest_expense_category']}
            💰 Daily Average Spending: ¥{metrics['average_daily_spending']:,.2f}
            """
            
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"spending_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col3:
        st.info("💡 Export your filtered data for external analysis or backup")

def show_loading_spinner(message: str = "Loading..."):
    """
    Display a loading spinner with message.
    
    Args:
        message: Message to show with spinner
    """
    return st.spinner(message)

def show_error_message(error: str, details: str = None):
    """
    Display a formatted error message.
    
    Args:
        error: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {error}")
    if details:
        with st.expander("Error Details"):
            st.code(details)

def show_success_message(message: str):
    """
    Display a success message.
    
    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")

def create_chart_container(title: str, chart_func, *args, **kwargs):
    """
    Create a container for charts with error handling.
    
    Args:
        title: Chart title
        chart_func: Function that creates the chart
        *args: Arguments for chart function
        **kwargs: Keyword arguments for chart function
    """
    try:
        with st.container():
            st.subheader(title)
            fig = chart_func(*args, **kwargs)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating chart '{title}': {str(e)}")
        with st.expander("Chart Error Details"):
            st.code(str(e))

def create_chart_configuration_controls() -> Dict[str, Any]:
    """
    Create chart configuration controls for enhanced features.
    
    Returns:
        Dictionary with chart configuration options
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Chart Options")
    
    config = {}
    
    # Savings rate chart options
    with st.sidebar.expander("💰 Savings Rate Chart", expanded=False):
        config["target_rate"] = st.slider(
            "Target Savings Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Set your target savings rate percentage"
        )
        
        config["show_dual_axis"] = st.checkbox(
            "Dual Axis View",
            value=False,
            help="Show both percentage and absolute amounts"
        )
        
        config["show_moving_average"] = st.checkbox(
            "Show Trend Line",
            value=True,
            help="Display moving average trend line"
        )
        
        if config["show_moving_average"]:
            config["ma_periods"] = st.selectbox(
                "Trend Periods",
                options=[2, 3, 4, 6],
                index=1,
                help="Number of months for trend calculation"
            )
        else:
            config["ma_periods"] = 3
    
    # General chart options
    with st.sidebar.expander("🎨 Display Options", expanded=False):
        config["chart_height"] = st.selectbox(
            "Chart Height",
            options=[300, 400, 500, 600],
            index=1,
            help="Adjust chart height for better viewing"
        )
        
        config["show_annotations"] = st.checkbox(
            "Show Annotations",
            value=True,
            help="Display target lines and reference marks"
        )
        
        config["compact_mode"] = st.checkbox(
            "Compact Mode",
            value=False,
            help="Reduce spacing for more charts on screen"
        )
    
    return config

def display_savings_rate_insights(df: pd.DataFrame, target_rate: float = 10.0):
    """
    Display insights and analysis for savings rate performance.
    
    Args:
        df: Processed DataFrame
        target_rate: Target savings rate percentage
    """
    metrics = get_enhanced_kpi_metrics(df, target_rate)
    
    if not metrics.get("savings_rate_stats") or not metrics.get("best_worst_analysis"):
        st.info("📊 Insufficient data for detailed savings rate analysis")
        return
    
    stats = metrics["savings_rate_stats"]
    analysis = metrics["best_worst_analysis"]
    
    # Performance overview
    st.subheader("📈 Savings Rate Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Rate",
            f"{stats['average_savings_rate']:.1f}%",
            delta=f"{stats['average_savings_rate'] - target_rate:+.1f}% vs target"
        )
    
    with col2:
        st.metric(
            "Best Month",
            f"{stats['max_savings_rate']:.1f}%",
            delta=f"{analysis['best_month']['month']}"
        )
    
    with col3:
        st.metric(
            "Target Achievement",
            f"{stats['pct_months_above_target_10']:.0f}%",
            delta=f"{stats['months_above_target_10']}/{stats['total_months']} months"
        )
    
    with col4:
        consistency_color = "normal" if stats['consistency_score'] > 80 else "inverse"
        st.metric(
            "Consistency Score",
            f"{stats['consistency_score']:.0f}/100",
            delta="Higher is better",
            delta_color=consistency_color
        )
    
    # Best and worst months
    col5, col6 = st.columns(2)
    
    with col5:
        st.success("🏆 **Best Performance**")
        best = analysis["best_month"]
        st.write(f"📅 **Month:** {best['month']}")
        st.write(f"📊 **Rate:** {best['savings_rate']:.1f}%")
        st.write(f"💰 **Saved:** ¥{best['net_savings']:,.0f}")
    
    with col6:
        st.error("📉 **Needs Improvement**")
        worst = analysis["worst_month"]
        st.write(f"📅 **Month:** {worst['month']}")
        st.write(f"📊 **Rate:** {worst['savings_rate']:.1f}%")
        st.write(f"💰 **Amount:** ¥{worst['net_savings']:,.0f}")
    
    # Improvement suggestions
    if "improvement_potential" in analysis:
        st.info("💡 **Improvement Opportunities**")
        potential = analysis["improvement_potential"]
        st.write(f"• {potential['months_below_average']} months below average")
        st.write(f"• Average shortfall: {potential['average_shortfall']:.1f}%")
