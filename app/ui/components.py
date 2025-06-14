"""
Reusable Streamlit UI components for the spending dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date

from ..features.kpis import get_kpi_cards_data
from ..data.preprocess import filter_data_by_selections

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
    
    # File uploader
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file",
        type=['xlsx', 'xls'],
        help="Upload your expense tracking Excel file"
    )
    filters["uploaded_file"] = uploaded_file
    
    if uploaded_file:
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("💡 Using sample data file")
    
    st.sidebar.divider()
    
    # Year filter
    if "years" in dim_tables and not dim_tables["years"].empty:
        st.sidebar.subheader("📅 Time Period")
        available_years = dim_tables["years"]["year"].tolist()
        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=available_years,
            default=available_years[-2:] if len(available_years) >= 2 else available_years,
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
        
        # Show top categories by default
        default_categories = available_categories[:10] if len(available_categories) > 10 else available_categories
        
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=available_categories,
            default=default_categories,
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
    Create section for data export functionality.
    
    Args:
        df: DataFrame to export
    """
    with st.expander("💾 Export Data", expanded=False):
        if df.empty:
            st.warning("No data to export")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"spending_data_{date.today().isoformat()}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary export
            if hasattr(df, 'describe'):
                summary_data = df.describe().to_csv()
                st.download_button(
                    label="📊 Download Summary Stats",
                    data=summary_data,
                    file_name=f"spending_summary_{date.today().isoformat()}.csv",
                    mime="text/csv"
                )

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
