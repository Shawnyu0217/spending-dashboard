"""
Main dashboard application entry point.
Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Import our modules
from app.pages import get_available_pages
from app.config import PAGE_CONFIG
from app.data.loader import get_data_for_dashboard
from app.data.preprocess import preprocess_data
from app.data.filters import filter_data_by_selections
from app.features.viz import (
    create_monthly_trend_chart,
    create_category_expense_chart,
    create_daily_spending_heatmap,
    create_savings_rate_gauge,
    create_monthly_savings_rate_chart,
    create_monthly_savings_rate_chart_enhanced,
    create_cumulative_balance_chart,
    create_expense_distribution_pie
)
from app.ui.components import (
    create_sidebar_filters,
    display_kpi_cards,
    display_data_summary,
    display_filter_summary,
    create_data_table_section,
    create_export_section,
    create_chart_container,
    create_chart_configuration_controls,
    display_savings_rate_insights,
    show_loading_spinner,
    show_error_message
)

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(**PAGE_CONFIG)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: transparent;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stMetric {
        background: transparent;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = pd.DataFrame()
    if "dim_tables" not in st.session_state:
        st.session_state.dim_tables = {}
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

def clear_cache_button():
    """Add a button to clear cache in sidebar."""
    if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.df_processed = pd.DataFrame()
        st.session_state.dim_tables = {}
        st.success("Cache cleared! Data will be reloaded.")
        st.rerun()

def load_and_process_data(uploaded_file=None) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load and process data with caching.
    
    Args:
        uploaded_file: Optional uploaded file
        
    Returns:
        Tuple of (processed_df, dimension_tables)
    """
    # Check if we need to reload data
    need_reload = (
        not st.session_state.data_loaded or
        uploaded_file is not None or
        st.session_state.df_processed.empty
    )
    
    if need_reload:
        with show_loading_spinner("Loading and processing data..."):
            try:
                # Load raw data
                df_raw = get_data_for_dashboard(uploaded_file)
                
                if df_raw.empty:
                    return pd.DataFrame(), {}
                
                # Process data
                df_processed, dim_tables = preprocess_data(df_raw)
                
                # Cache in session state
                st.session_state.df_processed = df_processed
                st.session_state.dim_tables = dim_tables
                st.session_state.data_loaded = True
                
                return df_processed, dim_tables
                
            except Exception as e:
                show_error_message("Failed to load data", str(e))
                return pd.DataFrame(), {}
    
    return st.session_state.df_processed, st.session_state.dim_tables

def render_tabbed_interface(df_filtered: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """
    Render the tabbed interface with available pages.
    
    Args:
        df_filtered: The filtered transaction data
        filters: Current filter selections
    """
    # Get pages that should be displayed
    pages = get_available_pages(df_filtered)
    
    if not pages:
        st.error("❌ No analysis pages available for the current data selection.")
        st.info("💡 Try adjusting your filters or uploading more data.")
        return
    
    # Create tab navigation
    tab_names = [f"{page.icon} {page.name}" for page in pages]
    tabs = st.tabs(tab_names)
    
    # Render each page in its corresponding tab
    for tab, page in zip(tabs, pages):
        with tab:
            with st.container():
                # Add loading state
                with st.spinner(f"Loading {page.name}..."):
                    try:
                        page.render(df_filtered, filters)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading {page.name}")
                        error_details = st.empty()
                        if st.button(f"Show details", key=f"details_{page.key}"):
                            error_details.exception(e)

def create_main_dashboard():
    """Create the main dashboard interface."""
    
    # Page header
    st.markdown('<h1 class="main-header">💰 Personal Spending Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load and process data first (with just file upload)
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Excel file",
        type=['xlsx', 'xls'],
        help="Upload your expense tracking Excel file"
    )
    
    # Add cache clear button
    clear_cache_button()
    
    if uploaded_file:
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("💡 Using sample data file")
    
    df_processed, dim_tables = load_and_process_data(uploaded_file)
    
    if df_processed.empty:
        st.warning("⚠️ No data available. Please check your data file or upload a new one.")
        st.stop()
    
    # Now create sidebar filters with actual data
    filters = create_sidebar_filters(dim_tables)
    # Add the uploaded file to filters for consistency
    filters["uploaded_file"] = uploaded_file
    
    # Apply filters to data
    df_filtered = filter_data_by_selections(
        df_processed,
        selected_years=filters.get("selected_years"),
        selected_months=filters.get("selected_months"),
        selected_categories=filters.get("selected_categories"),
        selected_accounts=filters.get("selected_accounts"),
        selected_members=filters.get("selected_members")
    )
    
    # Display filter summary
    display_filter_summary(filters, df_filtered, df_processed)
    
    # Create chart configuration controls (outside of tabs to avoid key conflicts)
    chart_config = create_chart_configuration_controls()
    # Add chart config to filters for pages to use
    filters["chart_config"] = chart_config
    
    st.divider()
    
    # NEW: Replace all the chart sections with tabbed interface
    render_tabbed_interface(df_filtered, filters)
    
    # IMPORTANT: Keep these sections commented for reference during migration
    # # Create chart columns
    # col1, col2 = st.columns(2)
    # 
    # with col1:
    #     create_chart_container(
    #         "📈 Monthly Trends",
    #         create_monthly_trend_chart,
    #         df_filtered
    #     )
    # 
    # with col2:
    #     create_chart_container(
    #         "🏆 Top Expense Categories",
    #         create_category_expense_chart,
    #         df_filtered
    #     )
    # 
    # # Second row of charts
    # col3, col4 = st.columns(2)
    # 
    # with col3:
    #     create_chart_container(
    #         "📊 Savings Rate",
    #         create_savings_rate_gauge,
    #         df_filtered
    #     )
    # 
    # with col4:
    #     create_chart_container(
    #         "🥧 Expense Distribution",
    #         create_expense_distribution_pie,
    #         df_filtered
    #     )
    # 
    # # Third row - Monthly Savings Rate Trends (full width with enhanced features)
    # def enhanced_savings_chart(df):
    #     return create_monthly_savings_rate_chart_enhanced(
    #         df,
    #         target_rate=chart_config.get("target_rate", 10.0),
    #         show_dual_axis=chart_config.get("show_dual_axis", False),
    #         show_moving_average=chart_config.get("show_moving_average", True),
    #         ma_periods=chart_config.get("ma_periods", 3)
    #     )
    # 
    # create_chart_container(
    #     "📊 Monthly Savings Rate Trends",
    #     enhanced_savings_chart,
    #     df_filtered
    # )
    # 
    # # Add insights section for savings rate
    # with st.expander("💡 Savings Rate Insights", expanded=False):
    #     display_savings_rate_insights(df_filtered, chart_config.get("target_rate", 10.0))
    # 
    # # Fourth row - Cumulative Balance (full width)
    # create_chart_container(
    #     "📈 Cumulative Balance",
    #     create_cumulative_balance_chart,
    #     df_filtered
    # )
    # 
    # # Optional: Daily spending heatmap (full width)
    # if len(df_filtered) > 30:  # Only show if we have enough data
    #     create_chart_container(
    #         "🔥 Daily Spending Heatmap",
    #         create_daily_spending_heatmap,
    #         df_filtered
    #     )
    # 
    # st.divider()
    # 
    # # Data sections
    # st.header("📋 Data Details")
    # 
    # # Data summary
    # display_data_summary(df_filtered)
    # 
    # # Detailed data table
    # create_data_table_section(df_filtered)
    # 
    # # Export functionality
    # create_export_section(df_filtered)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        💡 Personal Spending Dashboard | Built with Streamlit & Plotly<br>
        📊 Analyze your financial data with interactive charts and insights
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    try:
        configure_page()
        create_main_dashboard()
    except Exception as e:
        st.error("❌ Application Error")
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Show error details in development
        if st.checkbox("Show detailed error information"):
            st.exception(e)

if __name__ == "__main__":
    main()
