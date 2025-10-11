"""
Tab-based UI components for organizing dashboard content.
Each tab provides a focused view of different analytical aspects.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from ..features.viz import (
    create_monthly_trend_chart,
    create_category_expense_chart,
    create_daily_spending_heatmap,
    create_savings_rate_gauge,
    create_monthly_savings_rate_chart_enhanced,
    create_cumulative_balance_chart,
    create_expense_distribution_pie,
    create_hourly_spending_heatmap,
    create_time_of_day_spending_chart,
    create_category_by_time_heatmap,
    create_monthly_trend_with_yoy_chart,
    create_mom_yoy_comparison_metrics_chart,
    create_original_category_treemap,
    create_original_category_sunburst,
    create_original_subcategory_ranking
)
from .components import (
    display_kpi_cards,
    display_data_summary,
    create_data_table_section,
    create_export_section,
    create_chart_container,
    display_savings_rate_insights,
    display_time_pattern_insights,
    display_comparison_metrics
)


def render_overview_tab(df_filtered: pd.DataFrame, df_processed: pd.DataFrame, filters: Dict[str, Any]):
    """
    Render the Overview tab with KPIs and high-level summary.

    Args:
        df_filtered: Filtered DataFrame based on current selections
        df_processed: Full processed DataFrame
        filters: Current filter selections
    """
    st.markdown("### 📊 Financial Overview")
    st.markdown("Get a quick snapshot of your financial health with key metrics and primary charts.")

    st.divider()

    # Display KPI cards
    display_kpi_cards(df_filtered)

    st.divider()

    # Key charts in 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        create_chart_container(
            "📈 Monthly Trends",
            create_monthly_trend_chart,
            df_filtered,
            chart_key="overview_monthly_trends"
        )

    with col2:
        create_chart_container(
            "🏆 Top Expense Categories",
            create_category_expense_chart,
            df_filtered,
            chart_key="overview_top_categories"
        )

    # Summary statistics
    st.divider()
    st.markdown("### 📋 Quick Statistics")

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        total_records = len(df_filtered)
        st.metric("Filtered Records", f"{total_records:,}")

    with col4:
        if not df_filtered.empty and "date" in df_filtered.columns:
            date_range = (df_filtered["date"].max() - df_filtered["date"].min()).days
            st.metric("Date Range (days)", f"{date_range}")
        else:
            st.metric("Date Range (days)", "N/A")

    with col5:
        if not df_filtered.empty and "category_display" in df_filtered.columns:
            unique_cats = df_filtered["category_display"].nunique()
            st.metric("Active Categories", f"{unique_cats}")
        else:
            st.metric("Active Categories", "N/A")

    with col6:
        filter_ratio = (len(df_filtered) / len(df_processed) * 100) if len(df_processed) > 0 else 0
        st.metric("Data Coverage", f"{filter_ratio:.1f}%")


def render_trends_tab(df_filtered: pd.DataFrame):
    """
    Render the Trends Analysis tab with time-series visualizations.

    Args:
        df_filtered: Filtered DataFrame based on current selections
    """
    st.markdown("### 📈 Trends Analysis")
    st.markdown("Analyze your financial patterns over time with detailed trend charts.")

    st.divider()

    # Monthly trends - full width
    create_chart_container(
        "📊 Monthly Income, Expenses & Savings",
        create_monthly_trend_chart,
        df_filtered,
        chart_key="trends_monthly_trends"
    )

    st.divider()

    # NEW: Year-over-Year & Month-over-Month Comparisons
    with st.expander("📊 Period-over-Period Comparisons (MoM & YoY)", expanded=True):
        st.markdown("Compare current period performance against previous periods to identify growth trends.")

        # Display comparison metrics cards
        display_comparison_metrics(df_filtered)

        st.divider()

        # MoM vs YoY percentage change chart
        create_chart_container(
            "📈 MoM vs YoY Comparison",
            create_mom_yoy_comparison_metrics_chart,
            df_filtered,
            chart_key="trends_mom_yoy_comparison"
        )

        st.divider()

        # YoY comparison charts by metric
        st.markdown("#### Year-over-Year Detailed Comparison")

        metric_selector = st.selectbox(
            "Select metric to compare",
            options=["expenses", "income", "net_savings"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="yoy_metric_selector"
        )

        create_chart_container(
            f"Year-over-Year {metric_selector.replace('_', ' ').title()}",
            lambda df: create_monthly_trend_with_yoy_chart(df, metric=metric_selector),
            df_filtered,
            chart_key=f"trends_yoy_{metric_selector}"
        )

    st.divider()

    # Cumulative balance - full width
    create_chart_container(
        "📈 Cumulative Balance Over Time",
        create_cumulative_balance_chart,
        df_filtered,
        chart_key="trends_cumulative_balance"
    )

    st.divider()

    # === NEW: Time-based Spending Pattern Analysis ===
    st.markdown("### ⏰ Time-Based Spending Patterns")
    st.markdown("Discover when you spend most during the week and day.")

    # Check if we have time data
    has_time_data = "hour" in df_filtered.columns and "time_of_day" in df_filtered.columns

    if has_time_data and len(df_filtered) > 30:
        # Hourly heatmap - full width
        create_chart_container(
            "🔥 Hourly Spending Heatmap (Hour × Day of Week)",
            create_hourly_spending_heatmap,
            df_filtered,
            chart_key="trends_hourly_heatmap"
        )

        st.divider()

        # Time of day and category charts side by side
        col1, col2 = st.columns(2)

        with col1:
            create_chart_container(
                "🌅 Spending by Time of Day",
                create_time_of_day_spending_chart,
                df_filtered,
                chart_key="trends_time_of_day"
            )

        with col2:
            create_chart_container(
                "🏷️ Categories by Time",
                create_category_by_time_heatmap,
                df_filtered,
                chart_key="trends_category_time"
            )

        st.divider()

        # Time pattern insights
        with st.expander("💡 Detailed Time Pattern Insights", expanded=True):
            display_time_pattern_insights(df_filtered)

    elif has_time_data:
        st.info("📊 Time-based analysis requires at least 30 transactions. Add more data or adjust filters.")
    else:
        st.warning("⚠️ Time data not available. Please ensure your data includes timestamp information.")

    st.divider()

    # Original day-level heatmap (kept for backward compatibility)
    if len(df_filtered) > 30:
        with st.expander("📅 Day-Level Spending Heatmap (Week × Day)", expanded=False):
            st.markdown("*Legacy view showing spending by week of year and day of week*")
            create_chart_container(
                "Daily Spending Heatmap",
                create_daily_spending_heatmap,
                df_filtered,
                chart_key="trends_daily_heatmap_legacy"
            )
    else:
        st.info("📊 Daily spending heatmap requires at least 30 transactions. Add more data or adjust filters.")


def render_categories_tab(df_filtered: pd.DataFrame):
    """
    Render the Categories tab with expense category analysis.

    Args:
        df_filtered: Filtered DataFrame based on current selections
    """
    st.markdown("### 🏷️ Category Analysis")
    st.markdown("Deep dive into your spending categories to understand where your money goes.")

    st.divider()

    # Two-column layout for category charts (Mapped English categories)
    col1, col2 = st.columns(2)

    with col1:
        create_chart_container(
            "📊 Top Expense Categories",
            create_category_expense_chart,
            df_filtered,
            chart_key="categories_top_expenses"
        )

    with col2:
        create_chart_container(
            "🥧 Expense Distribution",
            create_expense_distribution_pie,
            df_filtered,
            chart_key="categories_expense_pie"
        )

    st.divider()

    # Category details section
    if not df_filtered.empty and "category_display" in df_filtered.columns:
        st.markdown("### 📋 Category Details")

        expenses_df = df_filtered[df_filtered["transaction_type"] == "expense"]

        if not expenses_df.empty:
            category_stats = expenses_df.groupby("category_display").agg({
                "amount": ["sum", "count", "mean", "min", "max"]
            }).reset_index()

            category_stats.columns = ["Category", "Total Amount", "Transactions", "Average", "Min", "Max"]
            category_stats = category_stats.sort_values("Total Amount", ascending=False)

            # Format currency columns
            for col in ["Total Amount", "Average", "Min", "Max"]:
                category_stats[col] = category_stats[col].apply(lambda x: f"${x:,.2f}")

            st.dataframe(
                category_stats,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No expense data available for category analysis.")
    else:
        st.warning("Category information not available in the dataset.")

    st.divider()

    # NEW: Original Category Analysis (Chinese names)
    st.markdown("### 🏷️ 原始分类分析 (Original Category Analysis)")
    st.markdown("Explore your spending using the original Chinese category and subcategory names from your financial app.")

    # Check if original category data is available
    if "category" not in df_filtered.columns:
        st.warning("⚠️ Original category data not available in the current dataset.")
        return

    # Transaction type selector
    analysis_type = st.radio(
        "选择交易类型 (Select Transaction Type)",
        options=["expense", "income", "all"],
        format_func=lambda x: {"expense": "支出 (Expenses)", "income": "收入 (Income)", "all": "全部 (All)"}[x],
        horizontal=True,
        key="original_category_type"
    )

    st.divider()

    # Treemap - full width
    create_chart_container(
        "🗺️ Category Hierarchy Treemap",
        lambda df: create_original_category_treemap(df, transaction_type=analysis_type),
        df_filtered,
        chart_key=f"categories_original_treemap_{analysis_type}"
    )

    st.divider()

    # Two-column layout for sunburst and ranking
    col1, col2 = st.columns(2)

    with col1:
        create_chart_container(
            "☀️ Category Sunburst",
            lambda df: create_original_category_sunburst(df, transaction_type=analysis_type),
            df_filtered,
            chart_key=f"categories_original_sunburst_{analysis_type}"
        )

    with col2:
        create_chart_container(
            "📊 Top Subcategories",
            lambda df: create_original_subcategory_ranking(df, n=15, transaction_type=analysis_type),
            df_filtered,
            chart_key=f"categories_original_ranking_{analysis_type}"
        )

    st.divider()

    # Detailed table with search
    st.markdown("### 📋 详细分类表格 (Detailed Category Table)")

    from ..features.kpis import analyze_original_categories

    category_data = analyze_original_categories(df_filtered, transaction_type=analysis_type)

    if not category_data.empty:
        # Add search box
        search_term = st.text_input(
            "搜索分类或子分类 (Search category or subcategory)",
            placeholder="输入关键词... (Enter keyword...)",
            key="category_search"
        )

        # Filter based on search
        if search_term:
            mask = (
                category_data["category"].str.contains(search_term, case=False, na=False) |
                category_data["subcategory"].astype(str).str.contains(search_term, case=False, na=False)
            )
            category_data = category_data[mask]

        # Format for display
        display_data = category_data.copy()
        display_data.columns = [
            "分类 (Category)",
            "子分类 (Subcategory)",
            "总金额 (Total)",
            "交易数 (Count)",
            "平均金额 (Avg)",
            "占比 (%)"]

        # Format currency columns
        for col in ["总金额 (Total)", "平均金额 (Avg)"]:
            display_data[col] = display_data[col].apply(lambda x: f"${x:,.2f}")

        display_data["占比 (%)"] = display_data["占比 (%)"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_cats = category_data["category"].nunique()
            st.metric("总分类数 (Total Categories)", unique_cats)
        with col2:
            unique_subs = category_data["subcategory"].nunique()
            st.metric("总子分类数 (Total Subcategories)", unique_subs)
        with col3:
            st.metric("显示记录数 (Displayed Records)", len(category_data))
    else:
        st.info("No original category data available for the selected transaction type.")


def render_insights_tab(df_filtered: pd.DataFrame, chart_config: Dict[str, Any]):
    """
    Render the Savings Insights tab with savings rate analysis.

    Args:
        df_filtered: Filtered DataFrame based on current selections
        chart_config: Chart configuration options
    """
    st.markdown("### 💰 Savings Insights")
    st.markdown("Track your savings performance with detailed metrics and actionable insights.")

    st.divider()

    # Two-column layout for savings visualizations
    col1, col2 = st.columns(2)

    with col1:
        create_chart_container(
            "📊 Savings Rate Gauge",
            create_savings_rate_gauge,
            df_filtered,
            chart_key="insights_savings_gauge"
        )

    with col2:
        # Summary savings metrics
        from ..features.kpis import get_kpi_metrics
        metrics = get_kpi_metrics(df_filtered)

        st.markdown("#### 💡 Savings Summary")
        st.metric("Net Savings", f"${metrics['net_savings']:,.2f}")
        st.metric("Savings Rate", f"{metrics['savings_rate']:.1f}%")
        st.metric("Daily Avg Spending", f"${metrics['average_daily_spending']:,.2f}")

    st.divider()

    # Enhanced monthly savings rate chart - full width
    def enhanced_savings_chart(df):
        return create_monthly_savings_rate_chart_enhanced(
            df,
            target_rate=chart_config.get("target_rate", 10.0),
            show_dual_axis=chart_config.get("show_dual_axis", False),
            show_moving_average=chart_config.get("show_moving_average", True),
            ma_periods=chart_config.get("ma_periods", 3)
        )

    create_chart_container(
        "📊 Monthly Savings Rate Trends",
        enhanced_savings_chart,
        df_filtered,
        chart_key="insights_monthly_savings_rate"
    )

    st.divider()

    # Detailed insights in expandable section
    with st.expander("💡 Detailed Savings Rate Analysis", expanded=True):
        display_savings_rate_insights(df_filtered, chart_config.get("target_rate", 10.0))


def render_data_tab(df_filtered: pd.DataFrame, df_processed: pd.DataFrame, filters: Dict[str, Any]):
    """
    Render the Data & Export tab with detailed data table and export options.

    Args:
        df_filtered: Filtered DataFrame based on current selections
        df_processed: Full processed DataFrame
        filters: Current filter selections
    """
    st.markdown("### 📋 Data & Export")
    st.markdown("Access detailed transaction data and export functionality.")

    st.divider()

    # Filter summary
    if not df_processed.empty:
        filter_ratio = len(df_filtered) / len(df_processed) * 100

        st.info(
            f"📊 Showing **{len(df_filtered):,}** of **{len(df_processed):,}** records "
            f"({filter_ratio:.1f}%) based on current filters"
        )

    # Data summary
    display_data_summary(df_filtered)

    st.divider()

    # Detailed data table
    create_data_table_section(df_filtered)

    st.divider()

    # Export functionality
    create_export_section(df_filtered)


def create_tabbed_dashboard(
    df_filtered: pd.DataFrame,
    df_processed: pd.DataFrame,
    filters: Dict[str, Any],
    chart_config: Dict[str, Any]
):
    """
    Create the main tabbed dashboard interface.

    Args:
        df_filtered: Filtered DataFrame based on current selections
        df_processed: Full processed DataFrame
        filters: Current filter selections
        chart_config: Chart configuration options
    """
    # Initialize session state for active tab
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"

    # Create tabs
    tab_names = ["📊 Overview", "📈 Trends", "🏷️ Categories", "💰 Savings", "📋 Data & Export"]
    tabs = st.tabs(tab_names)

    # Render each tab
    with tabs[0]:
        render_overview_tab(df_filtered, df_processed, filters)

    with tabs[1]:
        render_trends_tab(df_filtered)

    with tabs[2]:
        render_categories_tab(df_filtered)

    with tabs[3]:
        render_insights_tab(df_filtered, chart_config)

    with tabs[4]:
        render_data_tab(df_filtered, df_processed, filters)
