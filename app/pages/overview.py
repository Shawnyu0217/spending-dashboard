import streamlit as st
import pandas as pd
from typing import Dict, Any
from .base import BasePage

# Import all required UI components
from app.ui.components import (
    display_kpi_cards,
    create_chart_container,
    display_data_summary,
    create_data_table_section,
    create_export_section
)

# Import visualization functions
from app.features.viz import (
    create_monthly_trend_chart,
    create_category_expense_chart,
    create_savings_rate_gauge,
    create_expense_distribution_pie,
    create_monthly_savings_rate_chart_enhanced,
    create_cumulative_balance_chart,
    create_daily_spending_heatmap
)

class OverviewPage(BasePage):
    """Overview page showing KPIs and main charts."""
    
    @property
    def name(self) -> str:
        return "Overview"
    
    @property
    def icon(self) -> str:
        return "📊"
    
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render the overview page with KPIs and key charts."""
        try:
            # Display KPI cards at the top
            display_kpi_cards(df)
            
            st.divider()
            
            # Main charts section
            self._render_key_insights(df, filters)
            
            # Enhanced sections
            self._render_detailed_analysis(df, filters)
            
            # Data details section
            self._render_data_section(df)
            
        except Exception as e:
            self.render_error(str(e))
    
    def _render_key_insights(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render the key insights charts."""
        st.header("📊 Financial Analysis")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            create_chart_container(
                "📈 Monthly Trends",
                create_monthly_trend_chart,
                df
            )
        
        with col2:
            create_chart_container(
                "🏆 Top Expense Categories",
                create_category_expense_chart,
                df
            )
        
        # Second row of charts
        col3, col4 = st.columns(2)
        
        with col3:
            create_chart_container(
                "📊 Savings Rate",
                create_savings_rate_gauge,
                df
            )
        
        with col4:
            create_chart_container(
                "🥧 Expense Distribution",
                create_expense_distribution_pie,
                df
            )
    
    def _render_detailed_analysis(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render detailed analysis charts."""
        # Import chart configuration controls
        from app.ui.components import create_chart_configuration_controls, display_savings_rate_insights
        
        # Create chart configuration controls
        chart_config = create_chart_configuration_controls()
        
        # Third row - Monthly Savings Rate Trends (full width with enhanced features)
        def enhanced_savings_chart(df_input):
            return create_monthly_savings_rate_chart_enhanced(
                df_input,
                target_rate=chart_config.get("target_rate", 10.0),
                show_dual_axis=chart_config.get("show_dual_axis", False),
                show_moving_average=chart_config.get("show_moving_average", True),
                ma_periods=chart_config.get("ma_periods", 3)
            )
        
        create_chart_container(
            "📊 Monthly Savings Rate Trends",
            enhanced_savings_chart,
            df
        )
        
        # Add insights section for savings rate
        with st.expander("💡 Savings Rate Insights", expanded=False):
            display_savings_rate_insights(df, chart_config.get("target_rate", 10.0))
        
        # Fourth row - Cumulative Balance (full width)
        create_chart_container(
            "📈 Cumulative Balance",
            create_cumulative_balance_chart,
            df
        )
        
        # Optional: Daily spending heatmap (full width)
        if len(df) > 30:  # Only show if we have enough data
            create_chart_container(
                "🔥 Daily Spending Heatmap",
                create_daily_spending_heatmap,
                df
            )
    
    def _render_data_section(self, df: pd.DataFrame) -> None:
        """Render the data details section."""
        st.divider()
        
        st.header("📋 Data Details")
        
        # Data summary
        display_data_summary(df)
        
        # Detailed data table
        create_data_table_section(df)
        
        # Export functionality
        create_export_section(df)
