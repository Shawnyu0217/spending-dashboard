"""
Visualization functions for the spending dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import streamlit as st
import logging

from ..config import COLORS
from .kpis import monthly_summary, top_expense_categories

logger = logging.getLogger(__name__)

def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing monthly income, expenses, and net savings.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for monthly trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Add income line
    fig.add_trace(go.Scatter(
        x=monthly_data["month"],
        y=monthly_data["income"], 
        mode='lines+markers',
        name='Income',
        line=dict(color=COLORS["income"], width=3),
        marker=dict(size=8)
    ))
    
    # Add expenses line
    fig.add_trace(go.Scatter(
        x=monthly_data["month"],
        y=monthly_data["expenses"],
        mode='lines+markers', 
        name='Expenses',
        line=dict(color=COLORS["expense"], width=3),
        marker=dict(size=8),
        fill='tonexty'
    ))
    
    # Add net savings line
    fig.add_trace(go.Scatter(
        x=monthly_data["month"],
        y=monthly_data["net_savings"],
        mode='lines+markers',
        name='Net Savings',
        line=dict(color=COLORS["net"], width=3),
        marker=dict(size=8)
    ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Monthly Financial Trends",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_category_expense_chart(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """
    Create a bar chart showing top expense categories.
    
    Args:
        df: Processed DataFrame
        n: Number of top categories to show
        
    Returns:
        Plotly figure
    """
    top_cats = top_expense_categories(df, n)
    
    if top_cats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_cats["category"],
            y=top_cats["total_amount"],
            text=[f"${amt:,.0f}<br>({pct:.1f}%)" 
                  for amt, pct in zip(top_cats["total_amount"], top_cats["percentage"])],
            textposition='auto',
            marker_color=COLORS["categories"][:len(top_cats)],
            hovertemplate='<b>%{x}</b><br>' +
                         'Amount: $%{y:,.2f}<br>' +
                         'Transactions: %{customdata[0]}<br>' +
                         'Average: $%{customdata[1]:,.2f}<br>' +
                         '<extra></extra>',
            customdata=list(zip(top_cats["transaction_count"], top_cats["avg_amount"]))
        )
    ])
    
    fig.update_layout(
        title=f"Top {n} Expense Categories",
        xaxis_title="Category",
        yaxis_title="Total Amount ($)", 
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_daily_spending_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing daily spending patterns.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    if df.empty or "date" not in df.columns or "expense_amount" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily spending data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Create daily spending summary
    expenses_df = df[df["transaction_type"] == "expense"].copy()
    daily_spending = expenses_df.groupby("date")["amount"].sum().reset_index()
    
    # Add day of week
    daily_spending["weekday"] = daily_spending["date"].dt.day_name()
    daily_spending["week"] = daily_spending["date"].dt.isocalendar().week
    
    # Create pivot for heatmap
    heatmap_data = daily_spending.pivot_table(
        values="amount", 
        index="weekday", 
        columns="week", 
        aggfunc="sum", 
        fill_value=0
    )
    
    # Reorder weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(weekday_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Reds',
        hovertemplate='Week: %{x}<br>Day: %{y}<br>Spending: $%{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Daily Spending Heatmap",
        xaxis_title="Week of Year",
        yaxis_title="Day of Week",
        height=300
    )
    
    return fig

def create_savings_rate_gauge(df: pd.DataFrame) -> go.Figure:
    """
    Create a gauge chart for savings rate.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    from .kpis import savings_rate
    
    rate = savings_rate(df)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Savings Rate (%)"},
        delta={'reference': 20, 'position': "top"},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': COLORS["net"]},
            'steps': [
                {'range': [0, 10], 'color': "lightgray"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 50], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def create_monthly_savings_rate_chart(df: pd.DataFrame, target_rate: float = 10.0) -> go.Figure:
    """
    Create a line chart showing monthly savings rate trends over time.
    
    Args:
        df: Processed DataFrame
        target_rate: Target savings rate percentage (default: 10%)
        
    Returns:
        Plotly figure
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for monthly savings rate trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Define colors based on whether savings rate meets target
    colors = []
    for rate in monthly_data["savings_rate"]:
        if rate >= target_rate:
            colors.append(COLORS["net"])  # Green for above target
        else:
            colors.append(COLORS["expense"])  # Red for below target
    
    # Add savings rate line with conditional coloring
    fig.add_trace(go.Scatter(
        x=monthly_data["month"],
        y=monthly_data["savings_rate"],
        mode='lines+markers',
        name='Savings Rate',
        line=dict(color=COLORS["net"], width=3),
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{x}</b><br>' +
                     'Savings Rate: %{y:.1f}%<br>' +
                     'Income: $%{customdata[0]:,.0f}<br>' +
                     'Expenses: $%{customdata[1]:,.0f}<br>' +
                     'Net Savings: $%{customdata[2]:,.0f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(
            monthly_data["income"], 
            monthly_data["expenses"], 
            monthly_data["net_savings"]
        ))
    ))
    
    # Add target line
    fig.add_hline(
        y=target_rate, 
        line_dash="dash", 
        line_color="gray", 
        opacity=0.7,
        annotation_text=f"Target: {target_rate}%",
        annotation_position="top right"
    )
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
    
    # Color-code the background areas
    fig.add_hrect(
        y0=target_rate, y1=100,
        fillcolor="lightgreen", opacity=0.1,
        layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=0, y1=target_rate,
        fillcolor="lightcoral", opacity=0.1,
        layer="below", line_width=0,
    )
    
    fig.update_layout(
        title=f"Monthly Savings Rate Trends (Target: {target_rate}%)",
        xaxis_title="Month",
        yaxis_title="Savings Rate (%)",
        hovermode='x unified',
        showlegend=True,
        height=400,
        yaxis=dict(
            ticksuffix="%",
            range=[-10, max(50, monthly_data["savings_rate"].max() + 5)]
        )
    )
    
    return fig

def create_monthly_savings_rate_chart_enhanced(df: pd.DataFrame, 
                                             target_rate: float = 10.0,
                                             show_dual_axis: bool = False,
                                             show_moving_average: bool = True,
                                             ma_periods: int = 3) -> go.Figure:
    """
    Create an enhanced line chart showing monthly savings rate trends with advanced features.
    
    Args:
        df: Processed DataFrame
        target_rate: Target savings rate percentage
        show_dual_axis: Whether to show dual axis (rate % and absolute amounts)
        show_moving_average: Whether to show moving average trend line
        ma_periods: Number of periods for moving average (default: 3)
        
    Returns:
        Plotly figure with enhanced features
    """
    monthly_data = monthly_summary(df)
    
    if monthly_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for monthly savings rate trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Calculate moving averages
    if show_moving_average and len(monthly_data) >= ma_periods:
        monthly_data['savings_rate_ma'] = monthly_data['savings_rate'].rolling(
            window=ma_periods, center=True
        ).mean()
        monthly_data['net_savings_ma'] = monthly_data['net_savings'].rolling(
            window=ma_periods, center=True
        ).mean()
    
    # Define colors based on whether savings rate meets target
    colors = []
    for rate in monthly_data["savings_rate"]:
        if rate >= target_rate:
            colors.append(COLORS["net"])  # Green for above target
        else:
            colors.append(COLORS["expense"])  # Red for below target
    
    if show_dual_axis:
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add savings rate line (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=monthly_data["month"],
                y=monthly_data["savings_rate"],
                mode='lines+markers',
                name='Savings Rate (%)',
                line=dict(color=COLORS["net"], width=3),
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{x}</b><br>' +
                             'Savings Rate: %{y:.1f}%<br>' +
                             '<extra></extra>',
            ),
            secondary_y=False,
        )
        
        # Add net savings bars (secondary y-axis)
        fig.add_trace(
            go.Bar(
                x=monthly_data["month"],
                y=monthly_data["net_savings"],
                name='Net Savings ($)',
                marker_color=[color + '80' for color in colors],  # Add transparency
                opacity=0.6,
                hovertemplate='<b>%{x}</b><br>' +
                             'Net Savings: $%{y:,.0f}<br>' +
                             '<extra></extra>',
            ),
            secondary_y=True,
        )
        
        # Add moving average if requested
        if show_moving_average and len(monthly_data) >= ma_periods:
            fig.add_trace(
                go.Scatter(
                    x=monthly_data["month"],
                    y=monthly_data["savings_rate_ma"],
                    mode='lines',
                    name=f'{ma_periods}-Month MA',
                    line=dict(color='purple', width=2, dash='dot'),
                    hovertemplate='<b>%{x}</b><br>' +
                                 f'{ma_periods}-Month MA: %{{y:.1f}}%<br>' +
                                 '<extra></extra>',
                ),
                secondary_y=False,
            )
        
        # Add target line
        fig.add_hline(
            y=target_rate, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.7,
            annotation_text=f"Target: {target_rate}%",
            annotation_position="top right",
            secondary_y=False
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Savings Rate (%)", secondary_y=False, ticksuffix="%")
        fig.update_yaxes(title_text="Net Savings ($)", secondary_y=True, tickformat=",.0f")
        
        # Update layout
        fig.update_layout(
            title=f"Monthly Savings Rate & Amount Trends (Target: {target_rate}%)",
            xaxis_title="Month",
            hovermode='x unified',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
    else:
        # Single-axis chart (enhanced version of original)
        fig = go.Figure()
        
        # Add savings rate line with conditional coloring
        fig.add_trace(go.Scatter(
            x=monthly_data["month"],
            y=monthly_data["savings_rate"],
            mode='lines+markers',
            name='Savings Rate',
            line=dict(color=COLORS["net"], width=3),
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Savings Rate: %{y:.1f}%<br>' +
                         'Income: $%{customdata[0]:,.0f}<br>' +
                         'Expenses: $%{customdata[1]:,.0f}<br>' +
                         'Net Savings: $%{customdata[2]:,.0f}<br>' +
                         'Transactions: %{customdata[3]}<br>' +
                         '<extra></extra>',
            customdata=list(zip(
                monthly_data["income"], 
                monthly_data["expenses"], 
                monthly_data["net_savings"],
                monthly_data["transaction_count"]
            ))
        ))
        
        # Add moving average if requested
        if show_moving_average and len(monthly_data) >= ma_periods:
            fig.add_trace(go.Scatter(
                x=monthly_data["month"],
                y=monthly_data["savings_rate_ma"],
                mode='lines',
                name=f'{ma_periods}-Month Trend',
                line=dict(color='purple', width=2, dash='dot'),
                hovertemplate='<b>%{x}</b><br>' +
                             f'{ma_periods}-Month Trend: %{{y:.1f}}%<br>' +
                             '<extra></extra>',
            ))
        
        # Add target line
        fig.add_hline(
            y=target_rate, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.7,
            annotation_text=f"Target: {target_rate}%",
            annotation_position="top right"
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
        
        # Color-code the background areas
        fig.add_hrect(
            y0=target_rate, y1=100,
            fillcolor="lightgreen", opacity=0.1,
            layer="below", line_width=0,
        )
        fig.add_hrect(
            y0=0, y1=target_rate,
            fillcolor="lightcoral", opacity=0.1,
            layer="below", line_width=0,
        )
        
        fig.update_layout(
            title=f"Monthly Savings Rate Trends (Target: {target_rate}%)",
            xaxis_title="Month",
            yaxis_title="Savings Rate (%)",
            hovermode='x unified',
            showlegend=True,
            height=400,
            yaxis=dict(
                ticksuffix="%",
                range=[-10, max(50, monthly_data["savings_rate"].max() + 5)]
            )
        )
    
    return fig

def create_account_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a comparison chart of accounts.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    from .kpis import account_summary
    
    account_data = account_summary(df)
    
    if account_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No account data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Add income bars
    fig.add_trace(go.Bar(
        name='Income',
        x=account_data["account"],
        y=account_data["income"],
        marker_color=COLORS["income"],
        text=[f"${val:,.0f}" for val in account_data["income"]],
        textposition='auto'
    ))
    
    # Add expense bars (negative values)
    fig.add_trace(go.Bar(
        name='Expenses',
        x=account_data["account"],
        y=-account_data["expenses"],  # Negative for visual separation
        marker_color=COLORS["expense"],
        text=[f"${val:,.0f}" for val in account_data["expenses"]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Account Performance Comparison',
        xaxis_title='Account',
        yaxis_title='Amount ($)',
        barmode='relative',
        height=400,
        hovermode='x unified'
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_cumulative_balance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing cumulative balance over time.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    if df.empty or "date" not in df.columns or "running_balance" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No balance data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Sort by date
    balance_df = df.sort_values("date")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=balance_df["date"],
        y=balance_df["running_balance"],
        mode='lines',
        name='Cumulative Balance',
        line=dict(color=COLORS["net"], width=2),
        fill='tonexty',
        fillcolor=f'rgba({int(COLORS["net"][1:3], 16)}, {int(COLORS["net"][3:5], 16)}, {int(COLORS["net"][5:7], 16)}, 0.1)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Cumulative Balance Over Time',
        xaxis_title='Date',
        yaxis_title='Balance ($)',
        hovermode='x',
        height=400
    )
    
    return fig

def create_expense_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing expense distribution by top-level categories.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Plotly figure
    """
    if df.empty or "top_level_category" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    expenses_df = df[df["transaction_type"] == "expense"]
    
    if expenses_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Debug information using logger
    logger.debug("Expense transactions: %s", len(expenses_df))
    debug_counts = expenses_df["top_level_category"].value_counts()
    logger.debug("Top-level category distribution:\n%s", debug_counts)
    
    category_totals = expenses_df.groupby("top_level_category")["amount"].sum().reset_index()
    category_totals = category_totals.sort_values("amount", ascending=False)
    
    logger.debug("Category totals:\n%s", category_totals)
    
    fig = go.Figure(data=[go.Pie(
        labels=category_totals["top_level_category"],
        values=category_totals["amount"],
        hole=0.3,
        textinfo='label+percent',
        textposition='outside',
        marker_colors=COLORS["categories"][:len(category_totals)]
    )])
    
    fig.update_layout(
        title='Expense Distribution by Category Type',
        height=400,
        showlegend=True
    )

    return fig

def create_hourly_spending_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing spending patterns by hour and day of week.

    Args:
        df: Processed DataFrame with hour and weekday columns

    Returns:
        Plotly figure
    """
    if df.empty or "hour" not in df.columns or "weekday" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No hourly spending data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Filter to expenses only
    expenses_df = df[df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Create pivot table for heatmap (Weekday × Hour)
    heatmap_data = expenses_df.groupby(["weekday", "hour"])["amount"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot_table(
        values="amount",
        index="weekday",
        columns="hour",
        aggfunc="sum",
        fill_value=0
    )

    # Reorder weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_pivot = heatmap_pivot.reindex(weekday_order, fill_value=0)

    # Ensure all hours 0-23 are present
    all_hours = list(range(24))
    for hour in all_hours:
        if hour not in heatmap_pivot.columns:
            heatmap_pivot[hour] = 0
    heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]

    # Create custom hover text with transaction counts
    transaction_counts = expenses_df.groupby(["weekday", "hour"]).size().reset_index(name="count")
    count_pivot = transaction_counts.pivot_table(
        values="count",
        index="weekday",
        columns="hour",
        aggfunc="sum",
        fill_value=0
    )
    count_pivot = count_pivot.reindex(weekday_order, fill_value=0)
    for hour in all_hours:
        if hour not in count_pivot.columns:
            count_pivot[hour] = 0
    count_pivot = count_pivot[sorted(count_pivot.columns)]

    # Create hover text
    hover_text = []
    for i, day in enumerate(weekday_order):
        row_text = []
        for hour in range(24):
            amount = heatmap_pivot.loc[day, hour]
            count = count_pivot.loc[day, hour]
            text = f"{day}<br>Hour: {hour:02d}:00<br>Spending: ${amount:,.0f}<br>Transactions: {int(count)}"
            row_text.append(text)
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=heatmap_pivot.index,
        colorscale='Reds',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title="Spending ($)")
    ))

    fig.update_layout(
        title="Hourly Spending Heatmap (Hour × Day of Week)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400,
        xaxis=dict(side="bottom", tickangle=0),
        yaxis=dict(autorange="reversed")
    )

    return fig

def create_time_of_day_spending_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart comparing spending by time of day periods.

    Args:
        df: Processed DataFrame with time_of_day column

    Returns:
        Plotly figure
    """
    if df.empty or "time_of_day" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No time of day data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    expenses_df = df[df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Aggregate by time of day
    time_data = expenses_df.groupby("time_of_day").agg({
        "amount": ["sum", "count", "mean"]
    }).reset_index()
    time_data.columns = ["time_of_day", "total_amount", "transaction_count", "avg_amount"]

    # Ensure proper ordering
    time_order = ["Night", "Morning", "Afternoon", "Evening"]
    time_data["time_of_day"] = pd.Categorical(time_data["time_of_day"], categories=time_order, ordered=True)
    time_data = time_data.sort_values("time_of_day")

    # Define colors for each time period
    time_colors = {
        "Night": "#4a5568",      # Dark gray
        "Morning": "#f6ad55",    # Orange
        "Afternoon": "#fc8181",  # Red
        "Evening": "#9f7aea"     # Purple
    }

    colors = [time_colors.get(str(t), "#gray") for t in time_data["time_of_day"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=time_data["time_of_day"],
        y=time_data["total_amount"],
        marker_color=colors,
        text=[f"${amt:,.0f}<br>({cnt} txns)"
              for amt, cnt in zip(time_data["total_amount"], time_data["transaction_count"])],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                     'Total Spending: $%{y:,.0f}<br>' +
                     'Transactions: %{customdata[0]}<br>' +
                     'Avg per Transaction: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(time_data["transaction_count"], time_data["avg_amount"]))
    ))

    fig.update_layout(
        title="Spending by Time of Day",
        xaxis_title="Time Period",
        yaxis_title="Total Amount ($)",
        showlegend=False,
        height=400
    )

    return fig

def create_category_by_time_heatmap(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Create a heatmap showing which categories are spent on at different times of day.

    Args:
        df: Processed DataFrame
        top_n: Number of top categories to show

    Returns:
        Plotly figure
    """
    if df.empty or "time_of_day" not in df.columns or "category_display" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No category time data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    expenses_df = df[df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Get top N categories by total spending
    top_categories = expenses_df.groupby("category_display")["amount"].sum().nlargest(top_n).index.tolist()

    # Filter to top categories
    filtered_df = expenses_df[expenses_df["category_display"].isin(top_categories)]

    # Create pivot table
    heatmap_data = filtered_df.groupby(["category_display", "time_of_day"])["amount"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot_table(
        values="amount",
        index="category_display",
        columns="time_of_day",
        aggfunc="sum",
        fill_value=0
    )

    # Ensure proper time ordering
    time_order = ["Night", "Morning", "Afternoon", "Evening"]
    heatmap_pivot = heatmap_pivot.reindex(columns=time_order, fill_value=0)

    # Sort categories by total spending
    heatmap_pivot["total"] = heatmap_pivot.sum(axis=1)
    heatmap_pivot = heatmap_pivot.sort_values("total", ascending=False).drop("total", axis=1)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Blues',
        hovertemplate='Category: %{y}<br>Time: %{x}<br>Spending: $%{z:,.0f}<extra></extra>',
        colorbar=dict(title="Spending ($)")
    ))

    fig.update_layout(
        title=f"Top {top_n} Categories by Time of Day",
        xaxis_title="Time of Day",
        yaxis_title="Category",
        height=400,
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_monthly_trend_with_yoy_chart(df: pd.DataFrame, metric: str = "expenses") -> go.Figure:
    """
    Create a grouped bar chart comparing current year vs previous year by month.

    Args:
        df: Processed DataFrame
        metric: Metric to compare ('income', 'expenses', 'net_savings')

    Returns:
        Plotly figure with year-over-year comparison
    """
    from .kpis import monthly_summary

    monthly_data = monthly_summary(df, include_comparisons=True)

    if monthly_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for year-over-year comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Filter to only months with YoY data
    yoy_data = monthly_data[monthly_data[f"{metric}_yoy_prev"].notna()].copy()

    if yoy_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient historical data for year-over-year comparison<br>Need at least 2 years of data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Create month labels for x-axis (e.g., "Jan", "Feb")
    yoy_data["month_label"] = yoy_data["month"].str[5:7]  # Extract MM from YYYY-MM
    month_map = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
    }
    yoy_data["month_label"] = yoy_data["month_label"].map(month_map)

    fig = go.Figure()

    # Add previous year bars
    fig.add_trace(go.Bar(
        name=f'Previous Year',
        x=yoy_data["month_label"],
        y=yoy_data[f"{metric}_yoy_prev"],
        marker_color='lightblue',
        text=[f"${val:,.0f}" for val in yoy_data[f"{metric}_yoy_prev"]],
        textposition='auto',
        hovertemplate='<b>%{x} (Previous Year)</b><br>' +
                     'Amount: $%{y:,.2f}<br>' +
                     '<extra></extra>'
    ))

    # Add current year bars
    fig.add_trace(go.Bar(
        name=f'Current Year',
        x=yoy_data["month_label"],
        y=yoy_data[metric],
        marker_color=COLORS["expense"] if metric == "expenses" else COLORS["income"],
        text=[f"${val:,.0f}" for val in yoy_data[metric]],
        textposition='auto',
        hovertemplate='<b>%{x} (Current Year)</b><br>' +
                     'Amount: $%{y:,.2f}<br>' +
                     'YoY Change: %{customdata[0]:+,.0f} (%{customdata[1]:+.1f}%)<br>' +
                     '<extra></extra>',
        customdata=list(zip(yoy_data[f"{metric}_yoy_change"], yoy_data[f"{metric}_yoy_pct"]))
    ))

    metric_display = metric.replace("_", " ").title()
    fig.update_layout(
        title=f"Year-over-Year {metric_display} Comparison",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        barmode='group',
        hovermode='x unified',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_mom_yoy_comparison_metrics_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive comparison chart showing both MoM and YoY metrics.

    Args:
        df: Processed DataFrame

    Returns:
        Plotly figure with comparison metrics
    """
    from .kpis import calculate_mom_metrics, calculate_yoy_metrics

    mom_metrics = calculate_mom_metrics(df)
    yoy_metrics = calculate_yoy_metrics(df)

    if not mom_metrics or not yoy_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for comparison metrics",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Prepare data for comparison
    categories = ['Income', 'Expenses', 'Net Savings']
    mom_pcts = [
        mom_metrics['income']['change_pct'],
        mom_metrics['expenses']['change_pct'],
        mom_metrics['net_savings']['change_pct']
    ]

    # Handle potential NaN values in YoY metrics
    yoy_pcts = []
    for key in ['income', 'expenses', 'net_savings']:
        pct = yoy_metrics.get(key, {}).get('change_pct', 0)
        yoy_pcts.append(pct if pd.notna(pct) else 0)

    fig = go.Figure()

    # Add MoM bars
    fig.add_trace(go.Bar(
        name='Month-over-Month',
        x=categories,
        y=mom_pcts,
        marker_color='steelblue',
        text=[f"{val:+.1f}%" for val in mom_pcts],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                     'MoM Change: %{y:+.1f}%<br>' +
                     '<extra></extra>'
    ))

    # Add YoY bars
    fig.add_trace(go.Bar(
        name='Year-over-Year',
        x=categories,
        y=yoy_pcts,
        marker_color='darkorange',
        text=[f"{val:+.1f}%" for val in yoy_pcts],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                     'YoY Change: %{y:+.1f}%<br>' +
                     '<extra></extra>'
    ))

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="MoM vs YoY Percentage Change Comparison",
        xaxis_title="Metric",
        yaxis_title="Percentage Change (%)",
        barmode='group',
        hovermode='x unified',
        height=400,
        yaxis=dict(ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_original_category_treemap(df: pd.DataFrame, transaction_type: str = "expense") -> go.Figure:
    """
    Create a treemap showing spending by original Chinese category and subcategory.

    Args:
        df: Processed DataFrame with original 'category' and 'subcategory' columns
        transaction_type: Type of transaction to visualize

    Returns:
        Plotly treemap figure
    """
    from .kpis import analyze_original_categories

    category_data = analyze_original_categories(df, transaction_type)

    if category_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Prepare data for treemap
    labels = []
    parents = []
    values = []
    texts = []

    # Add root
    total_amount = category_data["total_amount"].sum()
    labels.append("Total")
    parents.append("")
    values.append(total_amount)
    texts.append(f"Total: ${total_amount:,.0f}")

    # Add categories and subcategories
    for _, row in category_data.iterrows():
        category = row["category"]
        subcategory = row.get("subcategory", "")
        amount = row["total_amount"]
        percentage = row["percentage"]

        # Add category (if not already added)
        if category not in labels:
            labels.append(category)
            parents.append("Total")
            # Sum all amounts for this category
            category_total = category_data[category_data["category"] == category]["total_amount"].sum()
            values.append(category_total)
            texts.append(f"{category}<br>${category_total:,.0f}")

        # Add subcategory (if exists and not empty)
        if subcategory and str(subcategory) != "nan" and str(subcategory).strip():
            subcategory_label = f"{category} - {subcategory}"
            labels.append(subcategory_label)
            parents.append(category)
            values.append(amount)
            texts.append(f"{subcategory}<br>${amount:,.0f} ({percentage:.1f}%)")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        textposition='middle center',
        hovertemplate='<b>%{label}</b><br>' +
                     'Amount: $%{value:,.2f}<br>' +
                     '<extra></extra>',
        marker=dict(
            colorscale='Blues',
            line=dict(width=2, color='white')
        )
    ))

    type_display = "支出" if transaction_type == "expense" else "收入" if transaction_type == "income" else "全部"
    fig.update_layout(
        title=f"原始分类分析 (Original Category Analysis) - {type_display}",
        height=500
    )

    return fig


def create_original_category_sunburst(df: pd.DataFrame, transaction_type: str = "expense") -> go.Figure:
    """
    Create a sunburst chart showing spending by original Chinese category hierarchy.

    Args:
        df: Processed DataFrame with original 'category' and 'subcategory' columns
        transaction_type: Type of transaction to visualize

    Returns:
        Plotly sunburst figure
    """
    from .kpis import analyze_original_categories

    category_data = analyze_original_categories(df, transaction_type)

    if category_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    texts = []

    # Add root
    total_amount = category_data["total_amount"].sum()
    labels.append("总计")
    parents.append("")
    values.append(total_amount)
    texts.append(f"${total_amount:,.0f}")

    # Track categories already added
    added_categories = set()

    # Add categories and subcategories
    for _, row in category_data.iterrows():
        category = row["category"]
        subcategory = row.get("subcategory", "")
        amount = row["total_amount"]
        percentage = row["percentage"]

        # Add category (if not already added)
        if category not in added_categories:
            labels.append(category)
            parents.append("总计")
            # Sum all amounts for this category
            category_total = category_data[category_data["category"] == category]["total_amount"].sum()
            values.append(category_total)
            pct = (category_total / total_amount * 100) if total_amount > 0 else 0
            texts.append(f"${category_total:,.0f}<br>({pct:.1f}%)")
            added_categories.add(category)

        # Add subcategory (if exists and not empty)
        if subcategory and str(subcategory) != "nan" and str(subcategory).strip():
            labels.append(subcategory)
            parents.append(category)
            values.append(amount)
            texts.append(f"${amount:,.0f}<br>({percentage:.1f}%)")

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        hovertemplate='<b>%{label}</b><br>' +
                     'Amount: $%{value:,.2f}<br>' +
                     'Percentage of Total: %{percentParent}<br>' +
                     '<extra></extra>',
        marker=dict(
            colorscale='RdYlBu_r',
            line=dict(width=2, color='white')
        )
    ))

    type_display = "支出" if transaction_type == "expense" else "收入" if transaction_type == "income" else "全部"
    fig.update_layout(
        title=f"原始分类层级图 (Category Hierarchy) - {type_display}",
        height=500
    )

    return fig


def create_original_subcategory_ranking(df: pd.DataFrame, n: int = 15, transaction_type: str = "expense") -> go.Figure:
    """
    Create a horizontal bar chart ranking top subcategories by amount (Chinese names).

    Args:
        df: Processed DataFrame
        n: Number of top subcategories to show
        transaction_type: Type of transaction to visualize

    Returns:
        Plotly figure
    """
    from .kpis import top_original_subcategories

    top_subs = top_original_subcategories(df, n, transaction_type)

    if top_subs.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No subcategory data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Reverse order for better visualization (highest at top)
    top_subs = top_subs.iloc[::-1]

    # Create labels with category context
    labels = [
        f"{row['subcategory']}<br><sup>({row['category']})</sup>"
        for _, row in top_subs.iterrows()
    ]

    # Assign colors by parent category
    unique_categories = top_subs["category"].unique()
    category_colors = {cat: COLORS["categories"][i % len(COLORS["categories"])]
                      for i, cat in enumerate(unique_categories)}
    bar_colors = [category_colors[cat] for cat in top_subs["category"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_subs["total_amount"],
        y=labels,
        orientation='h',
        marker_color=bar_colors,
        text=[f"${amt:,.0f}<br>({pct:.1f}%)"
              for amt, pct in zip(top_subs["total_amount"], top_subs["percentage"])],
        textposition='auto',
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                     'Category: %{customdata[1]}<br>' +
                     'Amount: $%{x:,.2f}<br>' +
                     'Transactions: %{customdata[2]}<br>' +
                     'Avg per Transaction: $%{customdata[3]:,.2f}<br>' +
                     'Percentage: %{customdata[4]:.1f}%<br>' +
                     '<extra></extra>',
        customdata=list(zip(
            top_subs["subcategory"],
            top_subs["category"],
            top_subs["transaction_count"],
            top_subs["avg_amount"],
            top_subs["percentage"]
        ))
    ))

    type_display = "支出" if transaction_type == "expense" else "收入" if transaction_type == "income" else "全部"
    fig.update_layout(
        title=f"Top {n} 子分类排名 (Subcategory Ranking) - {type_display}",
        xaxis_title="总金额 (Total Amount) $",
        yaxis_title="",
        showlegend=False,
        height=max(400, n * 25),  # Dynamic height based on number of categories
        margin=dict(l=200)  # Extra left margin for Chinese labels
    )

    return fig 