"""
Configuration constants and settings for the spending dashboard.
"""

import os
from typing import Dict, List

# File paths
DEFAULT_DATA_PATH = "data"
SAMPLE_FILE_NAME = "随手记家庭账本20250614165637.xlsx"
SAMPLE_FILE_PATH = os.path.join(DEFAULT_DATA_PATH, SAMPLE_FILE_NAME)

# Column name mappings (Chinese to English)
COLUMN_MAPPINGS = {
    "日期": "date",
    "交易类型": "transaction_type", 
    "分类": "category",
    "子分类": "subcategory",
    "账户": "account",
    "金额": "amount",
    "备注": "notes",
    "成员": "member"
}

# Category mappings for better display (Chinese to English)
CATEGORY_MAPPINGS = {
    "餐饮": "Food & Dining",
    "购物": "Shopping", 
    "交通": "Transportation",
    "生活服务": "Living Services",
    "医疗健康": "Healthcare",
    "教育": "Education",
    "娱乐": "Entertainment",
    "旅游": "Travel",
    "人情往来": "Social",
    "其他": "Other",
    "收入": "Income",
    "工资": "Salary",
    "奖金": "Bonus",
    "投资收益": "Investment Income"
}

# Transaction type mappings  
TRANSACTION_TYPE_MAPPINGS = {
    "支出": "expense",
    "收入": "income",
    "转账": "transfer"
}

# Color scheme for visualizations
COLORS = {
    "income": "#2E8B57",      # Sea Green
    "expense": "#DC143C",     # Crimson
    "net": "#4682B4",         # Steel Blue
    "categories": [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#FFB6C1",
        "#87CEEB", "#F0E68C", "#FFA07A", "#20B2AA"
    ]
}

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Personal Spending Dashboard",
    "page_icon": "💰", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Date format
DATE_FORMAT = "%Y-%m-%d"
DISPLAY_DATE_FORMAT = "%B %d, %Y"

# KPI display formats
CURRENCY_FORMAT = "${:,.2f}"
PERCENTAGE_FORMAT = "{:.1f}%" 