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
    # Food & Dining
    "餐饮": "Food & Dining",
    "食物": "Food & Dining",
    "饮食": "Food & Dining",
    "外卖": "Food & Dining",
    "餐厅": "Food & Dining",
    
    # Shopping & Retail
    "购物": "Shopping", 
    "日用品": "Shopping",
    "超市": "Shopping",
    "服饰": "Shopping",
    "家居": "Shopping",
    "电子产品": "Shopping",
    
    # Transportation
    "交通": "Transportation",
    "打车": "Transportation",
    "公交": "Transportation",
    "地铁": "Transportation",
    "汽油": "Transportation",
    "停车": "Transportation",
    
    # Living Services
    "生活服务": "Living Services",
    "通讯": "Living Services",
    "水电费": "Living Services",
    "房租": "Living Services",
    "物业": "Living Services",
    
    # Healthcare
    "医疗健康": "Healthcare",
    "医疗": "Healthcare",
    "药品": "Healthcare",
    "体检": "Healthcare",
    "保健": "Healthcare",
    
    # Education
    "教育": "Education",
    "培训": "Education",
    "书籍": "Education",
    "学费": "Education",
    
    # Entertainment
    "娱乐": "Entertainment",
    "游戏": "Entertainment",
    "电影": "Entertainment",
    "运动": "Entertainment",
    "健身": "Entertainment",
    
    # Travel
    "旅游": "Travel",
    "酒店": "Travel",
    "机票": "Travel",
    
    # Social & Personal
    "人情往来": "Social",
    "礼品": "Social",
    "美容": "Personal Care",
    "理发": "Personal Care",
    "宠物": "Pet Care",
    
    # Other
    "其他": "Other",
    "杂项": "Other",
    
    # Income
    "收入": "Income",
    "工资": "Salary",
    "奖金": "Bonus",
    "投资收益": "Investment Income",
    "理财": "Investment Income",
    "津贴": "Allowance",
    "补贴": "Subsidy"
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