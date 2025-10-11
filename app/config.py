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
    "食品酒水": "Food & Dining",
    "买菜": "Food & Dining",
    "外出美食": "Food & Dining",

    # Shopping
    "购物消费": "Shopping",
    "日常用品": "Shopping",
    "厨房用品": "Shopping",
    "衣裤鞋帽": "Shopping",
    "电子数码": "Shopping",
    "家具家电": "Shopping",
    "家居饰品": "Shopping",
    "洗护用品": "Shopping",
    "清洁用品": "Shopping",
    "美妆护肤": "Shopping",
    "珠宝首饰": "Shopping",

    # Transportation
    "行车交通": "Transportation",
    "交通费": "Transportation",
    "加油": "Transportation",
    "停车": "Transportation",
    "打车": "Transportation",
    "地铁": "Transportation",
    "汽车用品": "Transportation",
    "car insurance": "Transportation",
    "保养": "Transportation",
    "驾照": "Transportation",
    "违章罚款": "Transportation",

    # Household & Utilities
    "居家生活": "Household",
    "水费": "Utilities",
    "电费": "Utilities",
    "燃气费": "Utilities",
    "网费": "Utilities",
    "物业": "Utilities",
    "房贷": "Housing Loan",
    "租金": "Rent",

    # Renovation
    "装修费用": "Renovation",
    "装修人工": "Renovation",
    "装修材料": "Renovation",
    "Bunnings": "Renovation",

    # Communication
    "交流通讯": "Communication",
    "手机话费": "Communication",
    "快递费": "Communication",

    # Insurance & Financial
    "金融保险": "Insurance & Financial",
    "人身保险": "Insurance & Financial",
    "自住房保险": "Insurance & Financial",

    # Healthcare
    "医疗教育": "Healthcare & Education",
    "医疗护理": "Healthcare",
    "治疗费": "Healthcare",
    "药品费": "Healthcare",
    "学费": "Education",

    # Leisure & Entertainment
    "休闲娱乐": "Entertainment",
    "电影": "Entertainment",
    "运动": "Entertainment",
    "娱乐费": "Entertainment",
    "其他娱乐": "Entertainment",

    # Travel
    "出差旅游": "Travel",
    "住宿费": "Travel",
    "旅游费用": "Travel",
    "飞机": "Travel",

    # Social & Personal
    "人情费用": "Social",
    "人情收礼": "Social",
    "红包": "Social",
    "所收红包": "Social",
    "婚嫁": "Social",
    "孝敬长辈": "Social",
    "礼品": "Social",

    # Pet & Personal Care
    "理发💇": "Personal Care",

    # Investment & Property
    "投资房产": "Investment Property",
    "投资房贷": "Investment Property",
    "投资房杂项": "Investment Property",
    "Council Rate（投资）": "Property Tax",

    # Owner Occupied Housing
    "自助房": "Owner Occupied Housing",
    "Council Rate (自住)": "Owner Occupied Housing",

    # Income
    "职业收入": "Income",
    "工资收入": "Income",
    "奖金收入": "Income",
    "其他收入": "Income",
    "意外来钱": "Income",
    "津贴": "Income",
    "利息收入": "Income",
    "Cash rewards": "Income",
    "CPA资料出售": "Income",

    # Other / Exceptional
    "烂账损失": "Bad Debt",
    "其他杂项": "Other",
    "其他支出": "Other",
    "维修": "Maintenance",
    "维修费": "Maintenance",
    "Solar": "Maintenance",
    "immi": "Government Fees"
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