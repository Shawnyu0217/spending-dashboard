"""
Category mapping and classification transformations.
"""

import pandas as pd
from typing import Dict
from .base import BaseTransformer
from app.config import CATEGORY_MAPPINGS


class CategoryTransformer(BaseTransformer):
    """Adds category mappings and top-level category groupings."""
    
    def __init__(self, category_mappings: Dict[str, str] = None):
        """
        Initialize the category transformer.
        
        Args:
            category_mappings: Dictionary mapping original categories to display names
        """
        self.category_mappings = category_mappings or CATEGORY_MAPPINGS
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add category mappings and top-level category groupings.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with category mappings
        """
        if not self._validate_columns(df, ["category"]):
            return df
        
        df_result = df.copy()
        
        # Map Chinese categories to English display names
        df_result["category_display"] = df_result["category"].map(
            self.category_mappings
        ).fillna(df_result["category"])
        
        # Apply the enhanced mapping
        df_result["top_level_category"] = df_result.apply(
            lambda row: self._get_top_level_category(row["category_display"], row["category"]), 
            axis=1
        )
        
        return df_result
    
    def _get_top_level_category(self, category_display: str, original_category: str) -> str:
        """
        Get top-level category with fallback logic.
        
        Args:
            category_display: Display category name
            original_category: Original category name
            
        Returns:
            Top-level category classification
        """
        # First try mapping from display category
        top_level_mapping = {
            # Core living expenses
            "Food & Dining": "Living Expenses",
            "Shopping": "Living Expenses",
            "Transportation": "Living Expenses",
            "Personal Care": "Living Expenses",
            "Pet Care": "Living Expenses",
            "Maintenance": "Living Expenses",
            "Communication": "Living Expenses",
            
            # Essential fixed costs
            "Utilities": "Essential",
            "Insurance & Financial": "Essential",
            "Healthcare": "Essential",
            "Healthcare & Education": "Essential",
            "Education": "Essential",
            "Living Services": "Essential",

            # Property & Housing
            "Housing Loan": "Property & Financial",
            "Investment Property": "Property & Financial",
            "Rent": "Property & Financial",
            "Renovation": "Property & Financial",
            "Property Tax": "Property & Financial",
            "Government Fees": "Property & Financial",

            # Lifestyle / Discretionary
            "Entertainment": "Discretionary",
            "Travel": "Discretionary",
            "Social": "Discretionary",
            "Other": "Discretionary",
            "Bad Debt": "Discretionary",
            
            # Income Types
            "Income": "Income",
            "Salary": "Income",
            "Bonus": "Income",
            "Investment Income": "Income",
            "Other Income": "Income",
            "Allowance": "Income",
            "Subsidy": "Income"
        }
        
        # If display category is mapped, use it
        if category_display in top_level_mapping:
            return top_level_mapping[category_display]
        
        # Direct mapping from common Chinese categories to top-level
        chinese_to_top_level = {
            "餐饮": "Living Expenses",
            "购物": "Living Expenses",
            "交通": "Living Expenses", 
            "生活服务": "Living Expenses",
            "日用品": "Living Expenses",
            "超市": "Living Expenses",
            "服饰": "Living Expenses",
            "家居": "Living Expenses",
            "医疗健康": "Essential",
            "教育": "Essential",
            "医疗": "Essential",
            "娱乐": "Discretionary",
            "旅游": "Discretionary",
            "人情往来": "Discretionary",
            "通讯": "Essential",
            "运动": "Discretionary",
            "美容": "Discretionary",
            "宠物": "Living Expenses",
            "其他": "Other",
            "收入": "Income",
            "工资": "Income",
            "奖金": "Income",
            "投资收益": "Income",
            "理财": "Income"
        }
        
        # Try direct Chinese mapping
        if original_category in chinese_to_top_level:
            return chinese_to_top_level[original_category]
        
        # If it's an income transaction, classify as Income
        if any(income_word in str(original_category) for income_word in ['收入', '工资', '奖金', '薪', '津贴']):
            return "Income"
        
        # Default fallback based on common patterns
        category_str = str(original_category).lower()
        if any(food_word in category_str for food_word in ['餐', '食', '饮', '吃', '喝']):
            return "Living Expenses"
        elif any(shop_word in category_str for shop_word in ['购', '买', '商', '市']):
            return "Living Expenses"
        elif any(transport_word in category_str for transport_word in ['交通', '车', '油', '停车', '地铁', '公交']):
            return "Living Expenses"
        elif any(health_word in category_str for health_word in ['医', '药', '健康', '保健']):
            return "Essential"
        elif any(fun_word in category_str for fun_word in ['娱乐', '游戏', '电影', '旅游', '运动']):
            return "Discretionary"
        else:
            return "Other" 