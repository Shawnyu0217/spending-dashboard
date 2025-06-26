from abc import ABC, abstractmethod
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

class BasePage(ABC):
    """
    Abstract base class for all dashboard pages.
    Follows SOLID principles - specifically Open/Closed Principle.
    """
    
    def __init__(self):
        """Initialize page with default configuration."""
        self._config = self._load_page_config()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Page name for navigation."""
        pass
    
    @property
    @abstractmethod
    def icon(self) -> str:
        """Icon for the tab."""
        pass
    
    @property
    def key(self) -> str:
        """Unique key for the page."""
        return self.name.lower().replace(" ", "_")
    
    @abstractmethod
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Render the page content.
        
        Args:
            df: Filtered dataframe containing transaction data
            filters: Dictionary of current filter selections
        """
        pass
    
    def should_display(self, df: pd.DataFrame) -> bool:
        """
        Determine if this page should be displayed.
        
        Args:
            df: The filtered dataframe
            
        Returns:
            bool: True if page should be shown
        """
        if df.empty:
            return False
        
        min_rows = self.get_min_data_points()
        return len(df) >= min_rows
    
    def get_min_data_points(self) -> int:
        """
        Minimum number of data points required for this page.
        Override in child classes as needed.
        """
        return 0
    
    def _load_page_config(self) -> Dict[str, Any]:
        """Load page-specific configuration from config file."""
        from app.config import PAGE_SETTINGS
        return PAGE_SETTINGS.get(self.key, {})
    
    def render_error(self, error_msg: str) -> None:
        """Standardized error rendering for pages."""
        st.error(f"Error in {self.name} page: {error_msg}")
