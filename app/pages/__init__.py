from typing import List, Dict, Type, Optional
import pandas as pd
from .base import BasePage
from .overview import OverviewPage

# Page registry - add new pages here as they're created
PAGE_CLASSES: List[Type[BasePage]] = [
    OverviewPage,
]

def get_available_pages(df: pd.DataFrame) -> List[BasePage]:
    """
    Get list of pages that should be displayed based on data availability.
    
    Args:
        df: The filtered dataframe
        
    Returns:
        List of instantiated page objects
    """
    pages = []
    for page_class in PAGE_CLASSES:
        try:
            page = page_class()
            if page.should_display(df):
                pages.append(page)
        except Exception as e:
            # Log error but don't crash the app
            print(f"Error initializing {page_class.__name__}: {str(e)}")
    
    return pages

def get_page_by_key(key: str) -> Optional[BasePage]:
    """Get a specific page by its key."""
    for page_class in PAGE_CLASSES:
        try:
            page = page_class()
            if page.key == key:
                return page
        except Exception as e:
            # Log error but continue searching other pages
            print(f"Error initializing {page_class.__name__}: {str(e)}")
    return None
