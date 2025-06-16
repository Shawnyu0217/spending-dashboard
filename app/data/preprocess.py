"""
Data preprocessing and transformation functions.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Tuple

from ..config import CATEGORY_MAPPINGS

def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Main preprocessing function that applies all transformations using the new pipeline architecture.
    
    Args:
        df_raw: Raw DataFrame from loader
        
    Returns:
        Tuple of (processed_df, dimension_tables)
    """
    from .pipeline import PreprocessingPipeline
    
    # Create and configure pipeline
    pipeline = PreprocessingPipeline.create_default_pipeline(CATEGORY_MAPPINGS)
    
    # Set progress callback for UI notifications
    # pipeline.set_progress_callback(lambda msg: st.success(msg))
    
    # Process data using pipeline
    return pipeline.process(df_raw)

 