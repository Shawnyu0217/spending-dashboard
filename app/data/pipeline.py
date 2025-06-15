"""
Data preprocessing pipeline orchestrator.
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from .transformers import DataTransformer, DateTransformer, FinancialTransformer, CategoryTransformer
from .dimension_builder import DimensionBuilder


class PreprocessingPipeline:
    """Orchestrates data preprocessing with single responsibility."""
    
    def __init__(self, transformers: List[DataTransformer], dimension_builder: Optional[DimensionBuilder] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            transformers: List of data transformers to apply
            dimension_builder: Optional dimension builder for creating dimension tables
        """
        self.transformers = transformers
        self.dimension_builder = dimension_builder or DimensionBuilder()
        self.progress_callback: Optional[Callable[[str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """
        Set a callback function for progress reporting.
        
        Args:
            callback: Function to call with progress messages
        """
        self.progress_callback = callback
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Apply all transformations in sequence and create dimension tables.
        
        Args:
            df: Input DataFrame to process
            
        Returns:
            Tuple of (processed_df, dimension_tables)
        """
        if df.empty:
            return df, {}
        
        # Apply all transformations in sequence
        result = df.copy()
        for transformer in self.transformers:
            result = transformer.transform(result)
        
        # Create dimension tables
        dim_tables = self.dimension_builder.create_dimension_tables(result)
        
        # Report progress if callback is set
        if self.progress_callback:
            self.progress_callback(f"Preprocessing complete: {result.shape[0]} rows processed")
        
        return result, dim_tables
    
    @classmethod
    def create_default_pipeline(cls, category_mappings: Dict[str, str] = None) -> "PreprocessingPipeline":
        """
        Create a default preprocessing pipeline with standard transformers.
        
        Args:
            category_mappings: Optional category mappings for CategoryTransformer
            
        Returns:
            Configured preprocessing pipeline
        """
        transformers = [
            DateTransformer(),
            FinancialTransformer(),
            CategoryTransformer(category_mappings)
        ]
        
        return cls(transformers, DimensionBuilder()) 