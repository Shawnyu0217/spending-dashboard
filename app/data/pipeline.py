"""
Data preprocessing pipeline orchestrator.
"""

import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Callable
from .transformers import DataTransformer, DateTransformer, FinancialTransformer, CategoryTransformer
from .transformers.base import TransformationError
from .dimension_builder import DimensionBuilder


# Set up logger for pipeline
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates data preprocessing with single responsibility and enhanced error handling."""
    
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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
            
        Raises:
            TransformationError: If any transformation fails
        """
        if df.empty:
            self.logger.warning("Input DataFrame is empty")
            return df, {}
        
        self.logger.info(f"Starting pipeline processing with {len(df)} rows")
        
        try:
            # Apply all transformations in sequence
            result = df.copy()
            
            for i, transformer in enumerate(self.transformers):
                transformer_name = getattr(transformer, 'name', transformer.__class__.__name__)
                self.logger.info(f"Applying transformer {i+1}/{len(self.transformers)}: {transformer_name}")
                
                # Use safe_transform if available (for BaseTransformer instances)
                if hasattr(transformer, 'safe_transform'):
                    result = transformer.safe_transform(result)
                else:
                    # Fallback to regular transform for protocol implementations
                    result = transformer.transform(result)
                
                # Report progress if callback is set
                if self.progress_callback:
                    progress_msg = f"Applied {transformer_name} ({i+1}/{len(self.transformers)})"
                    self.progress_callback(progress_msg)
            
            # Create dimension tables
            self.logger.info("Creating dimension tables")
            dim_tables = self.dimension_builder.create_dimension_tables(result)
            
            # Final progress report
            if self.progress_callback:
                self.progress_callback(f"Preprocessing complete: {result.shape[0]} rows processed")
            
            self.logger.info(f"Pipeline processing completed successfully. Output: {len(result)} rows, {len(dim_tables)} dimension tables")
            
            return result, dim_tables
            
        except TransformationError:
            # Re-raise transformation errors as-is
            raise
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise TransformationError(error_msg) from e
    
    def get_pipeline_info(self) -> Dict[str, any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline metadata
        """
        transformer_info = []
        for transformer in self.transformers:
            if hasattr(transformer, 'get_transformation_info'):
                transformer_info.append(transformer.get_transformation_info())
            else:
                transformer_info.append({
                    "name": transformer.__class__.__name__,
                    "description": "Legacy transformer"
                })
        
        return {
            "transformer_count": len(self.transformers),
            "transformers": transformer_info,
            "has_dimension_builder": self.dimension_builder is not None
        }
    
    @classmethod
    def create_default_pipeline(cls, category_mappings: Dict[str, str] = None) -> "PreprocessingPipeline":
        """
        Create a default preprocessing pipeline with standard transformers.
        
        Args:
            category_mappings: Optional category mappings for CategoryTransformer
            
        Returns:
            Configured preprocessing pipeline
        """
        logger.info("Creating default preprocessing pipeline")
        
        transformers = [
            DateTransformer(),
            FinancialTransformer(),
            CategoryTransformer(category_mappings)
        ]
        
        pipeline = cls(transformers, DimensionBuilder())
        
        logger.info(f"Default pipeline created with {len(transformers)} transformers")
        return pipeline 