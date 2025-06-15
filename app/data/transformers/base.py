"""
Base classes and interfaces for data transformers.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Protocol, List, Optional, Any, Dict


# Set up logger for transformers
logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Custom exception for transformation errors."""
    pass


class DataTransformer(Protocol):
    """Protocol for data transformation steps."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        ...


class BaseTransformer(ABC):
    """Abstract base class for data transformers with enhanced error handling."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the transformer.
        
        Args:
            name: Optional name for the transformer (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def safe_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely transform the DataFrame with error handling and logging.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        if df.empty:
            self.logger.warning(f"{self.name}: Input DataFrame is empty, returning as-is")
            return df
        
        try:
            self.logger.info(f"{self.name}: Starting transformation on {len(df)} rows")
            
            # Validate input
            self._validate_input(df)
            
            # Perform transformation
            result = self.transform(df)
            
            # Validate output
            self._validate_output(result, df)
            
            self.logger.info(f"{self.name}: Transformation completed successfully, output has {len(result)} rows")
            return result
            
        except Exception as e:
            error_msg = f"{self.name}: Transformation failed - {str(e)}"
            self.logger.error(error_msg)
            raise TransformationError(error_msg) from e
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame before transformation.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            TransformationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise TransformationError(f"{self.name}: Input must be a pandas DataFrame")
        
        required_columns = self.get_required_columns()
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"{self.name}: Missing required columns: {missing_columns}")
                # Don't raise error, just log warning - let transformer decide how to handle
    
    def _validate_output(self, output_df: pd.DataFrame, input_df: pd.DataFrame) -> None:
        """
        Validate output DataFrame after transformation.
        
        Args:
            output_df: Output DataFrame to validate
            input_df: Original input DataFrame for comparison
            
        Raises:
            TransformationError: If validation fails
        """
        if not isinstance(output_df, pd.DataFrame):
            raise TransformationError(f"{self.name}: Output must be a pandas DataFrame")
        
        # Check for significant data loss (more than 50% of rows lost)
        if len(output_df) < len(input_df) * 0.5:
            self.logger.warning(
                f"{self.name}: Significant data loss detected - "
                f"input: {len(input_df)} rows, output: {len(output_df)} rows"
            )
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that required columns exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns exist, False otherwise
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"{self.name}: Missing columns: {missing_columns}")
            return False
        return True
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        return []
    
    def get_output_columns(self) -> List[str]:
        """
        Get list of columns that this transformer adds to the DataFrame.
        
        Returns:
            List of output column names
        """
        return []
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about this transformation.
        
        Returns:
            Dictionary with transformation metadata
        """
        return {
            "name": self.name,
            "required_columns": self.get_required_columns(),
            "output_columns": self.get_output_columns(),
            "description": self.__doc__ or "No description available"
        } 