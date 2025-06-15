# Personal Spending Dashboard - Refactoring Action Plan

## Executive Summary

This refactoring plan identifies key areas for improvement in the Personal Spending Dashboard project, prioritizing changes that will enhance code maintainability, performance, and adherence to KISS, YAGNI, and SOLID principles. Each priority includes concrete code examples demonstrating the improvements.

---

## 🔴 Priority Level 1: Critical Issues (Immediate Action Required)

### 1.1 Single Responsibility Principle Violations

**Issue**: Several modules contain functions doing multiple unrelated tasks

#### Example: Refactoring `preprocess.py`

**Before (Current Code)**:
```python
# app/data/preprocess.py - Function doing too many things
def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Main preprocessing function that applies all transformations."""
    if df_raw.empty:
        return df_raw, {}
    
    df_processed = df_raw.copy()
    
    # Multiple responsibilities in one function
    df_processed = add_derived_columns(df_processed)
    df_processed = add_financial_columns(df_processed)
    df_processed = add_category_mappings(df_processed)
    dim_tables = create_dimension_tables(df_processed)
    
    st.success(f"Preprocessing complete: {df_processed.shape[0]} rows processed")
    
    return df_processed, dim_tables
```

**After (Improved Code)**:
```python
# app/data/pipeline.py - New file with pipeline pattern
from abc import ABC, abstractmethod
from typing import Protocol

class DataTransformer(Protocol):
    """Protocol for data transformation steps."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class PreprocessingPipeline:
    """Orchestrates data preprocessing with single responsibility."""
    def __init__(self, transformers: List[DataTransformer]):
        self.transformers = transformers
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations in sequence."""
        result = df.copy()
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

# app/data/transformers.py - Each transformer has single responsibility
class DateColumnTransformer(DataTransformer):
    """Adds date-derived columns."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns:
            return df
        
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.month_name()
        df["ym"] = df["date"].dt.to_period("M")
        return df

class FinancialColumnTransformer(DataTransformer):
    """Adds financial calculation columns."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "amount" not in df.columns or "transaction_type" not in df.columns:
            return df
            
        df["net_amount"] = df.apply(self._calculate_net_amount, axis=1)
        df["income_amount"] = df["amount"].where(df["transaction_type"] == "income", 0)
        df["expense_amount"] = df["amount"].where(df["transaction_type"] == "expense", 0)
        return df
    
    def _calculate_net_amount(self, row):
        if row["transaction_type"] == "income":
            return row["amount"]
        elif row["transaction_type"] == "expense":
            return -row["amount"]
        return 0

# Usage
pipeline = PreprocessingPipeline([
    DateColumnTransformer(),
    FinancialColumnTransformer(),
    CategoryMappingTransformer()
])
df_processed = pipeline.process(df_raw)
```

### 1.2 Error Handling Standardization

**Issue**: Inconsistent error handling across modules

#### Example: Centralized Error Handling

**Before (Current Code)**:
```python
# Scattered error handling
def load_excel_file(file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        st.sidebar.success(f"Loaded {len(sheets_dict)} sheets")
        return sheets_dict
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        raise

def apply_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            try:
                # Silent failure - just prints
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    return df
```

**After (Improved Code)**:
```python
# app/core/exceptions.py - Centralized exception definitions
class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass

class DataLoadError(DashboardError):
    """Raised when data loading fails."""
    pass

class DataValidationError(DashboardError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(DashboardError):
    """Raised when configuration is invalid."""
    pass

# app/core/error_handler.py - Centralized error handling
import logging
from functools import wraps
from typing import TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar('T')

def handle_errors(default_return=None, reraise=True):
    """Decorator for consistent error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except DashboardError:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                if reraise:
                    raise DashboardError(f"Operation failed: {func.__name__}") from e
                return default_return
        return wrapper
    return decorator

# app/data/loader.py - Using centralized error handling
@handle_errors(default_return={})
def load_excel_file(file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load Excel file with proper error handling."""
    if not os.path.exists(file_path):
        raise DataLoadError(f"File not found: {file_path}")
    
    try:
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        logger.info(f"Successfully loaded {len(sheets_dict)} sheets from {file_path}")
        return sheets_dict
    except pd.errors.ExcelFileError as e:
        raise DataLoadError(f"Invalid Excel file format: {file_path}") from e
    except PermissionError as e:
        raise DataLoadError(f"Permission denied accessing file: {file_path}") from e

# app/data/schema.py - With validation and error handling
def apply_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data types with proper error handling and reporting."""
    conversion_errors = []
    df_typed = df.copy()
    
    for col, dtype in COLUMN_DTYPES.items():
        if col not in df_typed.columns:
            continue
            
        try:
            if dtype == "datetime64[ns]":
                df_typed[col] = pd.to_datetime(df_typed[col], errors='coerce')
            elif dtype == "float64" and df_typed[col].dtype == "object":
                # Clean currency symbols first
                df_typed[col] = df_typed[col].str.replace(r'[¥,$\s]', '', regex=True)
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
            else:
                df_typed[col] = df_typed[col].astype(dtype)
                
            logger.debug(f"Successfully converted column '{col}' to {dtype}")
            
        except Exception as e:
            conversion_errors.append(f"Column '{col}': {str(e)}")
            logger.warning(f"Failed to convert column '{col}' to {dtype}: {e}")
    
    if conversion_errors:
        raise DataValidationError(
            f"Data type conversion errors:\n" + "\n".join(conversion_errors)
        )
    
    return df_typed
```

### 1.3 Configuration Management

**Issue**: Hard-coded values scattered throughout code

#### Example: Comprehensive Configuration System

**Before (Current Code)**:
```python
# Scattered hard-coded values
def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig.update_layout(
        height=400  # Hard-coded
    )

def average_daily_spending(df: pd.DataFrame) -> float:
    if date_range == 0:
        date_range = 1  # Magic number

def top_expense_categories(df: pd.DataFrame, n: int = 10):  # Hard-coded default
    pass

# In components.py
selected_years = st.sidebar.multiselect(
    "Select Years",
    default=available_years[-2:] if len(available_years) >= 2 else available_years,  # Hard-coded logic
)
```

**After (Improved Code)**:
```python
# app/core/config.py - Centralized configuration with validation
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from pathlib import Path

@dataclass
class ChartConfig:
    """Chart-related configuration."""
    default_height: int = 400
    default_width: Optional[int] = None
    color_scheme: Dict[str, str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                "income": "#2E8B57",
                "expense": "#DC143C",
                "net": "#4682B4"
            }

@dataclass
class DataConfig:
    """Data processing configuration."""
    min_date_range_days: int = 1
    default_top_categories: int = 10
    default_year_selection: int = 2
    max_file_size_mb: int = 50
    supported_file_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = ["xlsx", "xls"]

@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_rows_preview: int = 1000
    chunk_size: int = 10000

@dataclass
class AppConfig:
    """Main application configuration."""
    chart: ChartConfig
    data: DataConfig
    performance: PerformanceConfig
    
    # Environment-specific settings
    debug_mode: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            chart=ChartConfig(
                default_height=int(os.getenv("CHART_HEIGHT", "400")),
            ),
            data=DataConfig(
                max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
                default_top_categories=int(os.getenv("DEFAULT_TOP_CATEGORIES", "10"))
            ),
            performance=PerformanceConfig(
                enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
                cache_ttl_seconds=int(os.getenv("CACHE_TTL", "3600"))
            ),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate(self):
        """Validate configuration values."""
        if self.chart.default_height < 100:
            raise ConfigurationError("Chart height must be at least 100px")
        if self.data.max_file_size_mb < 1:
            raise ConfigurationError("Max file size must be at least 1MB")
        if self.performance.cache_ttl_seconds < 0:
            raise ConfigurationError("Cache TTL must be non-negative")

# app/core/settings.py - Global settings instance
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Get application settings (singleton)."""
    config = AppConfig.from_env()
    config.validate()
    return config

# Usage in refactored code
from app.core.settings import get_settings

def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    settings = get_settings()
    
    fig.update_layout(
        height=settings.chart.default_height,
        # ... other settings
    )

def average_daily_spending(df: pd.DataFrame) -> float:
    settings = get_settings()
    
    if date_range == 0:
        date_range = settings.data.min_date_range_days

def top_expense_categories(df: pd.DataFrame, n: int = None):
    settings = get_settings()
    if n is None:
        n = settings.data.default_top_categories
```

---

## 🟡 Priority Level 2: Performance & Efficiency

### 2.1 Data Processing Optimization

**Issue**: Multiple passes over data during preprocessing

#### Example: Single-Pass Data Processing

**Before (Current Code)**:
```python
# Multiple iterations over the same data
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_derived = df.copy()
    df_derived["year"] = df_derived["date"].dt.year
    df_derived["month"] = df_derived["date"].dt.month
    # ... more operations
    return df_derived

def add_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_financial = df.copy()
    df_financial["net_amount"] = df_financial.apply(lambda row: ..., axis=1)
    # ... more operations
    return df_financial

def add_category_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df_categories = df.copy()
    df_categories["category_display"] = df_categories["category"].map(CATEGORY_MAPPINGS)
    # ... more operations
    return df_categories
```

**After (Improved Code)**:
```python
# app/data/optimized_processor.py - Single-pass processing
import numpy as np
from typing import Dict, Any

class OptimizedDataProcessor:
    """Efficient single-pass data processor."""
    
    def __init__(self, category_mappings: Dict[str, str]):
        self.category_mappings = category_mappings
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe in a single efficient pass."""
        if df.empty:
            return df
            
        # Copy once
        result = df.copy()
        
        # Vectorized operations instead of apply
        if "date" in result.columns:
            # All date operations at once
            result["year"] = result["date"].dt.year
            result["month"] = result["date"].dt.month
            result["day"] = result["date"].dt.day
            result["weekday"] = result["date"].dt.day_name()
            result["month_name"] = result["date"].dt.month_name()
            result["ym"] = result["date"].dt.to_period("M")
            result["ym_str"] = result["ym"].astype(str)
            result["quarter"] = result["date"].dt.quarter
        
        # Vectorized financial calculations
        if "amount" in result.columns and "transaction_type" in result.columns:
            # Use numpy where for better performance
            result["net_amount"] = np.where(
                result["transaction_type"] == "income", result["amount"],
                np.where(result["transaction_type"] == "expense", -result["amount"], 0)
            )
            
            result["income_amount"] = np.where(
                result["transaction_type"] == "income", result["amount"], 0
            )
            
            result["expense_amount"] = np.where(
                result["transaction_type"] == "expense", result["amount"], 0
            )
            
            # Sort once for all cumulative operations
            result = result.sort_values("date")
            result["running_balance"] = result["net_amount"].cumsum()
            
            # Group operations together
            result["monthly_cumulative"] = (
                result.groupby("ym_str")["net_amount"]
                .cumsum()
                .values  # Use values to avoid index alignment issues
            )
        
        # Category mappings with vectorized operation
        if "category" in result.columns:
            result["category_display"] = result["category"].map(
                self.category_mappings
            ).fillna(result["category"])
            
            # Pre-compute top-level categories
            result["top_level_category"] = self._vectorized_category_mapping(
                result["category_display"].values,
                result["category"].values
            )
        
        return result
    
    @staticmethod
    @np.vectorize
    def _vectorized_category_mapping(display_cat, original_cat):
        """Vectorized category mapping for performance."""
        # Mapping logic here (simplified)
        category_map = {
            "Food & Dining": "Living Expenses",
            "Shopping": "Living Expenses",
            # ... etc
        }
        return category_map.get(display_cat, "Other")

# Benchmark comparison
def benchmark_processing(df: pd.DataFrame):
    import time
    
    # Old method
    start = time.time()
    df1 = add_derived_columns(df)
    df1 = add_financial_columns(df1) 
    df1 = add_category_mappings(df1)
    old_time = time.time() - start
    
    # New method
    processor = OptimizedDataProcessor(CATEGORY_MAPPINGS)
    start = time.time()
    df2 = processor.process_dataframe(df)
    new_time = time.time() - start
    
    print(f"Old method: {old_time:.2f}s")
    print(f"New method: {new_time:.2f}s")
    print(f"Speedup: {old_time/new_time:.1f}x")
```

### 2.2 Caching Strategy Enhancement

**Issue**: Over-reliance on Streamlit's basic caching

#### Example: Multi-Level Caching

**Before (Current Code)**:
```python
@st.cache_data
def load_excel_file(file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    # Simple caching - all or nothing
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    return sheets_dict

def get_kpi_metrics(df: pd.DataFrame) -> Dict[str, any]:
    # No caching for expensive calculations
    metrics = {}
    metrics["total_income"] = total_income(df)
    metrics["total_expense"] = total_expense(df)
    # ... many calculations
    return metrics
```

**After (Improved Code)**:
```python
# app/core/cache.py - Advanced caching system
from functools import lru_cache, wraps
import hashlib
import pickle
from typing import Any, Callable, Optional
import streamlit as st

class CacheManager:
    """Manages multi-level caching with granular invalidation."""
    
    def __init__(self):
        self._memory_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = pickle.dumps((args, kwargs))
        return hashlib.md5(key_data).hexdigest()
    
    def cached(self, ttl: Optional[int] = None, maxsize: int = 128):
        """Decorator for caching with TTL and size limits."""
        def decorator(func: Callable) -> Callable:
            # Use LRU cache for memory efficiency
            cached_func = lru_cache(maxsize=maxsize)(func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check if caching is enabled
                if not get_settings().performance.enable_caching:
                    return func(*args, **kwargs)
                
                # Try memory cache first
                cache_key = self.cache_key(func.__name__, *args, **kwargs)
                
                if cache_key in self._memory_cache:
                    self._cache_stats["hits"] += 1
                    return self._memory_cache[cache_key]
                
                # Cache miss
                self._cache_stats["misses"] += 1
                result = cached_func(*args, **kwargs)
                self._memory_cache[cache_key] = result
                
                return result
            
            wrapper.cache_info = cached_func.cache_info
            wrapper.cache_clear = self._clear_cache
            return wrapper
        return decorator
    
    def _clear_cache(self):
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache_stats.copy()

# Global cache manager
cache_manager = CacheManager()

# app/features/optimized_kpis.py - Using granular caching
class KPICalculator:
    """Optimized KPI calculator with caching."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    @cache_manager.cached(maxsize=256)
    def _calculate_income(self, amounts: tuple, types: tuple) -> float:
        """Cached income calculation."""
        return sum(amt for amt, typ in zip(amounts, types) if typ == "income")
    
    @cache_manager.cached(maxsize=256)
    def _calculate_expenses(self, amounts: tuple, types: tuple) -> float:
        """Cached expense calculation."""
        return sum(amt for amt, typ in zip(amounts, types) if typ == "expense")
    
    def get_kpi_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate KPIs with intelligent caching."""
        # Convert to tuples for hashable cache keys
        amounts = tuple(df["amount"].values)
        types = tuple(df["transaction_type"].values)
        
        # Cached calculations
        total_income = self._calculate_income(amounts, types)
        total_expense = self._calculate_expenses(amounts, types)
        
        # Compute derived metrics
        metrics = {
            "total_income": total_income,
            "total_expense": total_expense,
            "net_savings": total_income - total_expense,
            "savings_rate": (total_income - total_expense) / total_income * 100 if total_income > 0 else 0
        }
        
        # Cache complex calculations separately
        metrics.update(self._calculate_category_metrics(df))
        metrics.update(self._calculate_trends(df))
        
        return metrics
    
    @cache_manager.cached(maxsize=128)
    def _calculate_category_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cached category calculations."""
        # Implementation here
        pass
    
    @cache_manager.cached(maxsize=64)
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cached trend calculations."""
        # Implementation here
        pass

# Usage with cache warming
def warm_cache(df: pd.DataFrame):
    """Pre-populate cache with common queries."""
    calculator = KPICalculator(cache_manager)
    
    # Warm up cache with common date ranges
    for year in df["year"].unique():
        year_df = df[df["year"] == year]
        calculator.get_kpi_metrics(year_df)
    
    # Warm up with common category filters
    for category in df["category_display"].value_counts().head(5).index:
        cat_df = df[df["category_display"] == category]
        calculator.get_kpi_metrics(cat_df)
```

### 2.3 Memory Optimization

**Issue**: Full dataset kept in memory unnecessarily

#### Example: Memory-Efficient Data Handling

**Before (Current Code)**:
```python
def filter_data_by_selections(df: pd.DataFrame, **filters) -> pd.DataFrame:
    # Creates full copy regardless of filters
    df_filtered = df.copy()
    
    if selected_years:
        df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]
    # More filters...
    
    return df_filtered  # Full copy even if minimal filtering

def create_dimension_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Creates multiple aggregations storing redundant data
    dim_tables = {}
    dim_tables["categories"] = df.groupby(["category_display", "top_level_category"]).agg({
        "amount": "sum",
        "transaction_type": "count"
    }).reset_index()
    # More dimension tables...
```

**After (Improved Code)**:
```python
# app/data/memory_efficient.py
import pandas as pd
from typing import Iterator, Optional, Dict, Any

class MemoryEfficientDataManager:
    """Manages data with minimal memory footprint."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self._data_view: Optional[pd.DataFrame] = None
        self._filters: Dict[str, Any] = {}
    
    def set_data(self, df: pd.DataFrame):
        """Set data using view instead of copy when possible."""
        # Use view for base data
        self._data_view = df
        
        # Optimize data types to reduce memory
        self._optimize_dtypes()
    
    def _optimize_dtypes(self):
        """Optimize data types for memory efficiency."""
        if self._data_view is None:
            return
            
        # Downcast numeric types
        for col in self._data_view.select_dtypes(include=['float']).columns:
            self._data_view[col] = pd.to_numeric(
                self._data_view[col], downcast='float'
            )
        
        for col in self._data_view.select_dtypes(include=['int']).columns:
            self._data_view[col] = pd.to_numeric(
                self._data_view[col], downcast='integer'
            )
        
        # Convert string columns with low cardinality to category
        for col in self._data_view.select_dtypes(include=['object']).columns:
            if self._data_view[col].nunique() / len(self._data_view) < 0.5:
                self._data_view[col] = self._data_view[col].astype('category')
    
    def get_filtered_view(self, **filters) -> pd.DataFrame:
        """Get filtered view without copying unless necessary."""
        if not filters:
            return self._data_view
        
        # Build boolean mask instead of copying
        mask = pd.Series(True, index=self._data_view.index)
        
        if 'selected_years' in filters and filters['selected_years']:
            mask &= self._data_view["year"].isin(filters['selected_years'])
        
        if 'selected_categories' in filters and filters['selected_categories']:
            mask &= self._data_view["category_display"].isin(filters['selected_categories'])
        
        # Return view, not copy
        return self._data_view.loc[mask]
    
    def iterate_chunks(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Iterate over data in chunks for memory efficiency."""
        for start in range(0, len(df), self.chunk_size):
            yield df.iloc[start:start + self.chunk_size]
    
    def create_lightweight_dimensions(self) -> Dict[str, pd.DataFrame]:
        """Create dimension tables with minimal memory usage."""
        dimensions = {}
        
        # Use categorical data for dimensions
        if "category_display" in self._data_view.columns:
            # Only store unique values and counts
            cat_counts = self._data_view["category_display"].value_counts()
            dimensions["categories"] = pd.DataFrame({
                "category": cat_counts.index,
                "count": cat_counts.values
            })
        
        if "year" in self._data_view.columns:
            # Store only unique years as int16
            dimensions["years"] = pd.DataFrame({
                "year": self._data_view["year"].unique().astype('int16')
            })
        
        return dimensions
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if self._data_view is None:
            return {}
            
        total_mb = self._data_view.memory_usage(deep=True).sum() / 1024 / 1024
        
        return {
            "total_mb": total_mb,
            "rows": len(self._data_view),
            "mb_per_row": total_mb / len(self._data_view) if len(self._data_view) > 0 else 0,
            "column_usage": {
                col: self._data_view[col].memory_usage(deep=True) / 1024 / 1024
                for col in self._data_view.columns
            }
        }

# Usage example
def process_large_file(file_path: str):
    """Process large files efficiently."""
    manager = MemoryEfficientDataManager()
    
    # Read in chunks for large files
    chunks = []
    for chunk in pd.read_excel(file_path, chunksize=10000):
        # Process each chunk
        processed_chunk = preprocess_chunk(chunk)
        chunks.append(processed_chunk)
    
    # Combine and optimize
    df = pd.concat(chunks, ignore_index=True)
    manager.set_data(df)
    
    # Get memory usage
    memory_stats = manager.get_memory_usage()
    st.info(f"Data loaded: {memory_stats['total_mb']:.1f} MB for {memory_stats['rows']:,} rows")
    
    return manager
```

---

## 🟢 Priority Level 3: Code Quality & Maintainability

### 3.1 Function Decomposition (KISS Principle)

**Issue**: Many functions exceed 50 lines

#### Example: Breaking Down Complex Functions

**Before (Current Code)**:
```python
def create_sidebar_filters(dim_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create sidebar filters and return selected values."""
    st.sidebar.header("📊 Dashboard Filters")
    
    filters = {}
    
    st.sidebar.divider()
    
    # Year filter - 20+ lines
    if "years" in dim_tables and not dim_tables["years"].empty:
        st.sidebar.subheader("📅 Time Period")
        available_years = dim_tables["years"]["year"].tolist()
        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=available_years,
            default=available_years[-2:] if len(available_years) >= 2 else available_years,
            help="Choose which years to include in analysis"
        )
        filters["selected_years"] = selected_years
    else:
        filters["selected_years"] = []
    
    # Month filter - 15+ lines
    if "months" in dim_tables and not dim_tables["months"].empty:
        # ... more code
    
    # Category filter - 20+ lines
    # Account filter - 15+ lines
    # Member filter - 15+ lines
    
    return filters  # Function is 100+ lines!
```

**After (Improved Code)**:
```python
# app/ui/filters.py - Decomposed filter components
from abc import ABC, abstractmethod
from typing import List, Any, Optional

class FilterComponent(ABC):
    """Base class for filter components."""
    
    @abstractmethod
    def render(self) -> Any:
        """Render the filter component."""
        pass
    
    @abstractmethod
    def get_value(self) -> Any:
        """Get the selected value."""
        pass

class YearFilter(FilterComponent):
    """Year selection filter."""
    
    def __init__(self, years: List[int], default_count: int = 2):
        self.years = years
        self.default_count = default_count
        self._selected = None
    
    def render(self) -> List[int]:
        """Render year selection widget."""
        default = self._get_default_years()
        
        self._selected = st.sidebar.multiselect(
            "Select Years",
            options=self.years,
            default=default,
            help="Choose which years to include in analysis"
        )
        return self._selected
    
    def _get_default_years(self) -> List[int]:
        """Get default year selection."""
        if len(self.years) >= self.default_count:
            return self.years[-self.default_count:]
        return self.years
    
    def get_value(self) -> List[int]:
        """Get selected years."""
        return self._selected or []

class CategoryFilter(FilterComponent):
    """Category selection filter."""
    
    def __init__(self, categories_df: pd.DataFrame, default_top_n: int = 10):
        self.categories = categories_df["category"].tolist()
        self.default_top_n = default_top_n
        self._selected = None
    
    def render(self) -> List[str]:
        """Render category selection widget."""
        default = self._get_default_categories()
        
        self._selected = st.sidebar.multiselect(
            "Select Categories",
            options=self.categories,
            default=default,
            help="Choose expense categories to analyze"
        )
        return self._selected
    
    def _get_default_categories(self) -> List[str]:
        """Get top N categories as default."""
        if len(self.categories) > self.default_top_n:
            return self.categories[:self.default_top_n]
        return self.categories
    
    def get_value(self) -> List[str]:
        """Get selected categories."""
        return self._selected or []

class FilterManager:
    """Manages all filter components."""
    
    def __init__(self, dim_tables: Dict[str, pd.DataFrame]):
        self.dim_tables = dim_tables
        self.filters: Dict[str, FilterComponent] = {}
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Initialize all filter components."""
        if "years" in self.dim_tables and not self.dim_tables["years"].empty:
            self.filters["years"] = YearFilter(
                self.dim_tables["years"]["year"].tolist()
            )
        
        if "categories" in self.dim_tables and not self.dim_tables["categories"].empty:
            self.filters["categories"] = CategoryFilter(
                self.dim_tables["categories"]
            )
        
        # Add more filters...
    
    def render_all(self) -> Dict[str, Any]:
        """Render all filters and return selections."""
        st.sidebar.header("📊 Dashboard Filters")
        st.sidebar.divider()
        
        selections = {}
        
        # Time filters
        if "years" in self.filters:
            st.sidebar.subheader("📅 Time Period")
            selections["selected_years"] = self.filters["years"].render()
        
        # Category filters
        if "categories" in self.filters:
            st.sidebar.subheader("🏷️ Categories")
            selections["selected_categories"] = self.filters["categories"].render()
        
        return selections

# Usage - clean and simple
def create_sidebar_filters(dim_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create sidebar filters using filter manager."""
    filter_manager = FilterManager(dim_tables)
    return filter_manager.render_all()
```

### 3.2 Type Safety Enhancement

**Issue**: Incomplete type hints

#### Example: Comprehensive Type Annotations

**Before (Current Code)**:
```python
def get_kpi_metrics(df: pd.DataFrame) -> Dict[str, any]:  # 'any' is not a type
    metrics = {}  # No type information
    # ...
    return metrics

def filter_data_by_selections(
    df: pd.DataFrame,
    selected_years: List[int] = None,  # Should use Optional
    selected_months: List[int] = None,
    # ... more parameters
) -> pd.DataFrame:
    pass

def monthly_trend(df: pd.DataFrame) -> Dict[str, float]:
    # Returns dict but structure is unclear
    return {
        "income_trend": 0.0,
        "expense_trend": 0.0,
        "savings_trend": 0.0
    }
```

**After (Improved Code)**:
```python
# app/types.py - Type definitions
from typing import TypedDict, Literal, Optional, List, Dict, Union
from datetime import datetime
import pandas as pd

# Define transaction types as literal
TransactionType = Literal["income", "expense", "transfer"]

# Define structured dictionaries
class KPIMetrics(TypedDict):
    """Type definition for KPI metrics."""
    total_income: float
    total_expense: float
    net_savings: float
    savings_rate: float
    largest_expense_category: str
    largest_expense_amount: float
    average_daily_spending: float
    total_transactions: int
    income_transactions: int
    expense_transactions: int

class MonthlyTrend(TypedDict):
    """Type definition for monthly trends."""
    income_trend: float
    expense_trend: float
    savings_trend: float

class FilterSelections(TypedDict, total=False):
    """Type definition for filter selections."""
    selected_years: Optional[List[int]]
    selected_months: Optional[List[int]]
    selected_categories: Optional[List[str]]
    selected_accounts: Optional[List[str]]
    selected_members: Optional[List[str]]
    date_range: Optional[tuple[datetime, datetime]]

# Type aliases for clarity
DataFrameDict = Dict[str, pd.DataFrame]
CategoryMapping = Dict[str, str]

# app/features/typed_kpis.py - Using proper types
from app.types import KPIMetrics, MonthlyTrend, TransactionType

def get_kpi_metrics(df: pd.DataFrame) -> KPIMetrics:
    """Get all KPI metrics with proper typing."""
    return KPIMetrics(
        total_income=calculate_total_income(df),
        total_expense=calculate_total_expense(df),
        net_savings=calculate_net_savings(df),
        savings_rate=calculate_savings_rate(df),
        largest_expense_category=get_largest_category(df),
        largest_expense_amount=get_largest_amount(df),
        average_daily_spending=calculate_daily_average(df),
        total_transactions=len(df),
        income_transactions=count_transactions(df, "income"),
        expense_transactions=count_transactions(df, "expense")
    )

def count_transactions(df: pd.DataFrame, transaction_type: TransactionType) -> int:
    """Count transactions of specific type."""
    return len(df[df["transaction_type"] == transaction_type])

def monthly_trend(df: pd.DataFrame) -> MonthlyTrend:
    """Calculate month-over-month trends with proper types."""
    # Implementation
    return MonthlyTrend(
        income_trend=calculate_trend(df, "income"),
        expense_trend=calculate_trend(df, "expense"),
        savings_trend=calculate_trend(df, "net")
    )

# app/data/typed_filters.py
from app.types import FilterSelections

def filter_data_by_selections(
    df: pd.DataFrame,
    filters: FilterSelections
) -> pd.DataFrame:
    """Filter DataFrame with typed selections."""
    result = df.copy()
    
    if filters.get("selected_years"):
        result = result[result["year"].isin(filters["selected_years"])]
    
    if filters.get("selected_months"):
        result = result[result["month"].isin(filters["selected_months"])]
    
    # ... more filtering
    
    return result

# Using Protocol for better interfaces
from typing import Protocol

class DataLoader(Protocol):
    """Protocol for data loading implementations."""
    
    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load data from source."""
        ...
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded data."""
        ...

class ExcelLoader:
    """Concrete implementation of DataLoader."""
    
    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        return pd.read_excel(source, sheet_name=None)
    
    def validate(self, df: pd.DataFrame) -> bool:
        required_columns = {"date", "amount", "transaction_type"}
        return required_columns.issubset(df.columns)
```

### 3.3 Code Duplication Removal (DRY Principle)

**Issue**: Similar patterns repeated across modules

#### Example: Creating Reusable Patterns

**Before (Current Code)**:
```python
# Repeated chart creation pattern
def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for monthly trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    # ... chart logic

def create_category_expense_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense data available",  # Similar pattern
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    # ... chart logic

# Repeated currency formatting
value=f"¥{x:,.2f}"  # In multiple places
```

**After (Improved Code)**:
```python
# app/features/chart_factory.py - Chart creation factory
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import plotly.graph_objects as go

class ChartBuilder(ABC):
    """Abstract base class for chart builders."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fig = go.Figure()
    
    def build(self) -> go.Figure:
        """Build chart with error handling."""
        if self.df.empty:
            return self._create_empty_chart()
        
        try:
            return self._create_chart()
        except Exception as e:
            return self._create_error_chart(str(e))
    
    @abstractmethod
    def _create_chart(self) -> go.Figure:
        """Create the actual chart."""
        pass
    
    @abstractmethod
    def _get_empty_message(self) -> str:
        """Get message for empty data."""
        pass
    
    def _create_empty_chart(self) -> go.Figure:
        """Create standardized empty chart."""
        fig = go.Figure()
        fig.add_annotation(
            text=self._get_empty_message(),
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False, 
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    def _create_error_chart(self, error: str) -> go.Figure:
        """Create standardized error chart."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {error}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

class MonthlyTrendChart(ChartBuilder):
    """Monthly trend chart builder."""
    
    def _get_empty_message(self) -> str:
        return "No data available for monthly trends"
    
    def _create_chart(self) -> go.Figure:
        monthly_data = self._prepare_data()
        
        # Add traces
        self._add_income_trace(monthly_data)
        self._add_expense_trace(monthly_data)
        self._add_savings_trace(monthly_data)
        
        # Configure layout
        self._configure_layout()
        
        return self.fig
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare monthly summary data."""
        return monthly_summary(self.df)
    
    def _add_income_trace(self, data: pd.DataFrame):
        """Add income trace to chart."""
        self.fig.add_trace(go.Scatter(
            x=data["month"],
            y=data["income"],
            mode='lines+markers',
            name='Income',
            line=dict(color=COLORS["income"], width=3),
            marker=dict(size=8)
        ))

# app/utils/formatters.py - Centralized formatting utilities
from typing import Union, Optional
from decimal import Decimal

class CurrencyFormatter:
    """Handles all currency formatting."""
    
    def __init__(self, symbol: str = "¥", decimal_places: int = 2):
        self.symbol = symbol
        self.decimal_places = decimal_places
    
    def format(self, value: Union[float, int, Decimal]) -> str:
        """Format value as currency."""
        return f"{self.symbol}{value:,.{self.decimal_places}f}"
    
    def format_short(self, value: Union[float, int, Decimal]) -> str:
        """Format large values in short form."""
        if abs(value) >= 1_000_000:
            return f"{self.symbol}{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{self.symbol}{value/1_000:.1f}K"
        else:
            return self.format(value)
    
    def parse(self, value: str) -> Optional[float]:
        """Parse currency string to float."""
        try:
            # Remove currency symbol and commas
            cleaned = value.replace(self.symbol, "").replace(",", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

# Global formatter instance
currency = CurrencyFormatter()

# Usage - DRY principle applied
def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create monthly trend chart using factory."""
    chart = MonthlyTrendChart(df)
    return chart.build()

def format_kpi_value(value: float, metric_type: str) -> str:
    """Format KPI value based on type."""
    if metric_type in ["amount", "income", "expense"]:
        return currency.format(value)
    elif metric_type == "percentage":
        return f"{value:.1f}%"
    else:
        return str(value)
```

---

## 🔵 Priority Level 4: Architecture & Design

### 4.1 Dependency Injection Implementation

**Issue**: Hard dependencies between modules

#### Example: Implementing Dependency Injection

**Before (Current Code)**:
```python
# Hard dependencies
from app.data.loader import load_excel_file
from app.features.kpis import get_kpi_metrics

def main():
    # Tightly coupled to specific implementations
    data = load_excel_file("data.xlsx")
    metrics = get_kpi_metrics(data)
```

**After (Improved Code)**:
```python
# app/core/container.py - Dependency injection container
from typing import Type, Dict, Any, Callable
from abc import ABC, abstractmethod

class ServiceContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
    
    def register(self, interface: Type, implementation: Any = None, factory: Callable = None):
        """Register a service or factory."""
        if implementation is not None:
            self._services[interface] = implementation
        elif factory is not None:
            self._factories[interface] = factory
        else:
            raise ValueError("Must provide either implementation or factory")
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service."""
        if interface in self._services:
            return self._services[interface]
        elif interface in self._factories:
            return self._factories[interface]()
        else:
            raise ValueError(f"No registration found for {interface}")

# app/interfaces/data.py - Define interfaces
from abc import ABC, abstractmethod

class IDataLoader(ABC):
    """Interface for data loading."""
    
    @abstractmethod
    def load(self, source: str) -> pd.DataFrame:
        """Load data from source."""
        pass

class IKPICalculator(ABC):
    """Interface for KPI calculations."""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> KPIMetrics:
        """Calculate KPIs from data."""
        pass

class IDataProcessor(ABC):
    """Interface for data processing."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data."""
        pass

# app/services/implementations.py - Concrete implementations
class ExcelDataLoader(IDataLoader):
    """Excel file data loader."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def load(self, source: str) -> pd.DataFrame:
        """Load data from Excel file."""
        max_size = self.config.data.max_file_size_mb * 1024 * 1024
        
        # Check file size
        if os.path.getsize(source) > max_size:
            raise DataLoadError(f"File exceeds maximum size of {self.config.data.max_file_size_mb}MB")
        
        return pd.read_excel(source, sheet_name=None)

class StandardKPICalculator(IKPICalculator):
    """Standard KPI calculator."""
    
    def __init__(self, formatter: CurrencyFormatter):
        self.formatter = formatter
    
    def calculate(self, df: pd.DataFrame) -> KPIMetrics:
        """Calculate standard KPIs."""
        # Implementation here
        pass

# app/core/bootstrap.py - Application bootstrap
def configure_services(container: ServiceContainer, config: AppConfig):
    """Configure all services."""
    # Register configuration
    container.register(AppConfig, config)
    
    # Register data loader
    container.register(
        IDataLoader,
        factory=lambda: ExcelDataLoader(config)
    )
    
    # Register KPI calculator
    container.register(
        IKPICalculator,
        factory=lambda: StandardKPICalculator(CurrencyFormatter())
    )
    
    # Register data processor
    container.register(
        IDataProcessor,
        factory=lambda: OptimizedDataProcessor(config.data.category_mappings)
    )

# app/dashboard_di.py - Using dependency injection
class Dashboard:
    """Main dashboard with dependency injection."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.data_loader = container.resolve(IDataLoader)
        self.kpi_calculator = container.resolve(IKPICalculator)
        self.data_processor = container.resolve(IDataProcessor)
    
    def run(self):
        """Run the dashboard."""
        # Load data
        raw_data = self.data_loader.load("data.xlsx")
        
        # Process data
        processed_data = self.data_processor.process(raw_data)
        
        # Calculate KPIs
        metrics = self.kpi_calculator.calculate(processed_data)
        
        # Display dashboard
        self.display(processed_data, metrics)

# Main entry point
def main():
    """Application entry point with DI."""
    # Create container
    container = ServiceContainer()
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Configure services
    configure_services(container, config)
    
    # Create and run dashboard
    dashboard = Dashboard(container)
    dashboard.run()
```

### 4.2 Plugin Architecture for Visualizations

**Issue**: Adding new chart types requires modifying core code

#### Example: Extensible Visualization System

**Before (Current Code)**:
```python
# All chart types hard-coded
def create_chart_container(title: str, chart_func, *args):
    # Limited to predefined chart functions
    fig = chart_func(*args)
    st.plotly_chart(fig)
```

**After (Improved Code)**:
```python
# app/plugins/base.py - Plugin base classes
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import plotly.graph_objects as go

class VisualizationPlugin(ABC):
    """Base class for visualization plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @property
    def config_schema(self) -> Dict[str, Any]:
        """Configuration schema for the plugin."""
        return {}
    
    @abstractmethod
    def create_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create the chart."""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate if data is suitable for this chart."""
        return True

# app/plugins/registry.py - Plugin registry
class PluginRegistry:
    """Registry for visualization plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, VisualizationPlugin] = {}
    
    def register(self, plugin: VisualizationPlugin):
        """Register a visualization plugin."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        
        self._plugins[plugin.name] = plugin
    
    def get(self, name: str) -> Optional[VisualizationPlugin]:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> Dict[str, str]:
        """List all available plugins."""
        return {
            name: plugin.description 
            for name, plugin in self._plugins.items()
        }
    
    def create_chart(self, name: str, df: pd.DataFrame, config: Dict[str, Any] = None) -> go.Figure:
        """Create chart using plugin."""
        plugin = self.get(name)
        if not plugin:
            raise ValueError(f"Plugin '{name}' not found")
        
        if not plugin.validate_data(df):
            raise ValueError(f"Data not suitable for plugin '{name}'")
        
        return plugin.create_chart(df, config or {})

# app/plugins/builtin/savings_goal.py - Example plugin
class SavingsGoalPlugin(VisualizationPlugin):
    """Savings goal tracking visualization."""
    
    @property
    def name(self) -> str:
        return "savings_goal_tracker"
    
    @property
    def description(self) -> str:
        return "Track progress towards savings goals"
    
    @property
    def config_schema(self) -> Dict[str, Any]:
        return {
            "goal_amount": {"type": "number", "default": 10000},
            "goal_date": {"type": "date", "required": True},
            "show_projection": {"type": "boolean", "default": True}
        }
    
    def create_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create savings goal chart."""
        goal_amount = config.get("goal_amount", 10000)
        goal_date = pd.to_datetime(config["goal_date"])
        
        # Calculate cumulative savings
        monthly_data = df.groupby("ym_str")["net_amount"].sum().cumsum()
        
        fig = go.Figure()
        
        # Actual savings line
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data.values,
            mode='lines+markers',
            name='Actual Savings',
            line=dict(color='blue', width=3)
        ))
        
        # Goal line
        fig.add_trace(go.Scatter(
            x=[monthly_data.index[0], goal_date.strftime("%Y-%m")],
            y=[0, goal_amount],
            mode='lines',
            name='Goal',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add projection if enabled
        if config.get("show_projection", True):
            self._add_projection(fig, monthly_data, goal_amount, goal_date)
        
        fig.update_layout(
            title=f"Progress Towards ¥{goal_amount:,.0f} Goal",
            xaxis_title="Month",
            yaxis_title="Cumulative Savings (¥)",
            hovermode='x unified'
        )
        
        return fig
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data has required columns."""
        return all(col in df.columns for col in ["ym_str", "net_amount"])

# app/plugins/loader.py - Plugin loader
import importlib
import pkgutil
from pathlib import Path

def load_builtin_plugins(registry: PluginRegistry):
    """Load all built-in plugins."""
    plugins_dir = Path(__file__).parent / "builtin"
    
    for module_info in pkgutil.iter_modules([str(plugins_dir)]):
        module = importlib.import_module(f"app.plugins.builtin.{module_info.name}")
        
        # Find all plugin classes in module
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, VisualizationPlugin) and 
                obj != VisualizationPlugin):
                
                # Instantiate and register
                plugin = obj()
                registry.register(plugin)

def load_user_plugins(registry: PluginRegistry, plugins_dir: str):
    """Load user-defined plugins from directory."""
    # Implementation for loading custom plugins
    pass

# Usage in dashboard
class VisualizationManager:
    """Manages chart creation with plugins."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        load_builtin_plugins(self.registry)
    
    def render_chart_selector(self) -> Optional[str]:
        """Render chart type selector."""
        plugins = self.registry.list_plugins()
        
        if not plugins:
            return None
        
        selected = st.selectbox(
            "Select Visualization Type",
            options=list(plugins.keys()),
            format_func=lambda x: plugins[x]
        )
        
        return selected
    
    def render_chart(self, chart_type: str, df: pd.DataFrame):
        """Render selected chart."""
        plugin = self.registry.get(chart_type)
        
        if not plugin:
            st.error(f"Unknown chart type: {chart_type}")
            return
        
        # Render configuration UI
        config = self._render_config_ui(plugin)
        
        # Create and display chart
        try:
            fig = plugin.create_chart(df, config)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    
    def _render_config_ui(self, plugin: VisualizationPlugin) -> Dict[str, Any]:
        """Render configuration UI for plugin."""
        config = {}
        schema = plugin.config_schema
        
        if not schema:
            return config
        
        with st.expander("Chart Configuration"):
            for key, spec in schema.items():
                if spec["type"] == "number":
                    config[key] = st.number_input(
                        key.replace("_", " ").title(),
                        value=spec.get("default", 0)
                    )
                elif spec["type"] == "boolean":
                    config[key] = st.checkbox(
                        key.replace("_", " ").title(),
                        value=spec.get("default", False)
                    )
                elif spec["type"] == "date":
                    config[key] = st.date_input(
                        key.replace("_", " ").title()
                    )
        
        return config
```

---

## ⚪ Priority Level 5: Infrastructure & DevOps

### 5.1 Logging System Implementation

**Issue**: No proper logging infrastructure

#### Example: Structured Logging System

**Before (Current Code)**:
```python
# Using print statements
print(f"Warning: Could not convert column {col} to {dtype}: {e}")
print(f"DEBUG: Expense transactions: {len(expenses_df)}")
```

**After (Improved Code)**:
```python
# app/core/logging_config.py - Logging configuration
import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(config: AppConfig):
    """Configure application logging."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    if config.debug_mode:
        # Detailed format for development
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Simple format for production
        console_format = logging.Formatter('%(levelname)s: %(message)s')
    
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(error_handler)

# app/core/logger.py - Logger utilities
class LoggerMixin:
    """Mixin to add logging to classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(
            f"Performance: {operation} took {duration:.3f}s",
            extra={"extra_fields": {"operation": operation, "duration_ms": duration * 1000, **kwargs}}
        )
    
    def log_data_operation(self, operation: str, rows: int, **kwargs):
        """Log data operations."""
        self.logger.info(
            f"Data operation: {operation} processed {rows} rows",
            extra={"extra_fields": {"operation": operation, "rows": rows, **kwargs}}
        )

# Usage examples
class DataProcessor(LoggerMixin):
    """Data processor with logging."""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data with logging."""
        start_time = time.time()
        self.logger.info(f"Starting data processing for {len(df)} rows")
        
        try:
            # Processing logic
            result = self._process_internal(df)
            
            duration = time.time() - start_time
            self.log_performance("data_processing", duration, rows=len(df))
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error processing data: {str(e)}",
                exc_info=True,
                extra={"extra_fields": {"rows": len(df), "columns": list(df.columns)}}
            )
            raise

# app/utils/audit.py - Audit logging
class AuditLogger:
    """Specialized logger for audit trails."""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        
        # Special handler for audit logs
        handler = logging.handlers.RotatingFileHandler(
            "logs/audit.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=10
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_data_access(self, user: str, operation: str, data_source: str, **kwargs):
        """Log data access for audit."""
        self.logger.info(
            f"Data access: {user} performed {operation} on {data_source}",
            extra={"extra_fields": {
                "user": user,
                "operation": operation,
                "data_source": data_source,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }}
        )
    
    def log_configuration_change(self, user: str, setting: str, old_value: Any, new_value: Any):
        """Log configuration changes."""
        self.logger.info(
            f"Configuration change: {user} changed {setting}",
            extra={"extra_fields": {
                "user": user,
                "setting": setting,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "timestamp": datetime.utcnow().isoformat()
            }}
        )

# Global audit logger
audit = AuditLogger()
```

### 5.2 Testing Infrastructure

**Issue**: Limited test coverage

#### Example: Comprehensive Testing Setup

**Before (Current Code)**:
```python
# Simple test without mocking or fixtures
def test_total_income():
    df = pd.DataFrame({"income_amount": [100, 200, 300]})
    assert total_income(df) == 600
```

**After (Improved Code)**:
```python
# tests/conftest.py - Pytest configuration and fixtures
import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import Mock

from app.core.container import ServiceContainer
from app.core.config import AppConfig

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample transaction data."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    for date in dates:
        # Generate random transactions
        n_transactions = np.random.randint(0, 5)
        for _ in range(n_transactions):
            transaction = {
                "date": date,
                "transaction_type": np.random.choice(["income", "expense"], p=[0.2, 0.8]),
                "amount": np.random.uniform(10, 1000),
                "category": np.random.choice(["Food", "Transport", "Shopping", "Utilities"]),
                "account": np.random.choice(["Cash", "Bank", "Credit Card"])
            }
            data.append(transaction)
    
    return pd.DataFrame(data)

@pytest.fixture
def test_config() -> AppConfig:
    """Test configuration."""
    return AppConfig(
        chart=ChartConfig(default_height=300),
        data=DataConfig(max_file_size_mb=10),
        performance=PerformanceConfig(enable_caching=False),
        debug_mode=True
    )

@pytest.fixture
def container(test_config) -> ServiceContainer:
    """Configured service container for testing."""
    container = ServiceContainer()
    configure_services(container, test_config)
    return container

# tests/unit/test_kpis.py - Comprehensive unit tests
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from app.features.kpis import KPICalculator, get_enhanced_kpi_metrics

class TestKPICalculator:
    """Test KPI calculations."""
    
    @pytest.fixture
    def calculator(self):
        """Create KPI calculator instance."""
        return KPICalculator(cache_manager=Mock())
    
    def test_total_income_calculation(self, calculator, sample_data):
        """Test income calculation accuracy."""
        # Arrange
        expected_income = sample_data[
            sample_data["transaction_type"] == "income"
        ]["amount"].sum()
        
        # Act
        result = calculator.calculate_income(sample_data)
        
        # Assert
        assert result == pytest.approx(expected_income, rel=0.01)
    
    @pytest.mark.parametrize("data_size", [100, 1000, 10000])
    def test_performance_scaling(self, calculator, data_size):
        """Test calculation performance with different data sizes."""
        # Generate data of specific size
        df = pd.DataFrame({
            "amount": np.random.uniform(0, 1000, data_size),
            "transaction_type": np.random.choice(["income", "expense"], data_size)
        })
        
        # Measure performance
        import time
        start = time.time()
        calculator.get_kpi_metrics(df)
        duration = time.time() - start
        
        # Assert linear scaling (roughly)
        assert duration < data_size / 10000  # Should process 10k rows/second
    
    def test_edge_cases(self, calculator):
        """Test edge cases and error conditions."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        metrics = calculator.get_kpi_metrics(empty_df)
        assert metrics["total_income"] == 0
        assert metrics["savings_rate"] == 0
        
        # All expenses, no income
        all_expense_df = pd.DataFrame({
            "amount": [100, 200],
            "transaction_type": ["expense", "expense"]
        })
        metrics = calculator.get_kpi_metrics(all_expense_df)
        assert metrics["savings_rate"] == 0  # Avoid division by zero
        
        # Single transaction
        single_df = pd.DataFrame({
            "amount": [1000],
            "transaction_type": ["income"]
        })
        metrics = calculator.get_kpi_metrics(single_df)
        assert metrics["total_income"] == 1000
        assert metrics["savings_rate"] == 100

# tests/integration/test_data_pipeline.py - Integration tests
class TestDataPipeline:
    """Test complete data processing pipeline."""
    
    @pytest.fixture
    def pipeline(self, container):
        """Create data pipeline from container."""
        return DataPipeline(
            loader=container.resolve(IDataLoader),
            processor=container.resolve(IDataProcessor),
            validator=container.resolve(IDataValidator)
        )
    
    @pytest.mark.integration
    def test_end_to_end_processing(self, pipeline, tmp_path):
        """Test complete data flow from file to metrics."""
        # Create test Excel file
        test_file = tmp_path / "test_data.xlsx"
        test_data = pd.DataFrame({
            "日期": pd.date_range("2024-01-01", periods=30),
            "交易类型": ["支出"] * 20 + ["收入"] * 10,
            "金额": np.random.uniform(100, 1000, 30),
            "分类": np.random.choice(["食品", "交通", "购物"], 30)
        })
        test_data.to_excel(test_file, index=False)
        
        # Process through pipeline
        result = pipeline.process(str(test_file))
        
        # Assertions
        assert len(result) == 30
        assert "date" in result.columns  # Column normalized
        assert "transaction_type" in result.columns
        assert result["transaction_type"].isin(["income", "expense"]).all()
        assert "category_display" in result.columns
    
    @pytest.mark.integration
    @patch('streamlit.error')
    def test_error_handling(self, mock_error, pipeline):
        """Test pipeline error handling."""
        # Test with invalid file
        with pytest.raises(DataLoadError):
            pipeline.process("nonexistent.xlsx")
        
        # Test with corrupted data
        corrupted_df = pd.DataFrame({"invalid": [1, 2, 3]})
        with pytest.raises(DataValidationError):
            pipeline.processor.process(corrupted_df)

# tests/ui/test_components.py - UI component tests
class TestUIComponents:
    """Test Streamlit UI components."""
    
    @pytest.fixture
    def mock_streamlit(self, monkeypatch):
        """Mock Streamlit functions."""
        mock_st = Mock()
        monkeypatch.setattr("streamlit.sidebar", mock_st.sidebar)
        monkeypatch.setattr("streamlit.columns", mock_st.columns)
        monkeypatch.setattr("streamlit.metric", mock_st.metric)
        return mock_st
    
    def test_kpi_card_rendering(self, mock_streamlit):
        """Test KPI card display."""
        from app.ui.components import display_kpi_cards
        
        test_df = pd.DataFrame({
            "income_amount": [1000, 2000],
            "expense_amount": [500, 700],
            "transaction_type": ["income", "expense"]
        })
        
        display_kpi_cards(test_df)
        
        # Verify streamlit calls
        assert mock_streamlit.columns.called
        assert mock_streamlit.metric.call_count >= 6  # At least 6 KPI cards
    
    def test_filter_creation(self, mock_streamlit):
        """Test filter component creation."""
        from app.ui.components import create_sidebar_filters
        
        dim_tables = {
            "years": pd.DataFrame({"year": [2023, 2024]}),
            "categories": pd.DataFrame({"category": ["Food", "Transport"]})
        }
        
        filters = create_sidebar_filters(dim_tables)
        
        # Verify filter structure
        assert "selected_years" in filters
        assert "selected_categories" in filters
        assert mock_streamlit.sidebar.multiselect.called

# tests/performance/test_benchmarks.py - Performance tests
class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_data_processing_speed(self, benchmark, sample_data):
        """Benchmark data processing speed."""
        processor = OptimizedDataProcessor(CATEGORY_MAPPINGS)
        
        result = benchmark(processor.process_dataframe, sample_data)
        
        # Verify result and performance
        assert len(result) == len(sample_data)
        assert benchmark.stats["mean"] < 0.1  # Should process in < 100ms
    
    @pytest.mark.benchmark
    def test_chart_generation_speed(self, benchmark, sample_data):
        """Benchmark chart generation speed."""
        from app.features.viz import create_monthly_trend_chart
        
        fig = benchmark(create_monthly_trend_chart, sample_data)
        
        # Verify chart created quickly
        assert fig is not None
        assert benchmark.stats["mean"] < 0.5  # Should generate in < 500ms
```

### 5.3 Development Environment

**Issue**: No standardized development setup

#### Example: Development Environment Configuration

**Before**: No development setup files

**After (Improved Code)**:

**.pre-commit-config.yaml**:
```yaml
# Pre-commit hooks for code quality
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: ['--ignore-missing-imports']
```

**pyproject.toml**:
```toml
[tool.poetry]
name = "personal-spending-dashboard"
version = "2.0.0"
description = "A comprehensive personal finance dashboard"
authors = ["Your Name <email@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
streamlit = "^1.28.0"
pandas = "^2.0.0"
plotly = "^5.15.0"
openpyxl = "^3.1.0"
pandera = "^0.17.0"
pydantic = "^2.0.0"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-benchmark = "^4.0.0"
pytest-mock = "^3.11.0"
black = "^23.0.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
mypy = "^1.3.0"
pre-commit = "^3.3.0"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-ra -q --strict-markers"
markers = [
    "integration: Integration tests",
    "benchmark: Performance benchmark tests",
    "slow: Slow running tests"
]

[tool.coverage.run]
source = ["app"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

**Makefile**:
```makefile
# Development automation
.PHONY: help install test lint format clean run docker

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	poetry install
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=app --cov-report=html

test-unit:  ## Run unit tests only
	pytest tests/unit -v

test-integration:  ## Run integration tests
	pytest tests/integration -v -m integration

lint:  ## Run linting
	flake8 app/ tests/
	mypy app/
	black --check app/ tests/
	isort --check-only app/ tests/

format:  ## Format code
	black app/ tests/
	isort app/ tests/

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf dist/ build/ *.egg-info

run:  ## Run the dashboard
	streamlit run app/dashboard.py

run-dev:  ## Run in development mode
	STREAMLIT_THEME_BASE="light" \
	STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
	LOG_LEVEL=DEBUG \
	streamlit run app/dashboard.py --server.runOnSave true

docker-build:  ## Build Docker image
	docker build -t spending-dashboard:latest .

docker-run:  ## Run Docker container
	docker run -p 8501:8501 -v $(PWD)/data:/app/data spending-dashboard:latest

quality-check:  ## Run all quality checks
	@echo "Running quality checks..."
	@make lint
	@make test
	@echo "All checks passed!"
```

**Dockerfile**:
```dockerfile
# Multi-stage Docker build for production
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==1.5.1

# Copy project files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Production stage
FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
WORKDIR /app
COPY app/ ./app/
COPY data/ ./data/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
```

**.github/workflows/ci.yml**:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: 1.5.1
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pypoetry
        key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      run: poetry install
    
    - name: Run linting
      run: |
        poetry run flake8 app/ tests/
        poetry run black --check app/ tests/
        poetry run isort --check-only app/ tests/
        poetry run mypy app/
    
    - name: Run tests
      run: poetry run pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Bandit security scan
      uses: gaurav-nelson/bandit-action@v1
      with:
        path: "app/"
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check

  build:
    needs: [quality, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t spending-dashboard:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push spending-dashboard:${{ github.sha }}
```

---

## 📋 Quick Wins (Can be done immediately)

Here are the updated quick wins with code examples:

### 1. Fix branch typo
```bash
git branch -m feture/saving-rate-visaul feature/saving-rate-visual
git push origin --delete feture/saving-rate-visaul
git push origin feature/saving-rate-visual
```

### 2. Add `.env.example`
```bash
# .env.example
# Application settings
DEBUG=false
LOG_LEVEL=INFO

# Chart configuration
CHART_HEIGHT=400
DEFAULT_TOP_CATEGORIES=10

# Data settings
MAX_FILE_SIZE_MB=50

# Performance
ENABLE_CACHING=true
CACHE_TTL=3600

# Feature flags
ENABLE_ADVANCED_CHARTS=true
ENABLE_EXPORT_FEATURES=true
```

### 3. Extract constants
```python
# Before
if date_range == 0:
    date_range = 1

# After
from app.core.config import MIN_DATE_RANGE_DAYS

if date_range == 0:
    date_range = MIN_DATE_RANGE_DAYS
```

### 4. Add logging
```python
# Before
print(f"Warning: Could not convert column {col}")

# After
import logging
logger = logging.getLogger(__name__)

logger.warning(f"Could not convert column '{col}' to {dtype}")
```

### 5. Improve docstrings
```python
# Before
def filter_data_by_selections(df, **filters):
    """Filter DataFrame based on user selections."""
    pass

# After
def filter_data_by_selections(
    df: pd.DataFrame,
    filters: FilterSelections
) -> pd.DataFrame:
    """
    Filter DataFrame based on user selections.
    
    Args:
        df: Input DataFrame with transaction data
        filters: Dictionary containing filter selections with keys:
            - selected_years: List of years to include
            - selected_categories: List of categories to include
            - selected_accounts: List of accounts to include
    
    Returns:
        Filtered DataFrame containing only matching transactions
    
    Raises:
        ValueError: If required columns are missing from DataFrame
    
    Example:
        >>> filters = {"selected_years": [2024], "selected_categories": ["Food"]}
        >>> filtered = filter_data_by_selections(df, filters)
    """
    pass
```

These code examples demonstrate concrete improvements for each priority level, showing exactly how to transform the current code into more maintainable, efficient, and well-structured solutions following KISS, YAGNI, and SOLID principles.