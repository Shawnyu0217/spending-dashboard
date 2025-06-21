# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based personal spending dashboard that analyzes financial data from Excel files. The application processes Chinese-language financial data and provides KPI metrics, interactive charts, and data filtering capabilities.

## Key Commands

### Running the Application
```bash
# Main entry point
streamlit run app/dashboard.py

# Alternative entry point
python run_dashboard.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_kpis.py
pytest tests/test_loader.py

# Run with verbose output
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - streamlit>=1.28.0 (Web framework)
# - pandas>=2.0.0 (Data processing)
# - plotly>=5.15.0 (Interactive charts)
# - openpyxl>=3.1.0 (Excel file reading)
# - pandera>=0.17.0 (Data validation)
# - pytest>=7.4.0 (Testing framework)
# - black, isort, flake8 (Code quality tools)
```

## Architecture

### Modular Structure
- **app/data/**: Data loading and preprocessing pipeline
  - `loader.py`: Excel file reading with multi-sheet support
  - `schema.py`: Column normalization and data validation
  - `preprocess.py`: Data transformation and filtering
  - `pipeline.py`: Pipeline orchestrator for data transformations
  - `dimension_builder.py`: Creates dimension tables for filters and analytics
  - `filters.py`: Data filtering utilities
  - `transformers/`: Transformer classes for modular data processing
    - `base.py`: Base classes and protocols for transformers
    - `date_transformer.py`: Date parsing and formatting
    - `financial_transformer.py`: Financial data processing
    - `category_transformer.py`: Category mapping and normalization
- **app/features/**: Business logic layer
  - `kpis.py`: Financial KPI calculations
  - `viz.py`: Chart creation using Plotly
- **app/ui/**: Streamlit UI components
  - `components.py`: Reusable UI elements and layout
- **app/config.py**: Configuration constants and mappings
- **data/**: Sample data files for testing and development
- **requirements.txt**: Python package dependencies

### Data Flow
1. Excel files loaded via `loader.py` with automatic sheet combination
2. Column names normalized from Chinese to English via `schema.py`
3. Data processed through `pipeline.py` using modular transformers:
   - `DateTransformer`: Date parsing and temporal feature extraction
   - `FinancialTransformer`: Amount processing and currency handling
   - `CategoryTransformer`: Category mapping and normalization
4. Dimension tables created via `dimension_builder.py` for filters
5. Data filtered using `filters.py` based on user selections
6. KPIs calculated in `kpis.py`
7. Charts generated in `viz.py`
8. UI assembled in `dashboard.py` using `components.py`

### Key Design Patterns
- **Pipeline Pattern**: Modular data processing using `PreprocessingPipeline` with pluggable transformers
- **Transformer Pattern**: Each data transformation step implemented as a separate `BaseTransformer` class
- **Protocol-based Design**: `DataTransformer` protocol ensures consistent transformer interfaces
- **Caching**: Streamlit `@st.cache_data` decorator used for data loading
- **Separation of Concerns**: Clear separation between data, business logic, and UI
- **Configuration-driven**: Chinese-to-English mappings centralized in `config.py`
- **Error Handling**: Comprehensive error handling with `TransformationError` and logging
- **Dimension Modeling**: Separate dimension tables created for efficient filtering and analytics

### Data Processing
- Supports Chinese financial app exports (随手记)
- Handles multiple transaction types: 收入 (Income), 支出 (Expense), 转账 (Transfer)
- Automatic currency symbol removal and data type conversion
- Category mapping from Chinese to English for better UX

### Testing Strategy
- Unit tests for data loading (`test_loader.py`)
- KPI calculation tests (`test_kpis.py`)
- Focus on edge cases and data validation