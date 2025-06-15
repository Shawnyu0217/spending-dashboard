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

## Architecture

### Modular Structure
- **app/data/**: Data loading and preprocessing pipeline
  - `loader.py`: Excel file reading with multi-sheet support
  - `schema.py`: Column normalization and data validation
  - `preprocess.py`: Data transformation and filtering
- **app/features/**: Business logic layer
  - `kpis.py`: Financial KPI calculations
  - `viz.py`: Chart creation using Plotly
- **app/ui/**: Streamlit UI components
  - `components.py`: Reusable UI elements and layout
- **app/config.py**: Configuration constants and mappings

### Data Flow
1. Excel files loaded via `loader.py` with automatic sheet combination
2. Column names normalized from Chinese to English via `schema.py`
3. Data preprocessed and filtered in `preprocess.py`
4. KPIs calculated in `kpis.py`
5. Charts generated in `viz.py`
6. UI assembled in `dashboard.py` using `components.py`

### Key Design Patterns
- **Caching**: Streamlit `@st.cache_data` decorator used for data loading
- **Separation of Concerns**: Clear separation between data, business logic, and UI
- **Configuration-driven**: Chinese-to-English mappings centralized in `config.py`
- **Error Handling**: Graceful handling of missing data and malformed files

### Data Processing
- Supports Chinese financial app exports (随手记)
- Handles multiple transaction types: 收入 (Income), 支出 (Expense), 转账 (Transfer)
- Automatic currency symbol removal and data type conversion
- Category mapping from Chinese to English for better UX

### Testing Strategy
- Unit tests for data loading (`test_loader.py`)
- KPI calculation tests (`test_kpis.py`)
- Focus on edge cases and data validation