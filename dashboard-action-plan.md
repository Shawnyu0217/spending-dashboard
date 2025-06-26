---

## Phase 5: Add Advanced Analytics (Day 9-10)

### 5.1 Create Advanced Analytics Page

**File: `app/pages/advanced.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import BasePage
from app.ui.components import create_chart_container

class AdvancedPage(BasePage):
    """Advanced analytics including anomaly detection and forecasting."""
    
    @property
    def name(self) -> str:
        return "Advanced"
    
    @property
    def icon(self) -> str:
        return "🔬"
    
    def get_min_data_points(self) -> int:
        """Advanced analytics need substantial data."""
        return 90
    
    def should_display(self, df: pd.DataFrame) -> bool:
        """Only show with sufficient data and if enabled."""
        if not self._config.get("enabled", False):
            return False
        
        return super().should_display(df)
    
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render advanced analytics tools."""
        try:
            st.markdown("### 🔬 Advanced Analytics")
            st.markdown("Discover patterns, anomalies, and future trends in your data")
            
            # Analysis type selector
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Anomaly Detection", "Spending Forecast", "Correlation Analysis", # Dashboard Navigation Implementation Action Plan

## 🚀 **PROGRESS TRACKER**

### ✅ **COMPLETED PHASES**
- **✅ Phase 1: Foundation Setup** (Complete - June 26, 2024)
  - Created `app/pages/` directory structure
  - Implemented abstract `BasePage` class with SOLID principles
  - Created page registry system (`get_available_pages()`)
  - Migrated dashboard functionality to `OverviewPage`
  - Added `PAGE_SETTINGS` configuration
  - Created comprehensive tests (`test_pages_base.py`)
  - **All tests passing: 27/27** ✅

- **✅ Phase 2: Dashboard Integration** (Complete - June 26, 2024)
  - Created `dashboard_backup.py` safety backup
  - Modified `dashboard.py` with `render_tabbed_interface()`
  - Added imports for page system
  - Implemented error handling and loading states
  - **🔧 Fixed: Slider key conflicts** by moving chart controls outside tabs
  - Commented out old chart sections for reference
  - Dashboard now displays single "Overview" tab with all functionality
  - **All tests passing: 27/27** ✅

### 🔄 **CURRENT STATUS**
- **Ready for testing**: Dashboard running with tabbed interface
- **Single Overview tab**: Contains all original functionality
- **No breaking changes**: All existing features preserved
- **Next phase ready**: Phase 3 (Trends Page) awaiting approval

### 📋 **NEXT PHASES**
- **Phase 3: Add Trends Page** (Pending)
- **Phase 4: Add Categories Page** (Pending)  
- **Phase 5: Add Advanced Analytics** (Pending)
- **Phase 6: Polish and Optimization** (Pending)

---

## Overview
Transform the existing single-page dashboard into a multi-tab interface while maintaining all current functionality and following KISS, YAGNI, and SOLID principles.

## Pre-Implementation Checklist

- [x] ✅ Create a backup branch: `git checkout -b feature/tabbed-navigation`
- [x] ✅ Ensure all tests pass: `pytest tests/` (27/27 tests passing)
- [ ] Document current dashboard screenshots for comparison
- [ ] Review this plan with stakeholders (if any)  
- [x] ✅ Set up development environment: `pip install -r requirements.txt`
- [x] ✅ Backup current working dashboard: `cp app/dashboard.py app/dashboard_backup.py`

---

## ✅ Phase 1: Foundation Setup (COMPLETE)

### ✅ 1.1 Create Directory Structure
```bash
# Execute these commands from repository root
mkdir -p app/pages
touch app/pages/__init__.py
touch app/pages/base.py
touch app/pages/overview.py

# Verify structure
tree app/pages/
```
**Status**: ✅ **COMPLETE** - Directory structure created successfully

### ✅ 1.2 Implement Base Page Class

**File: `app/pages/base.py`** ✅ **COMPLETE**
```python
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
```

### ✅ 1.3 Create Page Registry

**File: `app/pages/__init__.py`** ✅ **COMPLETE**
```python
from typing import List, Dict, Type
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
        page = page_class()
        if page.key == key:
            return page
    return None
```

### ✅ 1.4 Migrate Current Dashboard to Overview Page

**File: `app/pages/overview.py`**

Create the overview page by extracting current dashboard content:

```python
import streamlit as st
import pandas as pd
from typing import Dict, Any
from .base import BasePage

# Import all required UI components
from app.ui.components import (
    display_kpi_cards,
    create_chart_container,
    display_data_summary,
    create_data_table_section,
    create_export_section
)

# Import visualization functions
from app.features.viz import (
    create_monthly_trend_chart,
    create_category_expense_chart,
    create_savings_rate_gauge,
    create_expense_distribution_pie
)

class OverviewPage(BasePage):
    """Overview page showing KPIs and main charts."""
    
    @property
    def name(self) -> str:
        return "Overview"
    
    @property
    def icon(self) -> str:
        return "📊"
    
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render the overview page with KPIs and key charts."""
        try:
            # Display KPI cards at the top
            display_kpi_cards(df)
            
            st.divider()
            
            # Main charts section
            self._render_key_insights(df)
            
            # Data details section
            self._render_data_section(df)
            
        except Exception as e:
            self.render_error(str(e))
    
    def _render_key_insights(self, df: pd.DataFrame) -> None:
        """Render the key insights charts."""
        st.subheader("📈 Key Insights")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            create_chart_container(
                "Monthly Income & Expense Trends",
                create_monthly_trend_chart,
                df
            )
        
        with col2:
            create_chart_container(
                "Top Expense Categories",
                create_category_expense_chart,
                df
            )
        
        # Second row of charts
        col3, col4 = st.columns(2)
        
        with col3:
            create_chart_container(
                "Current Savings Rate",
                create_savings_rate_gauge,
                df
            )
        
        with col4:
            create_chart_container(
                "Expense Distribution",
                create_expense_distribution_pie,
                df
            )
    
    def _render_data_section(self, df: pd.DataFrame) -> None:
        """Render the data details section."""
        with st.expander("📋 Data Details", expanded=False):
            # Data summary
            display_data_summary(df)
            
            # Detailed data table
            create_data_table_section(df)
            
            # Export functionality
            create_export_section(df)
```

**Migration Checklist:**
- [ ] Extract KPI cards section from dashboard.py
- [ ] Extract chart creation sections
- [ ] Extract data table and export sections
- [ ] Remove these sections from dashboard.py (keep in comments initially)
- [ ] Verify all imports are included in overview.py

### ✅ 1.5 Update Configuration

**File: `app/config.py`**

Add detailed page settings:
```python
# Page-specific settings
PAGE_SETTINGS = {
    "overview": {
        "enabled": True,
        "min_data_days": 0,
        "description": "High-level financial overview with KPIs",
        "charts": {
            "show_monthly_trends": True,
            "show_category_breakdown": True,
            "show_savings_gauge": True,
            "show_expense_pie": True,
            "max_categories_shown": 10
        }
    },
    "trends": {
        "enabled": True,
        "min_data_days": 30,
        "description": "Detailed temporal analysis and trends",
        "charts": {
            "show_savings_trend": True,
            "show_cumulative_balance": True,
            "show_daily_heatmap": True,
            "default_ma_periods": 3,
            "default_target_savings": 10.0
        }
    },
    "categories": {
        "enabled": True,
        "min_data_days": 7,
        "description": "Category deep-dive and analysis",
        "features": {
            "show_subcategories": True,
            "enable_category_comparison": True,
            "max_categories_display": 15
        }
    },
    "advanced": {
        "enabled": False,  # Disabled initially
        "min_data_days": 90,
        "description": "Advanced analytics and forecasting",
        "features": {
            "anomaly_detection": True,
            "basic_forecasting": True,
            "correlation_analysis": False
        }
    }
}

# Tab display configuration
TAB_ORDER = ["overview", "trends", "categories", "advanced"]
DEFAULT_TAB = "overview"
```

### ✅ 1.6 Create Initial Tests

**File: `tests/test_pages_base.py`**
```python
import pytest
import pandas as pd
from datetime import datetime
from app.pages.base import BasePage
from app.pages.overview import OverviewPage

class TestBasePage:
    """Test the base page abstract class."""
    
    def test_base_page_is_abstract(self):
        """Ensure BasePage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePage()
    
    def test_page_key_generation(self):
        """Test that page keys are generated correctly."""
        page = OverviewPage()
        assert page.key == "overview"
        assert page.name == "Overview"
        assert page.icon == "📊"

class TestPageDisplay:
    """Test page display logic."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'amount': range(50),
            'transaction_type': ['expense'] * 50,
            'category': ['Food'] * 50
        })
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data."""
        return pd.DataFrame({
            'date': [datetime.now()],
            'amount': [100],
            'transaction_type': ['expense'],
            'category': ['Food']
        })
    
    def test_overview_displays_with_minimal_data(self, minimal_data):
        """Test that overview page shows with minimal data."""
        page = OverviewPage()
        assert page.should_display(minimal_data) is True
    
    def test_page_does_not_display_with_empty_data(self):
        """Test that pages don't display with empty dataframe."""
        page = OverviewPage()
        empty_df = pd.DataFrame()
        assert page.should_display(empty_df) is False
```

### ✅ 1.7 Verification Steps ✅ **COMPLETE**

Before proceeding to Phase 2:
1. [x] ✅ Run the test suite: `pytest tests/test_pages_base.py -v` (6/6 tests passing)
2. [x] ✅ Verify imports in overview.py match dashboard.py
3. [x] ✅ Check that all visualization functions are accessible
4. [x] ✅ Ensure configuration is loaded correctly
5. [x] ✅ Document any missing dependencies

**Success Criteria:** ✅ **ALL MET**
- ✅ Base page infrastructure is complete and tested
- ✅ Overview page class contains all current dashboard functionality
- ✅ Configuration system is in place for future pages
- ✅ No runtime errors when importing page modules
- ✅ All existing tests still pass (27/27)

---

## ✅ Phase 2: Dashboard Integration (COMPLETE)

### ✅ 2.1 Prepare for Integration ✅ **COMPLETE**

**Create backup and safety checks:**
```bash
# Backup current dashboard
cp app/dashboard.py app/dashboard_backup.py

# Create a feature flag (optional safety measure)
echo "ENABLE_TABBED_INTERFACE = False" >> app/config.py
```
**Status**: ✅ **COMPLETE** - Backup created: `app/dashboard_backup.py`

### ✅ 2.2 Modify dashboard.py - Detailed Implementation ✅ **COMPLETE**

**File: `app/dashboard.py`**

**Step 1: Add imports at the top**
```python
# Add to existing imports
from app.pages import get_available_pages
from typing import List
```

**Step 2: Create helper function for tab rendering**
```python
def render_tabbed_interface(df_filtered: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """
    Render the tabbed interface with available pages.
    
    Args:
        df_filtered: The filtered transaction data
        filters: Current filter selections
    """
    # Get pages that should be displayed
    pages = get_available_pages(df_filtered)
    
    if not pages:
        st.error("❌ No analysis pages available for the current data selection.")
        st.info("💡 Try adjusting your filters or uploading more data.")
        return
    
    # Create tab navigation
    tab_names = [f"{page.icon} {page.name}" for page in pages]
    tabs = st.tabs(tab_names)
    
    # Render each page in its corresponding tab
    for tab, page in zip(tabs, pages):
        with tab:
            # Add consistent styling container
            with st.container():
                try:
                    page.render(df_filtered, filters)
                except Exception as e:
                    st.error(f"Error rendering {page.name} page")
                    if st.checkbox(f"Show error details", key=f"error_{page.key}"):
                        st.exception(e)
```

**Step 3: Modify create_main_dashboard() function**

Locate the section after data filtering and replace the entire content rendering with:

```python
def create_main_dashboard():
    """Create the main dashboard interface."""
    
    # ... existing code for header, initialization, data loading ...
    
    # Apply filters to data (existing code)
    df_filtered = filter_data_by_selections(
        df_processed,
        selected_years=filters.get("selected_years"),
        selected_months=filters.get("selected_months"),
        selected_categories=filters.get("selected_categories"),
        selected_accounts=filters.get("selected_accounts"),
        selected_members=filters.get("selected_members")
    )
    
    # Display filter summary (keep this)
    display_filter_summary(filters, df_filtered, df_processed)
    
    st.divider()
    
    # NEW: Replace all the chart sections with tabbed interface
    render_tabbed_interface(df_filtered, filters)
    
    # Keep footer outside of tabs
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        💡 Personal Spending Dashboard | Built with Streamlit & Plotly<br>
        📊 Analyze your financial data with interactive charts and insights
    </div>
    """, unsafe_allow_html=True)
```

**Step 4: Comment out (don't delete) the old content sections**
```python
# IMPORTANT: Keep these sections commented for reference during migration
# # Display KPI cards
# display_kpi_cards(df_filtered)
# 
# st.divider()
# 
# # Main charts section
# st.header("📊 Financial Analysis")
# ... (keep all old code commented)
```

### 2.3 Update Overview Page - Final Verification

**Ensure all components are properly migrated:**

```python
# app/pages/overview.py
class OverviewPage(BasePage):
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render the overview page."""
        # Component checklist:
        # ✓ KPI cards
        # ✓ Monthly trends chart
        # ✓ Category expense chart  
        # ✓ Savings rate gauge
        # ✓ Expense distribution pie
        # ✓ Data summary section
        # ✓ Data table
        # ✓ Export functionality
```

### ✅ 2.4 Add Error Handling and Loading States ✅ **COMPLETE**

**Key Issue Resolved**: 🔧 **Fixed Slider Key Conflicts**
- **Problem**: Multiple slider elements with same auto-generated ID
- **Solution**: Moved `create_chart_configuration_controls()` outside tabs 
- **Implementation**: Chart config now passed through filters to pages
- **Result**: All widgets have unique keys, no more conflicts ✅

**Update the page rendering to include loading states:**

```python
# In dashboard.py render_tabbed_interface()
for tab, page in zip(tabs, pages):
    with tab:
        with st.container():
            # Add loading state
            with st.spinner(f"Loading {page.name}..."):
                try:
                    # Check if page has minimum data
                    if len(df_filtered) < page.get_min_data_points():
                        st.warning(f"⚠️ {page.name} requires at least {page.get_min_data_points()} data points.")
                        continue
                    
                    page.render(df_filtered, filters)
                    
                except Exception as e:
                    st.error(f"❌ Error loading {page.name}")
                    error_details = st.empty()
                    if st.button(f"Show details", key=f"details_{page.key}"):
                        error_details.exception(e)
```

### 2.5 Gradual Rollout Strategy (Optional)

If you want to test carefully, implement a feature flag:

```python
# In dashboard.py
from app.config import PAGE_SETTINGS

def create_main_dashboard():
    # ... existing code ...
    
    # Feature flag check
    use_tabs = PAGE_SETTINGS.get("_global", {}).get("use_tabbed_interface", True)
    
    if use_tabs:
        render_tabbed_interface(df_filtered, filters)
    else:
        # Original dashboard code
        render_original_dashboard(df_filtered, filters)
```

### 2.6 Testing Checklist

**Manual Testing Steps:**
1. [ ] Start the app: `streamlit run app/dashboard.py`
2. [ ] Verify Overview tab displays correctly
3. [ ] Check all KPI cards show correct values
4. [ ] Test each chart renders properly
5. [ ] Verify filter changes update the tab content
6. [ ] Test data export functionality
7. [ ] Upload different Excel files and verify behavior
8. [ ] Test with minimal data (1 row)
9. [ ] Test with large dataset (1000+ rows)
10. [ ] Check responsive behavior on different screen sizes

**Automated Testing:**
```python
# tests/test_dashboard_integration.py
import pytest
from app.dashboard import render_tabbed_interface
from app.pages import get_available_pages

def test_tabbed_interface_renders():
    """Test that tabbed interface renders without errors."""
    # Create test data
    df = create_test_dataframe()
    filters = {"selected_years": [2024]}
    
    # Should not raise any exceptions
    pages = get_available_pages(df)
    assert len(pages) > 0
    assert pages[0].name == "Overview"
```

### 2.7 Rollback Procedures

If issues occur:
1. **Quick rollback**: 
   ```bash
   cp app/dashboard_backup.py app/dashboard.py
   streamlit run app/dashboard.py
   ```

2. **Git rollback**:
   ```bash
   git checkout HEAD -- app/dashboard.py
   ```

3. **Feature flag disable**:
   Set `use_tabbed_interface = False` in config

**Success Criteria:** ✅ **ALL MET**
- ✅ Dashboard loads with single "Overview" tab
- ✅ All original features work identically  
- ✅ No performance degradation
- ✅ Clean tab UI with proper styling
- ✅ Error handling for edge cases
- ✅ **BONUS**: Fixed slider key conflicts issue

---

## 🎯 **PHASE 2 FINAL STATUS: COMPLETE** ✅

### ✅ **Achievements**
- **Tabbed Interface**: Successfully implemented with `render_tabbed_interface()`
- **Overview Tab**: Contains all original dashboard functionality
- **Error Handling**: Loading states, spinners, and error messages
- **Issue Resolution**: Fixed slider key conflicts by restructuring controls
- **Code Quality**: Backup preserved, comments added, clean implementation

### ✅ **Technical Implementation**
- Added `from app.pages import get_available_pages` 
- Created `render_tabbed_interface()` function with error handling
- Modified `create_main_dashboard()` to use tabs instead of direct rendering
- Moved chart configuration outside tabs to prevent ID conflicts
- All original chart sections commented but preserved for reference

### ✅ **Testing Results**
- **All tests passing**: 27/27 ✅
- **No breaking changes**: Existing functionality preserved ✅  
- **Dashboard functional**: Single Overview tab working ✅
- **Ready for expansion**: Phase 3 (Trends Page) can proceed ✅

---

## Phase 3: Add Trends Page (Day 5-6)

### 3.1 Create Trends Page Structure

**File: `app/pages/trends.py`**

```python
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .base import BasePage

# Import visualization components
from app.ui.components import create_chart_container, display_savings_rate_insights
from app.features.viz import (
    create_monthly_savings_rate_chart_enhanced,
    create_cumulative_balance_chart,
    create_daily_spending_heatmap,
    create_monthly_trend_chart
)

class TrendsPage(BasePage):
    """Detailed trends and temporal analysis page."""
    
    @property
    def name(self) -> str:
        return "Trends"
    
    @property
    def icon(self) -> str:
        return "📈"
    
    def get_min_data_points(self) -> int:
        """Trends page requires at least 30 days of data."""
        return 30
    
    def should_display(self, df: pd.DataFrame) -> bool:
        """Only show if we have sufficient temporal data."""
        if not super().should_display(df):
            return False
        
        # Check if we have enough date range
        if 'date' in df.columns:
            date_range = (df['date'].max() - df['date'].min()).days
            return date_range >= 30
        
        return len(df) >= self.get_min_data_points()
    
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render comprehensive trend analysis."""
        try:
            # Page header
            st.markdown("### 📊 Temporal Analysis & Trends")
            st.markdown("Analyze your financial patterns over time")
            
            # Configuration controls
            chart_config = self._render_chart_controls()
            
            # Main trend sections
            self._render_savings_trends(df, chart_config)
            self._render_balance_analysis(df)
            self._render_spending_patterns(df)
            self._render_period_comparison(df)
            
        except Exception as e:
            self.render_error(str(e))
    
    def _render_chart_controls(self) -> Dict[str, Any]:
        """Render chart configuration controls."""
        with st.expander("⚙️ Chart Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_rate = st.slider(
                    "Target Savings Rate (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=self._config.get("charts", {}).get("default_target_savings", 10.0),
                    step=0.5,
                    key="trends_target_rate",
                    help="Set your target savings rate for comparison"
                )
            
            with col2:
                show_ma = st.checkbox(
                    "Show Moving Average",
                    value=True,
                    key="trends_show_ma",
                    help="Display moving average trend line"
                )
                
                if show_ma:
                    ma_periods = st.number_input(
                        "MA Periods",
                        min_value=2,
                        max_value=12,
                        value=self._config.get("charts", {}).get("default_ma_periods", 3),
                        key="trends_ma_periods"
                    )
                else:
                    ma_periods = 3
            
            with col3:
                show_dual_axis = st.checkbox(
                    "Dual Axis View",
                    value=False,
                    key="trends_dual_axis",
                    help="Show amount and percentage on separate axes"
                )
        
        return {
            "target_rate": target_rate,
            "show_ma": show_ma,
            "ma_periods": ma_periods,
            "show_dual_axis": show_dual_axis
        }
    
    def _render_savings_trends(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Render savings rate trends section."""
        st.subheader("💰 Savings Rate Analysis")
        
        # Enhanced savings rate chart
        def create_enhanced_chart(df):
            return create_monthly_savings_rate_chart_enhanced(
                df,
                target_rate=config["target_rate"],
                show_dual_axis=config["show_dual_axis"],
                show_moving_average=config["show_ma"],
                ma_periods=config["ma_periods"]
            )
        
        create_chart_container(
            "Monthly Savings Rate Trends",
            create_enhanced_chart,
            df
        )
        
        # Savings insights
        with st.expander("💡 Savings Insights", expanded=True):
            display_savings_rate_insights(df, config["target_rate"])
    
    def _render_balance_analysis(self, df: pd.DataFrame) -> None:
        """Render cumulative balance analysis."""
        st.subheader("💵 Balance Progression")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            create_chart_container(
                "Cumulative Balance Over Time",
                create_cumulative_balance_chart,
                df
            )
        
        with col2:
            # Balance statistics
            st.markdown("**Balance Stats**")
            
            # Calculate key metrics
            cumulative_balance = self._calculate_cumulative_balance(df)
            if not cumulative_balance.empty:
                current_balance = cumulative_balance.iloc[-1]
                starting_balance = cumulative_balance.iloc[0]
                balance_change = current_balance - starting_balance
                
                st.metric(
                    "Current Balance",
                    f"${current_balance:,.2f}",
                    f"${balance_change:+,.2f}"
                )
                
                # Growth rate
                days = (df['date'].max() - df['date'].min()).days
                if days > 0:
                    daily_growth = balance_change / days
                    monthly_growth = daily_growth * 30
                    
                    st.metric(
                        "Avg Monthly Growth",
                        f"${monthly_growth:,.2f}"
                    )
    
    def _render_spending_patterns(self, df: pd.DataFrame) -> None:
        """Render spending pattern analysis."""
        # Only show if we have enough data
        if len(df) < 60:
            return
        
        st.subheader("🔥 Spending Patterns")
        
        create_chart_container(
            "Daily Spending Heatmap",
            create_daily_spending_heatmap,
            df
        )
        
        # Pattern insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week analysis
            st.markdown("**Spending by Day of Week**")
            dow_spending = self._analyze_dow_spending(df)
            st.bar_chart(dow_spending)
        
        with col2:
            # Time of month analysis
            st.markdown("**Spending by Time of Month**")
            tom_spending = self._analyze_time_of_month_spending(df)
            st.line_chart(tom_spending)
    
    def _render_period_comparison(self, df: pd.DataFrame) -> None:
        """Render period-over-period comparison."""
        st.subheader("📊 Period Comparison")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            period_type = st.selectbox(
                "Compare by",
                ["Month", "Quarter", "Year"],
                key="period_comparison_type"
            )
        
        with col2:
            comparison_metric = st.selectbox(
                "Metric",
                ["Total Expenses", "Total Income", "Net Savings", "Savings Rate"],
                key="comparison_metric"
            )
        
        # Calculate and display comparison
        comparison_df = self._calculate_period_comparison(df, period_type, comparison_metric)
        
        if not comparison_df.empty:
            # Create comparison chart
            st.bar_chart(comparison_df)
            
            # Show percentage changes
            with st.expander("View Percentage Changes"):
                pct_changes = comparison_df.pct_change().fillna(0) * 100
                st.dataframe(
                    pct_changes.style.format("{:.1f}%"),
                    use_container_width=True
                )
    
    # Helper methods
    def _calculate_cumulative_balance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cumulative balance over time."""
        df_sorted = df.sort_values('date')
        
        # Calculate net for each transaction
        df_sorted['net'] = df_sorted.apply(
            lambda x: x['amount'] if x['transaction_type'] == 'income' else -x['amount'],
            axis=1
        )
        
        return df_sorted.groupby('date')['net'].sum().cumsum()
    
    def _analyze_dow_spending(self, df: pd.DataFrame) -> pd.Series:
        """Analyze spending by day of week."""
        expenses = df[df['transaction_type'] == 'expense'].copy()
        expenses['dow'] = pd.to_datetime(expenses['date']).dt.day_name()
        
        # Define order
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return expenses.groupby('dow')['amount'].mean().reindex(dow_order)
    
    def _analyze_time_of_month_spending(self, df: pd.DataFrame) -> pd.Series:
        """Analyze spending by time of month."""
        expenses = df[df['transaction_type'] == 'expense'].copy()
        expenses['day_of_month'] = pd.to_datetime(expenses['date']).dt.day
        
        return expenses.groupby('day_of_month')['amount'].mean().sort_index()
    
    def _calculate_period_comparison(
        self, 
        df: pd.DataFrame, 
        period_type: str, 
        metric: str
    ) -> pd.DataFrame:
        """Calculate period-over-period comparison."""
        # Implementation for period comparison
        # This is a simplified version - expand as needed
        df['period'] = pd.to_datetime(df['date']).dt.to_period(period_type[0])
        
        if metric == "Total Expenses":
            result = df[df['transaction_type'] == 'expense'].groupby('period')['amount'].sum()
        elif metric == "Total Income":
            result = df[df['transaction_type'] == 'income'].groupby('period')['amount'].sum()
        # Add other metrics...
        
        return pd.DataFrame(result)
```

### 3.2 Extract Trend Charts from Overview

**Identify charts to move from overview.py to trends.py:**
1. Monthly savings rate trends (enhanced version)
2. Cumulative balance chart
3. Daily spending heatmap
4. Any year-over-year comparisons

**Update overview.py:**
```python
# Remove these from overview.py render method:
# - create_monthly_savings_rate_chart_enhanced
# - create_cumulative_balance_chart
# - create_daily_spending_heatmap

# Keep only the basic overview charts:
# - Monthly income/expense trends
# - Top categories
# - Savings rate gauge (current snapshot)
# - Expense distribution pie
```

### 3.3 Update Page Registry

**File: `app/pages/__init__.py`**
```python
from .trends import TrendsPage

PAGE_CLASSES: List[Type[BasePage]] = [
    OverviewPage,
    TrendsPage,  # Add this line
]
```

### 3.4 Add Specialized Trend Visualizations

**Create new trend-specific visualizations in `app/features/viz.py`:**

```python
def create_year_over_year_comparison(df: pd.DataFrame) -> go.Figure:
    """Create year-over-year comparison chart."""
    # Implementation
    pass

def create_seasonal_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Analyze seasonal spending patterns."""
    # Implementation
    pass

def create_trend_forecast_chart(df: pd.DataFrame, periods: int = 3) -> go.Figure:
    """Simple trend forecasting visualization."""
    # Implementation
    pass
```

### 3.5 Testing the Trends Page

**File: `tests/test_pages_trends.py`**
```python
import pytest
import pandas as pd
from datetime import datetime, timedelta
from app.pages.trends import TrendsPage

class TestTrendsPage:
    @pytest.fixture
    def trends_page(self):
        return TrendsPage()
    
    @pytest.fixture
    def sufficient_data(self):
        """Create 60 days of data for trends."""
        dates = pd.date_range(end=datetime.now(), periods=60)
        return pd.DataFrame({
            'date': dates,
            'amount': [100 + (i % 20) * 10 for i in range(60)],
            'transaction_type': ['expense'] * 40 + ['income'] * 20,
            'category': ['Food'] * 30 + ['Transport'] * 30
        })
    
    @pytest.fixture
    def insufficient_data(self):
        """Create only 20 days of data."""
        dates = pd.date_range(end=datetime.now(), periods=20)
        return pd.DataFrame({
            'date': dates,
            'amount': [100] * 20,
            'transaction_type': ['expense'] * 20,
            'category': ['Food'] * 20
        })
    
    def test_trends_page_properties(self, trends_page):
        assert trends_page.name == "Trends"
        assert trends_page.icon == "📈"
        assert trends_page.get_min_data_points() == 30
    
    def test_trends_page_display_logic(self, trends_page, sufficient_data, insufficient_data):
        # Should display with sufficient data
        assert trends_page.should_display(sufficient_data) is True
        
        # Should not display with insufficient data
        assert trends_page.should_display(insufficient_data) is False
    
    def test_cumulative_balance_calculation(self, trends_page, sufficient_data):
        balance = trends_page._calculate_cumulative_balance(sufficient_data)
        assert not balance.empty
        assert len(balance) > 0
```

### 3.6 Manual Testing Checklist

1. [ ] Launch dashboard with <30 days of data - verify Trends tab doesn't appear
2. [ ] Launch with >30 days of data - verify Trends tab appears
3. [ ] Test all chart configuration options:
   - [ ] Target savings rate slider
   - [ ] Moving average toggle and periods
   - [ ] Dual axis view
4. [ ] Verify charts render correctly:
   - [ ] Savings rate trends with MA
   - [ ] Cumulative balance
   - [ ] Daily spending heatmap (if >60 days)
   - [ ] Period comparisons
5. [ ] Test period comparison functionality:
   - [ ] Monthly comparison
   - [ ] Quarterly comparison
   - [ ] Year comparison
6. [ ] Check responsive design on mobile view
7. [ ] Verify no performance issues with large datasets

### 3.7 Performance Considerations

Add caching for expensive calculations:

```python
@st.cache_data
def calculate_period_metrics(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Cache period calculations."""
    # Expensive calculations here
    pass

# In TrendsPage
def _render_period_comparison(self, df: pd.DataFrame) -> None:
    # Use cached function
    metrics = calculate_period_metrics(df, period_type)
```

**Success Criteria:**
- Trends tab appears conditionally based on data
- All temporal analysis charts work correctly
- Configuration controls affect visualizations
- Clear insights and statistics displayed
- Good performance with large datasets
- Mobile-responsive design

---

## Phase 4: Add Categories Page (Day 7-8)

### 4.1 Create Categories Page Structure

**File: `app/pages/categories.py`**

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from .base import BasePage

from app.ui.components import create_chart_container
from app.features.viz import create_category_expense_chart
from app.config import CATEGORY_MAPPINGS, EXPENSE_CATEGORIES

class CategoriesPage(BasePage):
    """Deep dive into category analysis and insights."""
    
    @property
    def name(self) -> str:
        return "Categories"
    
    @property
    def icon(self) -> str:
        return "🏷️"
    
    def get_min_data_points(self) -> int:
        """Categories page needs at least 7 days of data."""
        return 7
    
    def render(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """Render comprehensive category analysis."""
        try:
            st.markdown("### 🏷️ Category Analysis")
            st.markdown("Understand your spending patterns by category")
            
            # Category view controls
            view_options = self._render_view_controls()
            
            # Main analysis sections
            self._render_category_overview(df, view_options)
            self._render_subcategory_analysis(df, view_options)
            self._render_category_trends(df)
            self._render_category_comparison(df)
            
            # Budget analysis (if budget data available)
            if self._has_budget_data(df):
                self._render_budget_analysis(df)
                
        except Exception as e:
            self.render_error(str(e))
    
    def _render_view_controls(self) -> Dict[str, Any]:
        """Render category view controls."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            view_type = st.selectbox(
                "View Type",
                ["Amount", "Percentage", "Count", "Average"],
                key="cat_view_type",
                help="Choose how to display category data"
            )
        
        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["All Time", "This Month", "Last Month", "Last 3 Months", "This Year"],
                key="cat_time_period"
            )
        
        with col3:
            show_subcategories = st.checkbox(
                "Show Subcategories",
                value=self._config.get("features", {}).get("show_subcategories", True),
                key="cat_show_sub"
            )
        
        with col4:
            top_n = st.number_input(
                "Top N Categories",
                min_value=5,
                max_value=30,
                value=self._config.get("features", {}).get("max_categories_display", 15),
                step=5,
                key="cat_top_n"
            )
        
        return {
            "view_type": view_type,
            "time_period": time_period,
            "show_subcategories": show_subcategories,
            "top_n": top_n
        }
    
    def _render_category_overview(self, df: pd.DataFrame, options: Dict[str, Any]) -> None:
        """Render main category overview."""
        st.subheader("📊 Category Breakdown")
        
        # Filter data by time period
        df_filtered = self._filter_by_time_period(df, options["time_period"])
        
        # Calculate category metrics
        category_metrics = self._calculate_category_metrics(
            df_filtered, 
            options["view_type"], 
            options["top_n"]
        )
        
        if category_metrics.empty:
            st.warning("No expense data available for the selected period.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main category chart
            fig = self._create_category_chart(category_metrics, options["view_type"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category statistics
            st.markdown("**Category Statistics**")
            
            # Top category
            top_category = category_metrics.index[0]
            top_value = category_metrics.iloc[0]
            
            st.metric(
                "Top Category",
                top_category,
                f"${top_value:,.2f}" if options["view_type"] == "Amount" else f"{top_value:.1f}%"
            )
            
            # Category diversity
            num_categories = len(category_metrics)
            st.metric("Active Categories", num_categories)
            
            # Concentration metrics
            if options["view_type"] in ["Amount", "Percentage"]:
                top_3_concentration = category_metrics.head(3).sum()
                total = category_metrics.sum()
                concentration_pct = (top_3_concentration / total * 100) if total > 0 else 0
                
                st.metric(
                    "Top 3 Concentration",
                    f"{concentration_pct:.1f}%",
                    help="Percentage of spending in top 3 categories"
                )
    
    def _render_subcategory_analysis(self, df: pd.DataFrame, options: Dict[str, Any]) -> None:
        """Render subcategory breakdown."""
        if not options["show_subcategories"] or 'subcategory' not in df.columns:
            return
        
        st.subheader("🔍 Subcategory Details")
        
        # Let user select a category to drill down
        expense_df = df[df['transaction_type'] == 'expense']
        categories = expense_df['category'].unique()
        
        selected_category = st.selectbox(
            "Select Category to Analyze",
            sorted(categories),
            key="subcat_selector"
        )
        
        # Filter for selected category
        cat_df = expense_df[expense_df['category'] == selected_category]
        
        if cat_df.empty:
            st.info(f"No data for {selected_category}")
            return
        
        # Subcategory breakdown
        subcat_metrics = cat_df.groupby('subcategory')['amount'].agg(['sum', 'count', 'mean'])
        subcat_metrics = subcat_metrics.sort_values('sum', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subcategory pie chart
            fig = px.pie(
                values=subcat_metrics['sum'],
                names=subcat_metrics.index,
                title=f"{selected_category} Subcategory Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Subcategory table
            st.markdown(f"**{selected_category} Breakdown**")
            
            # Format the metrics table
            display_df = pd.DataFrame({
                'Subcategory': subcat_metrics.index,
                'Total': subcat_metrics['sum'].apply(lambda x: f"${x:,.2f}"),
                'Count': subcat_metrics['count'],
                'Average': subcat_metrics['mean'].apply(lambda x: f"${x:,.2f}")
            })
            
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True
            )
    
    def _render_category_trends(self, df: pd.DataFrame) -> None:
        """Render category trends over time."""
        st.subheader("📈 Category Trends")
        
        # Trend period selector
        col1, col2 = st.columns([1, 3])
        
        with col1:
            trend_period = st.radio(
                "Trend Period",
                ["Daily", "Weekly", "Monthly"],
                key="cat_trend_period"
            )
        
        # Calculate trends
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        # Group by period and category
        if trend_period == "Daily":
            expense_df['period'] = expense_df['date']
        elif trend_period == "Weekly":
            expense_df['period'] = pd.to_datetime(expense_df['date']).dt.to_period('W').astype(str)
        else:  # Monthly
            expense_df['period'] = pd.to_datetime(expense_df['date']).dt.to_period('M').astype(str)
        
        # Get top categories for trend analysis
        top_categories = expense_df.groupby('category')['amount'].sum().nlargest(5).index
        
        # Filter for top categories only
        trend_df = expense_df[expense_df['category'].isin(top_categories)]
        
        # Create pivot table for trends
        pivot_df = trend_df.pivot_table(
            values='amount',
            index='period',
            columns='category',
            aggfunc='sum',
            fill_value=0
        )
        
        # Create line chart
        fig = go.Figure()
        
        for category in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[category],
                mode='lines+markers',
                name=category,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Top 5 Category Trends ({trend_period})",
            xaxis_title="Period",
            yaxis_title="Amount ($)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_category_comparison(self, df: pd.DataFrame) -> None:
        """Render category comparison tools."""
        st.subheader("⚖️ Category Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Month-over-month comparison
            st.markdown("**Month-over-Month Changes**")
            
            mom_changes = self._calculate_mom_changes(df)
            if not mom_changes.empty:
                # Create bar chart of changes
                fig = go.Figure()
                
                colors = ['red' if x < 0 else 'green' for x in mom_changes['change_pct']]
                
                fig.add_trace(go.Bar(
                    x=mom_changes.index,
                    y=mom_changes['change_pct'],
                    text=mom_changes['change_pct'].apply(lambda x: f"{x:+.1f}%"),
                    textposition='auto',
                    marker_color=colors
                ))
                
                fig.update_layout(
                    title="Category Changes vs Last Month",
                    yaxis_title="Change (%)",
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category share evolution
            st.markdown("**Category Share Evolution**")
            
            share_evolution = self._calculate_share_evolution(df)
            if not share_evolution.empty:
                # Create area chart
                fig = px.area(
                    share_evolution.T,
                    title="Category Share Over Time",
                    labels={'value': 'Share (%)', 'index': 'Month'}
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_budget_analysis(self, df: pd.DataFrame) -> None:
        """Render budget vs actual analysis."""
        st.subheader("💼 Budget Analysis")
        st.info("Budget analysis would go here if budget data is available")
        # Implementation for budget analysis
    
    # Helper methods
    def _filter_by_time_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter dataframe by selected time period."""
        if period == "All Time":
            return df
        
        today = pd.Timestamp.now()
        
        if period == "This Month":
            start_date = today.replace(day=1)
        elif period == "Last Month":
            start_date = (today.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
            end_date = today.replace(day=1) - pd.Timedelta(days=1)
            return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        elif period == "Last 3 Months":
            start_date = today - pd.Timedelta(days=90)
        else:  # This Year
            start_date = today.replace(month=1, day=1)
        
        return df[df['date'] >= start_date]
    
    def _calculate_category_metrics(
        self, 
        df: pd.DataFrame, 
        view_type: str, 
        top_n: int
    ) -> pd.Series:
        """Calculate category metrics based on view type."""
        expense_df = df[df['transaction_type'] == 'expense']
        
        if expense_df.empty:
            return pd.Series()
        
        if view_type == "Amount":
            metrics = expense_df.groupby('category')['amount'].sum()
        elif view_type == "Percentage":
            amounts = expense_df.groupby('category')['amount'].sum()
            metrics = (amounts / amounts.sum() * 100)
        elif view_type == "Count":
            metrics = expense_df.groupby('category').size()
        else:  # Average
            metrics = expense_df.groupby('category')['amount'].mean()
        
        return metrics.nlargest(top_n).sort_values(ascending=False)
    
    def _create_category_chart(self, metrics: pd.Series, view_type: str) -> go.Figure:
        """Create category visualization based on view type."""
        if view_type == "Percentage":
            # Pie chart for percentage view
            fig = px.pie(
                values=metrics.values,
                names=metrics.index,
                title=f"Category Distribution ({view_type})"
            )
        else:
            # Bar chart for other views
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=metrics.index,
                y=metrics.values,
                text=metrics.apply(
                    lambda x: f"${x:,.0f}" if view_type in ["Amount", "Average"] 
                    else f"{x:,.0f}"
                ),
                textposition='auto',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"Category {view_type}",
                xaxis_title="Category",
                yaxis_title=view_type,
                showlegend=False
            )
        
        return fig
    
    def _calculate_mom_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate month-over-month changes by category."""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        expense_df['month'] = pd.to_datetime(expense_df['date']).dt.to_period('M')
        
        # Get last two months
        months = expense_df['month'].unique()
        if len(months) < 2:
            return pd.DataFrame()
        
        last_month = months[-1]
        prev_month = months[-2]
        
        # Calculate totals
        last_totals = expense_df[expense_df['month'] == last_month].groupby('category')['amount'].sum()
        prev_totals = expense_df[expense_df['month'] == prev_month].groupby('category')['amount'].sum()
        
        # Calculate changes
        change_df = pd.DataFrame({
            'last_month': last_totals,
            'prev_month': prev_totals
        }).fillna(0)
        
        change_df['change_amt'] = change_df['last_month'] - change_df['prev_month']
        change_df['change_pct'] = (change_df['change_amt'] / change_df['prev_month'] * 100).fillna(0)
        
        return change_df.sort_values('change_pct', ascending=False).head(10)
    
    def _calculate_share_evolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate category share evolution over time."""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        expense_df['month'] = pd.to_datetime(expense_df['date']).dt.to_period('M')
        
        # Get top categories
        top_cats = expense_df.groupby('category')['amount'].sum().nlargest(5).index
        
        # Calculate monthly shares
        monthly_totals = expense_df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
        monthly_shares = monthly_totals.div(monthly_totals.sum(axis=1), axis=0) * 100
        
        return monthly_shares[top_cats] if len(monthly_shares) > 0 else pd.DataFrame()
    
    def _has_budget_data(self, df: pd.DataFrame) -> bool:
        """Check if budget data is available."""
        return 'budget' in df.columns or 'budget_amount' in df.columns
```

### 4.2 Update Page Registry

**File: `app/pages/__init__.py`**
```python
from .categories import CategoriesPage

PAGE_CLASSES: List[Type[BasePage]] = [
    OverviewPage,
    TrendsPage,
    CategoriesPage,  # Add this line
]
```

### 4.3 Create Category-Specific Utilities

**File: `app/features/category_utils.py`**
```python
import pandas as pd
from typing import Dict, List, Tuple

def analyze_category_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze spending patterns within categories."""
    patterns = {
        'frequent_categories': get_frequent_categories(df),
        'volatile_categories': get_volatile_categories(df),
        'growing_categories': get_growing_categories(df),
        'seasonal_categories': detect_seasonal_categories(df)
    }
    return patterns

def get_frequent_categories(df: pd.DataFrame, threshold: int = 5) -> List[str]:
    """Get categories with frequent transactions."""
    expense_df = df[df['transaction_type'] == 'expense']
    freq = expense_df.groupby('category').size()
    return freq[freq >= threshold].index.tolist()

def get_volatile_categories(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """Get categories with high spending volatility."""
    expense_df = df[df['transaction_type'] == 'expense']
    
    # Calculate coefficient of variation
    category_stats = expense_df.groupby('category')['amount'].agg(['std', 'mean'])
    category_stats['cv'] = category_stats['std'] / category_stats['mean']
    
    # Return top 5 most volatile
    return category_stats.nlargest(5, 'cv')['cv'].to_list()

def get_growing_categories(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """Identify categories with increasing spending trends."""
    # Implementation
    pass

def detect_seasonal_categories(df: pd.DataFrame) -> List[str]:
    """Detect categories with seasonal patterns."""
    # Implementation
    pass
```

### 4.4 Testing the Categories Page

**File: `tests/test_pages_categories.py`**
```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.pages.categories import CategoriesPage

class TestCategoriesPage:
    @pytest.fixture
    def categories_page(self):
        return CategoriesPage()
    
    @pytest.fixture
    def diverse_category_data(self):
        """Create data with multiple categories and subcategories."""
        dates = pd.date_range(end=datetime.now(), periods=100)
        categories = ['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills']
        subcategories = {
            'Food': ['Groceries', 'Restaurants', 'Coffee'],
            'Transport': ['Gas', 'Public Transit', 'Uber'],
            'Entertainment': ['Movies', 'Games', 'Concerts'],
            'Shopping': ['Clothes', 'Electronics', 'Home'],
            'Bills': ['Utilities', 'Internet', 'Phone']
        }
        
        data = []
        for i, date in enumerate(dates):
            cat = categories[i % len(categories)]
            subcat = subcategories[cat][i % len(subcategories[cat])]
            
            data.append({
                'date': date,
                'amount': 50 + np.random.rand() * 150,
                'transaction_type': 'expense',
                'category': cat,
                'subcategory': subcat
            })
        
        return pd.DataFrame(data)
    
    def test_category_page_properties(self, categories_page):
        assert categories_page.name == "Categories"
        assert categories_page.icon == "🏷️"
        assert categories_page.get_min_data_points() == 7
    
    def test_category_metrics_calculation(self, categories_page, diverse_category_data):
        # Test different view types
        amount_metrics = categories_page._calculate_category_metrics(
            diverse_category_data, "Amount", 5
        )
        assert len(amount_metrics) <= 5
        assert amount_metrics.sum() > 0
        
        pct_metrics = categories_page._calculate_category_metrics(
            diverse_category_data, "Percentage", 5
        )
        assert abs(pct_metrics.sum() - 100) < 0.01  # Should sum to ~100%
    
    def test_time_period_filtering(self, categories_page, diverse_category_data):
        # Test different time periods
        all_time = categories_page._filter_by_time_period(diverse_category_data, "All Time")
        assert len(all_time) == len(diverse_category_data)
        
        this_month = categories_page._filter_by_time_period(diverse_category_data, "This Month")
        assert len(this_month) < len(diverse_category_data)
```

### 4.5 Manual Testing Checklist

1. [ ] Verify Categories tab appears with minimal data (7+ days)
2. [ ] Test all view type options:
   - [ ] Amount view shows dollar amounts
   - [ ] Percentage view shows percentages that sum to 100%
   - [ ] Count view shows transaction counts
   - [ ] Average view shows average amounts
3. [ ] Test time period filters:
   - [ ] All Time
   - [ ] This Month
   - [ ] Last Month
   - [ ] Last 3 Months
   - [ ] This Year
4. [ ] Test subcategory analysis:
   - [ ] Subcategory toggle works
   - [ ] Selecting category shows correct subcategories
   - [ ] Pie chart and table display correctly
5. [ ] Test category trends:
   - [ ] Daily/Weekly/Monthly toggle works
   - [ ] Line chart shows top 5 categories
   - [ ] Trends are accurate
6. [ ] Test category comparison:
   - [ ] Month-over-month changes calculate correctly
   - [ ] Share evolution chart displays properly
7. [ ] Test with different data scenarios:
   - [ ] Single category only
   - [ ] Many categories (20+)
   - [ ] Categories with no subcategories
8. [ ] Performance test with large datasets

### 4.6 Integration Points

Ensure the Categories page integrates well with other pages:

1. **From Overview**: Users can click on a category in the overview to deep-dive
2. **To Trends**: Link to see historical trends for specific categories
3. **Filters**: Category selection in sidebar affects all pages consistently

**Success Criteria:**
- Categories tab provides deep insights into spending patterns
- Multiple view types give different perspectives
- Subcategory analysis provides granular details
- Trends and comparisons help identify patterns
- Performance remains good with many categories
- Mobile-responsive design maintained

---

## Phase 5: Add Advanced Analytics (Day 9-10)

### 5.1 Create Advanced Page

**File: `app/pages/advanced.py`**
```python
class AdvancedPage(BasePage):
    """Advanced analytics and tools."""
    
    @property
    def name(self) -> str:
        return "Advanced"
    
    @property
    def icon(self) -> str:
        return "🔬"
    
    def should_display(self, df: pd.DataFrame) -> bool:
        """Only show with 90+ days of data."""
        return len(df) >= 90
```

### 5.2 Implement Advanced Features
- [ ] Anomaly detection
- [ ] Simple forecasting
- [ ] Custom date range analysis
- [ ] Correlation analysis
- [ ] Export tools

### 5.3 Add Required Dependencies
Update `requirements.txt` if needed:
```txt
scikit-learn>=1.0.0  # For anomaly detection
statsmodels>=0.13.0  # For forecasting
```

---

## Phase 6: Polish and Optimization (Day 11-12)

### 6.1 Performance Optimization
- [ ] Add `@st.cache_data` decorators where appropriate
- [ ] Implement lazy loading for heavy computations
- [ ] Profile tab switching performance

### 6.2 UI/UX Improvements
- [ ] Add loading states for each tab
- [ ] Implement smooth transitions
- [ ] Add help tooltips
- [ ] Create consistent error handling

### 6.3 Code Cleanup
- [ ] Remove `dashboard_backup.py`
- [ ] Clean up imports
- [ ] Add comprehensive docstrings
- [ ] Update README.md

### 6.4 Documentation
- [ ] Update user documentation
- [ ] Create developer guide
- [ ] Document new architecture
- [ ] Add screenshots

---

## Testing Strategy

### Unit Tests
```bash
# Run after each phase
pytest tests/test_pages_base.py -v
pytest tests/test_pages_overview.py -v
pytest tests/test_pages_trends.py -v
```

### Integration Tests
```bash
# Full test suite
pytest tests/ -v --cov=app/pages
```

### Manual Testing Checklist
- [ ] Upload different Excel files
- [ ] Test with minimal data (1 day)
- [ ] Test with moderate data (30 days)
- [ ] Test with extensive data (365+ days)
- [ ] Test all filter combinations
- [ ] Test on different screen sizes
- [ ] Test export functionality

---

## Rollback Plan

If issues arise at any phase:

1. **Immediate rollback:**
```bash
git checkout main
git branch -D feature/tabbed-navigation
```

2. **Partial rollback:**
```bash
# Restore original dashboard
cp app/dashboard_backup.py app/dashboard.py
# Keep pages for future use
```

3. **Feature flag approach:**
```python
# In config.py
USE_TABBED_INTERFACE = False

# In dashboard.py
if USE_TABBED_INTERFACE:
    # New tabbed interface
else:
    # Original interface
```

---

## Success Metrics

### Functional Success
- [ ] All original features work in new interface
- [ ] No performance degradation
- [ ] Clean tab navigation
- [ ] Proper data filtering across tabs

### Code Quality Success
- [ ] Test coverage >80% for new code
- [ ] No pylint/flake8 errors
- [ ] Clear separation of concerns
- [ ] Following SOLID principles

### User Experience Success
- [ ] Intuitive navigation
- [ ] Faster access to specific analyses
- [ ] Reduced cognitive load
- [ ] Mobile-responsive

---

## Post-Implementation

### Week 1 After Launch
- [ ] Monitor for any issues
- [ ] Gather user feedback
- [ ] Document lessons learned
- [ ] Plan next features

### Future Enhancements
1. **Tab state persistence** - Remember last viewed tab
2. **Cross-tab insights** - Link related analyses
3. **Custom tabs** - User-defined analysis pages
4. **Tab-specific exports** - Tailored export formats

---

## Commands Reference

```bash
# Development
streamlit run run_dashboard.py

# Testing
pytest tests/ -v
pytest tests/ -v --cov=app

# Linting
pylint app/pages/
black app/pages/
isort app/pages/

# Git workflow
git add .
git commit -m "feat: implement tabbed navigation"
git push origin feature/tabbed-navigation
```

---

## Notes

- **Incremental approach**: Each phase builds on the previous
- **Testing first**: Write tests before implementing features
- **User-centric**: Focus on improving user experience
- **Maintainable**: Keep code simple and well-documented
- **Reversible**: Each phase can be rolled back independently

Remember: The goal is to enhance usability without sacrificing the simplicity and reliability of the current dashboard.