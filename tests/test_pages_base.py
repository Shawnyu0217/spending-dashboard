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

class TestOverviewPage:
    """Test the overview page specifically."""
    
    @pytest.fixture
    def overview_page(self):
        return OverviewPage()
    
    @pytest.fixture  
    def sample_transaction_data(self):
        """Create realistic transaction data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'amount': [100 + i * 10 for i in range(100)],
            'transaction_type': ['expense'] * 80 + ['income'] * 20,
            'category': ['Food'] * 40 + ['Transport'] * 30 + ['Shopping'] * 30,
            'account': ['Bank'] * 100,
            'member': ['User'] * 100
        })
    
    def test_overview_page_properties(self, overview_page):
        """Test overview page basic properties."""
        assert overview_page.name == "Overview"
        assert overview_page.icon == "📊"
        assert overview_page.key == "overview"
        assert overview_page.get_min_data_points() == 0
    
    def test_overview_page_should_display(self, overview_page, sample_transaction_data):
        """Test overview page display logic."""
        assert overview_page.should_display(sample_transaction_data) is True
        
        # Test with empty data
        empty_df = pd.DataFrame()
        assert overview_page.should_display(empty_df) is False 