# 💰 Personal Spending Dashboard

A comprehensive Streamlit-based dashboard for analyzing personal spending data from Excel files. Built following a modular architecture with clean separation of data processing, KPI calculations, and visualization components.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or download the files)
   ```bash
   git clone <repository-url>
   cd spending_dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run run_dashboard.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📊 Features

### Core Functionality
- **📁 File Upload**: Upload your own Excel spending data or use the sample file
- **🏷️ Smart Categorization**: Automatic mapping of Chinese categories to English
- **📈 KPI Metrics**: Key financial metrics with trend analysis
- **📊 Interactive Charts**: Multiple visualization types for different insights
- **🔍 Advanced Filtering**: Filter by date, category, account, and family members
- **💾 Export Options**: Download filtered data and summary statistics

### Dashboard Components

#### KPI Cards
- Total Income & Expenses
- Net Savings & Savings Rate
- Largest Expense Category
- Average Daily Spending

#### Interactive Charts
- **Monthly Trends**: Line chart showing income, expenses, and net savings over time
- **Category Analysis**: Bar chart of top expense categories
- **Savings Rate Gauge**: Visual indicator of your savings performance
- **Expense Distribution**: Pie chart of spending by category type
- **Account Comparison**: Compare performance across different accounts
- **Cumulative Balance**: Track your running balance over time
- **Daily Spending Heatmap**: Identify spending patterns by day/week

### Data Processing Features
- **Multi-sheet Excel Support**: Automatically combines data from multiple sheets
- **Column Normalization**: Converts Chinese column names to English
- **Data Type Conversion**: Proper handling of dates, currencies, and categories
- **Missing Data Handling**: Graceful handling of incomplete data
- **Currency Symbol Removal**: Automatic cleaning of currency formatting

## 📁 Project Structure

```
spending_dashboard/
├── app/
│   ├── data/                 # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py         # Excel file reading and normalization
│   │   ├── schema.py         # Data schema and validation
│   │   └── preprocess.py     # Data transformation and enrichment
│   ├── features/             # Business logic
│   │   ├── kpis.py          # KPI calculations
│   │   └── viz.py           # Chart creation functions
│   ├── ui/                  # User interface components
│   │   └── components.py    # Reusable Streamlit components
│   ├── dashboard.py         # Main application entry point
│   └── config.py           # Configuration and constants
├── data/                    # Sample data files
│   └── *.xlsx              # Excel files
├── tests/                   # Unit tests
│   ├── test_loader.py      # Data loading tests
│   └── test_kpis.py        # KPI calculation tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📝 Data Format

The dashboard expects Excel files with the following columns (Chinese or English):

### Required Columns
- **日期/Date**: Transaction date
- **交易类型/Transaction Type**: Income (收入), Expense (支出), or Transfer (转账)
- **金额/Amount**: Transaction amount
- **分类/Category**: Expense/income category

### Optional Columns
- **子分类/Subcategory**: More detailed categorization
- **账户/Account**: Account name (cash, bank, etc.)
- **备注/Notes**: Transaction notes
- **成员/Member**: Family member who made the transaction

### Supported Transaction Types
- **支出/Expense**: Money spent
- **收入/Income**: Money received
- **转账/Transfer**: Money moved between accounts

## 🎯 Usage Examples

### Basic Usage
1. Start the dashboard: `streamlit run app/dashboard.py`
2. Upload your Excel file using the sidebar
3. Use filters to focus on specific time periods or categories
4. Analyze your spending patterns using the charts and KPIs

### Advanced Filtering
- **Time Period**: Select specific years and months
- **Categories**: Focus on particular spending categories
- **Accounts**: Compare different accounts (cash, bank, credit cards)
- **Family Members**: Analyze spending by family member

### Data Export
- Download filtered transaction data as CSV
- Export summary statistics for external analysis
- Use the detailed data table for record-level inspection

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_kpis.py

# Run with verbose output
pytest tests/ -v
```

### Test Coverage
- Data loading and normalization
- KPI calculations and edge cases
- Column mapping and validation
- Error handling scenarios

## 🔧 Configuration

### Currency and Formatting
Edit `app/config.py` to customize:
- Currency symbols and formatting
- Category mappings (Chinese to English)
- Color schemes for charts
- Date formats

### Adding New Categories
To add new category mappings, update the `CATEGORY_MAPPINGS` dictionary in `config.py`:

```python
CATEGORY_MAPPINGS = {
    "新分类": "New Category",
    # ... existing mappings
}
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app/dashboard.py
```

### Production Deployment
The app can be deployed to various platforms:

- **Streamlit Cloud**: Connect your GitHub repository
- **Heroku**: Use the provided `requirements.txt`
- **Docker**: Create a Dockerfile with Streamlit base image

## 📊 Sample Data

A sample Excel file is included in the `data/` directory to demonstrate the dashboard functionality. The sample includes:
- Multiple months of transaction data
- Various expense categories
- Income transactions
- Different account types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code (optional)
black app/ tests/
isort app/ tests/
```

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**"Module not found" errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

**"File not found" errors**
- Verify Excel file format (`.xlsx` or `.xls`)
- Check file permissions
- Ensure file is not corrupted

**Empty or incorrect data**
- Verify column names match expected format
- Check for empty sheets in Excel file
- Review data type conversion in logs

**Performance issues**
- Large files (>10MB) may take time to process
- Consider filtering data to smaller date ranges
- Check available system memory

### Getting Help

1. Check the error messages in the Streamlit interface
2. Enable "Show detailed error information" for debugging
3. Review the test files for usage examples
4. Check the configuration in `app/config.py`

## 🔮 Future Enhancements

Planned features for future releases:
- Budget tracking and alerts
- Recurring transaction detection
- Multi-currency support
- Mobile-responsive design
- Data synchronization with financial APIs
- Advanced analytics and forecasting

---
*Built with ❤️ using Streamlit, Pandas, and Plotly*