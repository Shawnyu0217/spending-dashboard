3 – Build the First Working Dashboard (MVP)
The tasks below assume you have already finished Section 1 – Project Setup and Section 2 – Data Preparation.

3.1 Project Structure & Code Architecture
graphql
Copy
Edit
spending_dashboard/
├── app/
│   ├── data/                 # load / clean / transform
│   │   ├── __init__.py
│   │   ├── loader.py         # read_excel → pandas
│   │   ├── schema.py         # column names, dtypes, helpers
│   │   └── preprocess.py     # tidy-up, mappings, caches
│   ├── features/
│   │   ├── kpis.py           # reusable functions that return numbers
│   │   └── viz.py            # reusable functions that return figures
│   ├── ui/
│   │   └── components.py     # Streamlit components (sidebar, cards…)
│   ├── dashboard.py          # ⬅️ entry-point:  `streamlit run app/dashboard.py`
│   └── config.py             # constants & settings
├── data/                     # sample file lives here for dev
│   └── sample.xlsx
├── tests/                    # pytest unit tests
├── .gitignore
└── requirements.txt
Why this matters: separating data, features, and UI layers keeps the codebase small-but-clean, so adding new charts later is a single-file change.

3.2 Data Ingestion (app/data/loader.py)
Step	Detail	Tips
1	Read the uploaded Excel file (fall back to data/sample.xlsx when none is provided)	pd.read_excel(file, sheet_name=None) returns a dict of DataFrames.
2	Normalise column names to English snake-case	e.g. 交易类型 -> type, 日期 -> datetime, etc. Helps autocomplete & static typing.
3	Coerce datetimes pd.to_datetime, amounts to float, categories to category dtype	speeds later grouping.
4	Persist result in st.session_state["df_raw"] so re-uploads aren’t needed every rerun	wrap heavy parsing with @st.cache_data.

3.3 Minimal Pre-Processing (app/data/preprocess.py)
Combine sheets

python
Copy
Edit
df = pd.concat([df_expense.assign(io="out"), df_income.assign(io="in")])
Derived columns

year, month, day, ym (Period)

net_amount = amount * (+1 | -1)

Look-ups / mappings

Map Chinese category values to readable English display names (dictionary kept in config.py).

Add “top-level category” vs “sub-category” columns for quick roll-ups.

Return both df (tidy) and small dim tables for dropdown filters.

3.4 Core UI Skeleton (app/dashboard.py)
Section	Widget / Output	Data source
Sidebar	– File uploader
– Year / month multiselect
– Category multiselect	dim tables from 3.3
Header KPI cards	• Total Income
• Total Expense
• Net Savings
• Largest Expense Category	features.kpis.total_income(df_filtered) etc.
Main charts	1. Line chart: cumulative income, expense, net (x = date)
2. Bar chart: expense by top-10 categories (sortable)
3. Monthly heatmap (optional but easy with Plotly)	features.viz.*
Details	Expandable Ag-Grid or st.dataframe with the filtered records	show running balance column.

Keep it simple: one page, few charts — aim for <200 lines total.

3.5 Testing & Data-Quality Guardrails
Unit tests: pytest tests/

Loader doesn’t break when a new column appears.

Pre-process returns no nulls in year / month / amount.

Schema validation (optional): add pandera checks inside schema.py.

3.6 Git & Formatting
Add a pre-commit hook: black, isort, flake8.

Include a README with “How to run” & a screenshot.

Store large XLSX files with git lfs or ignore them (/data/*.xlsx).

3.7 Milestone Completion Checklist
✅ Item	Evidence
Project tree scaffolded	folders & empty __init__.py files committed
dashboard.py runs without errors on sample file	screenshot in README
Filters react and KPIs update	quick Loom / GIF or screenshot
Charts show sensible numbers	cross-check against Excel pivots
Tests pass	pytest -q returns 0

4 – What Comes Next (after MVP)
Sketch now; implement later

Budgets / Targets – user sets monthly budget per category.

Recurring transaction detection – flag subscription-type spend.

Currency support – convert CNY → AUD if needed via openexchangerates.

Authentication – optional password when you decide to deploy.

Dockerfile & GitHub Actions – one-click rebuild in the future.