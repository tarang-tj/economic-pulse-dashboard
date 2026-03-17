# 📈 Economic Pulse Dashboard

An end-to-end Python data pipeline that fetches, cleans, and visualizes key US and Washington State economic indicators using the FRED API — built to demonstrate real-world ETL, pandas data wrangling, statistical analysis, and interactive dashboarding.


**Live demo:** *https://economic-pulse-dashboard-bwxtaias7g4h7ctqpcntbg.streamlit.app/*

---

## What It Does

| Stage | What happens |
|-------|-------------|
| **Extract** | Pulls 7 macroeconomic time series from the FRED REST API with local 12-hour caching |
| **Transform** | Cleans missing values, casts types, computes YoY changes and rolling statistics with pandas |
| **Analyze** | Detects short-term trends via linear regression, builds correlation matrix, flags recession periods |
| **Load** | Renders an interactive Streamlit dashboard with Plotly charts |

---

## Indicators Tracked

| Series | Description |
|--------|-------------|
| `UNRATE` | US Civilian Unemployment Rate |
| `WAUR` | Washington State Unemployment Rate |
| `JTSJOL` | US Job Openings (JOLTS) |
| `PAYEMS` | Total Nonfarm Payrolls |
| `CPIAUCSL` | Consumer Price Index (Inflation) |
| `FEDFUNDS` | Federal Funds Rate |
| `GDPC1` | Real GDP |

---

## Tech Stack

- **Python 3.11+**
- **pandas** — data cleaning, resampling, rolling statistics
- **requests** — FRED API integration
- **scipy** — linear regression for trend detection
- **plotly** — interactive time series charts and heatmaps
- **Streamlit** — dashboard framework

---

## Project Structure

```
economic-pulse-dashboard/
├── app.py              # Streamlit dashboard (entry point)
├── pipeline.py         # ETL: fetch → validate → clean → cache
├── analysis.py         # Derived metrics: trends, correlations, stats
├── config.py           # Series definitions and constants
├── requirements.txt
├── .env.example        # API key template
└── .gitignore
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/economic-pulse-dashboard.git
cd economic-pulse-dashboard
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free FRED API key
1. Go to [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account and request an API key

### 5. Set up your environment
```bash
cp .env.example .env
# Edit .env and paste your FRED_API_KEY
```

### 6. Run the dashboard
```bash
streamlit run app.py
```

---

## Key Design Decisions

**Caching layer** — API responses are cached as local JSON for 12 hours (`pipeline.py`). This avoids hammering the FRED API during development and makes the app responsive.

**Trend detection** — Uses `scipy.stats.linregress` over a 6-month rolling window. A trend is only flagged if the regression is statistically significant (p < 0.10) and explains at least 30% of variance (R² ≥ 0.3). This avoids noisy false signals.

**YoY change logic** — For rate series (unemployment, fed funds), YoY is expressed as percentage-point change. For index/level series (CPI, GDP), it's expressed as a percentage change. This distinction matters for correct interpretation.

**Recession shading** — NBER-dated recessions are overlaid as translucent bands on all time series charts, making cyclical context immediately visible.

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add `FRED_API_KEY` as a **Secret** in the app settings
5. Deploy — your app gets a public URL instantly

---

## Extending the Project

Ideas for future enhancements:
- Add state-level comparisons beyond Washington (California, Texas, etc.)
- Pull BLS industry employment breakdowns
- Add a forecasting tab using `statsmodels` ARIMA
- Email/Slack alerts when an indicator crosses a threshold
- Deploy as a scheduled pipeline with Prefect or Airflow

---

## Data Source

All data sourced from the [Federal Reserve Bank of St. Louis (FRED)](https://fred.stlouisfed.org/). Free for personal and commercial use.

---

*Built by [Tarang (TJ) Jammalamadaka](https://tarang-tj.github.io) · [LinkedIn](https://linkedin.com/in/tarang-tj) · [GitHub](https://github.com/tarang-tj)*
