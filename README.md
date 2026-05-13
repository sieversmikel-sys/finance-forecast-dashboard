# Maschinenbauer Finance Forecast Dashboard

An interactive rolling 12-month forecast dashboard for a mid-size machinery manufacturer, demonstrating how finance teams can combine time-series modeling with scenario analysis. Built as a self-contained showcase: synthetic data generation, Prophet-based forecasting, rolling backtesting, and a Streamlit frontend — all in one repo.

> **Live demo:** *(add your Streamlit Community Cloud URL here after deployment)*

---

## Screenshot

![Dashboard screenshot](docs/screenshot.png)

*(Replace with an actual screenshot after deployment)*

---

## Features

- **Rolling 12-month forecast** – Facebook Prophet trained on a 36-month sliding window
- **Backtesting** – 12 consecutive 1-month-ahead forecasts (Jan–Dec 2024), MAPE per product line
- **Scenario simulation** – Sidebar sliders for order intake shock (−30 % → +20 %) and cost pressure (±15 %), with three chart lines: Base Case | Optimistic | Pessimistic
- **EBITDA delta KPIs** – Instant impact of each scenario on 12-month EBITDA
- **Auto-commentary** – Plain-text narrative generated from slider values and model output

---

## Built with

| Layer | Technology |
|-------|-----------|
| Data generation | Python · NumPy |
| Forecasting | [Facebook Prophet](https://facebook.github.io/prophet/) 1.3 |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Charts | [Plotly](https://plotly.com/python/) |
| Data wrangling | pandas |

---

## Project structure

```
forecast/
├── app.py                  ← Streamlit entry point
├── requirements.txt        ← Python dependencies
├── data/
│   ├── raw_data.csv        ← Synthetic monthly data (Jan 2021 – Dec 2024)
│   ├── forecast.csv        ← Prophet output (generated)
│   └── backtest_results.csv← Rolling backtest output (generated)
└── src/
    ├── generate_data.py    ← Synthetic data generator (seed 42, reproducible)
    ├── forecast_model.py   ← Prophet training & 12-month forecast
    └── backtest.py         ← Rolling backtesting routine
```

---

## Local setup

```bash
# 1 – Install dependencies
pip install -r forecast/requirements.txt

# 2 – Generate data & run models (first time only)
python forecast/src/generate_data.py
python forecast/src/forecast_model.py
python forecast/src/backtest.py

# 3 – Launch dashboard
streamlit run forecast/app.py --server.port 8502
```

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, set **Main file path** to `forecast/app.py`
4. Click **Deploy** — Streamlit Cloud auto-installs `forecast/requirements.txt`

> **Note:** `forecast.csv` and `backtest_results.csv` are not tracked in git (see `.gitignore`). Add a startup script or pre-generate and commit them if you want the full dashboard without the model-run step on first deploy.

---

## Data model

Synthetic data simulates three product lines over 48 months:

| Product line | Monthly revenue base | Trend p.a. | Notes |
|---|---|---|---|
| Standardmaschinen | 3,500 TEUR | +4.5 % | Strong Q4 peak |
| Sonderanlagen | 2,100 TEUR | +6.5 % | Long order lead time |
| Service | 1,200 TEUR | +8.0 % | Recurring, low seasonality |

COVID dampening factors applied to Q1–Q3 2021. Random seed fixed at 42 for reproducibility.

---

*Finance Showcase Project · May 2026*
