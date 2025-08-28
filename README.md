# Regime-Momentum Asset Allocation (RMAA)

This repository provides a Python implementation of the **Regime-Momentum Asset Allocation (RMAA)** strategy.  
RMAA is a **two-layer dynamic asset allocation framework**:  

1. **Macro regime detection** determines the stock/bond allocation.  
2. **Momentum scoring** within each asset class determines ETF weights.  

---

## 📊 Strategy Overview

### 1. Regime Detection
- **Indicators**
- OECD CLI (Composite Leading Indicator, growth proxy)  
- VIX (Volatility Index, risk proxy)  
- **Method**
- Rolling 36-month distribution to classify High / Medium / Low  
- Growth × Volatility → **9 regimes in total**  

### 2. Asset Allocation (Inter-asset)
- Optimize stock vs bond weights per regime to maximize Sharpe ratio  
- Constraint: Stock weight + Bond weight = 100%  
- Includes transaction costs and variance penalty  

### 3. Intra-Asset Momentum
- ETF momentum score = cumulative return from t-12 to t-1 (skip last month)  
- Negative scores truncated to zero (long-only allocation)  
- Normalized scores determine ETF weights  

### 4. Simulation & Reporting
- Split into **In-sample (2008–2019)** vs **Out-of-sample (2020–present)**  
- Performance metrics: CAGR, Annual Volatility, Sharpe Ratio  
- Outputs:
- Cumulative return plots  
- Optimal regime weights (stock/bond)  
- Final ETF allocation  
- DV01 (bond duration & rate sensitivity)  
- Results exported to Excel: `regime_allocation_report_YYYYMMDD.xlsx`  

---

## 📂 Project Structure
```

regime-momentum-allocation/
├─ regime\_strategy\_fixed.py              # main implementation
├─ docs/
│  └─ Regime-Momentum-Allocation\_Strategy\_20250430.pptx  # strategy slides
├─ requirements.txt
├─ .gitignore
└─ README.md

````

---

## ⚙️ Installation & Usage

### 1) Install dependencies
```bash
git clone https://github.com/<YOUR_ID>/regime-momentum-allocation.git
cd regime-momentum-allocation
pip install -r requirements.txt
````

### 2) Run the strategy

```bash
python regime_strategy_fixed.py
```

### 3) Outputs

* **Console**:

* In-sample & Out-of-sample performance metrics
* Current regime classification
* Optimal regime-specific weights
* Final ETF allocations & DV01 table
* **Desktop**:

* Excel report with performance, weights, DV01, and allocation tables

---

## 🧰 Requirements

* Python 3.10+
* pandas, numpy, scipy, matplotlib
* yfinance, pandas-datareader
* requests, beautifulsoup4, openpyxl

---

## 📈 Roadmap

* [ ] Add alternative macro indicators (yield curve, PMI, FCI)
* [ ] Volatility targeting (risk scaling)
* [ ] Multi-asset extension (commodities, gold, REITs)
* [ ] Interactive dashboard (Streamlit/Plotly)