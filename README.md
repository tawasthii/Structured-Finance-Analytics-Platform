# Structured Finance Analytics Platform

A full-stack fixed income and structured products analytics engine built from scratch, covering yield curve construction, MBS valuation, prepayment modeling, Monte Carlo simulation, and OAS-based risk analytics.

This project replicates the architecture used in institutional systems like Bloomberg PORT and Yield Book.

---

## 🚀 Project Overview

This platform builds a complete computational pipeline for pricing and risk analysis of mortgage-backed securities (MBS) and structured products:

Market Data → Curve Construction → Rate Simulation → Prepayment Modeling → Cash Flows → Discounting → OAS → Risk → Hedging → Portfolio Analytics

The key insight: **OAS is not standalone — it requires every upstream module to function.**

---

## 🧠 Key Features

### 🔹 Yield Curve Construction
- Bootstrapped Treasury & Swap curves
- Monotone convex interpolation (industry standard)
- Forward rate extraction
- Machine precision accuracy (≈ 0 bps error)

---

### 🔹 MBS Cash Flow Engine
- Monthly cash flow projection:
  - Scheduled interest & principal
  - Prepayments (CPR/SMM/PSA)
  - Defaults & loss modeling
- WAL sensitivity to prepayment speeds (5Y – 18Y range)

---

### 🔹 Multi-Factor Prepayment Model
- 4-component framework:
  - Refinancing (S-curve)
  - Turnover
  - Burnout (critical for seasoned MBS)
  - Seasonality & aging
- Achieved **41% reduction in RMSE** after calibration

---

### 🔹 CMO Structuring
- Sequential tranches with Z-bonds
- PAC / Companion tranche design
- IO/PO stripping with duration behavior:
  - IO → Negative duration
  - PO → Long duration

---

### 🔹 Volatility Modeling
- Black-76 & Bachelier frameworks
- SABR volatility surface calibration
- Excellent fit: RMSE ≈ 0.048%

---

### 🔹 Monte Carlo Rate Simulation
- 3-factor Hull-White model:
  - Level, slope, curvature factors
- Variance reduction:
  - Antithetic sampling
  - Moment matching
- 512 paths × 360 months simulated efficiently

---

### 🔹 OAS & Risk Analytics (Core)
- Full Monte Carlo OAS calculation
- Spread decomposition:
  - OAS
  - ZOAS
  - Option Cost

**Key Results:**
- OAS: ~118 bps  
- Option Cost: ~36 bps  
- OA Duration: ~4.5  
- Convexity: **-187 (negative convexity)**  

---

### 🔹 Scenario Analysis & Hedging
- Rate shock scenarios (±100bp, ±200bp)
- Volatility & prepayment sensitivity
- Treasury-based DV01 hedging

**Results:**
- Hedged risk reduction: **71%**
- P&L asymmetry: **2x downside vs upside**

---

### 🔹 Portfolio Analytics
- $20M multi-security MBS portfolio
- Risk metrics:
  - DV01: ~$9,384
  - WAL: ~7.2 years
  - CPR: ~9%

---

## 📊 Key Insights

- MBS exhibit **negative convexity** → upside capped, downside amplified  
- Majority of rate exposure lies around **10Y tenor (~60%)**  
- Prepayment behavior (burnout) is the **dominant driver of valuation**  
- OAS properly isolates **true spread vs embedded option cost**

---

## 🛠️ Tech Stack

- Python (NumPy, Pandas, SciPy)
- Monte Carlo Simulation
- Fixed Income Modeling
- Quantitative Risk Analytics

---

## 📁 Project Structure
Project Fixed Income/

├── utils/ # Core modules (curves, cashflows, OAS, etc.)
├── 01_Curve_Construction.ipynb
├── 02_MBS_Cashflows.ipynb
├── 03_Prepayment_Model.ipynb
├── 04_CMO_Structuring.ipynb
├── 05_Volatility_Surface.ipynb
├── 06_Monte_Carlo.ipynb
├── 07_OAS_Analysis.ipynb
├── 08_Scenario_Hedging.ipynb
├── 09_Portfolio_Analytics.ipynb
