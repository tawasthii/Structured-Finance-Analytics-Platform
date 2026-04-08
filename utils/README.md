# Fixed Income From Scratch

A ground-up implementation of structured finance analytics — curve construction, MBS cash flow modeling, prepayment analysis, spread calculations, and more — built entirely from first principles.

## Project Structure

```
fixed_income_from_scratch/
├── README.md
├── utils/                              # Core analytical modules
│   ├── __init__.py
│   ├── curves.py                       # Yield curve bootstrapping & interpolation
│   ├── cashflows.py                    # MBS & bond cash flow projection engines
│   ├── prepayment.py                   # Prepayment models & convention conversions
│   └── spreads.py                      # Spread calculations (nominal, Z-spread, swap)
├── notebooks/                          # Interactive analysis notebooks
│   ├── 01_Curve_Construction.ipynb     # Treasury & swap curve bootstrapping
│   └── 02_MBS_CashFlows_Static_Analysis.ipynb  # MBS cash flows & spread analysis
└── data/                               # Market data (FRED downloads, etc.)
```

## Phase 1 — Foundation (Complete)
- Treasury zero curve bootstrapping with verified round-trip consistency
- Swap curve bootstrapping from deposit rates and par swap rates
- Three interpolation methods: linear, cubic spline, monotone convex
- Forward rate extraction (discrete and instantaneous)
- Curve manipulation: parallel, twist, butterfly, key-rate shifts
- MBS pass-through cash flow projection engine
- All prepayment conventions: CPR, PSA, HEP, ABS, MHP, PPC
- Default/loss modeling with CDR, severity, recovery lag
- Spread hierarchy: nominal spread, Z-spread, swap spread
- Spread duration and comprehensive spread reports

## Upcoming Phases
- **Phase 2**: Multi-factor prepayment model, CMO waterfall engine, IO/PO strips
- **Phase 3**: BGM 3-factor model, Monte Carlo simulation, OAS with all risk measures
- **Phase 4**: Scenario analysis, attribution, hedging, batch processing

## Requirements
```
numpy
pandas
scipy
matplotlib
jupyter
```

## Getting Started
```bash
cd notebooks/
jupyter notebook 01_Curve_Construction.ipynb
```
