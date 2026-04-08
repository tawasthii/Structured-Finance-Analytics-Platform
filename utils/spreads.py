"""
Spread Calculations
====================

This module implements the hierarchy of spread measures used in 
fixed income analysis, from simplest to most sophisticated:

1. Nominal Spread (Interpolated Spread):
   Yield - Treasury yield at matching WAL
   Simple but misleading: ignores term structure shape, option cost, 
   and cash flow timing.

2. Z-Spread (Zero-Volatility Spread / Cash Flow Spread):
   Constant spread to the zero curve that discounts all projected 
   cash flows to the market price. Better than nominal because it 
   uses the entire term structure, but still assumes deterministic 
   cash flows (one prepayment scenario).
   
   Solving: find s such that:
   Price = sum( CF_i × exp(-(r(t_i) + s) × t_i) )
   
   This is a root-finding problem (Brent's method).

3. Swap Spread:
   Same as Z-spread but computed against the swap zero curve instead 
   of Treasuries. Important because swaps are the hedging instrument 
   for most institutional portfolios.

4. OAS (Option-Adjusted Spread):
   Spread to the zero curve across ALL Monte Carlo paths that equates 
   average PV to market price. This properly accounts for the 
   prepayment option. OAS > Z-spread never happens for a prepayable 
   bond (option cost is always non-negative).
   
   OAS is implemented in a later module (requires Monte Carlo).

The relationship: Z-Spread = OAS + Option Cost
"""

import numpy as np
from scipy.optimize import brentq
from typing import Optional
from .curves import ZeroCurve, pv_cashflows


def nominal_spread(bond_yield: float, curve: ZeroCurve, 
                    wal: float, curve_type: str = 'treasury') -> float:
    """
    Nominal (interpolated) spread = bond yield - benchmark yield at WAL.
    
    This is the simplest spread measure. Widely quoted for MBS as 
    "spread to the interpolated Treasury curve at average life."
    
    Parameters:
        bond_yield: Bond yield (semi-annual BEY)
        curve:      ZeroCurve (Treasury or Swap)
        wal:        Weighted Average Life in years
        curve_type: Label for reporting
    
    Returns:
        Spread in decimal (0.01 = 100 bps)
    
    Limitations:
        - Uses a single point on the curve (ignores term structure shape)
        - Assumes all principal returned at WAL (ignores cash flow timing)
        - Does not account for prepayment optionality
        - WAL itself depends on prepayment assumption (circular dependency)
    """
    # Get par rate at WAL point (convert from continuous to semi-annual)
    benchmark_rate_cont = float(curve.interpolate(np.array([wal]))[0])
    benchmark_rate_sa = 2.0 * (np.exp(benchmark_rate_cont / 2.0) - 1.0)
    
    return bond_yield - benchmark_rate_sa


def z_spread(price: float, cashflow_times: np.ndarray, 
              cashflows: np.ndarray, curve: ZeroCurve,
              face: float = 100.0,
              initial_guess: float = 0.005) -> float:
    """
    Z-Spread (Zero-Volatility Spread / Cash Flow Spread).
    
    Find the constant spread s added to each zero rate such that:
        Price = sum( CF_i × exp(-(r(t_i) + s) × t_i) )
    
    This is economically meaningful: it's the additional compensation 
    over the risk-free curve for credit risk, liquidity, and the 
    prepayment option (for MBS).
    
    Parameters:
        price:           Clean price per $100 face (e.g., 98.5)
        cashflow_times:  Array of cash flow times in years
        cashflows:       Array of cash flow amounts (dollar)
        curve:           ZeroCurve (Treasury or Swap)
        face:            Face value corresponding to the price convention
        initial_guess:   Starting spread for solver
    
    Returns:
        Z-spread in decimal (continuous compounding)
    
    Note: For MBS, the cash flows are projected under a specific 
    prepayment assumption. Different prepayment speeds give different 
    Z-spreads for the same bond — this is why Z-spread alone is 
    insufficient for securities with embedded options.
    """
    target_pv = price / 100.0 * face
    
    # Get zero rates at all cash flow times
    zero_rates = curve.interpolate(cashflow_times)
    
    def pv_diff(spread):
        dfs = np.exp(-(zero_rates + spread) * cashflow_times)
        return np.sum(cashflows * dfs) - target_pv
    
    try:
        spread = brentq(pv_diff, -0.05, 0.20, xtol=1e-8)
        return spread
    except ValueError:
        # Widen search range if initial fails
        try:
            spread = brentq(pv_diff, -0.10, 0.50, xtol=1e-8)
            return spread
        except ValueError:
            return np.nan


def z_spread_from_mbs(price: float, cf_result, curve: ZeroCurve) -> float:
    """
    Convenience function: compute Z-spread from an MBS CashFlowResult.
    
    Parameters:
        price:     Market price per $100 face
        cf_result: CashFlowResult from project_mbs_cashflows
        curve:     ZeroCurve for discounting
    
    Returns:
        Z-spread in decimal (continuous)
    """
    flows = cf_result.flows
    
    # Convert monthly periods to years
    times = flows['month'].values / 12.0
    cashflows = flows['total_cashflow'].values
    
    # Remove zero cash flow periods
    mask = cashflows > 0
    times = times[mask]
    cashflows = cashflows[mask]
    
    face = cf_result.pool.current_balance
    
    return z_spread(price, times, cashflows, curve, face=face)


def swap_spread(price: float, cashflow_times: np.ndarray,
                cashflows: np.ndarray, treasury_curve: ZeroCurve,
                swap_curve: ZeroCurve, face: float = 100.0) -> dict:
    """
    Compute both Treasury Z-spread and Swap Z-spread, plus the 
    swap spread (difference between the two benchmark curves).
    
    The swap spread component tells you how much of the total spread 
    is compensation for the Treasury-swap basis vs. security-specific risk.
    
    Returns:
        Dict with 'z_spread_tsy', 'z_spread_swap', 'swap_basis'
    """
    z_tsy = z_spread(price, cashflow_times, cashflows, treasury_curve, face)
    z_swap = z_spread(price, cashflow_times, cashflows, swap_curve, face)
    
    return {
        'z_spread_treasury': z_tsy,
        'z_spread_swap': z_swap,
        'swap_basis': z_tsy - z_swap if not (np.isnan(z_tsy) or np.isnan(z_swap)) else np.nan
    }


def spread_duration(price: float, cashflow_times: np.ndarray,
                     cashflows: np.ndarray, curve: ZeroCurve,
                     face: float = 100.0, bump_bps: float = 1.0) -> float:
    """
    Spread duration: sensitivity of price to a 1bp change in spread.
    
    SD = -(1/P) × dP/ds ≈ -(1/P) × (P(s+Δ) - P(s-Δ)) / (2Δ)
    
    This is the relevant risk measure for credit/spread positions 
    as opposed to rate duration.
    
    Returns:
        Spread duration in years
    """
    current_spread = z_spread(price, cashflow_times, cashflows, curve, face)
    if np.isnan(current_spread):
        return np.nan
    
    bump = bump_bps / 10000.0
    zero_rates = curve.interpolate(cashflow_times)
    
    pv_up = np.sum(cashflows * np.exp(-(zero_rates + current_spread + bump) * cashflow_times))
    pv_down = np.sum(cashflows * np.exp(-(zero_rates + current_spread - bump) * cashflow_times))
    
    target_pv = price / 100.0 * face
    
    return -(pv_up - pv_down) / (2 * bump) / target_pv


def spread_table(price: float, cashflow_times: np.ndarray,
                  cashflows: np.ndarray, 
                  treasury_curve: ZeroCurve,
                  swap_curve: ZeroCurve = None,
                  wal: float = None,
                  bond_yield: float = None,
                  face: float = 100.0) -> dict:
    """
    Comprehensive spread report for a security.
    
    Returns all available spread measures in a single dict.
    """
    results = {}
    
    # Z-spread to Treasury
    z_tsy = z_spread(price, cashflow_times, cashflows, treasury_curve, face)
    results['z_spread_treasury_bps'] = z_tsy * 10000 if not np.isnan(z_tsy) else None
    
    # Z-spread to Swap
    if swap_curve is not None:
        z_swp = z_spread(price, cashflow_times, cashflows, swap_curve, face)
        results['z_spread_swap_bps'] = z_swp * 10000 if not np.isnan(z_swp) else None
        if not (np.isnan(z_tsy) or np.isnan(z_swp)):
            results['swap_basis_bps'] = (z_tsy - z_swp) * 10000
    
    # Nominal spread
    if wal is not None and bond_yield is not None:
        nom_tsy = nominal_spread(bond_yield, treasury_curve, wal)
        results['nominal_spread_tsy_bps'] = nom_tsy * 10000
        
        if swap_curve is not None:
            nom_swp = nominal_spread(bond_yield, swap_curve, wal)
            results['nominal_spread_swap_bps'] = nom_swp * 10000
    
    # Spread duration
    sd = spread_duration(price, cashflow_times, cashflows, treasury_curve, face)
    results['spread_duration'] = sd if not np.isnan(sd) else None
    
    return results
