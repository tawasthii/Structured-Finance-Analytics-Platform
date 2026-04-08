"""
Option-Adjusted Spread (OAS) Analysis Engine
===============================================

This is the analytical culmination of the entire project. Everything 
built in previous modules feeds into this calculation:

    Curve Construction → Term Structure → Drift Calibration
    Volatility Surface → Factor Volatilities → Path Generation
    Prepayment Model → Path-Specific CPRs → Cash Flow Projection
    Cash Flow Engine → Path-Specific Flows → Discounting → OAS

The OAS Calculation:
    For each Monte Carlo path i:
        1. Extract the path of short rates and mortgage rates
        2. Feed mortgage rates into the prepayment model → CPR vector
        3. Feed CPR vector into the cash flow engine → cash flows
        4. Discount cash flows using path-specific short rates + OAS
        5. PV_i = sum( CF_t × exp(-sum(r_s + OAS) × dt) )
    
    OAS is the spread s such that:
        (1/N) × Σ PV_i(s) = Market Price
    
    This is a root-finding problem with a Monte Carlo inner loop.

Risk Measures (all by finite differences — bump and reprice):
    - OA Duration:          ∂Price/∂rates (parallel bump)
    - OA Convexity:         ∂²Price/∂rates² (second derivative)
    - Partial Duration:     Key-rate bumps (2Y, 5Y, 10Y, 30Y)
    - Prepayment Duration:  ∂Price/∂(prepay multiplier)
    - Volatility Duration:  ∂Price/∂(vol level)
    - Current Coupon Duration: ∂Price/∂(mortgage rate)
    - ZOAS:                 Z-spread on the forward path (no vol)
"""

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .curves import ZeroCurve
from .monte_carlo import (
    RateModelParams, SimulationConfig, RatePathResult,
    simulate_rate_paths
)
from .prepayment_model import (
    PrepaymentTuning, project_prepayment_rates
)
from .cashflows import MBSPool, project_mbs_cashflows


@dataclass
class OASResult:
    """
    Complete OAS analysis result.
    
    All spreads in basis points. All durations in years.
    """
    # Core OAS
    oas_bps: float = 0.0
    zoas_bps: float = 0.0
    option_cost_bps: float = 0.0     # ZOAS - OAS
    
    # Price
    model_price: float = 0.0          # Price at computed OAS
    market_price: float = 0.0
    
    # Duration measures
    oa_duration: float = 0.0
    oa_convexity: float = 0.0
    
    # Partial (key-rate) durations
    partial_durations: Dict[str, float] = field(default_factory=dict)
    
    # Specialty durations
    prepay_duration: float = 0.0
    vol_duration: float = 0.0
    current_coupon_duration: float = 0.0
    
    # Path statistics
    num_paths: int = 0
    avg_wal: float = 0.0
    avg_cpr: float = 0.0
    price_std: float = 0.0           # Standard error of price estimate
    
    def summary_dict(self) -> dict:
        """Return summary as ordered dict for display."""
        d = {
            'OAS (bps)': self.oas_bps,
            'ZOAS (bps)': self.zoas_bps,
            'Option Cost (bps)': self.option_cost_bps,
            'OA Duration': self.oa_duration,
            'OA Convexity': self.oa_convexity,
            'Prepay Duration': self.prepay_duration,
            'Vol Duration': self.vol_duration,
            'CC Duration': self.current_coupon_duration,
            'Avg WAL': self.avg_wal,
            'Avg CPR (%)': self.avg_cpr * 100,
            'Num Paths': self.num_paths,
            'Price Std Error': self.price_std,
        }
        for k, v in self.partial_durations.items():
            d[f'KRD {k}'] = v
        return d


# =============================================================================
# Core OAS Calculation
# =============================================================================

def compute_path_prices(pool: MBSPool, paths: RatePathResult,
                         oas_spread: float = 0.0,
                         tuning: PrepaymentTuning = None,
                         dt: float = 1/12) -> np.ndarray:
    """
    Compute the present value of MBS cash flows on each Monte Carlo path.
    
    This is the inner loop of OAS calculation. For each path:
    1. Extract the mortgage rate path
    2. Run the prepayment model to get path-specific CPRs
    3. Project cash flows using those CPRs
    4. Discount using path-specific short rates + OAS spread
    
    Parameters:
        pool:       MBS pool characteristics
        paths:      Simulated rate paths
        oas_spread: Spread to add to discount rates (in decimal, not bps)
        tuning:     Prepayment model tuning
        dt:         Time step
    
    Returns:
        Array of present values, one per path (per $1 of current face)
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    
    num_paths = paths.num_paths
    num_months = min(paths.num_months, pool.wam)
    pvs = np.zeros(num_paths)
    
    for p in range(num_paths):
        # 1. Get mortgage rate path for this simulation
        mtg_path = paths.mortgage_rates[p, :num_months]
        
        # 2. Run prepayment model on this path
        prepay_result = project_prepayment_rates(
            note_rate=pool.wac,
            current_coupon_path=mtg_path,
            num_months=num_months,
            loan_age=pool.age,
            original_term=pool.original_term,
            tuning=tuning
        )
        
        # 3. Project cash flows using model-generated SMMs
        cf = project_mbs_cashflows(pool, smm_vector=prepay_result.total_smm)
        cashflows = cf.flows['total_cashflow'].values
        
        # Ensure lengths match
        n = min(len(cashflows), num_months)
        
        # 4. Discount using path short rates + OAS
        short_rates = paths.short_rates[p, :n]
        cum_disc = np.cumsum((short_rates + oas_spread) * dt)
        disc_factors = np.exp(-cum_disc)
        
        # 5. PV on this path
        pvs[p] = np.sum(cashflows[:n] * disc_factors[:n])
    
    return pvs / pool.current_balance  # Return per $1 of face


def solve_oas(pool: MBSPool, paths: RatePathResult,
               market_price: float,
               tuning: PrepaymentTuning = None) -> float:
    """
    Find the OAS that equates average model price to market price.
    
    OAS is defined as the constant spread s such that:
        (1/N) × Σ_i PV_i(s) = Market_Price / 100
    
    Parameters:
        pool:         MBS pool
        paths:        Monte Carlo rate paths
        market_price: Market price per $100 face (e.g., 101.0)
        tuning:       Prepayment tuning
    
    Returns:
        OAS in decimal (multiply by 10000 for bps)
    """
    target = market_price / 100.0  # Convert to per $1 face
    
    def price_error(spread):
        pvs = compute_path_prices(pool, paths, spread, tuning)
        return np.mean(pvs) - target
    
    try:
        oas = brentq(price_error, -0.05, 0.15, xtol=1e-6)
        return oas
    except ValueError:
        # Widen search
        try:
            oas = brentq(price_error, -0.10, 0.30, xtol=1e-5)
            return oas
        except ValueError:
            return np.nan


def compute_zoas(pool: MBSPool, curve: ZeroCurve,
                  market_price: float,
                  tuning: PrepaymentTuning = None) -> float:
    """
    Compute Zero-Volatility OAS (ZOAS).
    
    ZOAS is computed on the FORWARD rate path only — no Monte Carlo.
    It's the spread to the forward curve that prices the security 
    using the prepayment model driven by forward rates.
    
    ZOAS - OAS = Option Cost
    The option cost is always non-negative for prepayable securities.
    """
    num_months = pool.wam
    dt = 1/12
    
    # Extract forward rates from the curve
    times = np.arange(1, num_months + 1) * dt
    forward_rates = curve.instantaneous_forward(times)
    
    # Mortgage rate on the forward path
    mortgage_spread = 0.015  # Same as in the MC model
    mortgage_rates = forward_rates * 0.8 + forward_rates.mean() * 0.2 + mortgage_spread
    
    # Run prepayment model on forward path
    prepay_result = project_prepayment_rates(
        note_rate=pool.wac,
        current_coupon_path=mortgage_rates,
        num_months=num_months,
        loan_age=pool.age,
        original_term=pool.original_term,
        tuning=tuning
    )
    
    # Project cash flows
    cf = project_mbs_cashflows(pool, smm_vector=prepay_result.total_smm)
    cashflows = cf.flows['total_cashflow'].values
    n = min(len(cashflows), num_months)
    
    target = market_price / 100.0 * pool.current_balance
    
    def price_error(spread):
        cum_disc = np.cumsum((forward_rates[:n] + spread) * dt)
        disc_factors = np.exp(-cum_disc)
        return np.sum(cashflows[:n] * disc_factors) - target
    
    try:
        zoas = brentq(price_error, -0.05, 0.15, xtol=1e-6)
        return zoas
    except ValueError:
        return np.nan


# =============================================================================
# Risk Measures
# =============================================================================

def compute_oa_duration_convexity(pool: MBSPool, curve: ZeroCurve,
                                    paths: RatePathResult,
                                    market_price: float,
                                    oas: float,
                                    tuning: PrepaymentTuning = None,
                                    bump_bps: float = 25.0,
                                    params: RateModelParams = None,
                                    config: SimulationConfig = None) -> dict:
    """
    OA Duration and Convexity by bump-and-reprice.
    
    Duration = -(P_up - P_down) / (2 × Δy × P)
    Convexity = (P_up + P_down - 2×P) / (Δy² × P)
    
    We bump the ENTIRE curve, regenerate paths, and reprice at the SAME OAS.
    """
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig(num_paths=paths.num_paths, num_months=paths.num_months)
    
    bump = bump_bps / 10000.0
    
    # Base price
    base_pvs = compute_path_prices(pool, paths, oas, tuning)
    base_price = np.mean(base_pvs) * 100
    
    # Up shift
    curve_up = curve.shift('parallel', shift_bps=bump_bps)
    paths_up = simulate_rate_paths(curve_up, params, config)
    pvs_up = compute_path_prices(pool, paths_up, oas, tuning)
    price_up = np.mean(pvs_up) * 100
    
    # Down shift
    curve_down = curve.shift('parallel', shift_bps=-bump_bps)
    paths_down = simulate_rate_paths(curve_down, params, config)
    pvs_down = compute_path_prices(pool, paths_down, oas, tuning)
    price_down = np.mean(pvs_down) * 100
    
    duration = -(price_up - price_down) / (2 * bump * base_price)
    convexity = (price_up + price_down - 2 * base_price) / (bump**2 * base_price)
    
    return {
        'duration': duration,
        'convexity': convexity,
        'price_up': price_up,
        'price_down': price_down,
        'base_price': base_price
    }


def compute_partial_durations(pool: MBSPool, curve: ZeroCurve,
                                paths: RatePathResult,
                                oas: float,
                                key_rates: list = None,
                                bump_bps: float = 25.0,
                                tuning: PrepaymentTuning = None,
                                params: RateModelParams = None,
                                config: SimulationConfig = None) -> dict:
    """
    Key-rate (partial) durations.
    
    Bump each key rate individually, regenerate paths, and measure 
    the price sensitivity. The sum of partial durations ≈ OA duration.
    """
    if key_rates is None:
        key_rates = [2, 5, 10, 20, 30]
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig(num_paths=paths.num_paths, num_months=paths.num_months)
    
    bump = bump_bps / 10000.0
    base_pvs = compute_path_prices(pool, paths, oas, tuning)
    base_price = np.mean(base_pvs) * 100
    
    partial_durs = {}
    for kr in key_rates:
        curve_bump = curve.shift('key_rate', shift_bps=bump_bps, 
                                  key_rate_tenor=kr, key_rate_width=2.0)
        paths_bump = simulate_rate_paths(curve_bump, params, config)
        pvs_bump = compute_path_prices(pool, paths_bump, oas, tuning)
        price_bump = np.mean(pvs_bump) * 100
        
        partial_durs[f'{kr}Y'] = -(price_bump - base_price) / (bump * base_price)
    
    return partial_durs


def compute_prepay_duration(pool: MBSPool, paths: RatePathResult,
                              oas: float,
                              tuning: PrepaymentTuning = None,
                              shift_pct: float = 0.10) -> float:
    """
    Prepayment duration: sensitivity to a shift in prepayment speed.
    
    Multiply all CPRs by (1 + shift_pct) and (1 - shift_pct), 
    reprice at the same OAS, and compute the sensitivity.
    
    This measures model risk — how much does the price change if 
    your prepayment assumptions are wrong?
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    
    # Base
    base_pvs = compute_path_prices(pool, paths, oas, tuning)
    base_price = np.mean(base_pvs) * 100
    
    # Faster prepays
    tuning_fast = PrepaymentTuning(**{**tuning.to_dict(), 
                                       'refinancing_multiplier': tuning.refinancing_multiplier * (1 + shift_pct),
                                       'turnover_multiplier': tuning.turnover_multiplier * (1 + shift_pct)})
    pvs_fast = compute_path_prices(pool, paths, oas, tuning_fast)
    price_fast = np.mean(pvs_fast) * 100
    
    # Slower prepays
    tuning_slow = PrepaymentTuning(**{**tuning.to_dict(),
                                       'refinancing_multiplier': tuning.refinancing_multiplier * (1 - shift_pct),
                                       'turnover_multiplier': tuning.turnover_multiplier * (1 - shift_pct)})
    pvs_slow = compute_path_prices(pool, paths, oas, tuning_slow)
    price_slow = np.mean(pvs_slow) * 100
    
    return -(price_fast - price_slow) / (2 * shift_pct * base_price)


def compute_vol_duration(pool: MBSPool, curve: ZeroCurve,
                           oas: float,
                           tuning: PrepaymentTuning = None,
                           vol_shift_pct: float = 0.10,
                           params: RateModelParams = None,
                           config: SimulationConfig = None) -> float:
    """
    Volatility duration: sensitivity to a parallel shift in volatility.
    
    Increase/decrease all factor volatilities by vol_shift_pct, 
    regenerate paths, and measure price sensitivity.
    """
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig()
    
    # Base
    paths_base = simulate_rate_paths(curve, params, config)
    base_pvs = compute_path_prices(pool, paths_base, oas, tuning)
    base_price = np.mean(base_pvs) * 100
    
    # Higher vol
    params_up = RateModelParams(
        sigma1=params.sigma1 * (1 + vol_shift_pct),
        sigma2=params.sigma2 * (1 + vol_shift_pct),
        sigma3=params.sigma3 * (1 + vol_shift_pct),
        a1=params.a1, a2=params.a2, a3=params.a3,
        rho12=params.rho12, rho13=params.rho13, rho23=params.rho23,
        mortgage_spread=params.mortgage_spread,
        mortgage_vol_mult=params.mortgage_vol_mult
    )
    paths_up = simulate_rate_paths(curve, params_up, config)
    pvs_up = compute_path_prices(pool, paths_up, oas, tuning)
    price_up = np.mean(pvs_up) * 100
    
    return -(price_up - base_price) / (vol_shift_pct * base_price)

def compute_cc_duration(pool: MBSPool, curve: ZeroCurve,
                          paths: RatePathResult,
                          oas: float,
                          tuning: PrepaymentTuning = None,
                          shift_bps: float = 25.0,
                          params: RateModelParams = None,
                          config: SimulationConfig = None) -> float:
    """
    Current coupon duration: sensitivity to a shift in the mortgage rate
    independent of Treasury rates.
    
    This captures basis risk — the mortgage-Treasury spread can move
    independently from the curve itself.
    """
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig()
    
    bump = shift_bps / 10000.0
    
    # Base price
    base_pvs = compute_path_prices(pool, paths, oas, tuning)
    base_price = np.mean(base_pvs) * 100
    
    # Bump mortgage spread up (wider spread = higher mortgage rates = less refi)
    params_up = RateModelParams(
        a1=params.a1, a2=params.a2, a3=params.a3,
        sigma1=params.sigma1, sigma2=params.sigma2, sigma3=params.sigma3,
        rho12=params.rho12, rho13=params.rho13, rho23=params.rho23,
        mortgage_spread=params.mortgage_spread + bump,
        mortgage_vol_mult=params.mortgage_vol_mult
    )
    paths_up = simulate_rate_paths(curve, params_up, config)
    pvs_up = compute_path_prices(pool, paths_up, oas, tuning)
    price_up = np.mean(pvs_up) * 100
    
    # Bump mortgage spread down
    params_down = RateModelParams(
        a1=params.a1, a2=params.a2, a3=params.a3,
        sigma1=params.sigma1, sigma2=params.sigma2, sigma3=params.sigma3,
        rho12=params.rho12, rho13=params.rho13, rho23=params.rho23,
        mortgage_spread=params.mortgage_spread - bump,
        mortgage_vol_mult=params.mortgage_vol_mult
    )
    paths_down = simulate_rate_paths(curve, params_down, config)
    pvs_down = compute_path_prices(pool, paths_down, oas, tuning)
    price_down = np.mean(pvs_down) * 100
    
    return -(price_up - price_down) / (2 * bump * base_price)


# =============================================================================
# Full OAS Analysis
# =============================================================================

def run_full_oas_analysis(pool: MBSPool, curve: ZeroCurve,
                            market_price: float,
                            tuning: PrepaymentTuning = None,
                            params: RateModelParams = None,
                            config: SimulationConfig = None,
                            compute_risk: bool = True) -> OASResult:
    """
    Run complete OAS analysis with all risk measures.
    
    This is the main entry point — it orchestrates:
    1. Monte Carlo path generation
    2. OAS solving
    3. ZOAS computation
    4. All duration/convexity measures
    
    Parameters:
        pool:          MBS pool characteristics
        curve:         Initial zero curve
        market_price:  Market price per $100 face
        tuning:        Prepayment model tuning
        params:        Rate model parameters
        config:        Simulation configuration
        compute_risk:  If True, compute all risk measures (slower)
    
    Returns:
        OASResult with full analytics
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig(num_paths=256, num_months=min(pool.wam, 360))
    
    result = OASResult()
    result.market_price = market_price
    result.num_paths = config.num_paths
    
    # 1. Generate paths
    paths = simulate_rate_paths(curve, params, config)
    
    # 2. Solve for OAS
    oas = solve_oas(pool, paths, market_price, tuning)
    result.oas_bps = oas * 10000 if not np.isnan(oas) else np.nan
    
    # 3. Compute ZOAS
    zoas = compute_zoas(pool, curve, market_price, tuning)
    result.zoas_bps = zoas * 10000 if not np.isnan(zoas) else np.nan
    
    # 4. Option cost
    if not (np.isnan(result.zoas_bps) or np.isnan(result.oas_bps)):
        result.option_cost_bps = result.zoas_bps - result.oas_bps
    
    # 5. Model price at OAS (verification)
    if not np.isnan(oas):
        pvs = compute_path_prices(pool, paths, oas, tuning)
        result.model_price = np.mean(pvs) * 100
        result.price_std = np.std(pvs) * 100 / np.sqrt(len(pvs))
        
        # Path statistics
        wals = []
        cprs = []
        for p in range(min(paths.num_paths, 100)):  # Sample paths for stats
            mtg_path = paths.mortgage_rates[p, :pool.wam]
            pr = project_prepayment_rates(pool.wac, mtg_path, pool.wam,
                                           pool.age, pool.original_term, tuning=tuning)
            cprs.append(np.mean(pr.total_cpr))
            cf = project_mbs_cashflows(pool, smm_vector=pr.total_smm)
            wals.append(cf.weighted_avg_life)
        result.avg_wal = np.mean(wals)
        result.avg_cpr = np.mean(cprs)
    
    # 6. Risk measures
    if compute_risk and not np.isnan(oas):
        # OA Duration & Convexity
        dur_conv = compute_oa_duration_convexity(
            pool, curve, paths, market_price, oas, tuning, 
            bump_bps=25, params=params, config=config
        )
        result.oa_duration = dur_conv['duration']
        result.oa_convexity = dur_conv['convexity']
        
        # Partial durations
        result.partial_durations = compute_partial_durations(
            pool, curve, paths, oas, [2, 5, 10, 30], 25, tuning, params, config
        )
        
        # Prepayment duration
        result.prepay_duration = compute_prepay_duration(
            pool, paths, oas, tuning, shift_pct=0.10
        )
        
        # Vol duration
        result.vol_duration = compute_vol_duration(
            pool, curve, oas, tuning, vol_shift_pct=0.10, params=params, config=config
        )
        
        # Current coupon duration
        result.current_coupon_duration = compute_cc_duration(
            pool, curve, paths, oas, tuning, params=params, config=config
        )
    
    return result
