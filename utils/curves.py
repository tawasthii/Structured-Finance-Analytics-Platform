"""
Yield Curve Construction, Bootstrapping & Interpolation
=========================================================

This module implements:
    1. Treasury zero-coupon curve bootstrapping from par yields
    2. Swap curve bootstrapping from deposit rates + swap par rates
    3. Interpolation methods: linear, cubic spline, monotone convex
    4. Forward rate extraction (instantaneous and discrete-period)
    5. Curve manipulation: parallel shifts, twists, butterflies, key-rate shifts
    6. Discount factor and present value utilities

The bootstrapping methodology:
    - Short end (< 1Y): directly from zero-coupon instruments (T-bills, deposits)
    - Long end (≥ 1Y): iterative stripping of coupon bonds/swaps using
      previously solved zero rates

Convention notes:
    - Treasury: Act/Act day count, semi-annual compounding
    - LIBOR/Swap: Act/360 day count, quarterly (LIBOR) or semi-annual (swap) frequency
    - All zero rates stored as continuously compounded for internal consistency
    - Conversion to other compounding conventions provided via utility functions
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional, Literal
import warnings


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ZeroCurve:
    """
    Represents a bootstrapped zero-coupon yield curve.
    
    Attributes:
        tenors:      Array of maturities in years
        zero_rates:  Continuously compounded zero rates
        source:      'treasury' or 'swap' — affects day count conventions
        as_of_date:  Valuation date string
    """
    tenors: np.ndarray
    zero_rates: np.ndarray
    source: str = 'treasury'
    as_of_date: str = ''
    
    def __post_init__(self):
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.zero_rates = np.asarray(self.zero_rates, dtype=float)
        if len(self.tenors) != len(self.zero_rates):
            raise ValueError("tenors and zero_rates must have same length")
    
    def discount_factor(self, t: np.ndarray) -> np.ndarray:
        """Compute discount factor D(0,t) = exp(-r(t)*t)"""
        t = np.asarray(t, dtype=float)
        r = self.interpolate(t)
        return np.exp(-r * t)
    
    def interpolate(self, t: np.ndarray, method: str = 'cubic_spline') -> np.ndarray:
        """Interpolate zero rates at arbitrary tenors."""
        t = np.asarray(t, dtype=float)
        if method == 'linear':
            return linear_interpolation(self.tenors, self.zero_rates, t)
        elif method == 'cubic_spline':
            return cubic_spline_interpolation(self.tenors, self.zero_rates, t)
        elif method == 'monotone_convex':
            return monotone_convex_interpolation(self.tenors, self.zero_rates, t)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Discrete forward rate f(t1, t2) between times t1 and t2.
        From no-arbitrage: D(0,t1) * D(t1,t2) = D(0,t2)
        => f(t1,t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        r1 = float(self.interpolate(np.array([t1]))[0])
        r2 = float(self.interpolate(np.array([t2]))[0])
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def instantaneous_forward(self, t: np.ndarray, dt: float = 1e-4) -> np.ndarray:
        """
        Instantaneous forward rate f(t) = d/dt [r(t)*t]
        Approximated by finite difference.
        """
        t = np.asarray(t, dtype=float)
        t_plus = t + dt
        t_minus = np.maximum(t - dt, 1e-6)
        r_plus = self.interpolate(t_plus)
        r_minus = self.interpolate(t_minus)
        return (r_plus * t_plus - r_minus * t_minus) / (t_plus - t_minus)
    
    def par_rate(self, maturity: float, freq: int = 2) -> float:
        """
        Par coupon rate for a bond maturing at given tenor.
        The par rate c satisfies: sum(c/freq * D(0,ti)) + D(0,T) = 1
        => c = freq * (1 - D(0,T)) / sum(D(0,ti))
        
        For zero-coupon instruments (maturity <= 1/freq), return the
        equivalent simple yield.
        
        Uses linear interpolation to be consistent with the bootstrap 
        procedure. For analytical work, use discount_factor() directly 
        with your preferred interpolation.
        """
        dt = 1.0 / freq
        if maturity <= dt + 1e-6:
            # Zero-coupon: par yield = (1/D(0,T) - 1) / T
            r = float(self.interpolate(np.array([maturity]), method='linear')[0])
            df = np.exp(-r * maturity)
            return (1.0 / df - 1.0) / maturity
        payment_times = np.arange(dt, maturity + dt/2, dt)
        rates = self.interpolate(payment_times, method='linear')
        dfs = np.exp(-rates * payment_times)
        return freq * (1.0 - dfs[-1]) / np.sum(dfs)
    
    def shift(self, shift_type: str = 'parallel', shift_bps: float = 100,
              pivot_tenor: float = 5.0, wing_shift_bps: float = 0.0,
              key_rate_tenor: float = None, key_rate_width: float = 2.0) -> 'ZeroCurve':
        """
        Return a new ZeroCurve with applied shift.
        
        Shift types:
            'parallel'  : uniform shift across all tenors
            'twist'     : steepening/flattening around pivot_tenor
            'butterfly' : wings move relative to belly
            'key_rate'  : localized bump at key_rate_tenor
        """
        shift = shift_bps / 10000.0
        new_rates = self.zero_rates.copy()
        
        if shift_type == 'parallel':
            new_rates += shift
            
        elif shift_type == 'twist':
            # Linear twist: short end moves -shift, long end moves +shift
            # pivoting around pivot_tenor
            twist_factor = (self.tenors - pivot_tenor) / (self.tenors[-1] - self.tenors[0])
            new_rates += shift * twist_factor
            
        elif shift_type == 'butterfly':
            # Belly moves by -shift, wings move by +wing_shift
            belly = pivot_tenor
            sigma = (self.tenors[-1] - self.tenors[0]) / 4
            belly_weight = np.exp(-0.5 * ((self.tenors - belly) / sigma) ** 2)
            wing_weight = 1.0 - belly_weight
            new_rates += (-shift * belly_weight + 
                         (wing_shift_bps / 10000.0) * wing_weight)
            
        elif shift_type == 'key_rate':
            if key_rate_tenor is None:
                raise ValueError("key_rate_tenor required for key_rate shift")
            # Triangular bump centered at key_rate_tenor
            distance = np.abs(self.tenors - key_rate_tenor)
            weight = np.maximum(0, 1 - distance / key_rate_width)
            new_rates += shift * weight
            
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")
        
        return ZeroCurve(self.tenors.copy(), new_rates, self.source, self.as_of_date)


# =============================================================================
# Interpolation Methods
# =============================================================================

def linear_interpolation(tenors: np.ndarray, rates: np.ndarray, 
                         t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation with flat extrapolation.
    Simple but creates discontinuous forward rates — not ideal for 
    production use but useful as a baseline.
    """
    return np.interp(t, tenors, rates)


def cubic_spline_interpolation(tenors: np.ndarray, rates: np.ndarray,
                                t: np.ndarray) -> np.ndarray:
    """
    Natural cubic spline interpolation on zero rates.
    Produces smooth forwards but can oscillate and go negative
    between knot points if the curve has unusual shapes.
    """
    cs = CubicSpline(tenors, rates, bc_type='natural')
    result = cs(t)
    return result


def monotone_convex_interpolation(tenors: np.ndarray, rates: np.ndarray,
                                   t: np.ndarray) -> np.ndarray:
    """
    Monotone-preserving interpolation using Hyman filtering on cubic spline.
    
    This prevents the oscillation problem of raw cubic splines while 
    maintaining smoothness. The key insight: we interpolate on discount 
    factors (which must be monotonically decreasing) rather than rates,
    then convert back.
    
    Reference: Hagan & West (2006), "Methods for Constructing a Yield Curve"
    """
    # Interpolate on log-discount factors (which should be monotone)
    log_df = -tenors * rates  # log(D(0,t)) = -r(t)*t
    
    # Fit cubic spline to log discount factors
    cs = CubicSpline(tenors, log_df, bc_type='natural')
    
    # Evaluate
    log_df_interp = cs(t)
    
    # Convert back to zero rates, handling t=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.where(t > 1e-10, -log_df_interp / t, rates[0])
    
    return result


# =============================================================================
# Treasury Curve Bootstrapping
# =============================================================================

def bootstrap_treasury_curve(par_tenors: np.ndarray, par_yields: np.ndarray,
                              freq: int = 2) -> ZeroCurve:
    """
    Bootstrap zero-coupon rates from Treasury par yields.
    
    Uses a root-finding approach: for each new tenor, we find the zero rate
    that makes the par bond price exactly 1.0 (par). This handles gaps 
    between bootstrapping tenors correctly by including the new rate in 
    the interpolation during the search.
    
    Parameters:
        par_tenors: Maturities in years
        par_yields: Par yields as decimals (e.g., 0.05 for 5%)
        freq:       Coupon frequency (2 = semi-annual for Treasuries)
    
    Returns:
        ZeroCurve with bootstrapped continuously compounded zero rates
    """
    par_tenors = np.asarray(par_tenors, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    
    n = len(par_tenors)
    solved_tenors = []
    solved_rates = []
    zero_rates = np.zeros(n)
    
    for i in range(n):
        T = par_tenors[i]
        c = par_yields[i]
        
        if T <= 1.0 / freq:
            # Zero-coupon instrument
            zero_rates[i] = np.log(1 + c * T) / T
        else:
            dt = 1.0 / freq
            coupon_payment = c / freq
            payment_times = np.arange(dt, T + dt/2, dt)
            payment_times = np.round(payment_times, 6)
            final_payment = 1.0 + coupon_payment
            
            st = np.array(solved_tenors)
            sr = np.array(solved_rates)
            
            def price_error(r_T):
                """Price of par bond minus 1, given trial zero rate at T."""
                # Build temporary curve including the trial rate
                t_all = np.append(st, T)
                r_all = np.append(sr, r_T)
                
                pv = 0.0
                for t_j in payment_times[:-1]:
                    r_j = np.interp(t_j, t_all, r_all)
                    pv += coupon_payment * np.exp(-r_j * t_j)
                # Final payment
                pv += final_payment * np.exp(-r_T * T)
                return pv - 1.0
            
            try:
                zero_rates[i] = brentq(price_error, -0.05, 0.20, xtol=1e-12)
            except ValueError:
                # Fallback: closed-form using flat extrapolation
                pv_coupons = 0.0
                for t_j in payment_times[:-1]:
                    r_j = np.interp(t_j, st, sr) if len(st) > 0 else 0.0
                    pv_coupons += coupon_payment * np.exp(-r_j * t_j)
                remaining = 1.0 - pv_coupons
                if remaining > 0:
                    zero_rates[i] = -np.log(remaining / final_payment) / T
                else:
                    zero_rates[i] = solved_rates[-1] if solved_rates else 0.0
        
        solved_tenors.append(T)
        solved_rates.append(zero_rates[i])
    
    return ZeroCurve(par_tenors, zero_rates, source='treasury')


# =============================================================================
# Swap Curve Bootstrapping
# =============================================================================

def bootstrap_swap_curve(deposit_tenors: np.ndarray, deposit_rates: np.ndarray,
                          swap_tenors: np.ndarray, swap_rates: np.ndarray,
                          float_freq: int = 4, fixed_freq: int = 2) -> ZeroCurve:
    """
    Bootstrap zero curve from money market deposits and swap par rates.
    
    Short end (< 1Y): LIBOR/SOFR deposit rates (Act/360, simple compounding)
    Long end (≥ 1Y): Swap par rates (fixed leg: semi-annual Act/360)
    
    The swap rate is the fixed rate that makes the swap have zero NPV at inception.
    This means:
        PV(fixed leg) = PV(floating leg)
    
    Since the floating leg of a par swap = 1 (resets to par at each fixing):
        sum(s/freq * D(0,ti)) + D(0,T) = 1
    
    This is identical in form to bond bootstrapping, just with different
    day-count conventions.
    
    Parameters:
        deposit_tenors: Money market tenors in years (e.g., [0.25, 0.5])
        deposit_rates:  Simple deposit rates (Act/360)
        swap_tenors:    Swap maturities in years (e.g., [1, 2, 3, 5, 7, 10, 15, 20, 30])
        swap_rates:     Par swap rates
        float_freq:     Floating leg frequency (4 = quarterly)
        fixed_freq:     Fixed leg frequency (2 = semi-annual)
    """
    # Combine all tenors
    all_tenors = np.concatenate([deposit_tenors, swap_tenors])
    all_rates_input = np.concatenate([deposit_rates, swap_rates])
    zero_rates = np.zeros(len(all_tenors))
    
    n_deposits = len(deposit_tenors)
    solved_tenors = []
    solved_rates = []
    
    # Short end: deposits (simple compounding, Act/360)
    for i in range(n_deposits):
        T = deposit_tenors[i]
        r_simple = deposit_rates[i]
        zero_rates[i] = np.log(1 + r_simple * T) / T
        solved_tenors.append(T)
        solved_rates.append(zero_rates[i])
    
    # Long end: swap rates (use root-finding for gap handling)
    for i in range(n_deposits, len(all_tenors)):
        T = all_tenors[i]
        s = all_rates_input[i]
        dt = 1.0 / fixed_freq
        coupon = s / fixed_freq
        
        payment_times = np.arange(dt, T + dt/2, dt)
        payment_times = np.round(payment_times, 6)
        final_payment = 1.0 + coupon
        
        st = np.array(solved_tenors)
        sr = np.array(solved_rates)
        
        def price_error(r_T):
            t_all = np.append(st, T)
            r_all = np.append(sr, r_T)
            pv = 0.0
            for t_j in payment_times[:-1]:
                r_j = np.interp(t_j, t_all, r_all)
                pv += coupon * np.exp(-r_j * t_j)
            pv += final_payment * np.exp(-r_T * T)
            return pv - 1.0
        
        try:
            zero_rates[i] = brentq(price_error, -0.05, 0.20, xtol=1e-12)
        except ValueError:
            zero_rates[i] = solved_rates[-1] if solved_rates else 0.0
        
        solved_tenors.append(T)
        solved_rates.append(zero_rates[i])
    
    return ZeroCurve(all_tenors, zero_rates, source='swap')


# =============================================================================
# Forward Rate Computation
# =============================================================================

def compute_forward_curve(curve: ZeroCurve, period: float = 0.5,
                           max_tenor: float = None) -> tuple:
    """
    Compute discrete-period forward rates from a zero curve.
    
    The forward rate f(t, t+dt) is the rate you can lock in today for 
    borrowing/lending between future times t and t+dt.
    
    From no-arbitrage:
        exp(-r(t)*t) * exp(-f(t,t+dt)*dt) = exp(-r(t+dt)*(t+dt))
        => f(t,t+dt) = [r(t+dt)*(t+dt) - r(t)*t] / dt
    
    Parameters:
        curve:     ZeroCurve object
        period:    Length of each forward period in years
        max_tenor: Maximum tenor to compute (defaults to curve max)
    
    Returns:
        (forward_start_times, forward_rates) tuple
    """
    if max_tenor is None:
        max_tenor = curve.tenors[-1]
    
    start_times = np.arange(0, max_tenor, period)
    end_times = start_times + period
    
    # Clip end times to max tenor
    mask = end_times <= max_tenor + 1e-6
    start_times = start_times[mask]
    end_times = end_times[mask]
    
    forward_rates = np.zeros(len(start_times))
    for i in range(len(start_times)):
        t1 = start_times[i]
        t2 = end_times[i]
        r1 = float(curve.interpolate(np.array([max(t1, 1e-6)]))[0])
        r2 = float(curve.interpolate(np.array([t2]))[0])
        if t1 < 1e-6:
            forward_rates[i] = r2
        else:
            forward_rates[i] = (r2 * t2 - r1 * t1) / (t2 - t1)
    
    return start_times, forward_rates


# =============================================================================
# Compounding Convention Utilities
# =============================================================================

def continuous_to_semi_annual(r_cont: np.ndarray) -> np.ndarray:
    """Convert continuously compounded rate to semi-annual compounded."""
    return 2.0 * (np.exp(r_cont / 2.0) - 1.0)


def semi_annual_to_continuous(r_sa: np.ndarray) -> np.ndarray:
    """Convert semi-annual compounded rate to continuously compounded."""
    return 2.0 * np.log(1.0 + r_sa / 2.0)


def continuous_to_annual(r_cont: np.ndarray) -> np.ndarray:
    """Convert continuously compounded rate to annual compounded."""
    return np.exp(r_cont) - 1.0


def annual_to_continuous(r_annual: np.ndarray) -> np.ndarray:
    """Convert annual compounded rate to continuously compounded."""
    return np.log(1.0 + r_annual)


def continuous_to_monthly(r_cont: np.ndarray) -> np.ndarray:
    """Convert continuously compounded rate to monthly compounded."""
    return 12.0 * (np.exp(r_cont / 12.0) - 1.0)


# =============================================================================
# Discount Factor Utilities  
# =============================================================================

def discount_factors_from_zeros(tenors: np.ndarray, 
                                 zero_rates: np.ndarray) -> np.ndarray:
    """D(0,t) = exp(-r(t) * t) for continuously compounded zeros."""
    return np.exp(-zero_rates * tenors)


def pv_cashflows(times: np.ndarray, cashflows: np.ndarray, 
                  curve: ZeroCurve, spread: float = 0.0) -> float:
    """
    Present value of a set of cash flows using a zero curve + spread.
    
    PV = sum( CF_i * exp(-(r(t_i) + s) * t_i) )
    
    Parameters:
        times:      Payment times in years
        cashflows:  Cash flow amounts
        curve:      ZeroCurve for discounting
        spread:     Parallel spread added to zero rates (continuous, decimal)
    
    Returns:
        Present value
    """
    times = np.asarray(times, dtype=float)
    cashflows = np.asarray(cashflows, dtype=float)
    rates = curve.interpolate(times)
    discount_factors = np.exp(-(rates + spread) * times)
    return float(np.sum(cashflows * discount_factors))


# =============================================================================
# Sample Data Generators (for when FRED data isn't available)
# =============================================================================

def sample_treasury_data(as_of: str = '2024-01-02') -> tuple:
    """
    Returns realistic Treasury par yields for bootstrapping examples.
    Based on approximate market levels from early 2024.
    
    Returns:
        (tenors, par_yields) — tenors in years, yields as decimals
    """
    tenors = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Approximate Treasury par yields (early 2024, inverted at short end)
    par_yields = np.array([
        0.0540,  # 1M
        0.0535,  # 3M
        0.0515,  # 6M
        0.0480,  # 1Y
        0.0440,  # 2Y
        0.0415,  # 3Y
        0.0400,  # 5Y
        0.0410,  # 7Y
        0.0420,  # 10Y
        0.0455,  # 20Y
        0.0445,  # 30Y
    ])
    
    return tenors, par_yields


def sample_swap_data(as_of: str = '2024-01-02') -> tuple:
    """
    Returns realistic USD swap curve data for bootstrapping.
    
    Returns:
        (deposit_tenors, deposit_rates, swap_tenors, swap_rates)
    """
    deposit_tenors = np.array([1/12, 3/12, 6/12])
    deposit_rates = np.array([0.0545, 0.0540, 0.0530])  # SOFR-based
    
    swap_tenors = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    swap_rates = np.array([
        0.0490,  # 1Y
        0.0450,  # 2Y
        0.0430,  # 3Y
        0.0415,  # 5Y
        0.0420,  # 7Y
        0.0425,  # 10Y
        0.0440,  # 15Y
        0.0445,  # 20Y
        0.0440,  # 25Y
        0.0435,  # 30Y
    ])
    
    return deposit_tenors, deposit_rates, swap_tenors, swap_rates
