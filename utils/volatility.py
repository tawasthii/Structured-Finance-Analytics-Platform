"""
Volatility Surfaces & Option Pricing Models
=============================================

This module implements:
    1. Swaption Volatility Surface: expiry × tenor grid with SABR smile
    2. SABR Model: stochastic volatility model for the smile/skew
    3. Black Model: closed-form pricing for European swaptions, caps, floors
    4. Cap/Floor Volatility: flat-to-spot vol stripping
    5. Synthetic Surface Generation: realistic vol surfaces for demonstration

Key Concepts:
    - NORMAL (basis point) vols vs LOGNORMAL (Black) vols: the market has
      shifted toward normal vols, especially post-2008. We support both.
    - The swaption vol surface has three dimensions: expiry, tenor, strike.
      At-the-money vols form a 2D grid; the strike dimension adds the smile.
    - SABR parameters (α, β, ρ, ν) have financial interpretations:
        α = ATM vol level
        β = backbone parameter (0=normal, 1=lognormal)
        ρ = correlation between rate and vol (controls skew)
        ν = vol-of-vol (controls smile curvature)
    
    These surfaces feed into:
    - BGM model calibration (Notebook 06): swaption vols constrain 
      factor volatilities
    - Black model for standalone derivative pricing
    - Volatility duration calculations in OAS risk decomposition
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import warnings


# =============================================================================
# Black Model (Lognormal and Normal)
# =============================================================================

def black76_call(forward: float, strike: float, vol: float,
                  expiry: float, df: float) -> float:
    """
    Black-76 call price (lognormal vol).
    
    P_call = df × [F × N(d1) - K × N(d2)]
    d1 = [ln(F/K) + 0.5σ²T] / (σ√T)
    d2 = d1 - σ√T
    """
    if vol <= 0 or expiry <= 0:
        return max(forward - strike, 0) * df
    
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    
    return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def black76_put(forward: float, strike: float, vol: float,
                 expiry: float, df: float) -> float:
    """Black-76 put price (lognormal vol)."""
    if vol <= 0 or expiry <= 0:
        return max(strike - forward, 0) * df
    
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    
    return df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def bachelier_call(forward: float, strike: float, normal_vol: float,
                    expiry: float, df: float) -> float:
    """
    Bachelier (normal) model call price.
    Used when rates can go negative.
    
    P_call = df × [(F-K)N(d) + σ√T × n(d)]
    d = (F-K) / (σ√T)
    """
    if normal_vol <= 0 or expiry <= 0:
        return max(forward - strike, 0) * df
    
    sqrt_t = np.sqrt(expiry)
    d = (forward - strike) / (normal_vol * sqrt_t)
    
    return df * ((forward - strike) * norm.cdf(d) + normal_vol * sqrt_t * norm.pdf(d))


def bachelier_put(forward: float, strike: float, normal_vol: float,
                   expiry: float, df: float) -> float:
    """Bachelier (normal) model put price."""
    if normal_vol <= 0 or expiry <= 0:
        return max(strike - forward, 0) * df
    
    sqrt_t = np.sqrt(expiry)
    d = (forward - strike) / (normal_vol * sqrt_t)
    
    return df * ((strike - forward) * norm.cdf(-d) + normal_vol * sqrt_t * norm.pdf(d))


def swaption_price(forward_swap_rate: float, strike: float,
                    vol: float, expiry: float, annuity: float,
                    is_payer: bool = True, vol_type: str = 'normal') -> float:
    """
    Price a European swaption.
    
    A payer swaption gives the right to PAY fixed (long rates).
    A receiver swaption gives the right to RECEIVE fixed (short rates).
    
    Parameters:
        forward_swap_rate: Forward par swap rate
        strike:            Strike rate
        vol:               Volatility (normal in bps or lognormal decimal)
        expiry:            Option expiry in years
        annuity:           PV of swap fixed-leg annuity (DV01-like)
        is_payer:          True for payer, False for receiver
        vol_type:          'normal' or 'lognormal'
    """
    if vol_type == 'normal':
        vol_decimal = vol / 10000.0  # Convert from bps to decimal
        if is_payer:
            return bachelier_call(forward_swap_rate, strike, vol_decimal, expiry, annuity)
        else:
            return bachelier_put(forward_swap_rate, strike, vol_decimal, expiry, annuity)
    else:
        if is_payer:
            return black76_call(forward_swap_rate, strike, vol, expiry, annuity)
        else:
            return black76_put(forward_swap_rate, strike, vol, expiry, annuity)


# =============================================================================
# SABR Model
# =============================================================================

def sabr_implied_vol(forward: float, strike: float, expiry: float,
                      alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    SABR implied lognormal volatility (Hagan et al. 2002 approximation).
    """
    # Guard against bad inputs
    if forward <= 0 or strike <= 0 or alpha <= 0 or expiry <= 0:
        return alpha / max(forward, 1e-10) ** (1 - beta)
    
    if abs(forward - strike) < 1e-10:
        # ATM formula
        f_mid = forward
        factor1 = alpha / (f_mid ** (1 - beta))
        
        term1 = ((1 - beta)**2 / 24) * (alpha**2 / (f_mid ** (2 - 2*beta)))
        term2 = 0.25 * rho * beta * nu * alpha / (f_mid ** (1 - beta))
        term3 = (2 - 3 * rho**2) / 24 * nu**2
        
        return factor1 * (1 + (term1 + term2 + term3) * expiry)
    
    # General formula
    f_mid = np.sqrt(forward * strike)
    log_fk = np.log(forward / strike)
    
    # z and x(z)
    z = (nu / alpha) * (f_mid ** (1 - beta)) * log_fk
    
    sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
    arg = (sqrt_term + z - rho) / (1 - rho)
    
    # Handle numerical edge cases
    if arg <= 0:
        return alpha / (f_mid ** (1 - beta))
    
    x_z = np.log(arg)
    
    if abs(x_z) < 1e-10:
        zx = 1.0
    else:
        zx = z / x_z
    
    # Prefactor
    fk_beta = (forward * strike) ** ((1 - beta) / 2)
    
    numer = alpha * zx
    denom = fk_beta * (1 + (1 - beta)**2 / 24 * log_fk**2 + 
                        (1 - beta)**4 / 1920 * log_fk**4)
    
    # Correction terms
    term1 = ((1 - beta)**2 / 24) * (alpha**2 / (fk_beta**2))
    term2 = 0.25 * rho * beta * nu * alpha / fk_beta
    term3 = (2 - 3 * rho**2) / 24 * nu**2
    
    result = (numer / denom) * (1 + (term1 + term2 + term3) * expiry)
    
    # Final safety check
    if np.isnan(result) or result <= 0:
        return alpha / (f_mid ** (1 - beta))
    
    return result


def sabr_normal_vol(forward: float, strike: float, expiry: float,
                     alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    SABR implied normal (basis point) volatility.
    Converts from lognormal SABR vol using the relationship:
    σ_N ≈ σ_LN × F (approximately, for ATM)
    """
    lognormal_vol = sabr_implied_vol(forward, strike, expiry, alpha, beta, rho, nu)
    # Approximate conversion: normal_vol ≈ lognormal_vol × forward
    f_mid = np.sqrt(forward * strike) if abs(forward - strike) > 1e-10 else forward
    return lognormal_vol * f_mid * 10000  # Return in bps


def calibrate_sabr(forward: float, expiry: float,
                    strikes: np.ndarray, market_vols: np.ndarray,
                    beta: float = 0.5,
                    vol_type: str = 'lognormal') -> dict:
    """
    Calibrate SABR parameters (α, ρ, ν) to market smile data.
    β is typically fixed (0.5 is common for rates).
    
    Parameters:
        forward:     Forward rate
        expiry:      Time to expiry
        strikes:     Array of strike rates
        market_vols: Array of market implied vols at those strikes
        beta:        Fixed backbone parameter
        vol_type:    'lognormal' or 'normal' (type of market_vols)
    
    Returns:
        Dict with 'alpha', 'beta', 'rho', 'nu', 'rmse'
    """
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        
        model_vols = np.array([
            sabr_implied_vol(forward, k, expiry, alpha, beta, rho, nu)
            for k in strikes
        ])
        
        if vol_type == 'normal':
            # Convert model lognormal to normal for comparison
            f_mids = np.sqrt(forward * strikes)
            model_vols = model_vols * f_mids * 10000
        
        return np.sum((model_vols - market_vols) ** 2)
    
    # Initial guess: alpha from ATM vol
    atm_idx = np.argmin(np.abs(strikes - forward))
    if vol_type == 'lognormal':
        alpha0 = market_vols[atm_idx]
    else:
        alpha0 = market_vols[atm_idx] / 10000 / forward
    
    result = minimize(
        objective,
        x0=[alpha0, -0.2, 0.3],
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8}
    )
    
    alpha, rho, nu = result.x
    rho = np.clip(rho, -0.999, 0.999)
    
    # Compute RMSE
    model_vols = np.array([
        sabr_implied_vol(forward, k, expiry, alpha, beta, rho, nu)
        for k in strikes
    ])
    if vol_type == 'normal':
        f_mids = np.sqrt(forward * strikes)
        model_vols = model_vols * f_mids * 10000
    
    rmse = np.sqrt(np.mean((model_vols - market_vols) ** 2))
    
    return {
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'nu': nu,
        'rmse': rmse
    }


# =============================================================================
# Swaption Volatility Surface
# =============================================================================

@dataclass
class SwaptionVolSurface:
    """
    Swaption ATM normal volatility surface.
    
    Attributes:
        expiries:   Option expiry tenors in years (e.g., [1, 2, 3, 5, 7, 10])
        tenors:     Underlying swap tenors in years (e.g., [1, 2, 5, 10, 20, 30])
        atm_vols:   2D array of ATM normal vols in bps [expiry × tenor]
        sabr_params: Dict of SABR params per (expiry, tenor) node (optional)
    """
    expiries: np.ndarray
    tenors: np.ndarray
    atm_vols: np.ndarray  # In basis points (normal vol)
    sabr_params: dict = field(default_factory=dict)
    
    def get_vol(self, expiry: float, tenor: float) -> float:
        """Interpolate ATM normal vol at arbitrary expiry/tenor (in bps)."""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.expiries, self.tenors), self.atm_vols,
            method='linear', bounds_error=False, fill_value=None
        )
        result = interp(np.array([[expiry, tenor]]))[0]
        return float(result)
    
    def get_smile_vol(self, expiry: float, tenor: float, 
                       strike: float, forward: float) -> float:
        """
        Get vol at a specific strike using SABR smile interpolation.
        Falls back to ATM vol if no SABR params calibrated.
        """
        key = (expiry, tenor)
        if key in self.sabr_params:
            params = self.sabr_params[key]
            return sabr_normal_vol(forward, strike, expiry,
                                    params['alpha'], params['beta'],
                                    params['rho'], params['nu'])
        return self.get_vol(expiry, tenor)
    
    def shift(self, shift_bps: float) -> 'SwaptionVolSurface':
        """Return a new surface with parallel vol shift."""
        return SwaptionVolSurface(
            self.expiries.copy(), self.tenors.copy(),
            self.atm_vols + shift_bps, dict(self.sabr_params)
        )


def generate_synthetic_swaption_surface() -> SwaptionVolSurface:
    """
    Generate a realistic synthetic swaption ATM normal vol surface.
    
    Based on typical market patterns:
    - Short expiries tend to have higher vols (steeper term structure)
    - Medium tenors (5-10Y) tend to have highest vols
    - Long expiry + long tenor = somewhat lower vols (mean reversion)
    
    Typical ATM normal vols are 50-120 bps for USD rates.
    """
    expiries = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    tenors = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    # Base vol levels (bps) — realistic for 2024 USD market
    # Shape: short expiry/short tenor highest, declining with expiry
    base_grid = np.array([
        # 1Y   2Y   3Y   5Y   7Y  10Y  15Y  20Y  30Y   (tenor)
        [105, 108, 110, 112, 110, 107, 103, 100,  95],  # 3M expiry
        [100, 103, 106, 108, 107, 104, 100,  97,  92],  # 6M
        [ 95,  98, 101, 104, 103, 101,  97,  94,  89],  # 1Y
        [ 88,  91,  94,  97,  97,  95,  92,  89,  85],  # 2Y
        [ 83,  86,  89,  93,  93,  91,  88,  86,  82],  # 3Y
        [ 76,  79,  82,  86,  87,  86,  83,  81,  77],  # 5Y
        [ 72,  75,  77,  81,  82,  82,  79,  77,  74],  # 7Y
        [ 68,  71,  73,  76,  78,  78,  76,  74,  71],  # 10Y
        [ 63,  65,  67,  70,  72,  73,  72,  70,  68],  # 15Y
        [ 60,  62,  64,  66,  68,  69,  69,  67,  65],  # 20Y
        [ 56,  58,  59,  61,  63,  64,  64,  63,  61],  # 30Y
    ], dtype=float)
    
    return SwaptionVolSurface(expiries, tenors, base_grid)


def generate_synthetic_sabr_params(forward: float = 0.04,
                                    expiry: float = 5.0) -> dict:
    """
    Generate realistic SABR parameters for a single node.
    Useful for smile demonstration.
    """
    return {
        'alpha': 0.025,
        'beta': 0.5,
        'rho': -0.25,   # Negative skew typical for rates
        'nu': 0.35
    }


# =============================================================================
# Cap/Floor Pricing
# =============================================================================

def price_caplet(forward_rate: float, strike: float, vol: float,
                  expiry: float, accrual: float, df: float,
                  vol_type: str = 'normal') -> float:
    """
    Price a single caplet (one period of a cap).
    A caplet pays max(L - K, 0) × accrual × notional at period end.
    """
    if vol_type == 'normal':
        return bachelier_call(forward_rate, strike, vol / 10000, expiry, df * accrual)
    else:
        return black76_call(forward_rate, strike, vol, expiry, df * accrual)


def price_cap(forward_rates: np.ndarray, strike: float,
               vols: np.ndarray, expiries: np.ndarray,
               accruals: np.ndarray, dfs: np.ndarray,
               vol_type: str = 'normal') -> float:
    """
    Price an interest rate cap (sum of caplets).
    
    Parameters:
        forward_rates: Forward rates for each caplet period
        strike:        Cap strike rate
        vols:          Spot vols for each caplet (normal bps or lognormal)
        expiries:      Expiry time for each caplet
        accruals:      Day count fraction for each period
        dfs:           Discount factors to each payment date
    """
    total = 0.0
    for i in range(len(forward_rates)):
        total += price_caplet(forward_rates[i], strike, vols[i],
                               expiries[i], accruals[i], dfs[i], vol_type)
    return total


def strip_cap_vols(flat_vols: np.ndarray, forward_rates: np.ndarray,
                    strike: float, expiries: np.ndarray,
                    accruals: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    """
    Strip spot (caplet) volatilities from flat (market-quoted) cap vols.
    
    The market quotes caps with a SINGLE flat vol that prices the entire 
    cap. But each caplet has its own vol. Stripping recovers the per-caplet
    vols by bootstrapping — analogous to zero curve bootstrapping.
    """
    n = len(flat_vols)
    spot_vols = np.zeros(n)
    spot_vols[0] = flat_vols[0]  # First caplet vol = flat vol
    
    for i in range(1, n):
        # Price cap at flat vol i
        target_price = price_cap(
            forward_rates[:i+1], strike, np.full(i+1, flat_vols[i]),
            expiries[:i+1], accruals[:i+1], dfs[:i+1]
        )
        
        # Price of known caplets (with already-stripped spot vols)
        known_price = price_cap(
            forward_rates[:i], strike, spot_vols[:i],
            expiries[:i], accruals[:i], dfs[:i]
        )
        
        # Remaining price = caplet i price
        remaining = target_price - known_price
        
        # Solve for spot vol of caplet i
        try:
            def obj(v):
                return price_caplet(forward_rates[i], strike, v,
                                     expiries[i], accruals[i], dfs[i]) - remaining
            spot_vols[i] = brentq(obj, 1, 300, xtol=0.01)
        except (ValueError, RuntimeError):
            spot_vols[i] = flat_vols[i]
    
    return spot_vols
