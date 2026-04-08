"""
Monte Carlo Interest Rate Simulation
======================================

This module implements interest rate path generation for OAS calculation.

We use a multi-factor Hull-White (Gaussian short rate) model rather than 
a full BGM/LMM implementation. The Hull-White model is:
    dr(t) = [θ(t) - a·r(t)]dt + σ(t)·dW(t)

Extended to multiple factors:
    r(t) = x1(t) + x2(t) + x3(t) + φ(t)
    
    dx1 = -a1·x1·dt + σ1·dW1  (level factor — parallel moves)
    dx2 = -a2·x2·dt + σ2·dW2  (slope factor — steepening/flattening)
    dx3 = -a3·x3·dt + σ3·dW3  (curvature factor — butterfly)
    
    φ(t) is a deterministic drift that ensures the model fits the 
    initial term structure exactly (no-arbitrage condition).

Why 3 factors:
    1-factor models can't capture curve reshaping (steepening, butterflies)
    which drives different prepayment behavior across the curve.
    3 factors capture level, slope, and curvature — the three principal 
    components that explain >99% of yield curve movements empirically.

This feeds into OAS:
    For each path → project rates → compute mortgage rate → run prepayment 
    model → generate cash flows → discount → average across paths.

Variance Reduction:
    - Antithetic variates (negate random draws for a paired path)
    - Moment matching (force sample mean/variance to match theoretical)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .curves import ZeroCurve


@dataclass
class RateModelParams:
    """
    Parameters for the 3-factor Hull-White model.
    
    Mean reversion speeds (a1, a2, a3):
        Higher = factor reverts faster = less persistent shocks
        Typical: a1 ~ 0.03 (slow), a2 ~ 0.10 (medium), a3 ~ 0.30 (fast)
    
    Factor volatilities (sigma1, sigma2, sigma3):
        Level factor has highest vol, curvature has lowest
        Typical: σ1 ~ 0.008 (80bp/yr), σ2 ~ 0.005, σ3 ~ 0.003
    
    Correlations:
        Level-slope typically negative (rates rise → curve flattens)
        Level-curvature near zero
        Slope-curvature can be slightly positive
    """
    # Mean reversion speeds
    a1: float = 0.03    # Level factor
    a2: float = 0.10    # Slope factor  
    a3: float = 0.30    # Curvature factor
    
    # Factor volatilities
    sigma1: float = 0.008   # Level (~80bp annual)
    sigma2: float = 0.005   # Slope (~50bp annual)
    sigma3: float = 0.003   # Curvature (~30bp annual)
    
    # Correlations
    rho12: float = -0.30    # Level-slope
    rho13: float = 0.05     # Level-curvature
    rho23: float = 0.15     # Slope-curvature
    
    # Mortgage rate model
    mortgage_spread: float = 0.015   # Spread of mortgage rate over 10Y (150bp)
    mortgage_vol_mult: float = 0.8   # Mortgage rate vol is 80% of Treasury vol
    
    @property
    def correlation_matrix(self) -> np.ndarray:
        """3×3 correlation matrix."""
        return np.array([
            [1.0,       self.rho12, self.rho13],
            [self.rho12, 1.0,       self.rho23],
            [self.rho13, self.rho23, 1.0      ]
        ])
    
    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky decomposition of correlation matrix."""
        return np.linalg.cholesky(self.correlation_matrix)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    num_paths: int = 512
    num_months: int = 360
    dt: float = 1/12          # Monthly time steps
    antithetic: bool = True   # Use antithetic variates
    moment_matching: bool = True  # Force moments to match
    seed: Optional[int] = 42


@dataclass
class RatePathResult:
    """
    Container for simulated rate paths.
    
    Attributes:
        short_rates:    [num_paths × num_months] short rate paths
        mortgage_rates: [num_paths × num_months] projected mortgage rates
        discount_factors: [num_paths × num_months] cumulative discount factors
        num_paths:      Number of paths (after antithetic doubling)
        num_months:     Number of time steps
    """
    short_rates: np.ndarray
    mortgage_rates: np.ndarray
    discount_factors: np.ndarray
    num_paths: int
    num_months: int
    
    def path_discount_factor(self, path_idx: int, month: int) -> float:
        """D(0, t_month) on a specific path."""
        return self.discount_factors[path_idx, month]
    
    def average_short_rate(self) -> np.ndarray:
        """Average short rate across all paths (should ≈ forward rates)."""
        return np.mean(self.short_rates, axis=0)
    
    def short_rate_distribution(self, month: int) -> Tuple[float, float]:
        """(mean, std) of short rate at a given month."""
        rates = self.short_rates[:, month]
        return float(np.mean(rates)), float(np.std(rates))


def generate_correlated_normals(num_paths: int, num_months: int,
                                  num_factors: int, cholesky: np.ndarray,
                                  config: SimulationConfig) -> np.ndarray:
    """
    Generate correlated standard normal random variables.
    
    Returns: [num_paths × num_months × num_factors] array
    
    With antithetic variates: generates num_paths/2 draws, then
    appends the negated draws. This reduces variance because for 
    any convex payoff, the average of f(Z) and f(-Z) has lower 
    variance than two independent draws.
    """
    rng = np.random.RandomState(config.seed)
    
    if config.antithetic:
        half_paths = num_paths // 2
        # Independent standard normals
        Z_indep = rng.standard_normal((half_paths, num_months, num_factors))
        # Correlate using Cholesky: Z_corr = Z_indep × L^T
        Z_half = np.einsum('ijk,lk->ijl', Z_indep, cholesky)
        # Antithetic
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z_indep = rng.standard_normal((num_paths, num_months, num_factors))
        Z = np.einsum('ijk,lk->ijl', Z_indep, cholesky)
    
    if config.moment_matching:
        # Force sample mean = 0, sample std = 1 per month per factor
        for m in range(num_months):
            for f in range(num_factors):
                z = Z[:, m, f]
                z = (z - z.mean()) / z.std()
                Z[:, m, f] = z
    
    return Z


def compute_drift_adjustment(curve: ZeroCurve, num_months: int,
                               dt: float,
                               params=None) -> np.ndarray:
    """
    Compute the deterministic drift φ(t) that ensures no-arbitrage.
    
    For Hull-White: φ(t) = f(0,t) + convexity_correction
    
    The convexity correction ensures E[D(0,T)] = D_curve(0,T).
    Without it, the Gaussian model systematically overprices long-dated
    discount factors because E[exp(-r)] > exp(-E[r]) (Jensen's inequality).
    
    For a multi-factor model:
        correction(t) = Σ_i (σ_i² / (2 a_i²)) × (1 - e^{-a_i × t})²
    
    Returns: Array of drift values, one per month.
    """
    times = np.arange(1, num_months + 1) * dt
    phi = curve.instantaneous_forward(times)
    
    # Add convexity correction for each factor
    if params is not None:
        for a, sigma in [(params.a1, params.sigma1), 
                          (params.a2, params.sigma2), 
                          (params.a3, params.sigma3)]:
            if a > 0:
                correction = (sigma**2 / (2 * a**2)) * (1 - np.exp(-a * times))**2
                phi += correction
    
    return phi


def simulate_rate_paths(curve: ZeroCurve,
                         params: RateModelParams = None,
                         config: SimulationConfig = None,
                         vol_surface=None) -> RatePathResult:
    """
    Simulate interest rate paths using the 3-factor Hull-White model.
    
    The simulation:
        1. Generate correlated normal random variables
        2. Evolve each factor: x_i(t+dt) = x_i(t)·(1-a_i·dt) + σ_i·√dt·Z_i
        3. Short rate: r(t) = x1(t) + x2(t) + x3(t) + φ(t)
        4. Mortgage rate: r_mtg(t) = function of short rate + spread
        5. Discount factors: D(0,t) = exp(-sum(r(s)·dt for s=0..t))
    
    Parameters:
        curve:       Initial zero curve (for drift calibration)
        params:      Rate model parameters
        config:      Simulation configuration
        vol_surface: Optional swaption vol surface for calibration
    
    Returns:
        RatePathResult with all paths
    """
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig()
    
    dt = config.dt
    num_months = config.num_months
    num_paths = config.num_paths
    
    # Adjust for antithetic
    effective_paths = num_paths
    
    # Generate correlated normals
    Z = generate_correlated_normals(num_paths, num_months, 3, 
                                     params.cholesky, config)
    
    actual_paths = Z.shape[0]
    
    # Compute drift adjustment from initial curve
    phi = compute_drift_adjustment(curve, num_months, dt, params)
    
    # Initialize factor paths
    x1 = np.zeros((actual_paths, num_months))
    x2 = np.zeros((actual_paths, num_months))
    x3 = np.zeros((actual_paths, num_months))
    
    sqrt_dt = np.sqrt(dt)
    
    # Evolve factors (Euler-Maruyama)
    for t in range(1, num_months):
        x1[:, t] = x1[:, t-1] * (1 - params.a1 * dt) + params.sigma1 * sqrt_dt * Z[:, t, 0]
        x2[:, t] = x2[:, t-1] * (1 - params.a2 * dt) + params.sigma2 * sqrt_dt * Z[:, t, 1]
        x3[:, t] = x3[:, t-1] * (1 - params.a3 * dt) + params.sigma3 * sqrt_dt * Z[:, t, 2]
    
    # Short rate = sum of factors + drift
    short_rates = x1 + x2 + x3 + phi[np.newaxis, :]
    
    # Mortgage rate = f(short_rate) + spread
    # In practice, the mortgage rate is related to the 10Y rate, not the short rate.
    # We approximate: mortgage_rate ≈ short_rate + term_premium + spread
    # The term premium dampens short rate movements for mortgage rates
    mortgage_rates = (short_rates * params.mortgage_vol_mult + 
                       phi[np.newaxis, :] * (1 - params.mortgage_vol_mult) +
                       params.mortgage_spread)
    
    # Cumulative discount factors: D(0,t) = exp(-Σ r(s)·dt)
    cumulative_rates = np.cumsum(short_rates * dt, axis=1)
    discount_factors = np.exp(-cumulative_rates)
    
    return RatePathResult(
        short_rates=short_rates,
        mortgage_rates=mortgage_rates,
        discount_factors=discount_factors,
        num_paths=actual_paths,
        num_months=num_months
    )


def validate_simulation(paths: RatePathResult, curve: ZeroCurve,
                          dt: float = 1/12) -> dict:
    """
    Validate Monte Carlo simulation against the initial curve.
    
    Key check: the average discount factor across all paths at each 
    tenor should approximately equal the discount factor from the 
    initial curve. Large deviations indicate drift calibration errors.
    
    Returns dict with validation metrics.
    """
    check_months = [12, 24, 60, 120, 240, 360]
    results = {}
    
    for m in check_months:
        if m > paths.num_months:
            continue
        m_idx = m - 1
        
        # Average simulated DF
        sim_df = float(np.mean(paths.discount_factors[:, m_idx]))
        
        # Curve DF
        t = m * dt
        curve_df = float(curve.discount_factor(np.array([t]))[0])
        
        error_bps = abs(sim_df - curve_df) / curve_df * 10000
        
        results[f'{m}mo'] = {
            'sim_df': sim_df,
            'curve_df': curve_df,
            'error_bps': error_bps
        }
    
    return results


# =============================================================================
# Calibration Helpers
# =============================================================================

def calibrate_to_swaption_vol(curve: ZeroCurve, target_vol_bps: float,
                                expiry: float, tenor: float,
                                params: RateModelParams = None) -> RateModelParams:
    """
    Simple calibration: adjust sigma1 so that the model approximately
    reproduces a target ATM swaption vol.
    
    This is a simplified calibration — production systems solve a 
    multi-dimensional optimization across the entire vol surface.
    """
    if params is None:
        params = RateModelParams()
    
    # Rough relationship: swaption_vol ≈ σ1 × sqrt(tenor) × 10000 / sqrt(3)
    # (the /sqrt(3) accounts for 3 factors)
    target_annual = target_vol_bps / 10000
    
    # Approximate adjustment
    params.sigma1 = target_annual / np.sqrt(tenor) * 0.7
    params.sigma2 = params.sigma1 * 0.6
    params.sigma3 = params.sigma1 * 0.35
    
    return params
