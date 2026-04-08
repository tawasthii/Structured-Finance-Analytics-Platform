"""
Multi-Factor Econometric Prepayment Model
==========================================

This module implements a rate-driven, multi-factor prepayment model 
decomposed into four economically motivated components:

1. REFINANCING (Rate-Driven):
   The primary driver. When current mortgage rates fall below the borrower's 
   note rate, they have an incentive to refinance. The response is nonlinear:
   - Small incentive (<50bp): minimal refinancing
   - Moderate incentive (50-150bp): accelerating response  
   - Large incentive (>150bp): saturation (not everyone refinances)
   
   Modeled as a logistic (S-curve) function of the refinancing incentive,
   where incentive = borrower_rate - current_market_rate.
   The "elbow" parameter controls where the curve inflects.

2. TURNOVER (Housing/Mobility-Driven):
   Rate-independent prepayments from home sales due to job changes,
   family events, divorce, death, etc. Produces a baseline prepayment 
   rate even when rates are rising. Modeled as a function of:
   - Loan age (peaks around months 20-60)
   - Seasonality (summer moves)
   - Lock-in effect (when rates rise, turnover drops because moving 
     means giving up a below-market rate)

3. BURNOUT (Population Heterogeneity):
   As a pool experiences refinancing waves, the most rate-sensitive 
   borrowers prepay first, leaving behind a less responsive population.
   This is the single most important dynamic in seasoned MBS valuation.
   
   Modeled as a declining function of cumulative refinancing exposure.
   A pool that has been "in the money" for years but still has a high 
   factor has burned out — the remaining borrowers are unlikely to 
   refinance regardless of incentive.

4. AGING (Seasoning Ramp):
   New loans prepay slowly (borrowers just moved in). Prepayment rates 
   ramp up over the first 30 months as borrowers settle in and some 
   begin to move or refinance. This is the PSA ramp generalized.

Total CPR = Refinancing × Burnout + Turnover × Aging_Ramp × Seasonality

Tuning Framework:
   Each component has multiplicative adjustments and key parameters that 
   can be tuned. Tuning parameter sets can be saved/loaded for reuse.
   This mirrors production systems where analysts maintain "house views" 
   on prepayment behavior.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import json
import warnings


# =============================================================================
# Tuning Parameters
# =============================================================================

@dataclass
class PrepaymentTuning:
    """
    Tuning parameters for the multi-factor prepayment model.
    
    Each multiplier scales its respective component. A multiplier of 1.0 
    means the base model; 1.2 means 20% faster than base; 0.8 means 20% slower.
    
    These are the "knobs" that analysts use to express views on borrower 
    behavior that diverge from the model's baseline calibration.
    """
    # Component multipliers
    refinancing_multiplier: float = 1.0
    turnover_multiplier: float = 1.0
    burnout_multiplier: float = 1.0
    aging_multiplier: float = 1.0
    
    # Refinancing curve parameters
    refi_elbow_shift: float = 0.0     # Shift the refinancing incentive elbow (in bps)
    refi_max_cpr: float = 0.65        # Maximum annualized refinancing CPR
    refi_steepness: float = 5.0       # Steepness of the S-curve
    refi_midpoint: float = 1.25       # Incentive (in %) where response = 50% of max
    
    # Turnover parameters
    turnover_base_cpr: float = 0.08   # Base annual turnover rate (8% CPR)
    turnover_peak_month: int = 40     # Month at which turnover peaks
    turnover_lockin_sensitivity: float = 0.5  # How much rising rates suppress turnover
    
    # Burnout parameters
    burnout_rate: float = 0.015       # Rate at which burnout accumulates
    burnout_floor: float = 0.20       # Minimum burnout factor (never fully burned out)
    
    # Aging ramp
    aging_ramp_months: int = 30       # Months to full seasoning
    
    # Seasonality (monthly multipliers, Jan=index 0)
    seasonality: np.ndarray = field(default_factory=lambda: np.array([
        0.85, 0.80, 0.90, 0.95, 1.05, 1.15,  # Jan-Jun
        1.20, 1.15, 1.05, 0.95, 0.85, 0.80   # Jul-Dec
    ]))
    
    # Global CPR floor and cap
    cpr_floor: float = 0.01           # Minimum CPR (1%)
    cpr_cap: float = 0.70             # Maximum CPR (70%)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for saving."""
        d = {k: v for k, v in self.__dict__.items() if k != 'seasonality'}
        d['seasonality'] = self.seasonality.tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PrepaymentTuning':
        """Deserialize from dictionary."""
        d = d.copy()
        d['seasonality'] = np.array(d['seasonality'])
        return cls(**d)
    
    def save(self, filepath: str):
        """Save tuning parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PrepaymentTuning':
        """Load tuning parameters from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Preset Tuning Configurations
# =============================================================================

def fast_prepay_tuning() -> PrepaymentTuning:
    """Aggressive prepayment assumptions (e.g., low-rate environment)."""
    return PrepaymentTuning(
        refinancing_multiplier=1.3,
        turnover_multiplier=1.1,
        refi_midpoint=1.0,  # Lower elbow = earlier refinancing response
    )

def slow_prepay_tuning() -> PrepaymentTuning:
    """Conservative prepayment assumptions (e.g., credit-impaired borrowers)."""
    return PrepaymentTuning(
        refinancing_multiplier=0.7,
        turnover_multiplier=0.8,
        burnout_floor=0.30,  # Faster burnout
        refi_midpoint=1.75,  # Higher elbow = less responsive
    )

def seasoned_discount_tuning() -> PrepaymentTuning:
    """Tuning for deeply seasoned discount pools (high burnout)."""
    return PrepaymentTuning(
        refinancing_multiplier=0.5,
        turnover_multiplier=1.0,
        burnout_floor=0.15,
        aging_multiplier=1.0,
    )


# =============================================================================
# Component Functions
# =============================================================================

def refinancing_component(incentive: float, tuning: PrepaymentTuning) -> float:
    """
    Compute the refinancing-driven CPR given the refinancing incentive.
    
    Uses a logistic (S-curve) function:
        refi_cpr = max_cpr / (1 + exp(-steepness × (incentive - midpoint)))
    
    The S-curve captures the empirical reality that:
    - Below ~50bp incentive: almost no one refinances (transaction costs)
    - 50-200bp: rapid acceleration (the "sweet spot")
    - Above 200bp: saturation (most rate-sensitive borrowers already left)
    
    Parameters:
        incentive: Refinancing incentive in percentage points
                   (borrower_rate - current_market_rate)
                   Positive = in the money (rates have fallen)
        tuning:    PrepaymentTuning parameters
    
    Returns:
        Annualized CPR from refinancing (before burnout adjustment)
    """
    # Apply elbow shift
    adjusted_incentive = incentive - (tuning.refi_elbow_shift / 100.0)
    
    # Logistic S-curve
    max_cpr = tuning.refi_max_cpr * tuning.refinancing_multiplier
    steepness = tuning.refi_steepness
    midpoint = tuning.refi_midpoint
    
    # Avoid overflow in exp
    x = -steepness * (adjusted_incentive - midpoint)
    x = np.clip(x, -50, 50)
    
    refi_cpr = max_cpr / (1.0 + np.exp(x))
    
    # Below zero incentive (rates have risen), minimal refinancing
    if adjusted_incentive < 0:
        refi_cpr *= max(0, 1.0 + adjusted_incentive * 0.5)  # Decay for disincentive
    
    return max(0, refi_cpr)


def turnover_component(loan_age: int, rate_incentive: float,
                        tuning: PrepaymentTuning,
                        start_month: int = 1) -> float:
    """
    Compute turnover-driven CPR (rate-independent base prepayment).
    
    Turnover captures housing mobility: job changes, family events, 
    upsizing/downsizing, relocation. It's relatively constant but has 
    two key dependencies:
    
    1. LOAN AGE: Turnover follows a hump-shaped pattern — very low for 
       new loans (who just moved in), peaks around month 40, then 
       slowly declines (long-tenured homeowners are less likely to move).
    
    2. LOCK-IN EFFECT: When rates have risen significantly above the 
       borrower's note rate, moving means giving up a below-market 
       mortgage. This suppresses turnover. The "golden handcuffs" effect.
    
    Parameters:
        loan_age:       Current loan age in months
        rate_incentive: Refinancing incentive (negative = rates have risen)
        tuning:         PrepaymentTuning parameters
        start_month:    Calendar month (1-12) for seasonality
    
    Returns:
        Annualized turnover CPR
    """
    base = tuning.turnover_base_cpr * tuning.turnover_multiplier
    
    # Age-dependent hump shape
    peak = tuning.turnover_peak_month
    if loan_age <= peak:
        age_factor = loan_age / peak
    else:
        # Slow decline after peak
        age_factor = 1.0 - 0.3 * (loan_age - peak) / (360 - peak)
        age_factor = max(age_factor, 0.4)  # Floor at 40% of peak
    
    # Lock-in effect: when rates have risen, turnover drops
    # rate_incentive < 0 means rates have risen above note rate
    if rate_incentive < 0:
        lock_in = 1.0 + rate_incentive * tuning.turnover_lockin_sensitivity
        lock_in = max(lock_in, 0.3)  # Floor: turnover never fully stops
    else:
        lock_in = 1.0
    
    # Seasonality
    month_idx = (start_month - 1) % 12
    seasonal = tuning.seasonality[month_idx]
    
    return base * age_factor * lock_in * seasonal


def burnout_factor(cumulative_incentive: float, pool_factor: float,
                    expected_factor: float, tuning: PrepaymentTuning) -> float:
    """
    Compute the burnout multiplier (0 to 1) that attenuates refinancing.
    
    Burnout captures population heterogeneity: a pool of mortgages contains 
    a mix of borrower types, from highly rate-sensitive (will refinance at 
    the first opportunity) to completely rate-insensitive (will never 
    refinance due to credit issues, inertia, or inability to qualify).
    
    As the pool experiences sustained refinancing incentive, the sensitive 
    borrowers prepay first. The remaining pool becomes progressively less 
    responsive — it has "burned out."
    
    We measure burnout by comparing the actual pool factor to what the 
    factor would be under scheduled amortization alone (no prepayment).
    A pool that has experienced heavy prepayment (low factor relative to 
    scheduled) has already lost its sensitive borrowers.
    
    Parameters:
        cumulative_incentive: Sum of positive monthly refinancing incentives 
                              over the pool's life (measures total exposure)
        pool_factor:          Current pool factor (balance / original)
        expected_factor:      Factor under zero voluntary prepayment
        tuning:               PrepaymentTuning parameters
    
    Returns:
        Burnout multiplier between burnout_floor and 1.0
    """
    # Ratio of actual factor to scheduled factor
    # Low ratio = heavy prepayment has occurred = high burnout
    if expected_factor > 0:
        factor_ratio = pool_factor / expected_factor
    else:
        factor_ratio = 0.0
    
    # Exponential decay based on cumulative incentive
    incentive_burnout = np.exp(-tuning.burnout_rate * cumulative_incentive * 
                                tuning.burnout_multiplier)
    
    # Combine: burnout is worse when both cumulative incentive is high 
    # AND the pool has already lost many borrowers
    raw_burnout = factor_ratio * incentive_burnout
    
    # Floor: even a heavily burned-out pool has some refinancing response
    return max(raw_burnout, tuning.burnout_floor)


def aging_ramp(loan_age: int, tuning: PrepaymentTuning) -> float:
    """
    Seasoning ramp: new loans prepay slowly, ramping up to full speed.
    
    This is the generalized PSA ramp. The ramp reflects the empirical 
    observation that borrowers who just closed on a mortgage are very 
    unlikely to prepay in the first few months (they just moved in, 
    incurred closing costs, etc.).
    
    Parameters:
        loan_age: Current loan age in months
        tuning:   PrepaymentTuning parameters
    
    Returns:
        Aging multiplier between 0 and 1
    """
    ramp_months = tuning.aging_ramp_months
    if loan_age >= ramp_months:
        return 1.0 * tuning.aging_multiplier
    else:
        return (loan_age / ramp_months) * tuning.aging_multiplier


# =============================================================================
# Main Model
# =============================================================================

@dataclass
class PrepaymentModelResult:
    """Container for prepayment model output over a projection period."""
    months: np.ndarray
    total_cpr: np.ndarray
    total_smm: np.ndarray
    refi_cpr: np.ndarray
    turnover_cpr: np.ndarray
    burnout_factors: np.ndarray
    aging_factors: np.ndarray
    incentives: np.ndarray
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for display."""
        import pandas as pd
        return pd.DataFrame({
            'month': self.months,
            'total_cpr': self.total_cpr,
            'total_smm': self.total_smm,
            'refi_cpr': self.refi_cpr,
            'turnover_cpr': self.turnover_cpr,
            'burnout': self.burnout_factors,
            'aging': self.aging_factors,
            'incentive': self.incentives
        })


def project_prepayment_rates(
    note_rate: float,
    current_coupon_path: np.ndarray,
    num_months: int,
    loan_age: int = 0,
    original_term: int = 360,
    pool_factor: float = 1.0,
    tuning: PrepaymentTuning = None,
    start_calendar_month: int = 1
) -> PrepaymentModelResult:
    """
    Project monthly prepayment rates using the multi-factor model.
    
    This is the main entry point. For each month in the projection:
        1. Compute refinancing incentive = note_rate - current_coupon
        2. Compute refinancing CPR from the S-curve
        3. Apply burnout adjustment to refinancing
        4. Compute turnover CPR (with lock-in effect)
        5. Apply aging ramp and seasonality
        6. Total CPR = Refinancing × Burnout + Turnover × Aging × Season
        7. Update cumulative state (burnout tracking)
    
    Parameters:
        note_rate:            Pool WAC (gross coupon rate, decimal)
        current_coupon_path:  Array of projected current coupon mortgage rates
                              (one per month). This is the rate a borrower 
                              would get if refinancing TODAY. In static mode,
                              it's constant. In Monte Carlo, it varies by path.
        num_months:           Number of months to project
        loan_age:             Current loan age in months (WALA)
        original_term:        Original loan term in months
        pool_factor:          Current pool factor
        tuning:               PrepaymentTuning parameters (None = defaults)
        start_calendar_month: Starting calendar month (1-12) for seasonality
    
    Returns:
        PrepaymentModelResult with full decomposition
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    
    # Ensure current coupon path is long enough
    cc_path = np.asarray(current_coupon_path)
    if len(cc_path) < num_months:
        cc_path = np.concatenate([cc_path, 
            np.full(num_months - len(cc_path), cc_path[-1])])
    
    # Compute scheduled factor path (no voluntary prepay, just amortization)
    monthly_rate = note_rate / 12.0
    scheduled_factors = np.ones(num_months)
    bal = pool_factor
    for i in range(num_months):
        remaining = original_term - loan_age - i
        if remaining <= 0 or bal <= 0:
            scheduled_factors[i] = 0
            continue
        if monthly_rate > 0:
            pmt = bal * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining))
            sched_prin = pmt - bal * monthly_rate
        else:
            sched_prin = bal / remaining
        bal -= sched_prin
        bal = max(bal, 0)
        scheduled_factors[i] = bal
    
    # Initialize output arrays
    total_cpr = np.zeros(num_months)
    total_smm = np.zeros(num_months)
    refi_cpr_arr = np.zeros(num_months)
    turnover_cpr_arr = np.zeros(num_months)
    burnout_arr = np.zeros(num_months)
    aging_arr = np.zeros(num_months)
    incentives = np.zeros(num_months)
    
    # State tracking
    cumulative_incentive = 0.0
    current_factor = pool_factor
    
    for i in range(num_months):
        age = loan_age + i + 1
        calendar_month = ((start_calendar_month - 1 + i) % 12) + 1
        
        # 1. Refinancing incentive
        incentive = (note_rate - cc_path[i]) * 100  # Convert to percentage points
        incentives[i] = incentive
        
        # Track cumulative positive incentive for burnout
        if incentive > 0:
            cumulative_incentive += incentive
        
        # 2. Refinancing CPR (before burnout)
        raw_refi = refinancing_component(incentive, tuning)
        
        # 3. Burnout factor
        bo = burnout_factor(
            cumulative_incentive, 
            current_factor,
            scheduled_factors[i] if i < len(scheduled_factors) else 0,
            tuning
        )
        burnout_arr[i] = bo
        
        # 4. Adjusted refinancing CPR
        adj_refi = raw_refi * bo
        refi_cpr_arr[i] = adj_refi
        
        # 5. Turnover CPR
        turn = turnover_component(age, incentive, tuning, calendar_month)
        turnover_cpr_arr[i] = turn
        
        # 6. Aging ramp
        age_mult = aging_ramp(age, tuning)
        aging_arr[i] = age_mult
        
        # 7. Total CPR = refinancing (with burnout) + turnover (with aging)
        raw_total = adj_refi + turn * age_mult
        
        # Apply floor and cap
        clamped = np.clip(raw_total, tuning.cpr_floor, tuning.cpr_cap)
        total_cpr[i] = clamped
        
        # Convert to SMM
        smm = 1.0 - (1.0 - clamped) ** (1.0 / 12.0)
        total_smm[i] = smm
        
        # Update factor for burnout tracking
        current_factor *= (1.0 - smm)
        
        # Account for scheduled amortization in factor
        if age < original_term and monthly_rate > 0:
            remaining = original_term - age
            if remaining > 0:
                pmt_rate = monthly_rate / (1 - (1 + monthly_rate) ** (-remaining))
                sched_smm = pmt_rate - monthly_rate
                current_factor *= (1.0 - sched_smm)
    
    return PrepaymentModelResult(
        months=np.arange(1, num_months + 1),
        total_cpr=total_cpr,
        total_smm=total_smm,
        refi_cpr=refi_cpr_arr,
        turnover_cpr=turnover_cpr_arr,
        burnout_factors=burnout_arr,
        aging_factors=aging_arr,
        incentives=incentives
    )


# =============================================================================
# S-Curve Analysis Utilities
# =============================================================================

def plot_scurve_data(tuning: PrepaymentTuning = None,
                      incentive_range: tuple = (-2, 4)) -> dict:
    """
    Generate S-curve data for visualization.
    
    Returns dict with 'incentives' and 'refi_cprs' arrays.
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    
    incentives = np.linspace(incentive_range[0], incentive_range[1], 200)
    cprs = np.array([refinancing_component(inc, tuning) for inc in incentives])
    
    return {
        'incentives': incentives,
        'refi_cprs': cprs * 100,
        'tuning': tuning
    }


def compute_speed_table(note_rate: float, 
                         rate_shifts: np.ndarray,
                         loan_age: int = 30,
                         tuning: PrepaymentTuning = None) -> dict:
    """
    Generate a prepayment speed table across rate scenarios.
    
    This is the standard "speed ramp" or "S-curve table" that traders 
    use to understand a pool's prepayment sensitivity.
    
    Parameters:
        note_rate:    Pool WAC
        rate_shifts:  Array of rate changes from current (e.g., [-200, -100, 0, 100, 200] bps)
        loan_age:     Current loan age
        tuning:       Model tuning parameters
    
    Returns:
        Dict with 'shifts_bps', 'cprs', 'smms'
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    
    base_cc = note_rate - 0.005  # Assume current coupon ≈ note rate - 50bp spread
    
    cprs = []
    for shift in rate_shifts:
        cc = base_cc + shift / 10000.0
        # Project one month at this rate
        result = project_prepayment_rates(
            note_rate=note_rate,
            current_coupon_path=np.array([cc]),
            num_months=1,
            loan_age=loan_age,
            tuning=tuning
        )
        cprs.append(result.total_cpr[0])
    
    cprs = np.array(cprs)
    smms = 1.0 - (1.0 - cprs) ** (1.0 / 12.0)
    
    return {
        'shifts_bps': rate_shifts,
        'cprs': cprs,
        'smms': smms,
        'current_coupons': base_cc + rate_shifts / 10000.0
    }


# =============================================================================
# Model Calibration Utilities
# =============================================================================

def calibration_error(actual_cprs: np.ndarray, model_cprs: np.ndarray) -> dict:
    """
    Compute calibration statistics between actual and model-predicted speeds.
    
    Returns:
        Dict with RMSE, MAE, mean_ratio, and per-month errors
    """
    errors = model_cprs - actual_cprs
    ratios = np.where(actual_cprs > 0.001, model_cprs / actual_cprs, np.nan)
    
    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'mean_ratio': np.nanmean(ratios),
        'median_ratio': np.nanmedian(ratios),
        'errors': errors,
        'ratios': ratios
    }


def generate_synthetic_historical_speeds(
    note_rate: float = 0.065,
    num_months: int = 60,
    rate_path_type: str = 'declining',
    noise_std: float = 0.005
) -> dict:
    """
    Generate synthetic "historical" prepayment data for calibration demos.
    
    Creates realistic-looking CPR data by running the model with known 
    parameters and adding noise. Used for demonstrating calibration 
    methodology when real Fannie Mae data isn't available.
    
    Parameters:
        note_rate:      Pool WAC
        num_months:     Number of months of history
        rate_path_type: 'declining', 'rising', 'volatile', or 'stable'
        noise_std:      Standard deviation of random noise added to CPRs
    
    Returns:
        Dict with 'months', 'actual_cprs', 'rate_path', 'true_tuning'
    """
    # Create a rate path
    base_cc = note_rate - 0.005
    months = np.arange(1, num_months + 1)
    
    if rate_path_type == 'declining':
        rate_path = base_cc - np.linspace(0, 0.02, num_months)
    elif rate_path_type == 'rising':
        rate_path = base_cc + np.linspace(0, 0.015, num_months)
    elif rate_path_type == 'volatile':
        rate_path = base_cc + 0.015 * np.sin(2 * np.pi * months / 24)
    elif rate_path_type == 'stable':
        rate_path = np.full(num_months, base_cc)
    else:
        rate_path = np.full(num_months, base_cc)
    
    # Generate "true" CPRs with known parameters
    true_tuning = PrepaymentTuning(
        refinancing_multiplier=1.05,
        turnover_multiplier=0.95,
        refi_midpoint=1.30,
    )
    
    result = project_prepayment_rates(
        note_rate=note_rate,
        current_coupon_path=rate_path,
        num_months=num_months,
        loan_age=12,
        tuning=true_tuning
    )
    
    # Add noise to simulate "actual" data
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, num_months)
    actual_cprs = np.clip(result.total_cpr + noise, 0.005, 0.60)
    
    return {
        'months': months,
        'actual_cprs': actual_cprs,
        'rate_path': rate_path,
        'true_tuning': true_tuning,
        'model_cprs': result.total_cpr
    }
