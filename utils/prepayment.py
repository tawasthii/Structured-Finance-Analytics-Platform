"""
Prepayment Models & Conventions
================================

This module implements prepayment rate conventions and models used across
the structured products universe.

Core Concepts:
    SMM (Single Monthly Mortality): Fraction of outstanding balance that 
        prepays in a given month. This is the atomic unit — all other 
        conventions convert to/from SMM for cash flow projection.
    
    CPR (Conditional Prepayment Rate): Annualized prepayment rate.
        SMM = 1 - (1 - CPR)^(1/12)
    
    PSA (Public Securities Association): Age-driven benchmark for agency MBS.
        100% PSA = CPR ramps from 0.2% at month 1 to 6% at month 30, 
        then stays at 6%. Other PSA speeds are multiples of this.
    
    HEP (Home Equity Prepayment): For home equity ABS. Similar age ramp
        but different parameters.
    
    ABS (Absolute Prepayment Speed): For auto ABS. Monthly prepayment rate 
        as a percentage of ORIGINAL balance (not current like CPR).
    
    MHP (Manufactured Housing Prepayment): For manufactured housing loans.
    
    PPC (Prospectus Prepayment Curve): For CMBS. Issuer-defined ramp.

This module provides Phase 1 (constant/vector CPR and PSA) and will be 
extended in Phase 2 with the multi-factor econometric model.
"""

import numpy as np
from typing import Union, Optional


# =============================================================================
# Convention Conversions
# =============================================================================

def cpr_to_smm(cpr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert annual CPR to monthly SMM.
    
    SMM = 1 - (1 - CPR)^(1/12)
    
    Intuition: if 6% of the pool prepays annually, the monthly rate 
    is NOT 0.5% — it's slightly less, because each month operates on 
    a smaller remaining balance. The geometric relationship ensures 
    consistency: (1-SMM)^12 = (1-CPR).
    """
    cpr = np.asarray(cpr)
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def smm_to_cpr(smm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert monthly SMM to annual CPR."""
    smm = np.asarray(smm)
    return 1.0 - (1.0 - smm) ** 12.0


def psa_to_cpr(psa_speed: float, month: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert PSA speed to CPR at a given loan age (month).
    
    The PSA benchmark (100% PSA):
        - Months 1-30: CPR increases linearly from 0.2% to 6.0%
          CPR = 0.2% * month (capped at month 30)
        - Month 30+: CPR = 6.0%
    
    At X% PSA, multiply the benchmark CPR by X/100.
    
    Parameters:
        psa_speed: PSA speed as percentage (e.g., 150 for 150% PSA)
        month:     Loan age in months (1-indexed)
    
    Returns:
        CPR as decimal
    """
    month = np.asarray(month, dtype=float)
    # Benchmark 100% PSA
    benchmark_cpr = np.minimum(month * 0.002, 0.06)
    return benchmark_cpr * (psa_speed / 100.0)


def psa_to_smm(psa_speed: float, month: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert PSA speed directly to SMM at given loan age."""
    return cpr_to_smm(psa_to_cpr(psa_speed, month))


def hep_to_cpr(hep_speed: float, month: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Home Equity Prepayment convention.
    
    HEP benchmark (100% HEP):
        - Months 1-10: CPR ramps from 0% to peak
        - Month 10+: flat at peak (typically higher than PSA due to 
          higher turnover in subprime/home equity populations)
    
    Standard HEP peaks at approximately 25-30% CPR.
    """
    month = np.asarray(month, dtype=float)
    peak_cpr = 0.25  # 25% CPR at 100% HEP
    benchmark_cpr = np.minimum(month / 10.0, 1.0) * peak_cpr
    return benchmark_cpr * (hep_speed / 100.0)


def abs_speed_to_smm(abs_speed: float, original_balance: float,
                      current_balance: float) -> float:
    """
    ABS prepayment convention for auto loans.
    
    ABS convention expresses prepayment as % of ORIGINAL balance,
    not current balance. This is fundamentally different from CPR.
    
    Monthly prepayment amount = ABS * original_balance / 12
    SMM equivalent = (ABS * original_balance / 12) / current_balance
    
    As the pool pays down, the same ABS speed implies increasing SMM 
    because the same dollar prepayment is a larger fraction of a 
    smaller remaining balance.
    """
    monthly_prepay = abs_speed * original_balance / 12.0
    if current_balance <= 0:
        return 0.0
    return min(monthly_prepay / current_balance, 1.0)


def mhp_to_cpr(mhp_speed: float, month: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Manufactured Housing Prepayment convention.
    
    MHP benchmark (100% MHP):
        - Months 1-24: CPR ramps from 3.7% to 6%
        - Month 24+: flat at 6%
    
    Manufactured housing prepays more slowly than conventional mortgages
    due to limited refinancing options and different borrower demographics.
    """
    month = np.asarray(month, dtype=float)
    start_cpr = 0.037
    peak_cpr = 0.06
    ramp_months = 24
    
    benchmark_cpr = np.where(
        month <= ramp_months,
        start_cpr + (peak_cpr - start_cpr) * (month / ramp_months),
        peak_cpr
    )
    return benchmark_cpr * (mhp_speed / 100.0)


def ppc_to_cpr(ppc_speed: float, month: Union[int, np.ndarray],
               ramp_months: int = 30, peak_cpr: float = 0.05) -> Union[float, np.ndarray]:
    """
    Prospectus Prepayment Curve for CMBS.
    
    PPC benchmark: linear ramp from 0 to peak_cpr over ramp_months,
    then flat. The peak and ramp are deal-specific.
    """
    month = np.asarray(month, dtype=float)
    benchmark = np.where(
        month <= ramp_months,
        peak_cpr * (month / ramp_months),
        peak_cpr
    )
    return benchmark * (ppc_speed / 100.0)


# =============================================================================
# Prepayment Vector Generators
# =============================================================================

def constant_cpr_vector(cpr: float, num_months: int) -> np.ndarray:
    """Generate a vector of constant CPR values."""
    return np.full(num_months, cpr)


def constant_smm_vector(cpr: float, num_months: int) -> np.ndarray:
    """Generate a vector of constant SMM values from a CPR."""
    smm = cpr_to_smm(cpr)
    return np.full(num_months, smm)


def psa_smm_vector(psa_speed: float, num_months: int,
                    wam: int = None, age: int = 0) -> np.ndarray:
    """
    Generate monthly SMM vector using PSA convention.
    
    Parameters:
        psa_speed:   PSA speed (e.g., 150 for 150% PSA)
        num_months:  Number of months to project
        wam:         Weighted average maturity (remaining months). 
                     If None, defaults to num_months
        age:         Current loan age in months (for seasoned pools)
    
    Returns:
        Array of SMM values, one per month
    """
    months = np.arange(1, num_months + 1) + age
    cprs = psa_to_cpr(psa_speed, months)
    return cpr_to_smm(cprs)


def vector_cpr_to_smm(cpr_vector: np.ndarray) -> np.ndarray:
    """Convert a vector of CPR values to SMM values."""
    return cpr_to_smm(np.asarray(cpr_vector))


# =============================================================================
# Default & Loss Models (Phase 1 — simple constant rate)
# =============================================================================

def cdr_to_mdr(cdr: float) -> float:
    """
    Convert annual CDR (Constant Default Rate) to monthly MDR.
    Same math as CPR/SMM: MDR = 1 - (1-CDR)^(1/12)
    """
    return 1.0 - (1.0 - cdr) ** (1.0 / 12.0)


def constant_default_vector(cdr: float, num_months: int,
                             severity: float = 0.40,
                             recovery_lag: int = 6) -> dict:
    """
    Generate default and loss vectors.
    
    Parameters:
        cdr:           Annual constant default rate (decimal)
        num_months:    Projection length
        severity:      Loss severity (1 - recovery rate)
        recovery_lag:  Months between default and loss realization
    
    Returns:
        Dict with 'mdr' (monthly default rate), 'severity', 'recovery_lag'
    """
    mdr = cdr_to_mdr(cdr)
    return {
        'mdr': np.full(num_months, mdr),
        'severity': severity,
        'recovery_lag': recovery_lag
    }


# =============================================================================
# Speed Equivalents Display
# =============================================================================

def speed_equivalents(smm: float, month: int = 30, 
                       original_balance: float = 100.0,
                       current_balance: float = 80.0) -> dict:
    """
    Given an SMM, compute equivalent speeds in all conventions.
    Useful for comparing across asset classes.
    
    Returns dict with CPR, PSA, HEP, ABS, MHP equivalents.
    """
    cpr = smm_to_cpr(smm)
    
    # PSA equivalent: what PSA speed gives this CPR at this month?
    psa_benchmark_cpr = min(month * 0.002, 0.06)
    psa_equiv = (cpr / psa_benchmark_cpr * 100) if psa_benchmark_cpr > 0 else 0
    
    # HEP equivalent
    peak_cpr_hep = 0.25
    hep_benchmark = min(month / 10.0, 1.0) * peak_cpr_hep
    hep_equiv = (cpr / hep_benchmark * 100) if hep_benchmark > 0 else 0
    
    # ABS equivalent (inverse of abs_speed_to_smm)
    monthly_prepay = smm * current_balance
    abs_equiv = (monthly_prepay * 12.0 / original_balance) if original_balance > 0 else 0
    
    return {
        'SMM': smm,
        'CPR': cpr,
        'PSA': psa_equiv,
        'HEP': hep_equiv,
        'ABS': abs_equiv,
        'month': month
    }
