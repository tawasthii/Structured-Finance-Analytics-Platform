"""
Cash Flow Projection Engines
==============================

This module projects month-by-month cash flows for mortgage-backed 
securities and related instruments.

Core Cash Flow Components (for a mortgage pass-through):
    1. Scheduled Interest:   Current balance × (net coupon / 12)
    2. Scheduled Principal:  Amortization payment minus interest
    3. Unscheduled Principal: Prepayment = SMM × (balance - scheduled_principal)
    4. Default:              MDR × current_balance (if modeling defaults)
    5. Loss:                 Default amount × severity (realized after lag)
    6. Recovery:             Default amount × (1 - severity)
    
    Total Principal = Scheduled + Unscheduled (Prepay) + Recovery - Loss
    Total Cash Flow = Interest + Total Principal

Key Insight:
    For a pass-through MBS, the investor receives cash flows NET of 
    servicing and guarantee fees. If the pool WAC is 6.5% and the 
    pass-through rate is 6.0%, the 50bp difference goes to the servicer 
    and guarantor. The cash flow engine must use the NET coupon for 
    interest, but the GROSS coupon (WAC) for amortization scheduling.

Conventions:
    - All rates are annualized decimals (0.06 = 6%)
    - Balances and cash flows in dollars
    - Month indexing: month 1 is the first payment month
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union
from .prepayment import cpr_to_smm, psa_smm_vector, cdr_to_mdr


@dataclass
class MBSPool:
    """
    Represents a mortgage pool (collateral backing an MBS).
    
    Attributes:
        original_balance:  Original face amount ($)
        current_balance:   Current remaining balance ($)
        wac:               Weighted Average Coupon (gross, annualized)
        net_coupon:        Pass-through rate to investors
        wam:               Weighted Average Maturity (remaining months)
        age:               Current loan age in months (WALA)
        original_term:     Original loan term in months (typically 360 or 180)
    """
    original_balance: float
    current_balance: float
    wac: float
    net_coupon: float
    wam: int
    age: int = 0
    original_term: int = 360
    
    @property
    def factor(self) -> float:
        """Pool factor = current_balance / original_balance"""
        return self.current_balance / self.original_balance
    
    @property
    def servicing_fee(self) -> float:
        """Servicing + guarantee fee strip"""
        return self.wac - self.net_coupon
    
    @property
    def monthly_wac(self) -> float:
        """Monthly gross coupon rate"""
        return self.wac / 12.0
    
    @property
    def monthly_net_coupon(self) -> float:
        """Monthly net coupon rate"""
        return self.net_coupon / 12.0


@dataclass
class CashFlowResult:
    """
    Container for projected cash flows.
    
    The DataFrame 'flows' contains monthly projections with columns:
        month, beg_balance, scheduled_interest, scheduled_principal,
        prepayment, default_amount, loss, recovery,
        total_principal, total_cashflow, end_balance, smm, cpr, factor
    """
    flows: pd.DataFrame
    pool: MBSPool
    total_interest: float = 0.0
    total_principal: float = 0.0
    weighted_avg_life: float = 0.0
    
    def __post_init__(self):
        if len(self.flows) > 0:
            self.total_interest = self.flows['net_interest'].sum()
            self.total_principal = self.flows['total_principal'].sum()
            self._compute_wal()
    
    def _compute_wal(self):
        """
        Weighted Average Life = sum(t_i × Principal_i) / sum(Principal_i)
        
        WAL tells you the average time to receive principal back.
        A critical metric because spread calculations interpolate the 
        benchmark curve at the WAL point.
        """
        months = self.flows['month'].values
        principal = self.flows['total_principal'].values
        total_prin = principal.sum()
        if total_prin > 0:
            self.weighted_avg_life = np.sum(months * principal) / total_prin / 12.0
        else:
            self.weighted_avg_life = 0.0


def project_mbs_cashflows(pool: MBSPool,
                           smm_vector: np.ndarray = None,
                           cpr: float = None,
                           psa_speed: float = None,
                           cdr: float = 0.0,
                           severity: float = 0.40,
                           recovery_lag: int = 6) -> CashFlowResult:
    """
    Project month-by-month cash flows for an MBS pass-through.
    
    This is the core engine. For each month:
        1. Compute scheduled payment (standard mortgage amortization)
        2. Split into interest and scheduled principal
        3. Apply prepayment model: prepay = SMM × (balance - sched_principal)
        4. Apply default model: default = MDR × balance
        5. Compute losses and recoveries
        6. Update remaining balance
    
    Parameters:
        pool:        MBSPool object with collateral characteristics
        smm_vector:  Explicit monthly SMM vector (overrides cpr/psa)
        cpr:         Constant CPR (used if smm_vector and psa_speed are None)
        psa_speed:   PSA speed (used if smm_vector is None and cpr is None)
        cdr:         Constant Default Rate (annual, decimal)
        severity:    Loss severity rate
        recovery_lag: Months between default and recovery/loss
    
    Returns:
        CashFlowResult with complete monthly projection
    
    Note: Prepayment and default are COMPETING RISKS — a loan that 
    defaults cannot also prepay. The standard approach is to apply 
    both to the beginning balance, which slightly overstates total 
    terminations, but this is the market convention.
    """
    n_months = pool.wam
    
    # Resolve prepayment vector
    if smm_vector is not None:
        smm_vec = np.asarray(smm_vector[:n_months])
        if len(smm_vec) < n_months:
            # Extend with last value
            smm_vec = np.concatenate([smm_vec, 
                np.full(n_months - len(smm_vec), smm_vec[-1])])
    elif psa_speed is not None:
        smm_vec = psa_smm_vector(psa_speed, n_months, age=pool.age)
    elif cpr is not None:
        smm_vec = np.full(n_months, cpr_to_smm(cpr))
    else:
        smm_vec = np.zeros(n_months)  # No prepayment
    
    # Default rates
    mdr = cdr_to_mdr(cdr) if cdr > 0 else 0.0
    
    # Monthly mortgage rate for amortization (use WAC, not net coupon)
    r_monthly = pool.wac / 12.0
    
    # Initialize arrays
    months = np.arange(1, n_months + 1)
    beg_balance = np.zeros(n_months)
    scheduled_interest = np.zeros(n_months)  # gross interest
    net_interest = np.zeros(n_months)        # net to investor
    scheduled_principal = np.zeros(n_months)
    prepayment = np.zeros(n_months)
    default_amount = np.zeros(n_months)
    loss = np.zeros(n_months)
    recovery = np.zeros(n_months)
    total_principal = np.zeros(n_months)
    total_cashflow = np.zeros(n_months)
    end_balance = np.zeros(n_months)
    
    balance = pool.current_balance
    
    # Compute the level monthly payment for a fully amortizing mortgage
    # PMT = Balance × r / (1 - (1+r)^(-N))
    # This is recalculated based on remaining balance and term
    
    for i in range(n_months):
        if balance <= 0.01:  # Pool paid down
            break
        
        beg_balance[i] = balance
        remaining_months = n_months - i
        
        # Scheduled mortgage payment (P&I) using standard amortization
        if r_monthly > 0:
            monthly_payment = balance * r_monthly / (1 - (1 + r_monthly) ** (-remaining_months))
        else:
            monthly_payment = balance / remaining_months
        
        # Interest (gross and net)
        gross_interest = balance * r_monthly
        scheduled_interest[i] = gross_interest
        net_interest[i] = balance * pool.monthly_net_coupon
        
        # Scheduled principal (amortization)
        sched_prin = monthly_payment - gross_interest
        sched_prin = min(sched_prin, balance)
        scheduled_principal[i] = sched_prin
        
        # Prepayment: SMM applied to balance AFTER scheduled principal
        # This is the market convention: prepayment is voluntary paydown 
        # BEYOND what's scheduled
        surviving_balance = balance - sched_prin
        prepay = smm_vec[i] * surviving_balance
        prepayment[i] = prepay
        
        # Default (applied to beginning balance)
        default_amt = mdr * balance
        default_amount[i] = default_amt
        
        # Losses and recoveries (with lag)
        if i >= recovery_lag and default_amount[i - recovery_lag] > 0:
            loss[i] = default_amount[i - recovery_lag] * severity
            recovery[i] = default_amount[i - recovery_lag] * (1 - severity)
        
        # Total principal to investor
        total_prin = sched_prin + prepay + recovery[i]
        total_principal[i] = total_prin
        
        # Total cash flow to investor (net interest + total principal)
        total_cashflow[i] = net_interest[i] + total_prin
        
        # Update balance
        balance = balance - sched_prin - prepay - default_amt
        balance = max(balance, 0)
        end_balance[i] = balance
    
    # Build DataFrame
    smm_display = np.zeros(n_months)
    cpr_display = np.zeros(n_months)
    smm_display[:len(smm_vec)] = smm_vec[:n_months]
    cpr_display[:len(smm_vec)] = 1.0 - (1.0 - smm_vec[:n_months]) ** 12
    
    df = pd.DataFrame({
        'month': months,
        'beg_balance': beg_balance,
        'scheduled_interest': scheduled_interest,
        'net_interest': net_interest,
        'scheduled_principal': scheduled_principal,
        'prepayment': prepayment,
        'default_amount': default_amount,
        'loss': loss,
        'recovery': recovery,
        'total_principal': total_principal,
        'total_cashflow': total_cashflow,
        'end_balance': end_balance,
        'smm': smm_display,
        'cpr': cpr_display,
        'factor': end_balance / pool.original_balance
    })
    
    # Trim trailing zero rows
    last_nonzero = df[df['beg_balance'] > 0.01].index.max()
    if last_nonzero is not None and not np.isnan(last_nonzero):
        df = df.iloc[:last_nonzero + 1].copy()
    
    return CashFlowResult(flows=df, pool=pool)


# =============================================================================
# Bond Cash Flow Generators (for Treasuries, Agency Debentures)
# =============================================================================

def treasury_cashflows(face: float, coupon_rate: float, 
                        maturity_years: float, freq: int = 2) -> tuple:
    """
    Generate cash flow schedule for a Treasury note/bond.
    
    Parameters:
        face:           Face value
        coupon_rate:    Annual coupon rate (decimal)
        maturity_years: Time to maturity in years
        freq:           Coupon frequency (2 = semi-annual)
    
    Returns:
        (payment_times, cashflows) — times in years, cashflows in dollars
    """
    dt = 1.0 / freq
    payment_times = np.arange(dt, maturity_years + dt/2, dt)
    payment_times = np.round(payment_times, 6)
    
    coupon = face * coupon_rate / freq
    cashflows = np.full(len(payment_times), coupon)
    cashflows[-1] += face  # Add principal at maturity
    
    return payment_times, cashflows


def amortizing_swap_cashflows(notional_schedule: np.ndarray,
                               fixed_rate: float,
                               float_rates: np.ndarray = None,
                               freq: int = 2) -> dict:
    """
    Generate cash flows for an amortizing interest rate swap.
    
    An amortizing swap has a declining notional that typically matches 
    a mortgage amortization schedule. The fixed payer pays fixed_rate 
    on the declining notional; the floating payer pays float_rate.
    
    Parameters:
        notional_schedule: Array of notional amounts for each period
        fixed_rate:        Annual fixed rate
        float_rates:       Array of floating rates for each period
                          (if None, only fixed leg is returned)
        freq:              Payment frequency
    
    Returns:
        Dict with 'fixed_leg', 'float_leg', 'net' cash flow arrays
    """
    n_periods = len(notional_schedule)
    dt = 1.0 / freq
    
    fixed_cf = notional_schedule * fixed_rate * dt
    
    if float_rates is not None:
        float_cf = notional_schedule * float_rates * dt
    else:
        float_cf = np.zeros(n_periods)
    
    return {
        'payment_times': np.arange(1, n_periods + 1) * dt,
        'notional': notional_schedule,
        'fixed_leg': fixed_cf,
        'float_leg': float_cf,
        'net': fixed_cf - float_cf  # From fixed payer's perspective
    }


# =============================================================================
# Yield-to-Maturity and Price Calculations
# =============================================================================

def bond_price(face: float, coupon_rate: float, ytm: float,
               maturity_years: float, freq: int = 2) -> float:
    """
    Price a bullet bond given yield-to-maturity.
    
    P = sum(C/(1+y/f)^t) + F/(1+y/f)^n
    
    where C = coupon, y = YTM, f = frequency, n = total periods
    """
    n_periods = int(maturity_years * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq
    
    if abs(y) < 1e-10:
        return coupon * n_periods + face
    
    pv_coupons = coupon * (1 - (1 + y) ** (-n_periods)) / y
    pv_face = face / (1 + y) ** n_periods
    
    return pv_coupons + pv_face


def mbs_price_from_cashflows(cf_result: CashFlowResult,
                              discount_rate: float) -> float:
    """
    Price an MBS from projected cash flows using a flat discount rate.
    
    This is a simplified pricing — proper pricing uses the zero curve.
    
    Parameters:
        cf_result:      CashFlowResult from project_mbs_cashflows
        discount_rate:  Annual discount rate (decimal)
    
    Returns:
        Dollar price per unit of current face
    """
    flows = cf_result.flows
    monthly_rate = discount_rate / 12.0
    
    months = flows['month'].values
    cashflows = flows['total_cashflow'].values
    
    discount_factors = (1 + monthly_rate) ** (-months)
    pv = np.sum(cashflows * discount_factors)
    
    return pv / cf_result.pool.current_balance * 100  # Price per $100 face


def mbs_yield(cf_result: CashFlowResult, price: float,
              initial_guess: float = 0.05) -> float:
    """
    Compute yield (cash flow yield) of an MBS given price.
    
    This is the constant monthly discount rate that equates PV of 
    projected cash flows to the market price. It's an IRR calculation.
    
    Note: This yield is conditional on the assumed prepayment speed.
    Change the speed, and the yield changes — that's why OAS exists.
    
    Parameters:
        cf_result:     CashFlowResult from project_mbs_cashflows  
        price:         Market price per $100 face
        initial_guess: Starting point for solver
    
    Returns:
        Annual yield (bond-equivalent, semi-annual compounding)
    """
    from scipy.optimize import brentq
    
    flows = cf_result.flows
    months = flows['month'].values
    cashflows = flows['total_cashflow'].values
    
    target_pv = price / 100.0 * cf_result.pool.current_balance
    
    def pv_diff(annual_rate):
        monthly_rate = annual_rate / 12.0
        if monthly_rate <= -1:
            return 1e10
        dfs = (1 + monthly_rate) ** (-months)
        return np.sum(cashflows * dfs) - target_pv
    
    try:
        y = brentq(pv_diff, -0.05, 0.50, xtol=1e-8)
        # Convert monthly to bond-equivalent yield
        # BEY = 2 * [(1 + monthly)^6 - 1]
        monthly = y / 12.0
        bey = 2.0 * ((1 + monthly) ** 6 - 1)
        return bey
    except ValueError:
        return np.nan
