"""
CMO Structuring & Waterfall Engine
=====================================

This module implements the cash flow allocation rules (waterfalls) for 
Collateralized Mortgage Obligations (CMOs).

A CMO takes a pool of mortgage collateral and redistributes the cash 
flows into tranches with different risk/return profiles. The waterfall 
rules determine which tranche receives principal and interest each month.

Structures Implemented:
    1. Sequential Pay:    Simplest — principal flows to tranches in order (A→B→C→Z)
    2. PAC / Companion:   PAC tranche has a defined sinking fund schedule protected
                          by companion tranches that absorb prepayment variability
    3. TAC:               Like PAC but with a single (lower) collar
    4. Z-Bond (Accrual):  Accrues interest (added to balance) until earlier tranches 
                          pay down, then receives all remaining cash flows
    5. IO / PO Strips:    Interest-Only and Principal-Only stripped from collateral

Key Concepts:
    - All tranches share the SAME collateral pool
    - Principal allocation follows the waterfall rules
    - Interest is paid to each tranche based on its outstanding balance 
      and coupon rate (unless it's a Z-bond or IO/PO)
    - PAC bands define the prepayment speed range over which the PAC 
      schedule is maintained
    - The companion tranche is the residual: it absorbs whatever the PAC 
      doesn't need (either excess prepayment or prepayment shortfall)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class TrancheType(Enum):
    SEQUENTIAL = "sequential"
    PAC = "pac"
    TAC = "tac"
    COMPANION = "companion"
    Z_BOND = "z_bond"
    IO = "io"
    PO = "po"


@dataclass
class Tranche:
    """
    Defines a single CMO tranche.
    
    Attributes:
        name:           Tranche identifier (e.g., 'A', 'B', 'PAC-1')
        tranche_type:   Type of tranche (determines waterfall behavior)
        original_balance: Original face amount
        coupon_rate:    Coupon rate paid to this tranche
        priority:       Payment priority (lower = paid first for sequential)
        pac_schedule:   For PAC tranches: array of monthly scheduled principal
        pac_upper_band: Upper PSA speed for PAC band
        pac_lower_band: Lower PSA speed for PAC band
    """
    name: str
    tranche_type: TrancheType
    original_balance: float
    coupon_rate: float
    priority: int = 0
    pac_schedule: np.ndarray = None
    pac_upper_band: float = 0.0
    pac_lower_band: float = 0.0


@dataclass
class TrancheFlows:
    """Cash flow results for a single tranche."""
    name: str
    tranche_type: str
    original_balance: float
    coupon_rate: float
    months: np.ndarray = None
    beg_balance: np.ndarray = None
    interest: np.ndarray = None
    principal: np.ndarray = None
    total_cashflow: np.ndarray = None
    end_balance: np.ndarray = None
    accrued_interest: np.ndarray = None  # For Z-bonds
    
    @property
    def weighted_avg_life(self) -> float:
        """WAL for this tranche. For IO strips, uses interest-weighted average."""
        if self.tranche_type == 'io':
            # IO has no principal — use interest-weighted average life
            if self.interest is None:
                return 0.0
            total_int = self.interest.sum()
            if total_int <= 0:
                return 0.0
            return np.sum(self.months * self.interest) / total_int / 12.0
        if self.principal is None:
            return 0.0
        total_prin = self.principal.sum()
        if total_prin <= 0:
            return 0.0
        return np.sum(self.months * self.principal) / total_prin / 12.0
    
    @property
    def total_interest(self) -> float:
        return self.interest.sum() if self.interest is not None else 0.0
    
    @property
    def total_principal(self) -> float:
        return self.principal.sum() if self.principal is not None else 0.0
    
    @property
    def window(self) -> tuple:
        """(first_principal_month, last_principal_month) — the payment window."""
        if self.principal is None:
            return (0, 0)
        prin_months = self.months[self.principal > 0.01]
        if len(prin_months) == 0:
            return (0, 0)
        return (int(prin_months[0]), int(prin_months[-1]))
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'month': self.months,
            'beg_balance': self.beg_balance,
            'interest': self.interest,
            'principal': self.principal,
            'total_cashflow': self.total_cashflow,
            'end_balance': self.end_balance
        })


@dataclass
class CMOResult:
    """Container for all tranche flows from a CMO deal."""
    tranches: Dict[str, TrancheFlows]
    collateral_balance: np.ndarray
    collateral_principal: np.ndarray
    collateral_interest: np.ndarray
    num_months: int
    
    def summary(self) -> pd.DataFrame:
        """Summary statistics for all tranches."""
        rows = []
        for name, tf in self.tranches.items():
            window = tf.window
            rows.append({
                'Tranche': name,
                'Type': tf.tranche_type,
                'Orig Balance': tf.original_balance,
                'Coupon (%)': tf.coupon_rate * 100,
                'WAL (Yrs)': tf.weighted_avg_life,
                'Total Interest': tf.total_interest,
                'Total Principal': tf.total_principal,
                'First Prin Mo': window[0],
                'Last Prin Mo': window[1],
            })
        return pd.DataFrame(rows).set_index('Tranche')
    
    def conservation_check(self) -> dict:
        """
        Verify total cash flow conservation.
        
        For deals WITHOUT Z-bonds: 
            sum(tranche principal) = collateral principal
            sum(tranche interest) = collateral interest
            
        For deals WITH Z-bonds:
            sum(tranche principal) > collateral principal (Z-bond converts interest→principal)
            sum(tranche interest) < collateral interest  
            sum(tranche total CF) = collateral total CF  (always conserved)
        """
        total_prin = sum(tf.total_principal for tf in self.tranches.values())
        total_int = sum(tf.total_interest for tf in self.tranches.values())
        total_cf = total_prin + total_int
        coll_cf = self.collateral_principal.sum() + self.collateral_interest.sum()
        
        return {
            'tranche_principal': total_prin,
            'tranche_interest': total_int,
            'tranche_total_cf': total_cf,
            'collateral_principal': self.collateral_principal.sum(),
            'collateral_interest': self.collateral_interest.sum(),
            'collateral_total_cf': coll_cf,
            'cf_error': abs(total_cf - coll_cf),
            'is_conserved': abs(total_cf - coll_cf) < 1.0
        }


# =============================================================================
# PAC Schedule Generation
# =============================================================================

def generate_pac_schedule(collateral_balance: float,
                           collateral_wac: float,
                           collateral_wam: int,
                           collateral_age: int,
                           lower_psa: float,
                           upper_psa: float,
                           pac_balance: float) -> np.ndarray:
    """
    Generate the PAC sinking fund schedule.
    
    The PAC schedule is the MINIMUM of principal at the lower PSA band 
    and principal at the upper PSA band, capped at the PAC balance.
    
    At slow speeds (lower band): collateral generates less principal → PAC 
    schedule tracks this slower pace
    At fast speeds (upper band): collateral generates more principal → PAC 
    schedule tracks this but the MINIMUM constraint from slow speeds limits it
    
    The PAC schedule is the "intersection" that works under both extremes.
    As long as actual prepayment stays within the bands, the PAC tranche 
    receives its scheduled principal — the companion absorbs the variability.
    
    Parameters:
        collateral_balance: Current collateral balance
        collateral_wac:     Weighted average coupon
        collateral_wam:     Remaining months
        collateral_age:     Current loan age
        lower_psa:          Lower PSA band (e.g., 100)
        upper_psa:          Upper PSA band (e.g., 300)
        pac_balance:        PAC tranche balance
    
    Returns:
        Array of monthly scheduled principal for the PAC tranche
    """
    from .prepayment import psa_smm_vector
    from .cashflows import MBSPool, project_mbs_cashflows
    
    pool = MBSPool(
        original_balance=collateral_balance,
        current_balance=collateral_balance,
        wac=collateral_wac,
        net_coupon=collateral_wac,
        wam=collateral_wam,
        age=collateral_age,
        original_term=collateral_wam + collateral_age
    )
    
    # Project at lower band
    cf_lower = project_mbs_cashflows(pool, psa_speed=lower_psa)
    prin_lower = cf_lower.flows['total_principal'].values
    
    # Project at upper band
    cf_upper = project_mbs_cashflows(pool, psa_speed=upper_psa)
    prin_upper = cf_upper.flows['total_principal'].values
    
    # Ensure same length
    n = max(len(prin_lower), len(prin_upper))
    p_low = np.zeros(n)
    p_high = np.zeros(n)
    p_low[:len(prin_lower)] = prin_lower
    p_high[:len(prin_upper)] = prin_upper
    
    # PAC schedule = minimum of lower and upper band principal
    pac_sched = np.minimum(p_low, p_high)
    
    # Cap cumulative schedule at PAC balance
    cumulative = np.cumsum(pac_sched)
    excess = cumulative - pac_balance
    for i in range(len(pac_sched)):
        if excess[i] > 0:
            pac_sched[i] = max(0, pac_sched[i] - excess[i])
            pac_sched[i+1:] = 0
            break
    
    return pac_sched


# =============================================================================
# Waterfall Engines
# =============================================================================

def run_sequential_cmo(tranches: List[Tranche],
                        collateral_principal: np.ndarray,
                        collateral_interest: np.ndarray,
                        collateral_balance_path: np.ndarray) -> CMOResult:
    """
    Run a sequential-pay CMO waterfall.
    
    Rules:
        - Interest: paid to ALL tranches pro-rata based on balance × coupon
          (except Z-bonds, which accrue)
        - Principal: paid to the HIGHEST priority tranche until it's paid off,
          then to the next, and so on
        - Z-bond: accrues interest (added to its balance) until all prior 
          tranches are paid off; then receives principal + accrued
    
    Parameters:
        tranches:                List of Tranche objects, sorted by priority
        collateral_principal:    Monthly principal from collateral
        collateral_interest:     Monthly interest from collateral
        collateral_balance_path: Monthly collateral balance
    
    Returns:
        CMOResult with all tranche cash flows
    """
    n_months = len(collateral_principal)
    tranches_sorted = sorted(tranches, key=lambda t: t.priority)
    
    # Initialize tranche flow containers
    flows = {}
    balances = {}
    accrued = {}
    
    for t in tranches_sorted:
        flows[t.name] = TrancheFlows(
            name=t.name,
            tranche_type=t.tranche_type.value,
            original_balance=t.original_balance,
            coupon_rate=t.coupon_rate,
            months=np.arange(1, n_months + 1),
            beg_balance=np.zeros(n_months),
            interest=np.zeros(n_months),
            principal=np.zeros(n_months),
            total_cashflow=np.zeros(n_months),
            end_balance=np.zeros(n_months),
            accrued_interest=np.zeros(n_months)
        )
        balances[t.name] = t.original_balance
        accrued[t.name] = 0.0
    
    for month in range(n_months):
        available_principal = collateral_principal[month]
        
        # Record beginning balances
        for t in tranches_sorted:
            flows[t.name].beg_balance[month] = balances[t.name]
        
        # Step 1: Interest allocation
        for t in tranches_sorted:
            if balances[t.name] <= 0:
                continue
            
            monthly_interest = balances[t.name] * t.coupon_rate / 12.0
            
            if t.tranche_type == TrancheType.Z_BOND:
                # Check if all prior tranches are paid off
                prior_active = any(
                    balances[pt.name] > 0.01 
                    for pt in tranches_sorted 
                    if pt.priority < t.priority and pt.tranche_type != TrancheType.Z_BOND
                )
                if prior_active:
                    # Accrue: add interest to Z-bond balance
                    accrued[t.name] += monthly_interest
                    balances[t.name] += monthly_interest
                    flows[t.name].interest[month] = 0
                    flows[t.name].accrued_interest[month] = accrued[t.name]
                    # The accrued interest becomes available principal for other tranches
                    available_principal += monthly_interest
                else:
                    # Z-bond is now active: pay interest normally
                    flows[t.name].interest[month] = monthly_interest
            elif t.tranche_type == TrancheType.IO:
                flows[t.name].interest[month] = monthly_interest
            else:
                flows[t.name].interest[month] = monthly_interest
        
        # Step 2: Principal allocation (sequential)
        for t in tranches_sorted:
            if available_principal <= 0:
                break
            if balances[t.name] <= 0.01:
                continue
            if t.tranche_type == TrancheType.IO:
                continue  # IO receives no principal
            
            # Check if prior tranches still active (for Z-bond)
            if t.tranche_type == TrancheType.Z_BOND:
                prior_active = any(
                    balances[pt.name] > 0.01 
                    for pt in tranches_sorted 
                    if pt.priority < t.priority and pt.tranche_type != TrancheType.Z_BOND
                )
                if prior_active:
                    continue  # Z-bond doesn't receive principal yet
            
            # Pay principal to this tranche
            principal_to_pay = min(available_principal, balances[t.name])
            flows[t.name].principal[month] = principal_to_pay
            balances[t.name] -= principal_to_pay
            available_principal -= principal_to_pay
        
        # Step 3: Update end balances and total cash flows
        for t in tranches_sorted:
            flows[t.name].end_balance[month] = balances[t.name]
            flows[t.name].total_cashflow[month] = (
                flows[t.name].interest[month] + flows[t.name].principal[month]
            )
    
    return CMOResult(
        tranches=flows,
        collateral_balance=collateral_balance_path,
        collateral_principal=collateral_principal,
        collateral_interest=collateral_interest,
        num_months=n_months
    )


def run_pac_companion_cmo(pac_tranche: Tranche,
                           companion_tranche: Tranche,
                           other_tranches: List[Tranche],
                           collateral_principal: np.ndarray,
                           collateral_interest: np.ndarray,
                           collateral_balance_path: np.ndarray,
                           pac_schedule: np.ndarray) -> CMOResult:
    """
    Run a PAC/Companion CMO waterfall.
    
    Rules:
        - PAC receives its scheduled principal as long as collateral 
          generates enough (within the PAC bands)
        - Companion absorbs excess principal (fast prepays) or receives 
          reduced principal (slow prepays) — it's the shock absorber
        - If the companion is fully paid down and prepays are still fast,
          the PAC schedule breaks ("busted PAC")
        - Other sequential tranches receive principal after PAC and companion
    
    This structure is the workhorse of the agency CMO market. PAC bonds 
    trade at tighter spreads because of their cash flow stability.
    """
    n_months = len(collateral_principal)
    all_tranches = [pac_tranche, companion_tranche] + other_tranches
    
    # Ensure PAC schedule is long enough
    pac_sched = np.zeros(n_months)
    pac_sched[:min(len(pac_schedule), n_months)] = pac_schedule[:n_months]
    
    # Initialize
    flows = {}
    balances = {}
    
    for t in all_tranches:
        flows[t.name] = TrancheFlows(
            name=t.name,
            tranche_type=t.tranche_type.value,
            original_balance=t.original_balance,
            coupon_rate=t.coupon_rate,
            months=np.arange(1, n_months + 1),
            beg_balance=np.zeros(n_months),
            interest=np.zeros(n_months),
            principal=np.zeros(n_months),
            total_cashflow=np.zeros(n_months),
            end_balance=np.zeros(n_months),
            accrued_interest=np.zeros(n_months)
        )
        balances[t.name] = t.original_balance
    
    for month in range(n_months):
        available_principal = collateral_principal[month]
        
        # Record beginning balances
        for t in all_tranches:
            flows[t.name].beg_balance[month] = balances[t.name]
        
        # Interest allocation
        for t in all_tranches:
            if balances[t.name] <= 0.01:
                continue
            if t.tranche_type == TrancheType.IO:
                flows[t.name].interest[month] = balances[t.name] * t.coupon_rate / 12.0
            elif t.tranche_type != TrancheType.PO:
                flows[t.name].interest[month] = balances[t.name] * t.coupon_rate / 12.0
        
        # Principal allocation: PAC first, then companion, then others
        
        # 1. PAC tranche: gets its scheduled amount (if available)
        pac_name = pac_tranche.name
        if balances[pac_name] > 0.01:
            scheduled = min(pac_sched[month], balances[pac_name])
            pac_prin = min(scheduled, available_principal)
            flows[pac_name].principal[month] = pac_prin
            balances[pac_name] -= pac_prin
            available_principal -= pac_prin
        
        # 2. Companion tranche: gets excess (or nothing if prepays are slow)
        comp_name = companion_tranche.name
        if available_principal > 0 and balances[comp_name] > 0.01:
            comp_prin = min(available_principal, balances[comp_name])
            flows[comp_name].principal[month] = comp_prin
            balances[comp_name] -= comp_prin
            available_principal -= comp_prin
        
        # 3. If companion is exhausted and there's still excess principal,
        #    it goes back to PAC (PAC schedule "busts")
        if available_principal > 0 and balances[pac_name] > 0.01:
            extra_pac = min(available_principal, balances[pac_name])
            flows[pac_name].principal[month] += extra_pac
            balances[pac_name] -= extra_pac
            available_principal -= extra_pac
        
        # 4. Other tranches (sequential among themselves)
        for t in sorted(other_tranches, key=lambda x: x.priority):
            if available_principal <= 0:
                break
            if balances[t.name] <= 0.01:
                continue
            if t.tranche_type == TrancheType.IO:
                continue
            prin = min(available_principal, balances[t.name])
            flows[t.name].principal[month] = prin
            balances[t.name] -= prin
            available_principal -= prin
        
        # Update end balances
        for t in all_tranches:
            flows[t.name].end_balance[month] = balances[t.name]
            flows[t.name].total_cashflow[month] = (
                flows[t.name].interest[month] + flows[t.name].principal[month]
            )
    
    return CMOResult(
        tranches=flows,
        collateral_balance=collateral_balance_path,
        collateral_principal=collateral_principal,
        collateral_interest=collateral_interest,
        num_months=n_months
    )


# =============================================================================
# IO/PO Strip Engine
# =============================================================================

def strip_io_po(collateral_principal: np.ndarray,
                collateral_interest: np.ndarray,
                collateral_balance_path: np.ndarray,
                notional_balance: float) -> CMOResult:
    """
    Create Interest-Only and Principal-Only strips from collateral.
    
    IO Strip: Receives ALL interest from the collateral, no principal.
        - Value DECREASES when prepays speed up (less balance → less interest)
        - Negative duration! This makes IOs a unique hedging instrument.
    
    PO Strip: Receives ALL principal (scheduled + prepayment), no interest.
        - Value INCREASES when prepays speed up (get principal back sooner)
        - Very long effective duration.
    
    Together, IO + PO = the full collateral cash flow. They're just 
    separated into two securities with opposite prepayment sensitivities.
    """
    n_months = len(collateral_principal)
    months = np.arange(1, n_months + 1)
    
    # IO strip
    io_balance = collateral_balance_path.copy()
    io_flows = TrancheFlows(
        name='IO',
        tranche_type='io',
        original_balance=notional_balance,
        coupon_rate=0.0,
        months=months,
        beg_balance=io_balance,
        interest=collateral_interest.copy(),
        principal=np.zeros(n_months),
        total_cashflow=collateral_interest.copy(),
        end_balance=io_balance,
        accrued_interest=np.zeros(n_months)
    )
    
    # PO strip
    po_balance = np.zeros(n_months)
    po_end = np.zeros(n_months)
    remaining = notional_balance
    for i in range(n_months):
        po_balance[i] = remaining
        remaining -= collateral_principal[i]
        remaining = max(remaining, 0)
        po_end[i] = remaining
    
    po_flows = TrancheFlows(
        name='PO',
        tranche_type='po',
        original_balance=notional_balance,
        coupon_rate=0.0,
        months=months,
        beg_balance=po_balance,
        interest=np.zeros(n_months),
        principal=collateral_principal.copy(),
        total_cashflow=collateral_principal.copy(),
        end_balance=po_end,
        accrued_interest=np.zeros(n_months)
    )
    
    return CMOResult(
        tranches={'IO': io_flows, 'PO': po_flows},
        collateral_balance=collateral_balance_path,
        collateral_principal=collateral_principal,
        collateral_interest=collateral_interest,
        num_months=n_months
    )


# =============================================================================
# Deal Construction Helpers
# =============================================================================

def create_sequential_deal(collateral_balance: float,
                            collateral_wac: float,
                            tranche_splits: Dict[str, float],
                            tranche_coupons: Dict[str, float],
                            z_bond_name: str = None) -> List[Tranche]:
    """
    Helper to create a sequential CMO deal.
    
    Parameters:
        collateral_balance: Total collateral balance
        tranche_splits:     Dict of {name: fraction} (must sum to 1.0)
        tranche_coupons:    Dict of {name: coupon_rate}
        z_bond_name:        Name of Z-bond tranche (if any)
    
    Returns:
        List of Tranche objects
    """
    tranches = []
    for i, (name, frac) in enumerate(tranche_splits.items()):
        ttype = TrancheType.Z_BOND if name == z_bond_name else TrancheType.SEQUENTIAL
        tranches.append(Tranche(
            name=name,
            tranche_type=ttype,
            original_balance=collateral_balance * frac,
            coupon_rate=tranche_coupons.get(name, collateral_wac),
            priority=i
        ))
    return tranches
