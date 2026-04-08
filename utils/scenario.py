"""
Scenario Analysis & Performance Attribution
=============================================

This module implements:
    1. Scenario Analysis: P&L under parallel/twist/vol/prepay shocks
    2. Performance Attribution: decompose period P&L into risk factors
    3. Hedging: compute Treasury/swap hedge ratios from partial durations
    4. Portfolio Aggregation: combine analytics across multiple securities

Scenario Analysis:
    - Parallel rate shifts (±50, ±100, ±200bp)
    - Yield curve twists (steepening/flattening)
    - Volatility shocks (±10%, ±25%)
    - Prepayment shocks (±10%, ±25% multiplier)
    - User-defined combined scenarios

Performance Attribution:
    Period P&L is decomposed into:
    1. Carry (coupon income + pull-to-par)
    2. Treasury curve change
    3. Spread change
    4. Prepayment model surprise
    5. Volatility change
    6. Residual (unexplained)

Hedging:
    Given partial durations, compute positions in benchmark 
    Treasuries/swaps that neutralize rate exposure.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .curves import ZeroCurve
from .cashflows import MBSPool, project_mbs_cashflows
from .prepayment_model import PrepaymentTuning, project_prepayment_rates
from .monte_carlo import RateModelParams, SimulationConfig, simulate_rate_paths
from .oas import compute_path_prices, solve_oas, compute_zoas


# =============================================================================
# Scenario Definitions
# =============================================================================

@dataclass
class Scenario:
    """Defines a single scenario for stress testing."""
    name: str
    rate_shift_bps: float = 0.0       # Parallel rate shift
    twist_bps: float = 0.0            # Twist (steepener +, flattener -)
    twist_pivot: float = 5.0          # Pivot tenor for twist
    vol_shift_pct: float = 0.0        # Vol shift as percentage (0.10 = +10%)
    prepay_shift_pct: float = 0.0     # Prepay multiplier shift
    spread_shift_bps: float = 0.0     # OAS shift for P&L calculation


def standard_scenarios() -> List[Scenario]:
    """Generate the standard scenario set used in production risk reports."""
    return [
        Scenario("Base Case"),
        Scenario("+50bp Parallel", rate_shift_bps=50),
        Scenario("+100bp Parallel", rate_shift_bps=100),
        Scenario("+200bp Parallel", rate_shift_bps=200),
        Scenario("-50bp Parallel", rate_shift_bps=-50),
        Scenario("-100bp Parallel", rate_shift_bps=-100),
        Scenario("-200bp Parallel", rate_shift_bps=-200),
        Scenario("Steepener +75bp", twist_bps=75),
        Scenario("Flattener -75bp", twist_bps=-75),
        Scenario("Vol Up 25%", vol_shift_pct=0.25),
        Scenario("Vol Down 25%", vol_shift_pct=-0.25),
        Scenario("Fast Prepay +25%", prepay_shift_pct=0.25),
        Scenario("Slow Prepay -25%", prepay_shift_pct=-0.25),
        Scenario("Bear Steepener", rate_shift_bps=100, twist_bps=50),
        Scenario("Bull Flattener", rate_shift_bps=-100, twist_bps=-50),
        Scenario("Crisis: Rates +200, Vol +50%", rate_shift_bps=200, vol_shift_pct=0.50),
    ]


# =============================================================================
# Scenario P&L Engine
# =============================================================================

@dataclass
class ScenarioResult:
    """Result from a single scenario analysis."""
    scenario: Scenario
    base_price: float
    scenario_price: float
    pnl_dollars: float          # Per $100 face
    pnl_bps: float              # In basis points of face
    scenario_oas: float         # OAS in the scenario
    scenario_wal: float         # WAL in the scenario


def run_scenario(pool: MBSPool, curve: ZeroCurve,
                  base_price: float, base_oas: float,
                  scenario: Scenario,
                  tuning: PrepaymentTuning = None,
                  params: RateModelParams = None,
                  config: SimulationConfig = None) -> ScenarioResult:
    """
    Run a single scenario and compute P&L.
    
    Methodology: 
    1. Apply rate/twist shifts to the curve
    2. Apply vol shifts to model params
    3. Apply prepay shifts to tuning
    4. Regenerate MC paths with shifted inputs
    5. Reprice at the SAME OAS → scenario price
    6. P&L = scenario price - base price
    """
    if tuning is None:
        tuning = PrepaymentTuning()
    if params is None:
        params = RateModelParams()
    if config is None:
        config = SimulationConfig(num_paths=128, num_months=min(pool.wam, 360))
    
    # 1. Shift curve
    shifted_curve = curve
    if scenario.rate_shift_bps != 0:
        shifted_curve = shifted_curve.shift('parallel', shift_bps=scenario.rate_shift_bps)
    if scenario.twist_bps != 0:
        shifted_curve = shifted_curve.shift('twist', shift_bps=scenario.twist_bps,
                                              pivot_tenor=scenario.twist_pivot)
    
    # 2. Shift vol
    shifted_params = RateModelParams(
        a1=params.a1, a2=params.a2, a3=params.a3,
        sigma1=params.sigma1 * (1 + scenario.vol_shift_pct),
        sigma2=params.sigma2 * (1 + scenario.vol_shift_pct),
        sigma3=params.sigma3 * (1 + scenario.vol_shift_pct),
        rho12=params.rho12, rho13=params.rho13, rho23=params.rho23,
        mortgage_spread=params.mortgage_spread,
        mortgage_vol_mult=params.mortgage_vol_mult
    )
    
    # 3. Shift prepay
    shifted_tuning_dict = tuning.to_dict()
    if scenario.prepay_shift_pct != 0:
        shifted_tuning_dict['refinancing_multiplier'] *= (1 + scenario.prepay_shift_pct)
        shifted_tuning_dict['turnover_multiplier'] *= (1 + scenario.prepay_shift_pct)
    shifted_tuning = PrepaymentTuning.from_dict(shifted_tuning_dict)
    
    # 4. Generate paths and price at base OAS
    paths = simulate_rate_paths(shifted_curve, shifted_params, config)
    pvs = compute_path_prices(pool, paths, base_oas, shifted_tuning)
    scenario_price = float(np.mean(pvs)) * 100
    
    # 5. Compute P&L
    pnl = scenario_price - base_price
    
    # 6. Estimate scenario WAL
    sample_wals = []
    for i in range(min(20, paths.num_paths)):
        mtg = paths.mortgage_rates[i, :pool.wam]
        pr = project_prepayment_rates(pool.wac, mtg, pool.wam, pool.age,
                                       pool.original_term, tuning=shifted_tuning)
        cf = project_mbs_cashflows(pool, smm_vector=pr.total_smm)
        sample_wals.append(cf.weighted_avg_life)
    
    return ScenarioResult(
        scenario=scenario,
        base_price=base_price,
        scenario_price=scenario_price,
        pnl_dollars=pnl,
        pnl_bps=pnl * 100,
        scenario_oas=base_oas * 10000,
        scenario_wal=np.mean(sample_wals)
    )


def run_scenario_table(pool: MBSPool, curve: ZeroCurve,
                        base_price: float, base_oas: float,
                        scenarios: List[Scenario] = None,
                        tuning: PrepaymentTuning = None,
                        params: RateModelParams = None,
                        config: SimulationConfig = None) -> pd.DataFrame:
    """Run all scenarios and return a summary DataFrame."""
    if scenarios is None:
        scenarios = standard_scenarios()
    
    results = []
    for scen in scenarios:
        res = run_scenario(pool, curve, base_price, base_oas, scen,
                            tuning, params, config)
        results.append({
            'Scenario': res.scenario.name,
            'Price': res.scenario_price,
            'P&L ($)': res.pnl_dollars,
            'P&L (bps)': res.pnl_bps,
            'WAL': res.scenario_wal,
        })
    
    return pd.DataFrame(results).set_index('Scenario')


# =============================================================================
# Performance Attribution
# =============================================================================

@dataclass
class AttributionResult:
    """P&L decomposition into risk factors."""
    total_pnl: float = 0.0
    carry: float = 0.0            # Coupon income + pull-to-par
    curve_pnl: float = 0.0       # Treasury curve change
    spread_pnl: float = 0.0      # Spread change (OAS widening/tightening)
    prepay_pnl: float = 0.0      # Prepayment model surprise
    vol_pnl: float = 0.0         # Volatility change
    residual: float = 0.0        # Unexplained
    
    def to_dict(self) -> dict:
        return {
            'Total P&L': self.total_pnl,
            'Carry': self.carry,
            'Curve': self.curve_pnl,
            'Spread': self.spread_pnl,
            'Prepay': self.prepay_pnl,
            'Volatility': self.vol_pnl,
            'Residual': self.residual,
        }


def compute_carry(pool: MBSPool, price: float, oas_bps: float,
                    horizon_months: int = 1) -> float:
    """
    Compute carry (income return) over a horizon.
    
    Carry = coupon income + (pull-to-par amortization)
    
    For a premium MBS: coupon > discount rate, but principal returned 
    at par means gradual price erosion. Net carry can be positive or 
    negative depending on the OAS level.
    
    Simplified: Carry ≈ (coupon - funding_cost) × dt + (100 - price) × (1/WAL) × dt
    """
    dt = horizon_months / 12.0
    
    # Coupon income (monthly net coupon rate × face × dt)
    coupon_return = pool.net_coupon * dt * 100  # Per $100 face
    
    # Pull-to-par (amortization of premium/discount)
    # For short horizon, approximate as linear
    # This is the "roll down" effect
    pull_to_par = 0  # Simplified: ignored for monthly horizon
    
    return coupon_return


def attribute_pnl(pool: MBSPool, curve: ZeroCurve,
                    price_begin: float, price_end: float,
                    oas_begin: float, oas_end: float,
                    curve_end: ZeroCurve = None,
                    oa_duration: float = 4.5,
                    spread_duration: float = 5.0,
                    vol_duration: float = 0.0,
                    prepay_duration: float = 0.0,
                    rate_change_bps: float = 0.0,
                    spread_change_bps: float = 0.0,
                    vol_change_pct: float = 0.0,
                    prepay_surprise_pct: float = 0.0,
                    horizon_months: int = 1) -> AttributionResult:
    """
    Decompose period P&L into risk factor contributions.
    
    Uses the duration-based approximation:
        P&L ≈ Carry 
              - OA_Duration × ΔRate 
              - Spread_Duration × ΔOAS 
              - Vol_Duration × ΔVol
              - Prepay_Duration × ΔPrepay
              + Residual
    
    Parameters:
        pool:               MBS pool
        curve:              Beginning-of-period curve
        price_begin/end:    Prices at start and end of period
        oas_begin/end:      OAS at start and end (in decimal)
        oa_duration:        OA Duration
        spread_duration:    Spread duration
        vol/prepay_duration: Other durations
        rate/spread/vol/prepay changes: Observed factor changes
        horizon_months:     Attribution period
    """
    result = AttributionResult()
    
    # Total P&L
    result.total_pnl = price_end - price_begin
    
    # 1. Carry
    result.carry = compute_carry(pool, price_begin, oas_begin * 10000, horizon_months)
    
    # 2. Curve P&L = -Duration × ΔRate
    result.curve_pnl = -oa_duration * (rate_change_bps / 10000) * price_begin
    
    # 3. Spread P&L = -SpreadDuration × ΔOAS
    result.spread_pnl = -spread_duration * (spread_change_bps / 10000) * price_begin
    
    # 4. Vol P&L
    result.vol_pnl = -vol_duration * vol_change_pct * price_begin
    
    # 5. Prepay P&L
    result.prepay_pnl = -prepay_duration * prepay_surprise_pct * price_begin
    
    # 6. Residual
    result.residual = (result.total_pnl - result.carry - result.curve_pnl - 
                        result.spread_pnl - result.vol_pnl - result.prepay_pnl)
    
    return result


# =============================================================================
# Hedging
# =============================================================================

@dataclass
class HedgeResult:
    """Hedge portfolio composition and residual risk."""
    hedge_positions: Dict[str, float]    # {instrument: DV01 or notional}
    hedge_dv01: Dict[str, float]         # DV01 contribution per instrument
    residual_dv01: float                 # Unhedged DV01
    hedge_cost: float                    # Carry cost of hedge
    hedged_duration: float               # Duration after hedging


def compute_hedge_ratios(partial_durations: Dict[str, float],
                           market_value: float,
                           hedge_instruments: Dict[str, float] = None) -> HedgeResult:
    """
    Compute hedge ratios to neutralize rate exposure.
    
    Uses partial durations to determine positions in benchmark 
    Treasuries or swaps at each key rate.
    
    The hedge ratio for each key rate:
        Notional_hedge = -(KRD_mbs × MV_mbs) / Duration_hedge
    
    Parameters:
        partial_durations: Key-rate durations of the MBS {tenor: duration}
        market_value:      Market value of the MBS position
        hedge_instruments: Duration of hedge instruments at each tenor
                          Default: use approximate Treasury durations
    """
    if hedge_instruments is None:
        # Approximate modified durations for on-the-run Treasuries
        hedge_instruments = {
            '2Y': 1.95,
            '5Y': 4.70,
            '10Y': 8.80,
            '30Y': 21.50,
        }
    
    positions = {}
    dv01s = {}
    
    for tenor, krd in partial_durations.items():
        if tenor in hedge_instruments:
            hedge_dur = hedge_instruments[tenor]
            # Notional needed to offset the KRD
            # DV01_mbs = KRD × MV / 10000
            # DV01_hedge = hedge_dur × Notional_hedge / 10000
            # Set equal: Notional_hedge = -(KRD × MV) / hedge_dur
            notional = -(krd * market_value) / hedge_dur
            positions[tenor] = notional
            dv01s[tenor] = krd * market_value / 10000  # MBS DV01 at this KR
    
    # Residual: sum of unhedged KRDs
    hedged_krd_sum = sum(partial_durations.get(t, 0) for t in hedge_instruments)
    total_krd = sum(partial_durations.values())
    residual_dv01 = (total_krd - hedged_krd_sum) * market_value / 10000
    
    # Hedge carry cost (simplified: assume funding at short rate)
    # In practice, this would be the net carry of short Treasury positions
    total_hedge_notional = sum(abs(v) for v in positions.values())
    hedge_cost = total_hedge_notional * 0.001 / 12  # ~10bp annual carry cost
    
    # Hedged duration
    hedged_duration = total_krd - hedged_krd_sum
    
    return HedgeResult(
        hedge_positions=positions,
        hedge_dv01=dv01s,
        residual_dv01=residual_dv01,
        hedge_cost=hedge_cost,
        hedged_duration=hedged_duration
    )


# =============================================================================
# Portfolio Analytics
# =============================================================================

@dataclass 
class SecurityAnalytics:
    """Analytics for a single security in a portfolio."""
    name: str
    cusip: str
    coupon: float
    price: float
    face_value: float
    market_value: float
    oas_bps: float
    zoas_bps: float
    option_cost_bps: float
    oa_duration: float
    oa_convexity: float
    wal: float
    avg_cpr: float
    partial_durations: Dict[str, float] = field(default_factory=dict)
    sector: str = ''


@dataclass
class PortfolioAnalytics:
    """Aggregated portfolio-level analytics."""
    securities: List[SecurityAnalytics]
    
    @property
    def total_market_value(self) -> float:
        return sum(s.market_value for s in self.securities)
    
    @property
    def total_face(self) -> float:
        return sum(s.face_value for s in self.securities)
    
    @property
    def weighted_oas(self) -> float:
        """Market-value weighted OAS."""
        mv = self.total_market_value
        if mv == 0: return 0
        return sum(s.oas_bps * s.market_value for s in self.securities) / mv
    
    @property
    def weighted_duration(self) -> float:
        """Market-value weighted OA Duration."""
        mv = self.total_market_value
        if mv == 0: return 0
        return sum(s.oa_duration * s.market_value for s in self.securities) / mv
    
    @property
    def weighted_convexity(self) -> float:
        mv = self.total_market_value
        if mv == 0: return 0
        return sum(s.oa_convexity * s.market_value for s in self.securities) / mv
    
    @property
    def weighted_wal(self) -> float:
        mv = self.total_market_value
        if mv == 0: return 0
        return sum(s.wal * s.market_value for s in self.securities) / mv
    
    @property
    def portfolio_dv01(self) -> float:
        """Total portfolio DV01 in dollars."""
        return sum(s.oa_duration * s.market_value / 10000 for s in self.securities)
    
    @property
    def aggregate_partial_durations(self) -> Dict[str, float]:
        """Market-value weighted partial durations."""
        mv = self.total_market_value
        if mv == 0: return {}
        result = {}
        for s in self.securities:
            for tenor, dur in s.partial_durations.items():
                if tenor not in result:
                    result[tenor] = 0
                result[tenor] += dur * s.market_value / mv
        return result
    
    def summary_table(self) -> pd.DataFrame:
        """Generate portfolio summary DataFrame."""
        rows = []
        for s in self.securities:
            rows.append({
                'Security': s.name,
                'Coupon (%)': s.coupon * 100,
                'Price': s.price,
                'Face ($K)': s.face_value / 1000,
                'MV ($K)': s.market_value / 1000,
                'OAS (bps)': s.oas_bps,
                'Opt Cost': s.option_cost_bps,
                'OA Dur': s.oa_duration,
                'OA Cvx': s.oa_convexity,
                'WAL': s.wal,
                'CPR (%)': s.avg_cpr * 100,
            })
        
        df = pd.DataFrame(rows).set_index('Security')
        
        # Add portfolio total row
        mv = self.total_market_value
        total = pd.DataFrame([{
            'Security': 'PORTFOLIO',
            'Coupon (%)': sum(s.coupon * s.market_value for s in self.securities) / mv * 100,
            'Price': mv / self.total_face * 100,
            'Face ($K)': self.total_face / 1000,
            'MV ($K)': mv / 1000,
            'OAS (bps)': self.weighted_oas,
            'Opt Cost': sum(s.option_cost_bps * s.market_value for s in self.securities) / mv,
            'OA Dur': self.weighted_duration,
            'OA Cvx': self.weighted_convexity,
            'WAL': self.weighted_wal,
            'CPR (%)': sum(s.avg_cpr * s.market_value for s in self.securities) / mv * 100,
        }]).set_index('Security')
        
        return pd.concat([df, total])
    
    def sector_summary(self) -> pd.DataFrame:
        """Summarize by sector/group."""
        sectors = {}
        for s in self.securities:
            sec = s.sector or 'Unclassified'
            if sec not in sectors:
                sectors[sec] = []
            sectors[sec].append(s)
        
        rows = []
        for sector, secs in sectors.items():
            mv = sum(s.market_value for s in secs)
            rows.append({
                'Sector': sector,
                'Count': len(secs),
                'MV ($K)': mv / 1000,
                'Weight (%)': mv / self.total_market_value * 100,
                'Wtd OAS': sum(s.oas_bps * s.market_value for s in secs) / mv,
                'Wtd Dur': sum(s.oa_duration * s.market_value for s in secs) / mv,
                'Wtd WAL': sum(s.wal * s.market_value for s in secs) / mv,
            })
        return pd.DataFrame(rows).set_index('Sector')


def create_sample_portfolio() -> List[dict]:
    """
    Generate a sample MBS portfolio for demonstration.
    Returns list of dicts with pool characteristics and prices.
    """
    return [
        {'name': 'FNMA 4.5%', 'wac': 0.050, 'net_cpn': 0.045, 'price': 95.0, 
         'face': 2_000_000, 'age': 36, 'sector': 'Discount'},
        {'name': 'FNMA 5.0%', 'wac': 0.055, 'net_cpn': 0.050, 'price': 97.5, 
         'face': 3_000_000, 'age': 24, 'sector': 'Current Coupon'},
        {'name': 'FNMA 5.5%', 'wac': 0.060, 'net_cpn': 0.055, 'price': 99.0, 
         'face': 5_000_000, 'age': 18, 'sector': 'Current Coupon'},
        {'name': 'FNMA 6.0%', 'wac': 0.065, 'net_cpn': 0.060, 'price': 101.0, 
         'face': 4_000_000, 'age': 12, 'sector': 'Premium'},
        {'name': 'FNMA 6.5%', 'wac': 0.070, 'net_cpn': 0.065, 'price': 103.0, 
         'face': 3_000_000, 'age': 6, 'sector': 'Premium'},
        {'name': 'GNMA 5.5%', 'wac': 0.060, 'net_cpn': 0.055, 'price': 99.5, 
         'face': 2_000_000, 'age': 12, 'sector': 'GNMA'},
        {'name': 'FNMA 7.0%', 'wac': 0.075, 'net_cpn': 0.070, 'price': 105.0, 
         'face': 1_000_000, 'age': 48, 'sector': 'Premium'},
    ]
