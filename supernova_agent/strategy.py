"""
Supernova Yield-Farming Agent – Strategy Engine
Scores pools, decides when to migrate, and orchestrates the migration flow.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from config import (
    MIN_APR_IMPROVEMENT, MIN_POOL_TVL_USD, MAX_DEPLOY_RATIO,
    SLIPPAGE_TOLERANCE, WETH, DRY_RUN, MAX_POOL_FEE,
)
from subgraph import BasicPoolData, ConcentratedPoolData

logger = logging.getLogger(__name__)


@dataclass
class PoolScore:
    """Unified score for any pool type."""
    pool_type: str  # "basic" or "concentrated"
    address: str
    symbol: str
    token0: str
    token0_symbol: str
    token1: str
    token1_symbol: str
    tvl_usd: float
    fee_apr: float
    emissions_apr: float
    total_apr: float
    score: float  # composite score used for ranking
    gauge: str = ""
    stable: bool = False
    # Concentrated-specific
    tick: int = 0
    tick_spacing: int = 0
    fee_bps: int = 0


@dataclass
class MigrationPlan:
    """Describes a planned migration from one pool to another."""
    from_pool: Optional[PoolScore]
    to_pool: PoolScore
    reason: str
    apr_improvement: float
    estimated_gas_cost_usd: float = 0.0


@dataclass
class AgentState:
    """Tracks the agent's current position."""
    current_pool: Optional[PoolScore] = None
    current_pool_address: str = ""
    current_gauge_address: str = ""
    current_lp_amount: int = 0
    current_nft_token_id: int = 0  # for concentrated positions
    total_rewards_claimed_usd: float = 0.0
    total_migrations: int = 0
    total_gas_spent_usd: float = 0.0


def score_basic_pool(pool: BasicPoolData, gauge_apr: float = 0.0) -> PoolScore:
    """Score a basic pool based on APR, TVL, and risk factors."""
    emissions_apr = gauge_apr

    total_apr = pool.apr_fee + emissions_apr

    # Composite score: weighted combination
    # - Higher APR is better
    # - Higher TVL is safer (less impermanent loss risk, more stable)
    # - Stable pools get a small bonus for lower IL risk
    tvl_factor = min(1.0, math.log10(max(pool.tvl_usd, 1)) / 7)  # normalize to ~0-1
    stability_bonus = 0.05 if pool.stable else 0.0

    score = total_apr * (0.7 + 0.3 * tvl_factor) + stability_bonus * 100

    return PoolScore(
        pool_type="basic",
        address=pool.address,
        symbol=pool.symbol,
        token0=pool.token0,
        token0_symbol=pool.token0_symbol,
        token1=pool.token1,
        token1_symbol=pool.token1_symbol,
        tvl_usd=pool.tvl_usd,
        fee_apr=pool.apr_fee,
        emissions_apr=emissions_apr,
        total_apr=total_apr,
        score=score,
        gauge=pool.gauge,
        stable=pool.stable,
    )


def score_concentrated_pool(pool: ConcentratedPoolData, gauge_apr: float = 0.0) -> PoolScore:
    """Score a concentrated pool."""
    emissions_apr = gauge_apr
    total_apr = pool.fee_apr + emissions_apr

    tvl_factor = min(1.0, math.log10(max(pool.tvl_usd, 1)) / 7)
    score = total_apr * (0.7 + 0.3 * tvl_factor)

    pair_symbol = f"{pool.token0_symbol}/{pool.token1_symbol}"

    return PoolScore(
        pool_type="concentrated",
        address=pool.address,
        symbol=pair_symbol,
        token0=pool.token0,
        token0_symbol=pool.token0_symbol,
        token1=pool.token1,
        token1_symbol=pool.token1_symbol,
        tvl_usd=pool.tvl_usd,
        fee_apr=pool.fee_apr,
        emissions_apr=emissions_apr,
        total_apr=total_apr,
        score=score,
        gauge=pool.gauge,
        tick=pool.tick,
        tick_spacing=pool.tick_spacing,
        fee_bps=pool.fee,
    )


def rank_pools(
    basic_pools: List[BasicPoolData],
    concentrated_pools: List[ConcentratedPoolData],
    gauge_aprs: Dict[str, float] = None,
) -> List[PoolScore]:
    """
    Score and rank all pools. Returns sorted list (best first).
    gauge_aprs: mapping pool_address -> emissions APR %
    """
    if gauge_aprs is None:
        gauge_aprs = {}

    scored: List[PoolScore] = []

    for bp in basic_pools:
        if bp.tvl_usd < MIN_POOL_TVL_USD:
            continue
        # fee 필터: fee_percent가 비율(0.01=1%)이면 그대로, 퍼센트(1.0=1%)이면 변환
        bp_fee = bp.fee_percent
        if bp_fee >= 1.0:
            bp_fee = bp_fee / 100  # 퍼센트 → 비율
        if bp_fee >= MAX_POOL_FEE:
            logger.debug("Basic pool %s 제외: fee %.4f >= %.4f", bp.symbol, bp_fee, MAX_POOL_FEE)
            continue
        g_apr = gauge_aprs.get(bp.address.lower(), 0.0)
        scored.append(score_basic_pool(bp, g_apr))

    for cp in concentrated_pools:
        if cp.tvl_usd < MIN_POOL_TVL_USD:
            continue
        # fee 필터: Algebra fee는 hundredths of a bip (1e-6)
        # fee=100 → 0.0001 (0.01%), fee=500 → 0.0005 (0.05%), fee=3000 → 0.003 (0.3%)
        cp_fee = cp.fee / 1_000_000
        if cp_fee >= MAX_POOL_FEE:
            logger.debug("CL pool %s/%s 제외: fee %.4f >= %.4f",
                         cp.token0_symbol, cp.token1_symbol, cp_fee, MAX_POOL_FEE)
            continue
        g_apr = gauge_aprs.get(cp.address.lower(), 0.0)
        scored.append(score_concentrated_pool(cp, g_apr))

    scored.sort(key=lambda p: p.score, reverse=True)
    return scored


def should_migrate(
    current: Optional[PoolScore],
    best: PoolScore,
    gas_cost_usd: float = 5.0,
) -> Optional[MigrationPlan]:
    """
    Decide whether to migrate from current pool to the best pool.
    Returns a MigrationPlan if migration is recommended, else None.
    """
    if current is None:
        return MigrationPlan(
            from_pool=None,
            to_pool=best,
            reason="No current position – entering best pool",
            apr_improvement=best.total_apr,
            estimated_gas_cost_usd=gas_cost_usd,
        )

    if best.address.lower() == current.address.lower():
        return None

    apr_diff = best.total_apr - current.total_apr
    if apr_diff < MIN_APR_IMPROVEMENT:
        logger.info(
            "Best pool %s (%.2f%%) vs current %s (%.2f%%) – diff %.2f%% < min %.2f%%",
            best.symbol, best.total_apr,
            current.symbol, current.total_apr,
            apr_diff, MIN_APR_IMPROVEMENT,
        )
        return None

    # Estimate if the APR improvement covers gas costs within 7 days
    # Assume $1000 position for calculation
    daily_improvement_usd = (apr_diff / 100 / 365) * 1000
    days_to_recover_gas = gas_cost_usd / daily_improvement_usd if daily_improvement_usd > 0 else 999
    if days_to_recover_gas > 7:
        logger.info(
            "Migration gas recovery %.1f days > 7 days – skipping",
            days_to_recover_gas,
        )
        return None

    return MigrationPlan(
        from_pool=current,
        to_pool=best,
        reason=(
            f"APR improvement: {current.symbol} ({current.total_apr:.2f}%) → "
            f"{best.symbol} ({best.total_apr:.2f}%) [+{apr_diff:.2f}%]"
        ),
        apr_improvement=apr_diff,
        estimated_gas_cost_usd=gas_cost_usd,
    )


def calculate_tick_range(
    current_tick: int,
    tick_spacing: int,
    range_multiplier: int = 10,
) -> Tuple[int, int]:
    """
    Calculate a tick range for concentrated liquidity around the current tick.
    range_multiplier: how many tick_spacing units above/below current tick.
    """
    lower = (current_tick // tick_spacing - range_multiplier) * tick_spacing
    upper = (current_tick // tick_spacing + range_multiplier) * tick_spacing
    return lower, upper


def estimate_gas_cost_usd(eth_price_usd: float = 2500.0, gas_units: int = 800_000, gas_gwei: float = 20.0) -> float:
    """Estimate total gas cost in USD for a migration (unstake + remove + swap + add + stake)."""
    gas_eth = gas_units * gas_gwei / 1e9
    return gas_eth * eth_price_usd


def format_pool_summary(pool: PoolScore) -> str:
    """Format a pool score for display."""
    lines = [
        f"🏊 {pool.symbol} ({pool.pool_type})",
        f"   Address: {pool.address[:10]}...{pool.address[-6:]}",
        f"   TVL: ${pool.tvl_usd:,.0f}",
        f"   Fee APR: {pool.fee_apr:.2f}%",
        f"   Emissions APR: {pool.emissions_apr:.2f}%",
        f"   Total APR: {pool.total_apr:.2f}%",
        f"   Score: {pool.score:.2f}",
    ]
    return "\n".join(lines)


def format_migration_plan(plan: MigrationPlan) -> str:
    """Format a migration plan for display."""
    lines = [
        "🔄 Migration Plan",
        f"   Reason: {plan.reason}",
        f"   APR Improvement: +{plan.apr_improvement:.2f}%",
        f"   Est. Gas Cost: ${plan.estimated_gas_cost_usd:.2f}",
    ]
    if plan.from_pool:
        lines.append(f"   From: {plan.from_pool.symbol} ({plan.from_pool.total_apr:.2f}%)")
    lines.append(f"   To: {plan.to_pool.symbol} ({plan.to_pool.total_apr:.2f}%)")
    return "\n".join(lines)
