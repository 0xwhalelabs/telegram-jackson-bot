"""
Supernova Yield-Farming Agent – Main Agent Loop
Continuously monitors pools, scores them, and migrates liquidity
to the highest-yield pool with acceptable fees.
"""
import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional

from config import (
    SCAN_INTERVAL_SECONDS, DRY_RUN, MIN_POOL_TVL_USD,
    WALLET_ADDRESS, WETH, GAUGE_MANAGER, GEMINI_API_KEY,
)
from state import save_state, load_state, state_to_dict
from brain import (
    analyze_pools_and_decide, evaluate_migration_safety,
    enforce_fee_rule,
)
from subgraph import (
    BasicPoolData, ConcentratedPoolData,
    fetch_basic_pools, fetch_concentrated_pools,
)
from chain import (
    get_w3, get_wallet, get_eth_balance, get_token_balance,
    get_gauge_for_pool, get_gauge_info,
    add_liquidity_basic, remove_liquidity_basic,
    stake_in_gauge, unstake_from_gauge, claim_gauge_rewards,
    swap_tokens, get_pair_address, get_token_info,
    mint_concentrated_position, remove_concentrated_position,
    get_pool_global_state, ensure_approval,
)
from strategy import (
    PoolScore, MigrationPlan, AgentState,
    rank_pools, should_migrate, calculate_tick_range,
    estimate_gas_cost_usd, format_pool_summary, format_migration_plan,
)
from notifier import (
    send_message, send_migration_alert, send_status_report,
    send_error_alert, send_startup_message, is_paused,
    register_callbacks, build_telegram_app,
)

logger = logging.getLogger(__name__)

# ── Custom Pool Deployers by tick spacing ─────────────────────
TICK_SPACING_DEPLOYERS: Dict[int, str] = {
    1: "0xc815b4e0abae3155f8f4f9e404f17c9fa6928eb8",
    10: "0x1c798614ef4c6a8f8a1aab25785714933e59e963",
    50: "0x44acd9579650d300ebbeac2e483b97fdcacdddc0",
    100: "0xd7b7cc843331cbdc857d5e7615d320b8b4ac090e",
    200: "0x42f5ecd6497d42e093de05bec73e33ceb82493da",
}
DEFAULT_DEPLOYER: str = "0x45bc0f9855a626743d57d37d989f3b9462deba45"  # poolDeployer


def _get_deployer_for_tick_spacing(tick_spacing: int) -> str:
    """Return the correct deployer contract for a given tick spacing."""
    return TICK_SPACING_DEPLOYERS.get(tick_spacing, DEFAULT_DEPLOYER)


# ── Global Agent State ────────────────────────────────────────
state = AgentState()
_ranked_pools: List[PoolScore] = []
_last_scan_time: float = 0


def _persist_state() -> None:
    """Save current agent state to disk."""
    pool = state.current_pool
    save_state(state_to_dict(
        current_pool_address=state.current_pool_address,
        current_pool_type=pool.pool_type if pool else "",
        current_pool_symbol=pool.symbol if pool else "",
        current_pool_apr=pool.total_apr if pool else 0.0,
        current_gauge_address=state.current_gauge_address,
        current_lp_amount=state.current_lp_amount,
        current_nft_token_id=state.current_nft_token_id,
        total_migrations=state.total_migrations,
        total_gas_spent_usd=state.total_gas_spent_usd,
        total_rewards_claimed_usd=state.total_rewards_claimed_usd,
    ))


def _restore_state() -> None:
    """Restore agent state from disk."""
    data = load_state()
    if not data:
        return
    state.current_pool_address = data.get("current_pool_address", "")
    state.current_gauge_address = data.get("current_gauge_address", "")
    state.current_lp_amount = int(data.get("current_lp_amount", 0))
    state.current_nft_token_id = int(data.get("current_nft_token_id", 0))
    state.total_migrations = int(data.get("total_migrations", 0))
    state.total_gas_spent_usd = float(data.get("total_gas_spent_usd", 0.0))
    state.total_rewards_claimed_usd = float(data.get("total_rewards_claimed_usd", 0.0))
    # Reconstruct a minimal PoolScore for the current pool
    sym = data.get("current_pool_symbol", "")
    if state.current_pool_address and sym:
        state.current_pool = PoolScore(
            pool_type=data.get("current_pool_type", "basic"),
            address=state.current_pool_address,
            symbol=sym,
            token0="", token0_symbol="",
            token1="", token1_symbol="",
            tvl_usd=0.0,
            fee_apr=0.0,
            emissions_apr=0.0,
            total_apr=float(data.get("current_pool_apr", 0.0)),
            score=0.0,
            gauge=state.current_gauge_address,
        )
        logger.info("Restored position: %s (%.2f%% APR)", sym, state.current_pool.total_apr)
    else:
        logger.info("No previous position found")


# ── Pool Scanning ─────────────────────────────────────────────

async def scan_all_pools() -> List[PoolScore]:
    """Fetch and rank all pools from subgraphs + on-chain gauge data."""
    global _ranked_pools, _last_scan_time

    logger.info("Scanning all pools...")

    basic_pools, concentrated_pools = await asyncio.gather(
        fetch_basic_pools(),
        fetch_concentrated_pools(),
    )

    logger.info("Found %d basic pools, %d concentrated pools",
                len(basic_pools), len(concentrated_pools))

    # Fetch gauge APRs for pools that have gauges
    gauge_aprs: Dict[str, float] = {}
    all_pool_addresses = (
        [p.address for p in basic_pools if p.tvl_usd >= MIN_POOL_TVL_USD] +
        [p.address for p in concentrated_pools if p.tvl_usd >= MIN_POOL_TVL_USD]
    )

    for addr in all_pool_addresses:
        try:
            gauge_addr = get_gauge_for_pool(addr)
            if gauge_addr and gauge_addr != "0x" + "0" * 40:
                info = get_gauge_info(gauge_addr)
                reward_rate = info.get("reward_rate", 0)
                total_supply = info.get("total_supply", 0)
                if total_supply > 0 and reward_rate > 0:
                    # Approximate emissions APR
                    # reward_rate is tokens/second, annualize it
                    annual_rewards = reward_rate * 365 * 24 * 3600
                    # This is a rough estimate – proper calculation needs token price
                    emissions_apr = (annual_rewards / total_supply) * 100
                    gauge_aprs[addr.lower()] = min(emissions_apr, 10000)  # cap at 10000%
        except Exception as e:
            logger.debug("Failed to get gauge for %s: %s", addr[:10], e)

    ranked = rank_pools(basic_pools, concentrated_pools, gauge_aprs)
    _ranked_pools = ranked
    _last_scan_time = time.time()

    if ranked:
        logger.info("Top 5 pools:")
        for i, p in enumerate(ranked[:5]):
            logger.info("  #%d: %s – APR: %.2f%% (fee: %.2f%% + emissions: %.2f%%) TVL: $%.0f",
                        i + 1, p.symbol, p.total_apr, p.fee_apr, p.emissions_apr, p.tvl_usd)

    return ranked


# ── Migration Execution ──────────────────────────────────────

async def execute_migration(plan: MigrationPlan) -> bool:
    """
    Execute a full migration:
    1. Claim rewards from current gauge
    2. Unstake from current gauge
    3. Remove liquidity from current pool
    4. Swap tokens if needed for the new pool
    5. Add liquidity to new pool
    6. Stake in new gauge
    """
    logger.info("Executing migration: %s", plan.reason)

    try:
        # Step 1 & 2: Exit current position
        if plan.from_pool and state.current_gauge_address:
            zero_addr = "0x" + "0" * 40
            if state.current_gauge_address != zero_addr:
                logger.info("Step 1: Claiming rewards from gauge %s", state.current_gauge_address[:10])
                try:
                    claim_gauge_rewards(state.current_gauge_address)
                except Exception as e:
                    logger.warning("Claim rewards failed (continuing): %s", e)

                if state.current_lp_amount > 0:
                    logger.info("Step 2: Unstaking %d LP from gauge", state.current_lp_amount)
                    unstake_from_gauge(state.current_gauge_address, state.current_lp_amount)

            # Step 3: Remove liquidity
            if plan.from_pool.pool_type == "basic" and state.current_lp_amount > 0:
                logger.info("Step 3: Removing basic liquidity")
                remove_liquidity_basic(
                    plan.from_pool.token0,
                    plan.from_pool.token1,
                    plan.from_pool.stable,
                    state.current_lp_amount,
                )
            elif plan.from_pool.pool_type == "concentrated" and state.current_nft_token_id > 0:
                logger.info("Step 3: Removing concentrated liquidity (NFT #%d)", state.current_nft_token_id)
                # Would need to get liquidity amount from position
                # For now, this is a placeholder
                pass

        # Step 4: Swap tokens to match new pool's requirements
        target = plan.to_pool
        wallet = get_wallet()

        # Get current balances of target pool tokens
        bal0 = get_token_balance(target.token0)
        bal1 = get_token_balance(target.token1)

        info0 = get_token_info(target.token0)
        info1 = get_token_info(target.token1)
        dec0 = info0["decimals"]
        dec1 = info1["decimals"]

        logger.info("Token balances: %s=%s, %s=%s",
                     target.token0_symbol, bal0 / 10**dec0,
                     target.token1_symbol, bal1 / 10**dec1)

        # If we have zero of one token, swap half of the other
        if bal0 > 0 and bal1 == 0:
            swap_amount = bal0 // 2
            logger.info("Step 4: Swapping %s %s → %s",
                        swap_amount / 10**dec0, target.token0_symbol, target.token1_symbol)
            swap_tokens(target.token0, target.token1, swap_amount, target.stable)
            bal0 = get_token_balance(target.token0)
            bal1 = get_token_balance(target.token1)
        elif bal1 > 0 and bal0 == 0:
            swap_amount = bal1 // 2
            logger.info("Step 4: Swapping %s %s → %s",
                        swap_amount / 10**dec1, target.token1_symbol, target.token0_symbol)
            swap_tokens(target.token1, target.token0, swap_amount, target.stable)
            bal0 = get_token_balance(target.token0)
            bal1 = get_token_balance(target.token1)

        if bal0 == 0 and bal1 == 0:
            logger.error("No tokens available for new position")
            return False

        # Step 5: Add liquidity to new pool
        if target.pool_type == "basic":
            logger.info("Step 5: Adding basic liquidity (%s + %s)",
                        target.token0_symbol, target.token1_symbol)
            tx_hash = add_liquidity_basic(
                target.token0, target.token1, target.stable,
                bal0, bal1,
            )

            # Get LP balance
            pair_addr = get_pair_address(target.token0, target.token1, target.stable)
            lp_balance = get_token_balance(pair_addr)

            # Step 6: Stake in gauge
            gauge_addr = get_gauge_for_pool(pair_addr)
            zero_addr = "0x" + "0" * 40
            if gauge_addr and gauge_addr != zero_addr and lp_balance > 0:
                logger.info("Step 6: Staking %d LP in gauge %s", lp_balance, gauge_addr[:10])
                stake_in_gauge(gauge_addr, lp_balance, pair_addr)
                state.current_gauge_address = gauge_addr
                state.current_lp_amount = lp_balance
            else:
                state.current_gauge_address = ""
                state.current_lp_amount = lp_balance

            state.current_pool_address = pair_addr

        elif target.pool_type == "concentrated":
            logger.info("Step 5: Adding concentrated liquidity")
            pool_state = get_pool_global_state(target.address)
            current_tick = pool_state["tick"]
            tick_lower, tick_upper = calculate_tick_range(
                current_tick, target.tick_spacing, range_multiplier=10
            )

            # Determine deployer based on tick spacing
            deployer = _get_deployer_for_tick_spacing(target.tick_spacing)
            logger.info("Using deployer %s for tick_spacing=%d", deployer[:10], target.tick_spacing)

            tx_hash = mint_concentrated_position(
                target.token0, target.token1, deployer,
                tick_lower, tick_upper,
                bal0, bal1,
            )
            state.current_pool_address = target.address

        # Update state
        state.current_pool = target
        state.total_migrations += 1
        _persist_state()

        # Send notification
        from_symbol = plan.from_pool.symbol if plan.from_pool else "None"
        from_apr = plan.from_pool.total_apr if plan.from_pool else 0.0
        await send_migration_alert(
            from_symbol=from_symbol,
            to_symbol=target.symbol,
            from_apr=from_apr,
            to_apr=target.total_apr,
            apr_diff=plan.apr_improvement,
            gas_cost=plan.estimated_gas_cost_usd,
            tx_hash=tx_hash if not DRY_RUN else "",
        )

        logger.info("Migration complete: now in %s (%.2f%% APR)", target.symbol, target.total_apr)
        return True

    except Exception as e:
        error_msg = f"Migration failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        await send_error_alert(str(e))
        return False


# ── Telegram Command Callbacks ────────────────────────────────

async def _handle_status(update, context):
    if not _ranked_pools:
        await update.message.reply_text("No pool data yet. Scanning...")
        return

    current = state.current_pool
    top_text = ""
    for i, p in enumerate(_ranked_pools[:5]):
        marker = " ← YOU" if current and p.address.lower() == current.address.lower() else ""
        top_text += f"#{i+1} {p.symbol}: {p.total_apr:.2f}% APR (${p.tvl_usd:,.0f}){marker}\n"

    await send_status_report(
        current_pool=current.symbol if current else "None",
        current_apr=current.total_apr if current else 0.0,
        tvl=current.tvl_usd if current else 0.0,
        earned_rewards=0.0,
        total_migrations=state.total_migrations,
        total_gas_spent=state.total_gas_spent_usd,
        top_pools_text=top_text,
    )


async def _handle_pools(update, context):
    if not _ranked_pools:
        await update.message.reply_text("No pool data yet.")
        return

    text = "<b>🏆 Top 10 Pools by Score</b>\n\n"
    for i, p in enumerate(_ranked_pools[:10]):
        text += (
            f"<b>#{i+1} {p.symbol}</b> ({p.pool_type})\n"
            f"  APR: {p.total_apr:.2f}% (fee {p.fee_apr:.2f}% + em {p.emissions_apr:.2f}%)\n"
            f"  TVL: ${p.tvl_usd:,.0f} | Score: {p.score:.1f}\n\n"
        )
    await send_message(text)


async def _handle_migrate(update, context):
    ranked = await scan_all_pools()
    if not ranked:
        await update.message.reply_text("No pools found.")
        return

    best = ranked[0]
    plan = should_migrate(state.current_pool, best)
    if plan is None:
        await update.message.reply_text(
            f"No migration needed. Current pool is optimal or close enough.\n"
            f"Best: {best.symbol} ({best.total_apr:.2f}%)"
        )
        return

    await update.message.reply_text(format_migration_plan(plan))
    success = await execute_migration(plan)
    if success:
        await update.message.reply_text("✅ Migration executed successfully!")
    else:
        await update.message.reply_text("❌ Migration failed. Check logs.")


# ── Main Agent Loop ───────────────────────────────────────────

async def agent_loop():
    """Main loop: scan → score → decide → migrate → sleep → repeat."""
    logger.info("Starting Supernova Yield Agent (DRY_RUN=%s)", DRY_RUN)

    # Restore previous state
    _restore_state()

    await send_startup_message()

    # Register Telegram callbacks
    register_callbacks(
        status_cb=_handle_status,
        pools_cb=_handle_pools,
        migrate_cb=_handle_migrate,
    )

    cycle = 0
    while True:
        cycle += 1
        try:
            if is_paused():
                logger.info("Agent is paused. Sleeping %ds...", SCAN_INTERVAL_SECONDS)
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                continue

            logger.info("=== Scan Cycle #%d ===", cycle)

            # 1. Scan and rank pools
            ranked = await scan_all_pools()
            if not ranked:
                logger.warning("No pools found, retrying in %ds", SCAN_INTERVAL_SECONDS)
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                continue

            best = ranked[0]
            gas_cost = estimate_gas_cost_usd()

            # 2. Gemini AI가 모든 판단 수행
            if GEMINI_API_KEY:
                logger.info("🧠 Gemini AI에게 판단 요청 중...")
                ai_decision = await analyze_pools_and_decide(
                    ranked_pools=ranked,
                    current_pool=state.current_pool,
                    gas_price_gwei=20.0,
                    eth_balance=0.0,
                )

                decision = ai_decision.get("decision", "hold")
                reasoning = ai_decision.get("reasoning", "")
                confidence = ai_decision.get("confidence", 0)
                risk = ai_decision.get("risk_assessment", "unknown")
                insight = ai_decision.get("market_insight", "")

                logger.info("🧠 Gemini 판단: %s (신뢰도: %.0f%%, 리스크: %s)",
                            decision, confidence * 100, risk)
                logger.info("🧠 근거: %s", reasoning)
                if insight:
                    logger.info("🧠 시장 분석: %s", insight)

                # Telegram으로 AI 판단 결과 전송
                ai_msg = (
                    f"🧠 <b>Gemini AI 판단</b>\n"
                    f"결정: <b>{decision.upper()}</b> (신뢰도: {confidence*100:.0f}%)\n"
                    f"리스크: {risk}\n"
                    f"근거: {reasoning}\n"
                )
                if insight:
                    ai_msg += f"시장 분석: {insight}\n"
                await send_message(ai_msg)

                if decision == "migrate":
                    target_addr = ai_decision.get("target_pool", "")
                    # AI가 추천한 풀 찾기
                    target_pool = None
                    for p in ranked:
                        if p.address.lower() == target_addr.lower():
                            target_pool = p
                            break
                    if target_pool is None:
                        target_pool = best  # AI가 주소를 못 줬으면 1등 풀 사용

                    # fee 규칙 강제 적용 (AI 판단과 무관)
                    if not enforce_fee_rule(target_pool):
                        logger.warning("🚫 AI가 추천한 풀 %s의 fee가 너무 높아 차단됨", target_pool.symbol)
                        await send_message(f"🚫 AI 추천 풀 <b>{target_pool.symbol}</b> fee 규칙 위반으로 차단")
                    else:
                        # 마이그레이션 전 최종 안전 검토
                        safety = await evaluate_migration_safety(
                            from_pool=state.current_pool,
                            to_pool=target_pool,
                            gas_cost_usd=gas_cost,
                        )
                        if safety.get("decision") == "migrate":
                            plan = MigrationPlan(
                                from_pool=state.current_pool,
                                to_pool=target_pool,
                                reason=f"🧠 AI 판단: {reasoning}",
                                apr_improvement=target_pool.total_apr - (state.current_pool.total_apr if state.current_pool else 0),
                                estimated_gas_cost_usd=gas_cost,
                            )
                            logger.info("Migration approved by AI:\n%s", format_migration_plan(plan))
                            success = await execute_migration(plan)
                            if success:
                                state.total_gas_spent_usd += gas_cost
                                _persist_state()
                        else:
                            logger.info("🧠 최종 검토에서 이동 취소: %s", safety.get("reasoning", ""))
                            await send_message(f"🧠 최종 검토 결과 이동 취소: {safety.get('reasoning', '')}")

                elif decision == "exit":
                    logger.info("🧠 AI가 포지션 퇴장 권고")
                    await send_message("🧠 AI가 현재 풀에서 퇴장을 권고합니다")
                    # 퇴장 로직은 현재 풀에서 유동성 제거만 수행

                else:  # hold
                    logger.info("🧠 AI 판단: 현재 포지션 유지")

            else:
                # Gemini 없으면 기존 규칙 기반 로직
                plan = should_migrate(state.current_pool, best, gas_cost)
                if plan:
                    # fee 규칙 강제 적용
                    if not enforce_fee_rule(plan.to_pool):
                        logger.warning("🚫 대상 풀 fee 규칙 위반으로 이동 취소")
                    else:
                        logger.info("Migration recommended:\n%s", format_migration_plan(plan))
                        success = await execute_migration(plan)
                        if success:
                            state.total_gas_spent_usd += gas_cost
                            _persist_state()
                else:
                    logger.info("No migration needed. Current position is optimal.")

            # Update current pool's APR from fresh data if we're in a pool
            if state.current_pool:
                for p in ranked:
                    if p.address.lower() == state.current_pool.address.lower():
                        state.current_pool = p
                        break

            # 3. Periodic status report (every 12 cycles ≈ 1 hour at 5min intervals)
            if cycle % 12 == 0:
                current = state.current_pool
                top_text = ""
                for i, p in enumerate(ranked[:5]):
                    marker = " ← YOU" if current and p.address.lower() == current.address.lower() else ""
                    top_text += f"#{i+1} {p.symbol}: {p.total_apr:.2f}% (${p.tvl_usd:,.0f}){marker}\n"

                await send_status_report(
                    current_pool=current.symbol if current else "None",
                    current_apr=current.total_apr if current else 0.0,
                    tvl=current.tvl_usd if current else 0.0,
                    earned_rewards=0.0,
                    total_migrations=state.total_migrations,
                    total_gas_spent=state.total_gas_spent_usd,
                    top_pools_text=top_text,
                )

        except Exception as e:
            error_msg = f"Agent loop error: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            await send_error_alert(str(e))

        logger.info("Sleeping %ds until next scan...", SCAN_INTERVAL_SECONDS)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
