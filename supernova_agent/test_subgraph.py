"""
Comprehensive dry-run test for the Supernova Yield-Farming Agent.
Tests: subgraph fetching, strategy scoring, migration decisions,
state persistence, deployer selection, and notification formatting.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label} — {detail}")


async def test_subgraph_fetching():
    print("\n[1] Subgraph Fetching")
    from subgraph import fetch_basic_pools, fetch_concentrated_pools

    basic = await fetch_basic_pools()
    check("Basic pools fetched", len(basic) > 0, f"got {len(basic)}")

    conc = await fetch_concentrated_pools()
    check("Concentrated pools fetched", len(conc) > 0, f"got {len(conc)}")
    check("Concentrated pools have TVL", any(p.tvl_usd > 0 for p in conc), "all TVL=0")
    check("Concentrated pools have fee APR", any(p.fee_apr > 0 for p in conc), "all APR=0")

    top = conc[0] if conc else None
    if top:
        check("Top pool has token symbols", bool(top.token0_symbol and top.token1_symbol),
              f"t0={top.token0_symbol} t1={top.token1_symbol}")
        print(f"    Top CL pool: {top.token0_symbol}/{top.token1_symbol} "
              f"TVL=${top.tvl_usd:,.0f} APR={top.fee_apr:.2f}%")

    return basic, conc


def test_strategy_scoring(basic, conc):
    print("\n[2] Strategy Scoring")
    from strategy import rank_pools, should_migrate, PoolScore, estimate_gas_cost_usd

    ranked = rank_pools(basic, conc)
    check("Pools ranked", len(ranked) > 0, f"got {len(ranked)}")
    check("Ranked by score desc", all(ranked[i].score >= ranked[i+1].score for i in range(len(ranked)-1)))

    for i, p in enumerate(ranked[:5]):
        print(f"    #{i+1} {p.symbol} ({p.pool_type}): "
              f"Score={p.score:.1f} APR={p.total_apr:.2f}% TVL=${p.tvl_usd:,.0f}")

    # Test migration decision: no current pool → should migrate
    best = ranked[0]
    plan = should_migrate(None, best)
    check("No position → recommends entry", plan is not None)
    if plan:
        check("Plan targets best pool", plan.to_pool.address == best.address)

    # Test migration decision: already in best pool → no migration
    plan2 = should_migrate(best, best)
    check("Already in best → no migration", plan2 is None)

    # Test migration decision: small APR diff → no migration
    if len(ranked) >= 2:
        fake_current = PoolScore(
            pool_type="concentrated", address="0xfake",
            symbol="FAKE", token0="", token0_symbol="",
            token1="", token1_symbol="",
            tvl_usd=100000, fee_apr=ranked[0].total_apr - 0.5,
            emissions_apr=0, total_apr=ranked[0].total_apr - 0.5,
            score=ranked[0].score - 1,
        )
        plan3 = should_migrate(fake_current, best, gas_cost_usd=5.0)
        check("Small APR diff (<2%) → no migration", plan3 is None,
              f"diff={best.total_apr - fake_current.total_apr:.2f}%")

    # Gas cost estimation
    gas = estimate_gas_cost_usd(eth_price_usd=2500, gas_units=800_000, gas_gwei=20)
    check("Gas cost estimate reasonable", 0 < gas < 200, f"${gas:.2f}")

    return ranked


def test_deployer_selection():
    print("\n[3] Deployer Selection")
    from agent import _get_deployer_for_tick_spacing, TICK_SPACING_DEPLOYERS, DEFAULT_DEPLOYER

    for ts, expected in TICK_SPACING_DEPLOYERS.items():
        result = _get_deployer_for_tick_spacing(ts)
        check(f"tick_spacing={ts} → correct deployer", result == expected)

    fallback = _get_deployer_for_tick_spacing(999)
    check("Unknown tick_spacing → default deployer", fallback == DEFAULT_DEPLOYER)


def test_state_persistence():
    print("\n[4] State Persistence")
    from state import save_state, load_state, state_to_dict, STATE_FILE

    test_data = state_to_dict(
        current_pool_address="0xabc123",
        current_pool_type="concentrated",
        current_pool_symbol="WBTC/WETH",
        current_pool_apr=72.5,
        current_gauge_address="0xgauge1",
        current_lp_amount=1000000,
        current_nft_token_id=42,
        total_migrations=3,
        total_gas_spent_usd=15.50,
        total_rewards_claimed_usd=120.0,
    )

    save_state(test_data)
    check("State file created", os.path.exists(STATE_FILE))

    loaded = load_state()
    check("State loaded", bool(loaded))
    check("Pool address preserved", loaded.get("current_pool_address") == "0xabc123")
    check("Pool symbol preserved", loaded.get("current_pool_symbol") == "WBTC/WETH")
    check("APR preserved", loaded.get("current_pool_apr") == 72.5)
    check("Migrations count preserved", loaded.get("total_migrations") == 3)
    check("NFT token ID preserved", loaded.get("current_nft_token_id") == 42)

    # Cleanup
    try:
        os.remove(STATE_FILE)
    except Exception:
        pass


def test_notification_formatting():
    print("\n[5] Notification Formatting")
    from strategy import format_pool_summary, format_migration_plan, PoolScore, MigrationPlan

    pool = PoolScore(
        pool_type="concentrated", address="0x1234567890abcdef1234567890abcdef12345678",
        symbol="WBTC/USDT", token0="0xa", token0_symbol="WBTC",
        token1="0xb", token1_symbol="USDT",
        tvl_usd=5_000_000, fee_apr=22.5, emissions_apr=5.0,
        total_apr=27.5, score=25.0,
    )
    summary = format_pool_summary(pool)
    check("Pool summary contains symbol", "WBTC/USDT" in summary)
    check("Pool summary contains APR", "27.50%" in summary)

    plan = MigrationPlan(
        from_pool=pool,
        to_pool=PoolScore(
            pool_type="concentrated", address="0xnew",
            symbol="WBTC/WETH", token0="", token0_symbol="",
            token1="", token1_symbol="",
            tvl_usd=1_000_000, fee_apr=70.0, emissions_apr=0,
            total_apr=70.0, score=68.0,
        ),
        reason="APR improvement",
        apr_improvement=42.5,
        estimated_gas_cost_usd=8.0,
    )
    plan_text = format_migration_plan(plan)
    check("Migration plan contains reason", "APR improvement" in plan_text)
    check("Migration plan contains gas cost", "$8.00" in plan_text)


def test_tick_range():
    print("\n[6] Tick Range Calculation")
    from strategy import calculate_tick_range

    lower, upper = calculate_tick_range(current_tick=200000, tick_spacing=60, range_multiplier=10)
    check("Tick lower < current", lower < 200000)
    check("Tick upper > current", upper > 200000)
    check("Tick range aligned to spacing", lower % 60 == 0 and upper % 60 == 0)

    lower2, upper2 = calculate_tick_range(current_tick=-50000, tick_spacing=200, range_multiplier=5)
    check("Negative tick handled", lower2 < -50000 and upper2 > -50000)
    check("Tick range aligned (200)", lower2 % 200 == 0 and upper2 % 200 == 0)


async def main():
    print("=" * 60)
    print("  Supernova Yield Agent – Comprehensive Test")
    print("=" * 60)

    basic, conc = await test_subgraph_fetching()
    test_strategy_scoring(basic, conc)
    test_deployer_selection()
    test_state_persistence()
    test_notification_formatting()
    test_tick_range()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
