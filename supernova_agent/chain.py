"""
Supernova Yield-Farming Agent – On-chain interactions
Handles Web3 connections, contract calls, and transaction building.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from web3 import Web3
from eth_account import Account

try:
    from web3.middleware import ExtraDataToPOAMiddleware
except ImportError:
    try:
        from web3.middleware import geth_poa_middleware as ExtraDataToPOAMiddleware
    except ImportError:
        ExtraDataToPOAMiddleware = None

from config import (
    RPC_URL, PRIVATE_KEY, WALLET_ADDRESS,
    ROUTER_V2, PAIR_FACTORY, GAUGE_MANAGER,
    NONFUNGIBLE_POSITION_MANAGER, ALGEBRA_SWAP_ROUTER,
    QUOTER_V2, PAIR_API_V2, WETH,
    SLIPPAGE_TOLERANCE, MAX_GAS_PRICE_GWEI, DRY_RUN,
)
from abis import (
    ERC20_ABI, ROUTER_V2_ABI, PAIR_FACTORY_ABI, PAIR_ABI,
    GAUGE_MANAGER_ABI, GAUGE_ABI, GAUGE_CL_ABI,
    NFT_POSITION_MANAGER_ABI, ALGEBRA_POOL_ABI,
    QUOTER_V2_ABI, PAIR_API_V2_ABI,
)

logger = logging.getLogger(__name__)

# ── Web3 singleton ────────────────────────────────────────────
_w3: Optional[Web3] = None


def get_w3() -> Web3:
    global _w3
    if _w3 is None:
        _w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"timeout": 30}))
        if ExtraDataToPOAMiddleware is not None:
            try:
                _w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except Exception:
                pass
        if not _w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC: {RPC_URL}")
        logger.info("Connected to chain: %s (block %d)", RPC_URL, _w3.eth.block_number)
    return _w3


def get_account() -> Account:
    if not PRIVATE_KEY:
        raise ValueError("PRIVATE_KEY not set")
    return Account.from_key(PRIVATE_KEY)


def get_wallet() -> str:
    if WALLET_ADDRESS:
        return Web3.to_checksum_address(WALLET_ADDRESS)
    return get_account().address


def _deadline(seconds: int = 300) -> int:
    return int(time.time()) + seconds


def _slippage_min(amount: int, slippage_pct: float = SLIPPAGE_TOLERANCE) -> int:
    return int(amount * (1 - slippage_pct / 100))


def _check_gas_price(w3: Web3) -> bool:
    gas_price_wei = w3.eth.gas_price
    gas_price_gwei = gas_price_wei / 1e9
    if gas_price_gwei > MAX_GAS_PRICE_GWEI:
        logger.warning("Gas price %.1f gwei exceeds max %d gwei", gas_price_gwei, MAX_GAS_PRICE_GWEI)
        return False
    return True


def _contract(w3: Web3, address: str, abi: list):
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)


# ── Token helpers ─────────────────────────────────────────────

def get_token_info(token_address: str) -> Dict[str, Any]:
    w3 = get_w3()
    token = _contract(w3, token_address, ERC20_ABI)
    try:
        symbol = token.functions.symbol().call()
    except Exception:
        symbol = "???"
    try:
        decimals = token.functions.decimals().call()
    except Exception:
        decimals = 18
    balance = token.functions.balanceOf(get_wallet()).call()
    return {"symbol": symbol, "decimals": decimals, "balance": balance}


def get_token_balance(token_address: str) -> int:
    w3 = get_w3()
    token = _contract(w3, token_address, ERC20_ABI)
    return token.functions.balanceOf(get_wallet()).call()


def get_eth_balance() -> int:
    w3 = get_w3()
    return w3.eth.get_balance(get_wallet())


def ensure_approval(token_address: str, spender: str, amount: int) -> Optional[str]:
    """Approve spender if current allowance is insufficient. Returns tx hash or None."""
    w3 = get_w3()
    token = _contract(w3, token_address, ERC20_ABI)
    wallet = get_wallet()
    current = token.functions.allowance(wallet, Web3.to_checksum_address(spender)).call()
    if current >= amount:
        return None

    approve_amount = 2**256 - 1
    tx = token.functions.approve(
        Web3.to_checksum_address(spender), approve_amount
    ).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 80_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def _sign_and_send(w3: Web3, tx: Dict) -> str:
    if DRY_RUN:
        logger.info("[DRY RUN] Would send tx: %s", {k: v for k, v in tx.items() if k != "data"})
        return "0x_dry_run"
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    hex_hash = tx_hash.hex()
    logger.info("Tx sent: %s", hex_hash)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] != 1:
        raise RuntimeError(f"Tx failed: {hex_hash}")
    logger.info("Tx confirmed: %s (gas used: %d)", hex_hash, receipt["gasUsed"])
    return hex_hash


# ── Basic Pool Operations ────────────────────────────────────

def get_pair_address(token_a: str, token_b: str, stable: bool) -> str:
    w3 = get_w3()
    factory = _contract(w3, PAIR_FACTORY, PAIR_FACTORY_ABI)
    return factory.functions.getPair(
        Web3.to_checksum_address(token_a),
        Web3.to_checksum_address(token_b),
        stable,
    ).call()


def get_pair_reserves(pair_address: str) -> Tuple[int, int]:
    w3 = get_w3()
    pair = _contract(w3, pair_address, PAIR_ABI)
    r0, r1, _ = pair.functions.getReserves().call()
    return r0, r1


def get_gauge_for_pool(pool_address: str) -> str:
    w3 = get_w3()
    gm = _contract(w3, GAUGE_MANAGER, GAUGE_MANAGER_ABI)
    return gm.functions.gauges(Web3.to_checksum_address(pool_address)).call()


def get_gauge_info(gauge_address: str) -> Dict[str, Any]:
    """Get reward rate, total supply, and earned for our wallet."""
    w3 = get_w3()
    gauge = _contract(w3, gauge_address, GAUGE_ABI)
    wallet = get_wallet()
    try:
        reward_rate = gauge.functions.rewardRate().call()
    except Exception:
        reward_rate = 0
    try:
        total_supply = gauge.functions.totalSupply().call()
    except Exception:
        total_supply = 0
    try:
        earned = gauge.functions.earned(wallet).call()
    except Exception:
        earned = 0
    try:
        staked = gauge.functions.balanceOf(wallet).call()
    except Exception:
        staked = 0
    return {
        "reward_rate": reward_rate,
        "total_supply": total_supply,
        "earned": earned,
        "staked": staked,
    }


def add_liquidity_basic(
    token_a: str, token_b: str, stable: bool,
    amount_a: int, amount_b: int,
) -> str:
    """Add liquidity to a basic pool via RouterV2."""
    w3 = get_w3()
    if not _check_gas_price(w3):
        raise RuntimeError("Gas price too high")

    router = _contract(w3, ROUTER_V2, ROUTER_V2_ABI)
    wallet = get_wallet()

    ensure_approval(token_a, ROUTER_V2, amount_a)
    ensure_approval(token_b, ROUTER_V2, amount_b)

    tx = router.functions.addLiquidity(
        Web3.to_checksum_address(token_a),
        Web3.to_checksum_address(token_b),
        stable,
        amount_a,
        amount_b,
        _slippage_min(amount_a),
        _slippage_min(amount_b),
        wallet,
        _deadline(),
    ).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 500_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def remove_liquidity_basic(
    token_a: str, token_b: str, stable: bool,
    lp_amount: int,
) -> str:
    """Remove liquidity from a basic pool via RouterV2."""
    w3 = get_w3()
    if not _check_gas_price(w3):
        raise RuntimeError("Gas price too high")

    pair_addr = get_pair_address(token_a, token_b, stable)
    ensure_approval(pair_addr, ROUTER_V2, lp_amount)

    router = _contract(w3, ROUTER_V2, ROUTER_V2_ABI)
    wallet = get_wallet()

    tx = router.functions.removeLiquidity(
        Web3.to_checksum_address(token_a),
        Web3.to_checksum_address(token_b),
        stable,
        lp_amount,
        0,
        0,
        wallet,
        _deadline(),
    ).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 500_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def swap_tokens(
    token_in: str, token_out: str, amount_in: int,
    stable: bool = False,
) -> str:
    """Swap tokens via RouterV2."""
    w3 = get_w3()
    if not _check_gas_price(w3):
        raise RuntimeError("Gas price too high")

    ensure_approval(token_in, ROUTER_V2, amount_in)
    router = _contract(w3, ROUTER_V2, ROUTER_V2_ABI)
    wallet = get_wallet()

    routes = [(
        Web3.to_checksum_address(token_in),
        Web3.to_checksum_address(token_out),
        stable,
    )]

    amounts_out = router.functions.getAmountsOut(amount_in, routes).call()
    min_out = _slippage_min(amounts_out[-1])

    tx = router.functions.swapExactTokensForTokens(
        amount_in,
        min_out,
        routes,
        wallet,
        _deadline(),
    ).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 350_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


# ── Gauge Staking ─────────────────────────────────────────────

def stake_in_gauge(gauge_address: str, lp_amount: int, lp_token: str) -> str:
    """Stake LP tokens in a gauge to earn emissions."""
    w3 = get_w3()
    ensure_approval(lp_token, gauge_address, lp_amount)
    gauge = _contract(w3, gauge_address, GAUGE_ABI)
    wallet = get_wallet()

    tx = gauge.functions.deposit(lp_amount).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 300_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def unstake_from_gauge(gauge_address: str, amount: int) -> str:
    """Unstake LP tokens from a gauge."""
    w3 = get_w3()
    gauge = _contract(w3, gauge_address, GAUGE_ABI)
    wallet = get_wallet()

    tx = gauge.functions.withdraw(amount).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 300_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def claim_gauge_rewards(gauge_address: str) -> str:
    """Claim pending rewards from a gauge."""
    w3 = get_w3()
    gauge = _contract(w3, gauge_address, GAUGE_ABI)
    wallet = get_wallet()

    tx = gauge.functions.getReward().build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 200_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


# ── Concentrated Pool Operations ─────────────────────────────

def get_pool_global_state(pool_address: str) -> Dict[str, Any]:
    w3 = get_w3()
    pool = _contract(w3, pool_address, ALGEBRA_POOL_ABI)
    state = pool.functions.globalState().call()
    return {
        "price": state[0],
        "tick": state[1],
        "fee": state[2],
        "unlocked": state[6],
    }


def mint_concentrated_position(
    token0: str, token1: str, deployer: str,
    tick_lower: int, tick_upper: int,
    amount0: int, amount1: int,
) -> str:
    """Mint a new concentrated liquidity position."""
    w3 = get_w3()
    if not _check_gas_price(w3):
        raise RuntimeError("Gas price too high")

    nft_mgr = _contract(w3, NONFUNGIBLE_POSITION_MANAGER, NFT_POSITION_MANAGER_ABI)
    wallet = get_wallet()

    ensure_approval(token0, NONFUNGIBLE_POSITION_MANAGER, amount0)
    ensure_approval(token1, NONFUNGIBLE_POSITION_MANAGER, amount1)

    params = (
        Web3.to_checksum_address(token0),
        Web3.to_checksum_address(token1),
        Web3.to_checksum_address(deployer),
        tick_lower,
        tick_upper,
        amount0,
        amount1,
        _slippage_min(amount0),
        _slippage_min(amount1),
        wallet,
        _deadline(),
    )

    tx = nft_mgr.functions.mint(params).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 600_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx)


def remove_concentrated_position(token_id: int, liquidity: int) -> str:
    """Remove liquidity from a concentrated position and collect tokens."""
    w3 = get_w3()
    if not _check_gas_price(w3):
        raise RuntimeError("Gas price too high")

    nft_mgr = _contract(w3, NONFUNGIBLE_POSITION_MANAGER, NFT_POSITION_MANAGER_ABI)
    wallet = get_wallet()

    decrease_params = (token_id, liquidity, 0, 0, _deadline())
    tx1 = nft_mgr.functions.decreaseLiquidity(decrease_params).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 400_000,
        "gasPrice": w3.eth.gas_price,
    })
    _sign_and_send(w3, tx1)

    max_uint128 = 2**128 - 1
    collect_params = (token_id, wallet, max_uint128, max_uint128)
    tx2 = nft_mgr.functions.collect(collect_params).build_transaction({
        "from": wallet,
        "nonce": w3.eth.get_transaction_count(wallet),
        "gas": 300_000,
        "gasPrice": w3.eth.gas_price,
    })
    return _sign_and_send(w3, tx2)


# ── Batch pool info via PairAPIV2 ────────────────────────────

def fetch_all_pairs_onchain() -> List[Dict[str, Any]]:
    """Fetch all basic pairs via the on-chain PairAPIV2 contract."""
    w3 = get_w3()
    api = _contract(w3, PAIR_API_V2, PAIR_API_V2_ABI)
    try:
        raw = api.functions.getAllPair().call()
    except Exception as e:
        logger.error("getAllPair failed: %s", e)
        return []

    results = []
    for p in raw:
        results.append({
            "pair_address": p[0],
            "symbol": p[1],
            "stable": p[4],
            "total_supply": p[5],
            "token0": p[6],
            "token0_symbol": p[7],
            "token0_decimals": p[8],
            "reserve0": p[9],
            "token1": p[11],
            "token1_symbol": p[12],
            "token1_decimals": p[13],
            "reserve1": p[14],
            "gauge": p[16],
            "gauge_total_supply": p[17],
            "fee": p[18],
            "emissions": p[20],
            "emissions_token": p[21],
            "emissions_token_decimals": p[22],
        })
    return results
