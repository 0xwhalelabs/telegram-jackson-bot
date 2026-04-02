"""
Supernova Yield-Farming Agent – Subgraph data fetching
Queries Goldsky-hosted subgraphs for Basic and Concentrated pool data.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config import BASIC_SUBGRAPH_URL, ALGEBRA_SUBGRAPH_URL

logger = logging.getLogger(__name__)


@dataclass
class BasicPoolData:
    """Data for a Uniswap V2-style basic pool."""
    address: str
    symbol: str
    stable: bool
    token0: str
    token0_symbol: str
    token0_decimals: int
    token1: str
    token1_symbol: str
    token1_decimals: int
    reserve0: float
    reserve1: float
    total_supply: float
    volume_usd_24h: float = 0.0
    fee_percent: float = 0.0
    tvl_usd: float = 0.0
    apr_fee: float = 0.0
    apr_emissions: float = 0.0
    apr_total: float = 0.0
    gauge: str = ""
    emissions_per_second: float = 0.0


@dataclass
class ConcentratedPoolData:
    """Data for an Algebra-based concentrated liquidity pool."""
    address: str
    token0: str
    token0_symbol: str
    token0_decimals: int
    token1: str
    token1_symbol: str
    token1_decimals: int
    fee: int
    tick: int
    tick_spacing: int
    liquidity: int
    sqrt_price: int
    tvl_usd: float = 0.0
    volume_usd_24h: float = 0.0
    fee_apr: float = 0.0
    apr_emissions: float = 0.0
    apr_total: float = 0.0
    gauge: str = ""


async def _query_subgraph(url: str, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
    """Execute a GraphQL query against a subgraph endpoint."""
    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("Subgraph query failed (%d): %s", resp.status, text[:500])
                return {}
            data = await resp.json()
            if "errors" in data:
                logger.error("Subgraph errors: %s", data["errors"])
            return data.get("data", {})


# ── Basic Pools ───────────────────────────────────────────────

# Well-known Ethereum mainnet tokens: address -> (symbol, decimals)
_KNOWN_TOKENS: Dict[str, Tuple[str, int]] = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", 18),
    "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6),
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6),
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC", 8),
    "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18),
    "0x514910771af9ca656af840dff83e8264ecf986ca": ("LINK", 18),
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": ("UNI", 18),
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": ("AAVE", 18),
    "0x00da8466b296e382e5da2bf20962d0cb87200c78": ("NOVA", 18),
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": ("cbBTC", 8),
    "0x45804880de22913dafe09f4980848ece6ecbaf78": ("PAXG", 18),
    "0x68749665ff8d2d112fa859aa293f07a622782f38": ("XAUt", 6),
    "0xe53ec727dbdeb9e2d5456c3be40cff031ab40a55": ("AAVE", 18),  # may vary
}

# Token info cache: address -> (symbol, decimals)
_TOKEN_CACHE: Dict[str, Tuple[str, int]] = {}

# Pre-populate cache with known tokens
for _addr, _info in _KNOWN_TOKENS.items():
    _TOKEN_CACHE[_addr.lower()] = _info


def _cache_token(address: str, symbol: str, decimals: int) -> None:
    _TOKEN_CACHE[address.lower()] = (symbol, decimals)


def _get_cached_token(address: str) -> Tuple[str, int]:
    return _TOKEN_CACHE.get(address.lower(), (address[:8], 18))


async def _resolve_token_info_batch(addresses: List[str]) -> None:
    """Resolve token symbols/decimals via on-chain calls for uncached addresses."""
    uncached = [a for a in set(addresses) if a.lower() not in _TOKEN_CACHE]
    if not uncached:
        return
    try:
        from chain import get_w3, _contract
        from abis import ERC20_ABI
        w3 = get_w3()
        for addr in uncached:
            try:
                token = _contract(w3, addr, ERC20_ABI)
                sym = token.functions.symbol().call()
                dec = token.functions.decimals().call()
                _cache_token(addr, sym, dec)
            except Exception:
                _cache_token(addr, addr[:6] + "..", 18)
    except Exception as e:
        logger.warning("RPC unavailable for token resolution, using address prefixes: %s", e)
        for addr in uncached:
            if addr.lower() not in _TOKEN_CACHE:
                _cache_token(addr, addr[:6] + "..", 18)


BASIC_POOLS_QUERY = """
{
  pairs(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
    id
    stable
    token0
    token1
    reserve0
    reserve1
    reserveUSD
    totalSupply
    totalValueLockedUSD
    volumeUSD
    feesUSD
    fee
    gaugeAddress
  }
}
"""

BASIC_PAIR_DAY_QUERY = """
{
  pairDayDatas(
    first: 200
    orderBy: date
    orderDirection: desc
  ) {
    pair {
      id
    }
    date
    volumeUSD
    feesUSD
    reserveUSD
  }
}
"""


async def fetch_basic_pools() -> List[BasicPoolData]:
    """Fetch top basic pools from the subgraph."""
    data = await _query_subgraph(BASIC_SUBGRAPH_URL, BASIC_POOLS_QUERY)
    pairs_raw = data.get("pairs", [])
    if not pairs_raw:
        logger.warning("No basic pools returned from subgraph")
        return []

    # Fetch daily data for fee APR calculation
    day_data = await _query_subgraph(BASIC_SUBGRAPH_URL, BASIC_PAIR_DAY_QUERY)
    day_entries = day_data.get("pairDayDatas", [])

    fees_24h_map: Dict[str, float] = {}
    volume_24h_map: Dict[str, float] = {}
    seen_pairs: set = set()
    for entry in day_entries:
        pid = entry.get("pair", {}).get("id", "")
        if pid in seen_pairs:
            continue
        seen_pairs.add(pid)
        fees_24h_map[pid] = float(entry.get("feesUSD", 0))
        volume_24h_map[pid] = float(entry.get("volumeUSD", 0))

    # Collect all token addresses for batch resolution
    all_tokens: List[str] = []
    for p in pairs_raw:
        t0 = p.get("token0", "")
        t1 = p.get("token1", "")
        if t0:
            all_tokens.append(t0)
        if t1:
            all_tokens.append(t1)
    await _resolve_token_info_batch(all_tokens)

    pools: List[BasicPoolData] = []
    for p in pairs_raw:
        try:
            t0_addr = p.get("token0", "")
            t1_addr = p.get("token1", "")
            t0_sym, t0_dec = _get_cached_token(t0_addr)
            t1_sym, t1_dec = _get_cached_token(t1_addr)

            tvl = float(p.get("totalValueLockedUSD", 0) or p.get("reserveUSD", 0))
            fees_24h = fees_24h_map.get(p["id"], 0.0)
            vol_24h = volume_24h_map.get(p["id"], 0.0)

            fee_pct = float(p.get("fee", 0.3))
            fee_apr = (fees_24h * 365 / tvl * 100) if tvl > 0 else 0.0

            gauge_addr = p.get("gaugeAddress", "")
            if gauge_addr and gauge_addr == "0x" + "0" * 40:
                gauge_addr = ""

            symbol = f"{t0_sym}/{t1_sym}"
            is_stable = bool(p.get("stable"))
            if is_stable:
                symbol += " (stable)"

            pool = BasicPoolData(
                address=p["id"],
                symbol=symbol,
                stable=is_stable,
                token0=t0_addr,
                token0_symbol=t0_sym,
                token0_decimals=t0_dec,
                token1=t1_addr,
                token1_symbol=t1_sym,
                token1_decimals=t1_dec,
                reserve0=float(p.get("reserve0", 0)),
                reserve1=float(p.get("reserve1", 0)),
                total_supply=float(p.get("totalSupply", 0)),
                volume_usd_24h=vol_24h,
                fee_percent=fee_pct,
                tvl_usd=tvl,
                apr_fee=fee_apr,
                gauge=gauge_addr or "",
            )
            pools.append(pool)
        except Exception as e:
            logger.warning("Failed to parse basic pool %s: %s", p.get("id"), e)
            continue

    return pools


# ── Concentrated Pools ────────────────────────────────────────

CONCENTRATED_POOLS_QUERY = """
{
  pools(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
    id
    token0 {
      id
      symbol
      decimals
    }
    token1 {
      id
      symbol
      decimals
    }
    fee
    tick
    tickSpacing
    liquidity
    sqrtPrice
    totalValueLockedUSD
    volumeUSD
    feesUSD
    poolDayData(first: 1, orderBy: date, orderDirection: desc) {
      date
      volumeUSD
      feesUSD
      tvlUSD
    }
  }
}
"""


async def fetch_concentrated_pools() -> List[ConcentratedPoolData]:
    """Fetch top concentrated pools from the Algebra subgraph."""
    data = await _query_subgraph(ALGEBRA_SUBGRAPH_URL, CONCENTRATED_POOLS_QUERY)
    pools_raw = data.get("pools", [])
    if not pools_raw:
        logger.warning("No concentrated pools returned from subgraph")
        return []

    pools: List[ConcentratedPoolData] = []
    for p in pools_raw:
        try:
            t0 = p.get("token0", {})
            t1 = p.get("token1", {})
            tvl = float(p.get("totalValueLockedUSD", 0))

            day_data_list = p.get("poolDayData", [])
            vol_24h = 0.0
            fees_24h = 0.0
            if day_data_list:
                vol_24h = float(day_data_list[0].get("volumeUSD", 0))
                fees_24h = float(day_data_list[0].get("feesUSD", 0))

            fee_apr = (fees_24h * 365 / tvl * 100) if tvl > 0 else 0.0

            pool = ConcentratedPoolData(
                address=p["id"],
                token0=t0.get("id", ""),
                token0_symbol=t0.get("symbol", ""),
                token0_decimals=int(t0.get("decimals", 18)),
                token1=t1.get("id", ""),
                token1_symbol=t1.get("symbol", ""),
                token1_decimals=int(t1.get("decimals", 18)),
                fee=int(p.get("fee", 0)),
                tick=int(p.get("tick", 0)),
                tick_spacing=int(p.get("tickSpacing", 60)),
                liquidity=int(p.get("liquidity", 0)),
                sqrt_price=int(p.get("sqrtPrice", 0)),
                tvl_usd=tvl,
                volume_usd_24h=vol_24h,
                fee_apr=fee_apr,
            )
            pools.append(pool)
        except Exception as e:
            logger.warning("Failed to parse concentrated pool %s: %s", p.get("id"), e)
            continue

    return pools
