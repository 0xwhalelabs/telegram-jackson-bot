"""
Supernova Yield-Farming Agent – Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Ethereum RPC ──────────────────────────────────────────────
RPC_URL: str = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")

# ── Wallet ────────────────────────────────────────────────────
PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")

# ── Gemini AI ─────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Telegram ──────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("SN_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("SN_TELEGRAM_CHAT_ID", "")

# ── Agent Parameters ──────────────────────────────────────────
# How often (seconds) to scan pools for better yield
SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))

# Minimum APR improvement (%) to trigger migration
MIN_APR_IMPROVEMENT: float = float(os.getenv("MIN_APR_IMPROVEMENT", "2.0"))

# Maximum gas price (gwei) – skip migration if gas is too expensive
MAX_GAS_PRICE_GWEI: int = int(os.getenv("MAX_GAS_PRICE_GWEI", "50"))

# Slippage tolerance (0.5 = 0.5%)
SLIPPAGE_TOLERANCE: float = float(os.getenv("SLIPPAGE_TOLERANCE", "0.5"))

# Minimum TVL ($) for a pool to be considered
MIN_POOL_TVL_USD: float = float(os.getenv("MIN_POOL_TVL_USD", "10000"))

# Maximum portion of wallet to deploy (0.0-1.0)
MAX_DEPLOY_RATIO: float = float(os.getenv("MAX_DEPLOY_RATIO", "0.95"))

# Maximum pool fee allowed (0.01 = 1%). Pools with fee >= this are excluded
MAX_POOL_FEE: float = float(os.getenv("MAX_POOL_FEE", "0.01"))

# Dry-run mode – simulate without executing transactions
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

# ── Subgraph URLs ─────────────────────────────────────────────
BASIC_SUBGRAPH_URL: str = (
    "https://api.goldsky.com/api/public/project_cm8gyxv0x02qv01uphvy69ey6"
    "/subgraphs/sn-basic-pools-mainnet/basicsnmainnet/gn"
)
ALGEBRA_SUBGRAPH_URL: str = (
    "https://api.goldsky.com/api/public/project_cm8gyxv0x02qv01uphvy69ey6"
    "/subgraphs/core/algebrasnmainnet/gn"
)

# ── Contract Addresses (Mainnet) ──────────────────────────────
ROUTER_V2: str = "0xbFAe8E87053309fDe07ab3cA5f4B5345f8e3058f"
PAIR_FACTORY: str = "0x5aef44edfc5a7edd30826c724ea12d7be15bdc30"
GAUGE_MANAGER: str = "0x19a410046afc4203aece5fbfc7a6ac1a4f517ae2"
VOTER_V3: str = "0x1c7bf2532dfa34eeea02c3759e0ca8d87b1d8171"
VOTING_ESCROW: str = "0x4c3e7640b3e3a39a2e5d030a0c1412d80fee1d44"
NONFUNGIBLE_POSITION_MANAGER: str = "0x00d5bbd0fe275efee371a2b34d0a4b95b0c8aaaa"
ALGEBRA_FACTORY: str = "0x44b7fbd4d87149efa5347c451e74b9fd18e89c55"
ALGEBRA_SWAP_ROUTER: str = "0x72d63a5b080e1b89cc93f9b9f50cbfa5e291c8ac"
QUOTER_V2: str = "0x8217550d36823b1194b58562dac55d7fe8efb727"
ALGEBRA_POOL_API: str = "0x0ee8553a64edf161b3daa6907a4ff45b0a12ea59"
FARMING_CENTER: str = "0x428ea5b4ac84ab687851e6a2688411bdbd6c91af"
ALGEBRA_ETERNAL_FARMING: str = "0x1e862624eda92b8fe532c16253356d17dd70a337"
PAIR_API_V2: str = "0x2B9FC4714589544Aa1e0a75596c611a1364963Dc"
MINTER: str = "0xfe29ea1348f0990273db5e19ad521e45acda84a2"
SUPERNOVA_TOKEN: str = "0x00da8466b296e382e5da2bf20962d0cb87200c78"

# WETH on Ethereum mainnet
WETH: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
