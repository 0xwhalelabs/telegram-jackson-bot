"""
Supernova Yield-Farming Agent – State Persistence
Saves and loads agent state to/from a JSON file so the agent
can resume after restarts.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "agent_state.json")


def save_state(data: Dict[str, Any]) -> None:
    """Save agent state to disk."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("State saved to %s", STATE_FILE)
    except Exception as e:
        logger.error("Failed to save state: %s", e)


def load_state() -> Dict[str, Any]:
    """Load agent state from disk. Returns empty dict if no state file."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("State loaded from %s", STATE_FILE)
        return data
    except Exception as e:
        logger.error("Failed to load state: %s", e)
        return {}


def state_to_dict(
    current_pool_address: str,
    current_pool_type: str,
    current_pool_symbol: str,
    current_pool_apr: float,
    current_gauge_address: str,
    current_lp_amount: int,
    current_nft_token_id: int,
    total_migrations: int,
    total_gas_spent_usd: float,
    total_rewards_claimed_usd: float,
) -> Dict[str, Any]:
    """Convert agent state fields to a serializable dict."""
    return {
        "current_pool_address": current_pool_address,
        "current_pool_type": current_pool_type,
        "current_pool_symbol": current_pool_symbol,
        "current_pool_apr": current_pool_apr,
        "current_gauge_address": current_gauge_address,
        "current_lp_amount": current_lp_amount,
        "current_nft_token_id": current_nft_token_id,
        "total_migrations": total_migrations,
        "total_gas_spent_usd": total_gas_spent_usd,
        "total_rewards_claimed_usd": total_rewards_claimed_usd,
    }
