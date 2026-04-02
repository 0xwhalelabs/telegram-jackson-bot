"""
Supernova Yield-Farming Agent – Telegram Notifier
Sends status updates, migration alerts, and allows basic control via Telegram.
"""
import asyncio
import logging
from typing import Optional

from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DRY_RUN

logger = logging.getLogger(__name__)

_bot: Optional[Bot] = None
_app: Optional[Application] = None

# Callback for pause/resume from Telegram
_agent_paused: bool = False


def is_paused() -> bool:
    return _agent_paused


def set_paused(paused: bool) -> None:
    global _agent_paused
    _agent_paused = paused


def _get_bot() -> Optional[Bot]:
    global _bot
    if _bot is None and TELEGRAM_BOT_TOKEN:
        _bot = Bot(token=TELEGRAM_BOT_TOKEN)
    return _bot


async def send_message(text: str, parse_mode: str = "HTML") -> None:
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping message")
        return
    bot = _get_bot()
    if bot is None:
        return
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=parse_mode,
        )
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)


async def send_migration_alert(
    from_symbol: str,
    to_symbol: str,
    from_apr: float,
    to_apr: float,
    apr_diff: float,
    gas_cost: float,
    tx_hash: str = "",
) -> None:
    """Send a migration notification."""
    status = "🔄 DRY RUN" if DRY_RUN else "✅ EXECUTED"
    text = (
        f"<b>{status} – Pool Migration</b>\n\n"
        f"📤 From: <code>{from_symbol}</code> ({from_apr:.2f}% APR)\n"
        f"📥 To: <code>{to_symbol}</code> ({to_apr:.2f}% APR)\n"
        f"📈 Improvement: <b>+{apr_diff:.2f}%</b>\n"
        f"⛽ Gas Cost: ~${gas_cost:.2f}\n"
    )
    if tx_hash and tx_hash != "0x_dry_run":
        text += f"🔗 <a href='https://etherscan.io/tx/{tx_hash}'>View on Etherscan</a>\n"
    await send_message(text)


async def send_status_report(
    current_pool: str,
    current_apr: float,
    tvl: float,
    earned_rewards: float,
    total_migrations: int,
    total_gas_spent: float,
    top_pools_text: str,
) -> None:
    """Send periodic status report."""
    paused_tag = "⏸ PAUSED" if _agent_paused else "▶️ RUNNING"
    dry_tag = " (DRY RUN)" if DRY_RUN else ""
    text = (
        f"<b>📊 Supernova Agent Status {paused_tag}{dry_tag}</b>\n\n"
        f"🏊 Current Pool: <code>{current_pool}</code>\n"
        f"📈 Current APR: {current_apr:.2f}%\n"
        f"💰 Pool TVL: ${tvl:,.0f}\n"
        f"🎁 Pending Rewards: ${earned_rewards:.4f}\n"
        f"🔄 Total Migrations: {total_migrations}\n"
        f"⛽ Total Gas Spent: ${total_gas_spent:.2f}\n\n"
        f"<b>🏆 Top Pools:</b>\n{top_pools_text}"
    )
    await send_message(text)


async def send_error_alert(error_msg: str) -> None:
    """Send an error notification."""
    text = f"🚨 <b>Supernova Agent Error</b>\n\n<code>{error_msg[:500]}</code>"
    await send_message(text)


async def send_startup_message() -> None:
    """Send agent startup notification."""
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    text = (
        f"🚀 <b>Supernova Yield Agent Started</b>\n"
        f"Mode: <b>{mode}</b>\n\n"
        f"Commands:\n"
        f"/sn_status – Current status\n"
        f"/sn_pools – Top pools\n"
        f"/sn_pause – Pause agent\n"
        f"/sn_resume – Resume agent\n"
        f"/sn_migrate – Force migration check\n"
    )
    await send_message(text)


# ── Telegram Command Handlers ─────────────────────────────────

# These will be populated by the main agent loop
_status_callback = None
_pools_callback = None
_migrate_callback = None


def register_callbacks(status_cb=None, pools_cb=None, migrate_cb=None):
    global _status_callback, _pools_callback, _migrate_callback
    _status_callback = status_cb
    _pools_callback = pools_cb
    _migrate_callback = migrate_cb


async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _status_callback:
        await _status_callback(update, context)
    else:
        await update.message.reply_text("Agent status not available yet.")


async def _cmd_pools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _pools_callback:
        await _pools_callback(update, context)
    else:
        await update.message.reply_text("Pool data not available yet.")


async def _cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    set_paused(True)
    await update.message.reply_text("⏸ Agent paused. Use /sn_resume to resume.")


async def _cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    set_paused(False)
    await update.message.reply_text("▶️ Agent resumed.")


async def _cmd_migrate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _migrate_callback:
        await update.message.reply_text("🔍 Checking for migration opportunity...")
        await _migrate_callback(update, context)
    else:
        await update.message.reply_text("Migration check not available yet.")


def build_telegram_app() -> Optional[Application]:
    """Build the Telegram application with command handlers."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set – Telegram features disabled")
        return None

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("sn_status", _cmd_status))
    app.add_handler(CommandHandler("sn_pools", _cmd_pools))
    app.add_handler(CommandHandler("sn_pause", _cmd_pause))
    app.add_handler(CommandHandler("sn_resume", _cmd_resume))
    app.add_handler(CommandHandler("sn_migrate", _cmd_migrate))

    return app
