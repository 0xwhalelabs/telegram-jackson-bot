"""
Supernova Yield-Farming Agent – Entry Point
Runs the agent loop and optionally the Telegram bot in parallel.
"""
import asyncio
import logging
import sys

from config import DRY_RUN, TELEGRAM_BOT_TOKEN, SCAN_INTERVAL_SECONDS
from agent import agent_loop
from notifier import build_telegram_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("supernova_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("supernova")


async def run_agent_only():
    """Run the agent loop without Telegram bot polling."""
    await agent_loop()


async def run_with_telegram():
    """Run both the agent loop and Telegram bot concurrently."""
    app = build_telegram_app()
    if app is None:
        logger.warning("Telegram app not built – running agent only")
        await run_agent_only()
        return

    # Initialize the Telegram application
    await app.initialize()
    await app.start()

    # Start polling in background
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling started")

    try:
        await agent_loop()
    finally:
        logger.info("Shutting down Telegram bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main():
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    logger.info("=" * 60)
    logger.info("  Supernova Yield-Farming Agent")
    logger.info("  Mode: %s", mode)
    logger.info("  Scan interval: %ds", SCAN_INTERVAL_SECONDS)
    logger.info("=" * 60)

    if TELEGRAM_BOT_TOKEN:
        logger.info("Telegram bot enabled – starting with bot polling")
        asyncio.run(run_with_telegram())
    else:
        logger.info("No Telegram token – running agent loop only")
        asyncio.run(run_agent_only())


if __name__ == "__main__":
    main()
