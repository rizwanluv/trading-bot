import os
import zipfile
import sqlite3
import asyncio
import logging
from typing import Optional
from datetime import datetime, timezone
from telegram import Bot

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "trading_bot.db")
JSON_STORES = [
    "autotrade_store.json",
    "learning_store.json",
    "alerts_store.json",
    "memory_store.json",
]


def _get_backup_chat_id() -> Optional[str]:
    return (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("BACKUP_CHANNEL_ID")
        or os.getenv("ADMIN_CHAT_ID")
    )


async def send_database_backup_to_telegram(bot: Bot):
    """
    Takes a safe hot snapshot of the database and JSON trading state stores,
    bundles them, and sends the backup archive to the configured Telegram chat.
    """
    chat_id = _get_backup_chat_id()
    if not chat_id:
        logger.debug("[Backup] Skipped: No TELEGRAM_CHAT_ID or BACKUP_CHANNEL_ID configured.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_zip = f"backup_trading_{timestamp}.zip"
    snapshot_db = f"backup_snapshot_{timestamp}.db"
    files_to_clean = []

    try:
        has_content = False
        with zipfile.ZipFile(backup_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 1. Safe SQLite hot backup if DB file exists
            if os.path.exists(DB_PATH):
                try:
                    src_conn = sqlite3.connect(DB_PATH)
                    dst_conn = sqlite3.connect(snapshot_db)
                    with dst_conn:
                        src_conn.backup(dst_conn)
                    dst_conn.close()
                    src_conn.close()
                    zipf.write(snapshot_db, arcname="trading_bot.db")
                    files_to_clean.append(snapshot_db)
                    has_content = True
                except Exception as db_err:
                    logger.warning(f"[Backup] Failed to snapshot SQLite DB: {db_err}")

            # 2. Add JSON persistent stores if they exist
            for store in JSON_STORES:
                if os.path.exists(store):
                    try:
                        zipf.write(store, arcname=store)
                        has_content = True
                    except Exception as store_err:
                        logger.warning(f"[Backup] Failed to add {store} to archive: {store_err}")

        files_to_clean.append(backup_zip)

        if not has_content:
            logger.debug("[Backup] No database or state files exist yet to back up.")
            return

        with open(backup_zip, "rb") as doc:
            caption = (
                f"💾 <b>AUTOMATED TRADING STATE BACKUP</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Timestamp:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>\n"
                f"• <b>Archive:</b> <code>{backup_zip}</code>\n"
                f"#BACKUP #TRADING_BOT"
            )
            await bot.send_document(
                chat_id=chat_id,
                document=doc,
                caption=caption,
                parse_mode="HTML"
            )
        logger.info(f"[Backup] Successfully sent {backup_zip} to Telegram chat {chat_id}.")

    except Exception as e:
        logger.error(f"[Backup] Error during backup dispatch: {e}")
        try:
            if chat_id:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ <b>Backup Warning:</b> Failed to export backup archive.\n<code>{str(e)}</code>",
                    parse_mode="HTML"
                )
        except Exception:
            pass
    finally:
        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


async def schedule_backups(bot: Bot, interval_seconds: int = 86400):
    """
    Runs continuously in the background to dispatch periodic backups.
    Default interval: 86400s (24 hours).
    """
    logger.info(f"[Backup] Scheduler started (interval: {interval_seconds}s).")
    await asyncio.sleep(60)

    while True:
        try:
            await send_database_backup_to_telegram(bot)
        except Exception as e:
            logger.error(f"[Backup] Loop error: {e}")

        await asyncio.sleep(interval_seconds)
