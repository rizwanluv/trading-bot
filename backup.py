import os
import sqlite3
import asyncio
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKUP_CHANNEL_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_PATH = os.getenv("DB_PATH", "/app/data/trading_bot.db")

async def send_database_backup_to_telegram(bot: Bot):
    """Takes a hot snapshot of the SQLite DB and sends it to Telegram."""
    if not os.path.exists(DB_PATH):
        print(f"[Backup] Skipped: Database file {DB_PATH} does not exist yet.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"backup_trading_{timestamp}.db"

    try:
        # Create a non-blocking safe live copy of the SQLite DB
        src_conn = sqlite3.connect(DB_PATH)
        dst_conn = sqlite3.connect(snapshot_filename)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()

        # Send the database file to your Telegram channel
        with open(snapshot_filename, "rb") as doc:
            caption = (
                f"💾 *AUTOMATED DATABASE BACKUP*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"• *Timestamp:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`\n"
                f"• *File:* `{snapshot_filename}`\n"
                f"#BACKUP #DATABASE"
            )
            await bot.send_document(
                chat_id=BACKUP_CHANNEL_ID,
                document=doc,
                caption=caption,
                parse_mode="Markdown"
            )
        print(f"[Backup] Successfully sent {snapshot_filename} to Telegram.")

    except Exception as e:
        print(f"[Backup] Error during backup dispatch: {e}")
        try:
            await bot.send_message(
                chat_id=BACKUP_CHANNEL_ID,
                text=f"⚠️ *Backup Warning:* Failed to export DB backup.\n`{str(e)}`",
                parse_mode="Markdown"
            )
        except Exception:
            pass
    finally:
        # Delete the temporary local snapshot file to keep disk space clean
        if os.path.exists(snapshot_filename):
            os.remove(snapshot_filename)


async def schedule_backups(bot: Bot, interval_seconds: int = 86400):
    """
    Runs continuously in the background.
    Default interval: 86400s (24 hours). For 6 hours, set to 21600.
    """
    print("[Backup] Backup scheduler task started.")
    # Optional: wait 60 seconds after startup before the very first backup
    await asyncio.sleep(60)
    
    while True:
        try:
            await send_database_backup_to_telegram(bot)
        except Exception as e:
            print(f"[Backup] Loop error: {e}")
            
        await asyncio.sleep(interval_seconds)
