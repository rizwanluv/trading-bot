import os
import zipfile
import sqlite3
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
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


def _get_backup_chat_id(override_chat_id: Optional[str] = None) -> Optional[str]:
    """
    Resolve target channel/chat for backups.
    Dedicated backup channel has top priority over regular chat ID.
    """
    if override_chat_id:
        return str(override_chat_id).strip()

    # 1. Environment BACKUP_CHANNEL_ID (Explicit dedicated backup channel)
    env_backup = os.getenv("BACKUP_CHANNEL_ID")
    if env_backup and env_backup.strip():
        return env_backup.strip()

    # 2. Persisted setting in autotrade_store.json
    if os.path.exists("autotrade_store.json"):
        try:
            import json
            with open("autotrade_store.json", "r", encoding="utf-8") as f:
                d = json.load(f)
                val = d.get("backup_channel_id")
                if val and str(val).strip():
                    return str(val).strip()
        except Exception:
            pass

    # 3. Fallbacks
    return (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("ADMIN_CHAT_ID")
    )


def set_backup_channel_id(channel_id: str, env_path: str = ".env") -> bool:
    """Persist dedicated backup channel to runtime env, .env file, and autotrade_store.json."""
    clean_id = str(channel_id).strip()
    os.environ["BACKUP_CHANNEL_ID"] = clean_id

    # 1. Update .env
    lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Failed to read {env_path}: {e}")

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("BACKUP_CHANNEL_ID="):
            new_lines.append(f"BACKUP_CHANNEL_ID={clean_id}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"BACKUP_CHANNEL_ID={clean_id}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        logger.error(f"Failed to write BACKUP_CHANNEL_ID to {env_path}: {e}")

    # 2. Update autotrade_store.json
    if os.path.exists("autotrade_store.json"):
        try:
            import json
            with open("autotrade_store.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            data["backup_channel_id"] = clean_id
            with open("autotrade_store.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as err:
            logger.warning(f"Failed to update autotrade_store.json with backup channel: {err}")

    return True


def get_backup_status() -> Dict[str, Any]:
    """Return status of backup destinations and available database/store files."""
    chat_id = _get_backup_chat_id()
    files_available = []
    if os.path.exists(DB_PATH):
        files_available.append(DB_PATH)
    for s in JSON_STORES:
        if os.path.exists(s):
            files_available.append(s)

    return {
        "backup_channel_id": chat_id,
        "is_configured": bool(chat_id),
        "db_exists": os.path.exists(DB_PATH),
        "files_available": files_available,
        "files_count": len(files_available),
    }


async def send_database_backup_to_telegram(bot: Bot, chat_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Takes a safe hot snapshot of the database and JSON trading state stores,
    bundles them, and sends the backup archive to the configured Telegram chat/channel.
    Returns result dictionary.
    """
    target_id = _get_backup_chat_id(override_chat_id=chat_id)
    if not target_id:
        msg = "No TELEGRAM_CHAT_ID or BACKUP_CHANNEL_ID configured."
        logger.debug(f"[Backup] Skipped: {msg}")
        return {"success": False, "error": msg}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_zip = f"backup_trading_{timestamp}.zip"
    snapshot_db = f"backup_snapshot_{timestamp}.db"
    files_to_clean = []
    archived_files = []

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
                    archived_files.append("trading_bot.db")
                    has_content = True
                except Exception as db_err:
                    logger.warning(f"[Backup] Failed to snapshot SQLite DB: {db_err}")

            # 2. Add JSON persistent stores if they exist
            for store in JSON_STORES:
                if os.path.exists(store):
                    try:
                        zipf.write(store, arcname=store)
                        archived_files.append(store)
                        has_content = True
                    except Exception as store_err:
                        logger.warning(f"[Backup] Failed to add {store} to archive: {store_err}")

        files_to_clean.append(backup_zip)

        if not has_content:
            logger.debug("[Backup] No database or state files exist yet to back up.")
            return {"success": False, "error": "No database or state files exist yet to back up."}

        file_size_kb = round(os.path.getsize(backup_zip) / 1024.0, 2)
        with open(backup_zip, "rb") as doc:
            caption = (
                f"💾 <b>AUTOMATED TRADING STATE BACKUP</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Timestamp:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>\n"
                f"• <b>Archive:</b> <code>{backup_zip}</code> ({file_size_kb} KB)\n"
                f"• <b>Files:</b> <code>{', '.join(archived_files)}</code>\n"
                f"• <b>Destination:</b> <code>{target_id}</code>\n"
                f"#BACKUP #TRADING_BOT"
            )
            await bot.send_document(
                chat_id=target_id,
                document=doc,
                caption=caption,
                parse_mode="HTML"
            )
        logger.info(f"[Backup] Successfully sent {backup_zip} ({file_size_kb} KB) to Telegram chat/channel {target_id}.")
        return {
            "success": True,
            "chat_id": target_id,
            "archive_name": backup_zip,
            "size_kb": file_size_kb,
            "files": archived_files,
        }

    except Exception as e:
        logger.error(f"[Backup] Error during backup dispatch: {e}")
        try:
            if target_id:
                await bot.send_message(
                    chat_id=target_id,
                    text=f"⚠️ <b>Backup Warning:</b> Failed to export backup archive.\n<code>{str(e)}</code>",
                    parse_mode="HTML"
                )
        except Exception:
            pass
        return {"success": False, "error": str(e), "chat_id": target_id}
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
