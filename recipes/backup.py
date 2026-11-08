"""Daily off-site backup of saved recipes to a private GitHub repo.

The Render disk is otherwise the only working copy. The export is plain JSON, so
the repo's history doubles as a version history.
"""

import asyncio
import base64
import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from recipes.store import RecipeStore

logger = logging.getLogger(__name__)

BACKUP_FILE = "recipes.json"
API_ROOT = "https://api.github.com"
CHECK_INTERVAL_SECONDS = 60 * 60
BACKUP_INTERVAL = timedelta(hours=24)
# A backup this old means something is wrong, even if no attempt has failed.
STALE_AFTER = timedelta(hours=48)
LAST_ATTEMPT = "backup_last_attempt"
LAST_SUCCESS = "backup_last_success"
LAST_ERROR = "backup_last_error"


class BackupError(Exception):
    """A failed backup, with a message that's safe to show and has no token in it."""


@dataclass(frozen=True)
class BackupConfig:
    repo: str
    token: str


def backup_config() -> BackupConfig | None:
    repo = os.environ.get("RECIPES_BACKUP_REPO")
    token = os.environ.get("RECIPES_BACKUP_TOKEN")
    return BackupConfig(repo, token) if repo and token else None


def _git_blob_sha(content: bytes) -> str:
    """GitHub's ID for a file, to tell whether its content has changed."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def push_export(
    export: str, config: BackupConfig, transport: httpx.BaseTransport | None = None
) -> bool:
    """Commit an export if it changed. Returns whether a commit was made."""
    content = export.encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "prashand-recipe-tool",
    }
    try:
        with httpx.Client(
            base_url=API_ROOT, headers=headers, transport=transport, timeout=30.0
        ) as client:
            repo = client.get(f"/repos/{config.repo}")
            if repo.status_code != 200:
                raise BackupError(
                    f"GitHub returned status {repo.status_code} for the backup repo."
                )
            # Saved recipes are private, and the site's own repo is public.
            if repo.json().get("private") is not True:
                raise BackupError("The backup repo isn't private, so nothing was sent.")

            path = f"/repos/{config.repo}/contents/{BACKUP_FILE}"
            current = client.get(path)
            if current.status_code not in (200, 404):
                raise BackupError(
                    f"GitHub returned status {current.status_code} reading the backup."
                )
            sha = current.json().get("sha") if current.status_code == 200 else None
            if sha == _git_blob_sha(content):
                return False

            body = {
                "message": f"Back up recipes, {_now().date().isoformat()}",
                "content": base64.b64encode(content).decode("ascii"),
            }
            if sha:
                body["sha"] = sha
            result = client.put(path, json=body)
            if result.status_code not in (200, 201):
                raise BackupError(
                    f"GitHub returned status {result.status_code} saving the backup."
                )
    except httpx.HTTPError as e:
        raise BackupError("GitHub couldn't be reached.") from e
    except ValueError as e:
        raise BackupError("GitHub sent an unreadable response.") from e
    return True


def run_backup(
    store: RecipeStore,
    config: BackupConfig,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Run one backup and record the outcome where the tool's pages can show it."""
    store.set_setting(LAST_ATTEMPT, _now().isoformat())
    try:
        changed = push_export(store.export_json(), config, transport)
    except BackupError as e:
        store.set_setting(LAST_ERROR, str(e))
        logger.error("Recipe backup failed: %s", e)
        return
    store.set_setting(LAST_SUCCESS, _now().isoformat())
    store.set_setting(LAST_ERROR, "")
    logger.info("Recipe backup finished, changed=%s", changed)


def backup_is_due(store: RecipeStore) -> bool:
    """Due daily, and hourly while the last attempt has failed."""
    attempt = store.get_setting(LAST_ATTEMPT)
    if not attempt or store.get_setting(LAST_ERROR):
        return True
    return _now() - datetime.fromisoformat(attempt) >= BACKUP_INTERVAL


def backup_status(store: RecipeStore) -> str | None:
    """Say when recipes last reached the backup repo, and when they next will."""
    success = store.get_setting(LAST_SUCCESS)
    attempt = store.get_setting(LAST_ATTEMPT)
    if backup_config() is None or not success or not attempt:
        return None
    last = datetime.fromisoformat(success).strftime("%d %b %Y, %H:%M UTC")
    # The hourly check can run up to an hour after the backup is due.
    next_due = datetime.fromisoformat(attempt) + BACKUP_INTERVAL
    latest = next_due + timedelta(seconds=CHECK_INTERVAL_SECONDS)
    return (
        f"Last backup: {last}. Recipes saved since then go in the next one, "
        f"by {latest.strftime('%d %b, %H:%M UTC')}."
    )


def backup_warning(store: RecipeStore) -> str | None:
    """Explain what's wrong with backups, if anything is."""
    if backup_config() is None:
        return "Backups aren't set up, so the Render disk holds your only copy."
    error = store.get_setting(LAST_ERROR)
    if error:
        return f"The last backup failed: {error}"
    success = store.get_setting(LAST_SUCCESS)
    if success and _now() - datetime.fromisoformat(success) > STALE_AFTER:
        return "No backup has succeeded in the last 48 hours."
    return None


async def run_forever(db_path: Path, config: BackupConfig) -> None:
    """Check hourly, for the life of the app. A failure never stops the site."""
    while True:
        try:
            store = await asyncio.to_thread(RecipeStore, db_path)
            if await asyncio.to_thread(backup_is_due, store):
                await asyncio.to_thread(run_backup, store, config)
        except (OSError, sqlite3.Error, ValueError):
            logger.exception("Recipe backup check failed")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
