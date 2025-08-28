"""
Utility functions for Monica Bot
"""
import time
import json
import os
import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger("Monica.Utils")

# Colors
THEME_COLOR = 0x9155FD
OK_COLOR = 0x2ECC71
ERR_COLOR = 0xE74C3C

def format_duration(sec: Optional[int]) -> str:
    """Format duration in seconds to readable string."""
    if sec is None:
        return "??:??"
    if sec == 0:
        return "LIVE"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def truncate(text: Optional[str], n: int = 60) -> str:
    """Truncate text to specified length with ellipsis."""
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

def make_progress_bar(elapsed: float, total: Optional[float], width: int = 18) -> str:
    """Create a progress bar string."""
    if not total or total <= 0:
        return f"{format_duration(int(elapsed))}"
    frac = min(max(elapsed / total, 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {format_duration(int(elapsed))}/{format_duration(int(total))}"

def write_snapshot_file(snap: dict):
    """Blocking atomic write of snapshot to disk (use in executor or sync contexts)."""
    try:
        tmp = f"queues_snapshot.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            os.replace(tmp, "queues_snapshot.json")
        except Exception:
            # fallback to rename
            try:
                os.remove("queues_snapshot.json")
            except Exception:
                pass
            os.rename(tmp, "queues_snapshot.json")
    except Exception:
        logger.exception("Failed to write snapshot file")


def _safe_create_task(coro, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.Task:
    """Create an asyncio.Task in a safe way using provided loop or current running loop.

    Falls back to loop.call_soon_threadsafe/asyncio.ensure_future when needed.
    """
    try:
        if loop is None:
            loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        # Not in event loop thread; schedule creation via call_soon_threadsafe
        if loop is None:
            loop = asyncio.get_event_loop()
        fut = asyncio.Future()

        def _create():
            try:
                t = asyncio.ensure_future(coro)
                fut.set_result(t)
            except Exception as e:
                fut.set_exception(e)

        try:
            loop.call_soon_threadsafe(_create)
            return asyncio.get_event_loop().run_until_complete(fut)
        except Exception:
            # Worst-case fallback: ensure_future in current context
            return asyncio.ensure_future(coro)
