"""Queue persistence (optional).
Non-intrusive: only used if queue_persistence_enabled True in config.
Stores per-guild queue snapshots periodically and on shutdown.
"""
import json, time, asyncio, logging
from typing import Dict, Any

logger = logging.getLogger("Monica.QueuePersist")

_PERSIST_PATH = "queues_persisted.json"
_SAVE_INTERVAL = 60  # seconds
_running = False
_task = None

async def _periodic_dump(get_players_callable):
    global _running
    while _running:
        try:
            await asyncio.sleep(_SAVE_INTERVAL)
            await save_now(get_players_callable())
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Periodic dump failed", exc_info=True)

async def start(get_players_callable):
    """Start periodic persistence (idempotent)."""
    global _running, _task
    if _running:
        return
    _running = True
    _task = asyncio.create_task(_periodic_dump(get_players_callable))

async def stop():
    global _running, _task
    _running = False
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except Exception:
            pass

async def save_now(players: Dict[int, Any]):
    data = {}
    for gid, p in players.items():
        try:
            snap = p.queue.snapshot(limit=50)
            arr = []
            for item in snap:
                if isinstance(item, dict):
                    arr.append({k: item.get(k) for k in ("title","webpage_url","duration","is_live")})
                else:
                    arr.append(str(item))
            data[str(gid)] = {"queue": arr, "ts": int(time.time())}
        except Exception:
            logger.debug("Snapshot fail guild %s", gid, exc_info=True)
    try:
        loop = asyncio.get_event_loop()
        def _write():
            try:
                with open(_PERSIST_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed writing %s", _PERSIST_PATH)
        await loop.run_in_executor(None, _write)
    except Exception:
        logger.exception("Failed scheduling persist write", exc_info=True)

async def load_into(players: Dict[int, Any], make_player_func, guild_lookup_func):
    try:
        with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0
    except Exception:
        logger.exception("Failed reading %s", _PERSIST_PATH); return 0
    restored = 0
    for gid_s, obj in data.items():
        try:
            gid = int(gid_s)
            guild = guild_lookup_func(gid)
            if not guild: continue
            if gid in players:
                continue
            qitems = obj.get("queue") or []
            # create a placeholder player (text_channel selection left to runtime when user interacts)
            # cannot fully restore without a channel; skip if none.
            restored += 1
        except Exception:
            logger.debug("Restore fail for guild %s", gid_s, exc_info=True)
    return restored
