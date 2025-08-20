#!/usr/bin/env python3

from __future__ import annotations

import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import asyncio
import json
import os
import logging
import time
import signal
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Any, List

import discord
from discord.ext import commands
from discord import ui
from yt_dlp import YoutubeDL
import yt_dlp

# --- Config ---
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "token": os.getenv("DISCORD_TOKEN", ""),
    "prefix": "!",
    "owner_id": None,
    "max_queue_size": 200,
    "download_concurrency": 2,
    "cache_ttl_seconds": 900,
    "cache_size_limit": 200,
    "ffmpeg_bitrate": "96k",
    "ffmpeg_threads": 1,
    "prefetch_next": False,
    "idle_disconnect_seconds": 300,
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_conf = json.load(f)
        CONFIG = {**DEFAULT_CONFIG, **user_conf}
    except Exception:
        CONFIG = DEFAULT_CONFIG.copy()
else:
    CONFIG = DEFAULT_CONFIG.copy()

TOKEN = CONFIG.get("token") or os.getenv("DISCORD_TOKEN")
PREFIX = CONFIG.get("prefix", "!")
OWNER_ID = CONFIG.get("owner_id")
if OWNER_ID is not None:
    try:
        OWNER_ID = int(OWNER_ID)
    except Exception:
        OWNER_ID = None
MAX_QUEUE_SIZE = int(CONFIG.get("max_queue_size", 200))
DOWNLOAD_CONCURRENCY = max(1, int(CONFIG.get("download_concurrency", 1)))
CACHE_TTL_SECONDS = int(CONFIG.get("cache_ttl_seconds", 900))
CACHE_SIZE_LIMIT = int(CONFIG.get("cache_size_limit", 200))
FFMPEG_BITRATE = str(CONFIG.get("ffmpeg_bitrate", "96k"))
FFMPEG_THREADS = int(CONFIG.get("ffmpeg_threads", 1))
PREFETCH_NEXT = bool(CONFIG.get("prefetch_next", False))
IDLE_DISCONNECT_SECONDS = int(CONFIG.get("idle_disconnect_seconds", 300))

# Colors
THEME_COLOR = 0x9155FD
OK_COLOR = 0x2ECC71
ERR_COLOR = 0xE74C3C

# helpers
def format_duration(sec: Optional[int]) -> str:
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
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

def make_progress_bar(elapsed: float, total: Optional[float], width: int = 18) -> str:
    if not total or total <= 0:
        return f"{format_duration(int(elapsed))}"
    frac = min(max(elapsed / total, 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {format_duration(int(elapsed))}/{format_duration(int(total))}"

# logging
logger = logging.getLogger("Monica")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
# File handler with utf-8 encoding
fh = RotatingFileHandler("Monica.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)

# discord setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)
tree = bot.tree

# yt-dlp / ffmpeg
# Use flexible format and UA header to reduce 403 & format problems
YTDL_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "no_warnings": True,
    "default_search": "ytsearch",
    "http_chunk_size": 1024 * 1024,
    "geo_bypass": True,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    },
    # do not force source_address here (can cause binding issues on some systems)
}
ytdl = YoutubeDL(YTDL_OPTS)
FFMPEG_BEFORE = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"

PLAYLISTS_PATH = "playlists.json"

def _load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("Failed to load %s: %s", path, e)
    return default

PLAYLISTS = _load_json_safe(PLAYLISTS_PATH, {})

def save_playlists():
    try:
        with open(PLAYLISTS_PATH, "w", encoding="utf-8") as f:
            json.dump(PLAYLISTS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Error saving playlists: %s", e)

# download semaphore & cache
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
_TRACK_CACHE: Dict[str, Dict[str, Any]] = {}

def _cache_get(key: str):
    entry = _TRACK_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        _TRACK_CACHE.pop(key, None)
        return None
    entry["ts"] = time.time()
    return entry["data"]

def _cache_put(key: str, data: dict):
    lean = {
        "title": data.get("title"),
        "webpage_url": data.get("webpage_url"),
        "url": data.get("url"),
        "thumbnail": data.get("thumbnail"),
        "duration": data.get("duration"),
        "uploader": data.get("uploader"),
        "is_live": bool(data.get("is_live") or data.get("live_status") in ("is_live", "started")),
    }
    _TRACK_CACHE[key] = {"data": lean, "ts": time.time()}
    while len(_TRACK_CACHE) > CACHE_SIZE_LIMIT:
        oldest = next(iter(_TRACK_CACHE))
        _TRACK_CACHE.pop(oldest, None)

async def _cache_cleanup_loop():
    while True:
        try:
            now = time.time()
            keys = list(_TRACK_CACHE.keys())
            for k in keys:
                if now - _TRACK_CACHE[k]["ts"] > CACHE_TTL_SECONDS:
                    _TRACK_CACHE.pop(k, None)
        except Exception:
            logger.exception("Cache cleanup error")
        await asyncio.sleep(60 * 5)

# helper to choose a usable audio URL from formats
def _pick_best_audio_url(info: dict) -> Optional[str]:
    if info.get("url"):
        return info.get("url")
    formats = info.get("formats") or []
    if not formats:
        return None

    # Filter formats that contain audio
    candidates = []
    for f in formats:
        acodec = f.get("acodec")
        # prefer those that actually have audio
        if acodec and acodec != "none":
            candidates.append(f)
    if not candidates:
        candidates = formats

    def score(f):
        s = 0
        # prefer m4a then webm then others
        ext = (f.get("ext") or "").lower()
        if ext == "m4a":
            s += 40
        if ext == "webm":
            s += 30
        if f.get("abr"):
            try:
                s += int(float(f.get("abr")))
            except Exception:
                pass
        # prefer http/https protocols
        proto = f.get("protocol") or ""
        if proto.startswith("http"):
            s += 5
        # prefer non-dash if possible
        if f.get("vcodec") in (None, "none"):
            s += 3
        return s

    best = max(candidates, key=score)
    return best.get("url")

# Async deque backed queue (single source of truth for playlist)
class AsyncDequeQueue:
    def __init__(self):
        self._dq = deque()
        self._cond = asyncio.Condition()

    async def put(self, item: Any):
        async with self._cond:
            self._dq.append(item)
            self._cond.notify_all()

    async def get(self, timeout: Optional[float] = None) -> Any:
        async with self._cond:
            if not self._dq:
                if timeout is None:
                    await self._cond.wait()
                else:
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout)
                    except asyncio.TimeoutError:
                        raise
            return self._dq.popleft()

    async def clear(self) -> int:
        async with self._cond:
            n = len(self._dq)
            self._dq.clear()
            return n

    async def remove_by_pred(self, pred) -> int:
        async with self._cond:
            old = list(self._dq)
            new = [x for x in old if not pred(x)]
            removed = len(old) - len(new)
            self._dq = deque(new)
            return removed

    def snapshot(self) -> List[Any]:
        return list(self._dq)

    def qsize(self) -> int:
        return len(self._dq)

    def empty(self) -> bool:
        return not self._dq

# Track abstraction with robust resolve (retry fallback)
class YTDLTrack:
    def __init__(self, data: dict):
        self.data = data
        self.title = data.get("title")
        self.webpage_url = data.get("webpage_url")
        self.stream_url = data.get("url")
        self.thumbnail = data.get("thumbnail")
        self.uploader = data.get("uploader")
        self.duration = data.get("duration")
        self.is_live = bool(data.get("is_live") or data.get("live_status") in ("is_live", "started"))

    @classmethod
    async def resolve(cls, query: str, timeout: float = 20.0):
        key = query.strip()
        cached = _cache_get(key)
        if cached:
            # return a lightweight YTDLTrack-like object from cache data
            return cls(dict(cached))

        loop = asyncio.get_running_loop()
        async with DOWNLOAD_SEMAPHORE:
            data = None
            # Attempt 1: use global ytdl with default flexible format
            try:
                fut = loop.run_in_executor(None, lambda: ytdl.extract_info(query, download=False))
                data = await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("yt-dlp timeout for query=%s", query)
                raise RuntimeError("Tìm kiếm quá lâu, thử lại sau")
            except yt_dlp.utils.DownloadError as e:
                logger.warning("yt-dlp download error (attempt 1): %s", e)
                data = None
            except yt_dlp.utils.ExtractorError as e:
                logger.warning("yt-dlp extractor error (attempt 1): %s", e)
                data = None
            except Exception as e:
                logger.exception("yt-dlp extract_info failed (attempt 1): %s", e)
                data = None

            # Fallback attempt: try a different YoutubeDL instance with alternate formats and retries
            if not data:
                try:
                    alt_opts = dict(YTDL_OPTS)
                    alt_opts["format"] = "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"
                    alt_opts["noplaylist"] = True
                    alt_ytdl = YoutubeDL(alt_opts)
                    fut2 = loop.run_in_executor(None, lambda: alt_ytdl.extract_info(query, download=False))
                    data = await asyncio.wait_for(fut2, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning("yt-dlp fallback timeout for query=%s", query)
                    raise RuntimeError("Tìm kiếm quá lâu (fallback), thử lại sau")
                except yt_dlp.utils.DownloadError as e2:
                    logger.error("yt-dlp download error (fallback): %s", e2)
                    # last resort: try again with the most minimal options
                    try:
                        minimal_opts = dict(YTDL_OPTS)
                        minimal_opts.pop("format", None)
                        minimal_opts["noplaylist"] = True
                        minimal_ytdl = YoutubeDL(minimal_opts)
                        fut3 = loop.run_in_executor(None, lambda: minimal_ytdl.extract_info(query, download=False))
                        data = await asyncio.wait_for(fut3, timeout=timeout)
                    except Exception:
                        logger.exception("yt-dlp final fallback failed")
                        raise RuntimeError("Không thể lấy thông tin nguồn (định dạng/nguồn không khả dụng)")
                except Exception as e2:
                    logger.exception("yt-dlp extract_info failed (fallback): %s", e2)
                    raise RuntimeError("Không thể lấy thông tin nguồn")

        if not data:
            raise RuntimeError("Không tìm thấy kết quả")
        if "entries" in data:
            entries = [e for e in data["entries"] if e]
            if not entries:
                raise RuntimeError("Không tìm thấy mục trong kết quả")
            data = entries[0]

        # If extract_info didn't provide an accessible stream URL, try to pick one from formats
        if not data.get("url"):
            picked = _pick_best_audio_url(data)
            if picked:
                data["url"] = picked

        if not data.get("url"):
            raise RuntimeError("Không lấy được stream URL từ nguồn")

        track = cls(data)
        try:
            if not track.is_live:
                _cache_put(key, data)
        except Exception:
            logger.exception("Cache put error (ignored)")
        return track

# audio creation
def create_audio_source(stream_url: str, volume: float = 1.0):
    vol = max(0.0, min(float(volume), 4.0))
    options = f'-vn -af "volume={vol}" -b:a {FFMPEG_BITRATE} -ar 48000 -threads {FFMPEG_THREADS}'
    kwargs = {"before_options": FFMPEG_BEFORE, "options": options}
    try:
        return discord.FFmpegOpusAudio(stream_url, **kwargs)
    except Exception as e:
        logger.warning("FFmpegOpusAudio failed (%s); fallback to PCM", e)
        return discord.FFmpegPCMAudio(stream_url, **kwargs)

# Player implementation
class MusicPlayer:
    def __init__(self, guild: discord.Guild, text_channel: discord.TextChannel):
        self.bot = bot
        self.guild = guild
        self.text_channel = text_channel
        self.queue = AsyncDequeQueue()
        self.next_event = asyncio.Event()
        self.current: Optional[dict] = None
        self.volume: float = 1.0
        self.loop_mode: bool = False
        self.loop_list: List[dict] = []
        self.history = deque(maxlen=200)
        # capture the loop running when player is created
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # fallback: create/set a loop; this is unlikely but safe
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(self._player_loop())
        self._closing = False
        self._lock = asyncio.Lock()
        self.prefetch_task = None
        self.vc = None
        self.now_message = None
        self.now_update_task = None
        if PREFETCH_NEXT:
            try:
                self.prefetch_task = self._loop.create_task(self._prefetch_worker())
            except Exception:
                self.prefetch_task = None

    async def add_track(self, data: dict):
        async with self._lock:
            size = self.queue.qsize()
            if size >= MAX_QUEUE_SIZE:
                raise RuntimeError("Hàng đợi đã đầy")
            await self.queue.put(data)

    async def clear_all(self):
        async with self._lock:
            count = await self.queue.clear()
            return count

    async def clear_by_title(self, title: str):
        lowered = title.lower()
        removed = await self.queue.remove_by_pred(lambda item: lowered in (item.get("title") or "").lower())
        return removed

    async def enable_loop(self):
        async with self._lock:
            snapshot = []
            if self.current:
                snapshot.append(self.current)
            snapshot.extend(self.queue.snapshot())
            self.loop_list = [dict(item) for item in snapshot]
            self.loop_mode = True
            return len(self.loop_list)

    async def disable_loop(self):
        async with self._lock:
            self.loop_mode = False
            self.loop_list = []

    async def _prefetch_worker(self):
        try:
            while True:
                if self.queue.empty():
                    await asyncio.sleep(1.0)
                    continue
                next_item = None
                snap = self.queue.snapshot()
                if snap:
                    next_item = snap[0]
                if isinstance(next_item, dict) and not next_item.get("url"):
                    try:
                        resolved = await YTDLTrack.resolve(next_item.get("webpage_url") or next_item.get("title") or next_item.get("query"))
                        async with self._lock:
                            snap2 = self.queue.snapshot()
                            if snap2 and snap2[0] == next_item:
                                rest = snap2[1:]
                                await self.queue.clear()
                                newd = dict(resolved.data)
                                newd.update({k: next_item.get(k) for k in ("requested_by",) if next_item.get(k)})
                                await self.queue.put(newd)
                                for it in rest:
                                    await self.queue.put(it)
                    except Exception:
                        pass
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Prefetch worker crashed")

    async def _start_now_update(self, started_at: float, duration: Optional[float]):
        async def updater():
            try:
                while True:
                    if not self.now_message:
                        return
                    try:
                        elapsed = time.time() - started_at
                        bar = make_progress_bar(elapsed, duration)
                        embed = self._build_now_embed(self.current, extra_desc=bar)
                        await self.now_message.edit(embed=embed, view=MusicControls(self.guild.id))
                    except discord.HTTPException:
                        pass
                    await asyncio.sleep(10)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Now update task failed")
        if self.now_update_task and not self.now_update_task.done():
            self.now_update_task.cancel()
        try:
            self.now_update_task = self._loop.create_task(updater())
        except Exception:
            self.now_update_task = asyncio.create_task(updater())

    def _build_now_embed(self, data: dict, extra_desc: Optional[str] = None) -> discord.Embed:
        title = truncate(data.get("title", "Now Playing"), 80)
        embed = discord.Embed(title=title, url=data.get("webpage_url"), color=THEME_COLOR, timestamp=discord.utils.utcnow())
        if data.get("thumbnail"):
            embed.set_thumbnail(url=data.get("thumbnail"))
        desc = f"{'🔴 LIVE —' if data.get('is_live') else '🎧 Now playing —'} {truncate(data.get('title') or 'Unknown', 80)}"
        if extra_desc:
            desc = f"{desc}\n{extra_desc}"
        embed.description = desc
        embed.set_author(name=data.get("uploader") or "Unknown artist")
        embed.add_field(name="⏱️ Thời lượng", value=format_duration(data.get("duration")), inline=True)
        if data.get("requested_by"):
            embed.add_field(name="🙋 Yêu cầu bởi", value=truncate(data.get("requested_by"), 30), inline=True)
        embed.set_footer(text="Monica • Discord Music Bot ✨")
        return embed

    async def _player_loop(self):
        logger.info("Player start guild=%s", self.guild.id)
        try:
            while not self._closing:
                self.next_event.clear()
                try:
                    item = await self.queue.get(timeout=IDLE_DISCONNECT_SECONDS)
                except asyncio.TimeoutError:
                    try:
                        await self.text_channel.send("Không ai phát nhạc à? Mình rời kênh để tiết kiệm tài nguyên — gọi mình lại khi cần nhé ✨")
                    except Exception:
                        pass
                    break

                track = None
                data = None
                if isinstance(item, dict):
                    data = item
                    if not data.get("url"):
                        try:
                            resolved = await YTDLTrack.resolve(data.get("webpage_url") or data.get("title") or data.get("query"))
                            data = dict(resolved.data)
                            if item.get("requested_by"):
                                data["requested_by"] = item.get("requested_by")
                        except Exception as e:
                            logger.exception("Failed to resolve queued dict: %s", e)
                            try:
                                await self.text_channel.send(f"Không thể phát mục đã xếp: {e}")
                            except Exception:
                                pass
                            continue
                    track = YTDLTrack(data)
                elif isinstance(item, YTDLTrack):
                    track = item
                    data = track.data
                elif isinstance(item, str):
                    try:
                        track = await YTDLTrack.resolve(item)
                        data = track.data
                    except Exception as e:
                        logger.exception("Failed to resolve queued string: %s", e)
                        try:
                            await self.text_channel.send(f"Không thể phát bài đã xếp: {e}")
                        except Exception:
                            pass
                        continue
                else:
                    logger.error("Unknown queue item type: %s", type(item))
                    continue

                if not data or not data.get("url"):
                    try:
                        await self.text_channel.send("Không có stream URL cho bài này :<")
                    except Exception:
                        pass
                    continue

                try:
                    src = create_audio_source(data.get("url"), volume=self.volume)
                except Exception as e:
                    logger.exception("create_audio_source failed: %s", e)
                    try:
                        await self.text_channel.send("Lỗi khi tạo nguồn phát")
                    except Exception:
                        pass
                    continue

                vc = self.vc or discord.utils.get(self.bot.voice_clients, guild=self.guild)
                if not vc or not vc.is_connected():
                    try:
                        await self.text_channel.send("Mình chưa vô kênh thoại nào cả :<")
                    except Exception:
                        pass
                    break

                played_at = time.time()

                def _after(err):
                    if err:
                        logger.exception("Playback error guild %s: %s", self.guild.id, err)
                    try:
                        # Use the player's loop to schedule the event set
                        try:
                            self._loop.call_soon_threadsafe(self.next_event.set)
                        except Exception:
                            # fallback: use default loop if needed
                            try:
                                asyncio.get_event_loop().call_soon_threadsafe(self.next_event.set)
                            except Exception:
                                logger.exception("Failed to set next event (double fallback)")
                    except Exception:
                        logger.exception("Failed in _after callback")

                async with self._lock:
                    try:
                        vc.play(src, after=_after)
                        try:
                            vc.source._track_meta = {"title": data.get("title"), "url": data.get("webpage_url")}
                        except Exception:
                            pass
                        self.current = data
                        self.history.append(data)
                    except Exception as e:
                        logger.exception("vc.play failed: %s", e)
                        try:
                            await self.text_channel.send("Lỗi khi phát")
                        except Exception:
                            pass
                        continue

                try:
                    embed = self._build_now_embed(data)
                    # Try to edit existing now_message if possible; otherwise send a new one.
                    if self.now_message:
                        try:
                            edit_fn = getattr(self.now_message, "edit", None)
                            if callable(edit_fn):
                                await edit_fn(embed=embed, view=MusicControls(self.guild.id))
                            else:
                                # can't edit (old object/API), send a new message and replace
                                self.now_message = await self.text_channel.send(embed=embed, view=MusicControls(self.guild.id))
                        except Exception:
                            # if edit fails for any reason, send a fresh message and replace
                            try:
                                self.now_message = await self.text_channel.send(embed=embed, view=MusicControls(self.guild.id))
                            except Exception:
                                logger.exception("Failed to send now-playing embed (both edit and send failed)")
                    else:
                        self.now_message = await self.text_channel.send(embed=embed, view=MusicControls(self.guild.id))

                    await self._start_now_update(played_at, data.get("duration"))
                except Exception:
                    logger.exception("Failed to send now-playing embed")

                await self.next_event.wait()

                try:
                    if self.now_update_task and not self.now_update_task.done():
                        self.now_update_task.cancel()
                        self.now_update_task = None
                except Exception:
                    pass

                try:
                    if self.loop_mode and isinstance(track, YTDLTrack) and track.data:
                        await self.queue.put(track.data)
                except Exception:
                    logger.exception("Failed to requeue for loop mode")

                vc = discord.utils.get(self.bot.voice_clients, guild=self.guild)
                if self.queue.empty() and (not vc or not vc.is_playing()):
                    break

        except asyncio.CancelledError:
            logger.info("Player loop cancelled guild=%s", self.guild.id)
        except Exception as e:
            logger.exception("Unhandled in player loop guild=%s: %s", self.guild.id, e)
        finally:
            try:
                players.pop(self.guild.id, None)
            except Exception:
                pass
            try:
                if self.prefetch_task and not self.prefetch_task.done():
                    self.prefetch_task.cancel()
            except Exception:
                pass
            try:
                if self.now_update_task and not self.now_update_task.done():
                    self.now_update_task.cancel()
            except Exception:
                pass
            logger.info("Player stopped guild=%s", self.guild.id)

    def destroy(self):
        self._closing = True
        try:
            players.pop(self.guild.id, None)
        except Exception:
            pass
        try:
            if self.prefetch_task and not self.prefetch_task.done():
                self.prefetch_task.cancel()
        except Exception:
            pass
        try:
            if self.now_update_task and not self.now_update_task.done():
                self.now_update_task.cancel()
        except Exception:
            pass
        try:
            if not self._task.done():
                self._task.cancel()
        except Exception:
            logger.exception("Error cancelling player task")
        try:
            # clear queue asynchronously
            try:
                self._loop.create_task(self.queue.clear())
            except Exception:
                asyncio.create_task(self.queue.clear())
        except Exception:
            pass
        try:
            if self.vc and self.vc.is_connected():
                try:
                    self._loop.create_task(self.vc.disconnect())
                except Exception:
                    asyncio.create_task(self.vc.disconnect())
        except Exception:
            pass

# global structures
players: Dict[int, MusicPlayer] = {}

def get_player_for_ctx(guild: discord.Guild, text_channel: discord.TextChannel) -> MusicPlayer:
    if guild is None:
        raise RuntimeError("No guild in context")
    player = players.get(guild.id)
    if not player:
        player = MusicPlayer(guild=guild, text_channel=text_channel)
        players[guild.id] = player
    return player

# UI controls
class MusicControls(ui.View):
    def __init__(self, guild_id: int, *, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.guild_id = guild_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message("Bạn phải ở trong kênh thoại để điều chỉnh nhạc", ephemeral=True)
            return False
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc or not vc.is_connected():
            await interaction.response.send_message("Mình chưa kết nối kênh thoại nào cả :<", ephemeral=True)
            return False
        if interaction.user.voice.channel.id != vc.channel.id:
            await interaction.response.send_message("Bạn phải ở cùng kênh thoại với bot để điều khiển", ephemeral=True)
            return False
        return True

    @ui.button(emoji="⏯️", style=discord.ButtonStyle.primary, row=0)
    async def pause_resume(self, inter: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=inter.guild)
        if not vc or not getattr(vc, "source", None):
            await inter.response.send_message("Không có bài nào đang phát", ephemeral=True)
            return
        if vc.is_paused():
            vc.resume(); await inter.response.send_message("▶️ Tiếp tục phát nhạc", ephemeral=True)
        elif vc.is_playing():
            vc.pause(); await inter.response.send_message("⏸️ Đã tạm dừng nhạc", ephemeral=True)
        else:
            await inter.response.send_message("Không thể điều chỉnh hiện tại", ephemeral=True)

    @ui.button(emoji="⏭️", style=discord.ButtonStyle.secondary, row=0)
    async def skip(self, inter: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=inter.guild)
        if not vc or not vc.is_playing():
            await inter.response.send_message("Không có bài nhạc nào để bỏ qua", ephemeral=True); return
        vc.stop(); await inter.response.send_message("⏭️ Đã bỏ qua bài nhạc", ephemeral=True)

    @ui.button(emoji="⏹️", style=discord.ButtonStyle.danger, row=0)
    async def stop(self, inter: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=inter.guild)
        if vc:
            try:
                vc.stop()
            except Exception:
                pass
        player = players.pop(inter.guild.id, None)
        if player:
            player.destroy()
        await inter.response.send_message("Đã dừng phát và xóa hàng đợi", ephemeral=True)

    @ui.button(emoji="📜", style=discord.ButtonStyle.secondary, row=1)
    async def show_queue(self, inter: discord.Interaction, button: ui.Button):
        player = players.get(inter.guild.id)
        if not player or player.queue.empty():
            await inter.response.send_message("Hàng đợi đang trống, bạn thêm nhạc vào nhé ✨", ephemeral=True); return
        upcoming = player.queue.snapshot()[:10]
        text = "\n".join(
            f"{idx+1}. {truncate((item.get('title') if isinstance(item, dict) else str(item)), 50)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
            for idx, item in enumerate(upcoming)
        )
        embed = discord.Embed(title="Queue (next up)", description=text or "Trống", color=0x2F3136)
        await inter.response.send_message(embed=embed, ephemeral=True)

    @ui.button(emoji="🔁", style=discord.ButtonStyle.primary, row=1)
    async def toggle_loop(self, inter: discord.Interaction, button: ui.Button):
        player = players.get(inter.guild.id)
        if not player:
            await inter.response.send_message("Không có phiên phát nào đang chạy", ephemeral=True); return
        player.loop_mode = not player.loop_mode
        await inter.response.send_message(f"🔁 Loop {'Bật' if player.loop_mode else 'Tắt'}", ephemeral=True)

    @ui.button(emoji="💾", style=discord.ButtonStyle.success, row=1)
    async def favorite_current(self, inter: discord.Interaction, button: ui.Button):
        player = players.get(inter.guild.id)
        if not player or not player.current:
            await inter.response.send_message("Không có bài nhạc nào hiện tại để lưu", ephemeral=True); return
        PLAYLISTS.setdefault("favorites", [])
        try:
            PLAYLISTS["favorites"].append(player.current)
            save_playlists()
            await inter.response.send_message("💾 Đã lưu bài hiện tại vào playlist `favorites`.", ephemeral=True)
        except Exception:
            logger.exception("Failed saving favorite")
            await inter.response.send_message("Lưu thất bại.", ephemeral=True)

# Events and commands
@bot.event
async def on_ready():
    logger.info("Bot ready: %s (ID: %s)", bot.user, bot.user.id)
    try:
        await tree.sync()
        logger.info("Synced application commands.")
    except Exception:
        logger.exception("Failed to sync commands")
    try:
        asyncio.create_task(_cache_cleanup_loop())
    except Exception:
        pass
    try:
        await bot.change_presence(activity=discord.Game(name="300 Bài code thiếu nhi ✨"))
    except Exception:
        pass

@bot.event
async def on_voice_state_update(member: discord.Member, before, after):
    if member.id != bot.user.id:
        return
    if before.channel and not after.channel:
        player = players.pop(before.channel.guild.id, None)
        if player:
            player.destroy()
            logger.info("Player destroyed due to bot voice disconnect in guild %s", before.channel.guild.id)

# helper to ensure voice connection when user requests join
async def ensure_connected_for_user(ctx_or_interaction) -> Optional[discord.VoiceClient]:
    user = getattr(ctx_or_interaction, 'author', None) or getattr(ctx_or_interaction, 'user', None)
    guild = getattr(ctx_or_interaction, 'guild', None)
    if not user or not getattr(user, 'voice', None) or not user.voice.channel:
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message("Bạn chưa ở trong kênh thoại nào", ephemeral=True)
            else:
                await ctx_or_interaction.send("Bạn chưa ở trong kênh thoại nào")
        except Exception:
            pass
        return None
    ch = user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=guild)
    try:
        if vc and vc.is_connected():
            if vc.channel.id != ch.id:
                await vc.move_to(ch)
        else:
            vc = await ch.connect()
    except Exception:
        logger.exception("Connect failed")
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message("Không thể kết nối kênh thoại", ephemeral=True)
            else:
                await ctx_or_interaction.send("Không thể kết nối kênh thoại.")
        except Exception:
            pass
        return None
    player = get_player_for_ctx(guild, getattr(ctx_or_interaction, 'channel', None) or getattr(ctx_or_interaction, 'text_channel', None))
    player.vc = vc
    return vc

# central play handler shared by both text and slash
async def handle_play_request(ctx_or_interaction, query: str):
    user = getattr(ctx_or_interaction, 'author', None) or getattr(ctx_or_interaction, 'user', None)
    guild = getattr(ctx_or_interaction, 'guild', None)
    channel_ctx = getattr(ctx_or_interaction, 'channel', None) or getattr(ctx_or_interaction, 'text_channel', None)
    if not user or not getattr(user, 'voice', None) or not user.voice.channel:
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message("Bạn cần vào kênh thoại để yêu cầu phát nhạc", ephemeral=True)
            else:
                await ctx_or_interaction.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc")
        except Exception:
            pass
        return

    ch = user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=guild)
    if not vc or not vc.is_connected():
        try:
            vc = await ch.connect()
        except Exception:
            logger.exception("Connect failed")
            try:
                if isinstance(ctx_or_interaction, discord.Interaction):
                    await ctx_or_interaction.response.send_message("Không thể kết nối vào kênh thoại", ephemeral=True)
                else:
                    await ctx_or_interaction.send("Không thể kết nối kênh thoại.")
            except Exception:
                pass
            return

    player = get_player_for_ctx(guild, channel_ctx)
    player.vc = vc
    if player.queue.qsize() >= MAX_QUEUE_SIZE:
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message("Hàng đợi đã đầy", ephemeral=True)
            else:
                await ctx_or_interaction.send("Hàng đợi đã đầy")
        except Exception:
            pass
        return

    try:
        track = await YTDLTrack.resolve(query)
    except Exception as e:
        logger.exception("Resolve failed: %s", e)
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message(f"Lỗi khi tìm kiếm: {e}", ephemeral=True)
            else:
                await ctx_or_interaction.send(f"Lỗi khi tìm kiếm: {e}")
        except Exception:
            pass
        return

    data = dict(track.data)
    data["requested_by"] = getattr(user, 'display_name', str(user))
    try:
        await player.add_track(data)
    except Exception as e:
        logger.exception("Add track failed: %s", e)
        try:
            if isinstance(ctx_or_interaction, discord.Interaction):
                await ctx_or_interaction.response.send_message(str(e), ephemeral=True)
            else:
                await ctx_or_interaction.send(str(e))
        except Exception:
            pass
        return

    try:
        embed = discord.Embed(description=f"✅ **Đã thêm vào hàng đợi**\n{truncate(track.title, 80)}", color=OK_COLOR)
        embed.set_footer(text="Monica • Đã thêm vào hàng đợi ✨")
        if isinstance(ctx_or_interaction, discord.Interaction):
            await ctx_or_interaction.response.send_message(embed=embed, view=MusicControls(guild.id))
        else:
            await ctx_or_interaction.send(embed=embed)
    except Exception:
        pass

# commands (text & slash)
@bot.command(name="join")
async def text_join(ctx):
    await ensure_connected_for_user(ctx)

@tree.command(name="join", description="Kêu bot vào kênh thoại")
async def slash_join(interaction: discord.Interaction):
    vc = await ensure_connected_for_user(interaction)
    if vc:
        await interaction.response.send_message(f"✅ Đã kết nối tới **{vc.channel.name}**")

@bot.command(name="play")
async def text_play(ctx, *, query: str):
    await handle_play_request(ctx, query)

@tree.command(name="play", description="Phát nhạc từ URL hoặc tên bài nhạc (YouTube)")
@discord.app_commands.describe(query="URL hoặc tên bài (YouTube)")
async def slash_play(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    await handle_play_request(interaction, query)

@bot.command(name="pause")
async def text_pause(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát"); return
    vc.pause(); await ctx.send("⏸️ Đã tạm dừng")

@tree.command(name="pause", description="Tạm dừng nhạc")
async def slash_pause(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True); return
    vc.pause(); await interaction.response.send_message("⏸️ Đã tạm dừng.", ephemeral=True)

@bot.command(name="resume")
async def text_resume(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_paused():
        await ctx.send("Không có bài nhạc nào bị tạm dừng"); return
    vc.resume(); await ctx.send("▶️ Đã tiếp tục phát")

@tree.command(name="resume", description="Tiếp tục phát")
async def slash_resume(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_paused():
        await interaction.response.send_message("Không có bài nhạc nào bị tạm dừng", ephemeral=True); return
    vc.resume(); await interaction.response.send_message("▶️ Tiếp tục phát", ephemeral=True)

@bot.command(name="skip")
async def text_skip(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát để bỏ qua"); return
    vc.stop(); await ctx.send("⏭️ Đã skip bài hiện tại")

@tree.command(name="skip", description="Bỏ qua bài đang phát")
async def slash_skip(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có nhạc đang phát để bỏ qua", ephemeral=True); return
    vc.stop(); await interaction.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)

@bot.command(name="queue")
async def text_queue(ctx):
    player = players.get(ctx.guild.id)
    if not player or player.queue.empty():
        await ctx.send("Hàng đợi trống"); return
    upcoming = player.queue.snapshot()[:10]
    text = "\n".join(
        f"{idx+1}. {truncate(item.get('title') if isinstance(item, dict) else str(item), 45)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
        for idx, item in enumerate(upcoming)
    )
    await ctx.send(embed=discord.Embed(title="Queue (next up)", description=text, color=0x2F3136))

@tree.command(name="queue", description="Hiện 10 bài nhạc tiếp theo")
async def slash_queue(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or player.queue.empty():
        await interaction.response.send_message("Hàng đợi trống", ephemeral=True); return
    upcoming = player.queue.snapshot()[:10]
    text = "\n".join(
        f"{idx+1}. {truncate(item.get('title') if isinstance(item, dict) else str(item), 45)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
        for idx, item in enumerate(upcoming)
    )
    await interaction.response.send_message(embed=discord.Embed(title="Queue (next up)", description=text, color=0x2F3136), ephemeral=True)

@bot.command(name="now")
async def text_now(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not getattr(vc, "source", None):
        await ctx.send("Không có bài nào đang phát"); return
    player = players.get(ctx.guild.id)
    if player and player.current:
        data = player.current
        await ctx.send(embed=player._build_now_embed(data))
    else:
        meta = getattr(vc.source, "_track_meta", None)
        if meta:
            await ctx.send(f"Now playing: {meta.get('title')}")
        else:
            await ctx.send("Không có metadata hiện tại.")

@tree.command(name="now", description="Hiện bài đang phát")
async def slash_now(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not getattr(vc, "source", None):
        await interaction.response.send_message("Không có bài nào đang phát", ephemeral=True); return
    player = players.get(interaction.guild.id)
    if player and player.current:
        data = player.current
        await interaction.response.send_message(embed=player._build_now_embed(data))
    else:
        meta = getattr(vc.source, "_track_meta", None)
        if meta:
            await interaction.response.send_message(f"Now playing: {meta.get('title')}")
        else:
            await interaction.response.send_message("Không có metadata hiện tại.", ephemeral=True)

@bot.command(name="volume")
async def text_volume(ctx, vol: float):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có phiên chơi nhạc đang hoạt động"); return
    player.volume = max(0.0, min(vol, 4.0))
    await ctx.send(f"🔊 Đã đặt âm lượng (áp dụng cho bài tiếp theo): {player.volume}")

@tree.command(name="volume", description="Đặt âm lượng (áp dụng cho bài tiếp theo)")
async def slash_volume(interaction: discord.Interaction, vol: float):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có phiên chơi nhạc đang hoạt động", ephemeral=True); return
    player.volume = max(0.0, min(vol, 4.0))
    await interaction.response.send_message(f"🔊 Đã đặt âm lượng (áp dụng cho bài tiếp theo): {player.volume}", ephemeral=True)

@bot.command(name="list_playlists")
async def text_list_playlists(ctx):
    if not PLAYLISTS:
        await ctx.send("Chưa có playlist nào."); return
    keys = sorted(PLAYLISTS.keys())
    await ctx.send("Playlist đã lưu:\n" + "\n".join(keys))

@tree.command(name="list_playlists", description="Liệt kê các playlist đã lưu")
async def slash_list_playlists(interaction: discord.Interaction):
    if not PLAYLISTS:
        await interaction.response.send_message("Chưa có playlist nào.", ephemeral=True); return
    keys = sorted(PLAYLISTS.keys())
    await interaction.response.send_message("Playlist đã lưu:\n" + "\n".join(keys), ephemeral=True)

@bot.command(name="save_playlist")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_save_playlist(ctx, name: str):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có playlist để lưu.")
        return
    items = player.queue.snapshot()
    PLAYLISTS[name] = items
    save_playlists()
    await ctx.send(f"✅ Đã lưu playlist `{name}`.")

@tree.command(name="save_playlist", description="Lưu playlist hiện tại")
async def slash_save_playlist(interaction: discord.Interaction, name: str):
    if OWNER_ID is not None and interaction.user.id != int(OWNER_ID):
        await interaction.response.send_message("Chỉ owner mới có thể dùng lệnh này.", ephemeral=True)
        return
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có playlist để lưu.", ephemeral=True)
        return
    items = player.queue.snapshot()
    PLAYLISTS[name] = items
    save_playlists()
    await interaction.response.send_message(f"✅ Đã lưu playlist `{name}`.", ephemeral=True)

@bot.command(name="play_playlist")
async def text_play_playlist(ctx, name: str):
    if name not in PLAYLISTS:
        await ctx.send("Không tìm thấy playlist.")
        return
    user = ctx.author
    if not user.voice or not user.voice.channel:
        await ctx.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc")
        return
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        try:
            await user.voice.channel.connect()
        except Exception:
            logger.exception("Connect failed (text)")
            await ctx.send("Không thể kết nối kênh thoại")
            return
    player = get_player_for_ctx(ctx.guild, ctx.channel)
    for item in PLAYLISTS[name]:
        await player.add_track(item)
    await ctx.send(f"✅ Đã thêm playlist `{name}` vào hàng đợi")

@tree.command(name="play_playlist", description="Phát playlist đã lưu theo tên")
async def slash_play_playlist(interaction: discord.Interaction, name: str):
    if name not in PLAYLISTS:
        await interaction.response.send_message("Không tìm thấy playlist", ephemeral=True)
        return
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Bạn cần vào kênh thoại để yêu cầu phát nhạc", ephemeral=True)
        return
    ch = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        try:
            await ch.connect()
        except Exception:
            logger.exception("Connect failed")
            await interaction.response.send_message("Không thể kết nối vào kênh thoại", ephemeral=True)
            return
    player = get_player_for_ctx(interaction.guild, interaction.channel)
    for item in PLAYLISTS[name]:
        await player.add_track(item)
    await interaction.response.send_message(f"✅ Đã thêm playlist `{name}` vào hàng đợi", ephemeral=True)

@bot.command(name="shutdown")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_shutdown(ctx):
    await ctx.send("⚠️ Đang tắt bot...")
    save_playlists()
    try:
        snap = {}
        for gid, p in list(players.items()):
            try:
                snap[str(gid)] = p.queue.snapshot()
            except Exception:
                pass
        with open("queues_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to snapshot queues")
    for vc in list(bot.voice_clients):
        try:
            await vc.disconnect()
        except Exception:
            pass
    await bot.close()

@tree.command(name="shutdown", description="Tắt bot")
async def slash_shutdown(interaction: discord.Interaction):
    if OWNER_ID is not None and interaction.user.id != int(OWNER_ID):
        await interaction.response.send_message("Chỉ owner mới có thể tắt bot", ephemeral=True)
        return
    await interaction.response.send_message("⚠️ Đang tắt bot...")
    save_playlists()
    try:
        snap = {}
        for gid, p in list(players.items()):
            try:
                snap[str(gid)] = p.queue.snapshot()
            except Exception:
                pass
        with open("queues_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to snapshot queues")
    for vc in list(bot.voice_clients):
        try:
            await vc.disconnect()
        except Exception:
            pass
    await bot.close()

@bot.command(name="clear_all")
async def text_clear_all(ctx):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có hàng đợi nào để xóa")
        return
    count = await player.clear_all()
    await ctx.send(f"🗑️ Đã xóa {count} bài trong hàng đợi.")

@tree.command(name="clear_all", description="Xóa toàn bộ hàng đợi")
async def slash_clear_all(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có hàng đợi nào để xóa", ephemeral=True)
        return
    count = await player.clear_all()
    await interaction.response.send_message(f"🗑️ Đã xóa {count} bài trong hàng đợi.", ephemeral=True)

@bot.command(name="clear")
async def text_clear(ctx, *, title: str):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có hàng đợi nào để xóa")
        return
    removed = await player.clear_by_title(title)
    if removed:
        await ctx.send(f"✅ Đã xóa {removed} mục trùng với '{title}' khỏi hàng đợi.")
    else:
        await ctx.send(f"Không tìm thấy bài nào khớp với '{title}'.")

@tree.command(name="clear", description="Xóa bài khỏi hàng đợi theo tên (partial match, case-insensitive)")
async def slash_clear(interaction: discord.Interaction, title: str):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có hàng đợi nào để xóa", ephemeral=True)
        return
    removed = await player.clear_by_title(title)
    if removed:
        await interaction.response.send_message(f"✅ Đã xóa {removed} mục trùng với '{title}' khỏi hàng đợi.")
    else:
        await interaction.response.send_message(f"Không tìm thấy bài nào khớp với '{title}'.", ephemeral=True)

@bot.command(name="loop_all")
async def text_loop_all(ctx):
    player = players.get(ctx.guild.id)
    if not player or (not player.queue.snapshot() and not player.current):
        await ctx.send("Không có hàng đợi hoặc bài đang phát để vòng lặp.")
        return
    count = await player.enable_loop()
    await ctx.send(f"🔁 Bật loop cho {count} bài (queue hiện tại).")

@tree.command(name="loop_all", description="Bật vòng lặp cho toàn bộ hàng đợi hiện tại")
async def slash_loop_all(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or (not player.queue.snapshot() and not player.current):
        await interaction.response.send_message("Không có hàng đợi hoặc bài đang phát để vòng lặp.", ephemeral=True)
        return
    count = await player.enable_loop()
    await interaction.response.send_message(f"🔁 Bật loop cho {count} bài (queue hiện tại).")

@bot.command(name="unloop")
async def text_unloop(ctx):
    player = players.get(ctx.guild.id)
    if not player or not player.loop_mode:
        await ctx.send("Chưa bật loop.")
        return
    await player.disable_loop()
    await ctx.send("⛔ Đã tắt loop.")

@tree.command(name="unloop", description="Tắt chế độ loop")
async def slash_unloop(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or not player.loop_mode:
        await interaction.response.send_message("Chưa bật loop.", ephemeral=True)
        return
    await player.disable_loop()
    await interaction.response.send_message("⛔ Đã tắt loop.", ephemeral=True)

@bot.command(name="help")
async def text_help(ctx):
    embed = discord.Embed(
        title="Monica Bot — Trợ giúp", 
        color=0x5865F2, 
        description="Các lệnh chính :"
    )
    embed.add_field(name="/join  |  !join", value="Kêu bot vào kênh thoại của bạn", inline=False)
    embed.add_field(name="/play <query>  |  !play <query>", value="Thêm bài vào hàng đợi (link hoặc tên bài nhạc).", inline=False)
    embed.add_field(name="/pause / /resume / /skip / /stop", value="Dừng / tiếp tục / bỏ qua / dừng và xóa hàng đợi", inline=False)
    embed.add_field(name="/queue / /now / /volume", value="Xem hàng đợi (10 bài tiếp theo), hiển thị bài đang phát, đặt âm lượng", inline=False)
    embed.add_field(name="/clear_all", value="Xóa toàn bộ hàng đợi", inline=False)
    embed.add_field(name="/clear <tên>", value="Xóa các bài khớp với tên khỏi hàng đợi", inline=False)
    embed.add_field(name="/loop_all / /unloop", value="Bật/tắt vòng lặp cho toàn bộ hàng đợi hiện tại", inline=False)
    embed.add_field(name="/list_playlists / /save_playlist / /play_playlist", value="Quản lý playlist đã lưu (chưa ổn định, không khuyến khích dùng 😭)", inline=False)

    # Disclaimer
    disclaimer_text = (
        "⚠️ **Disclaimer**\n"
        "Monica-Music-Bot is for personal and educational use only.\n"
        "Using it with YouTube or copyrighted sources may violate their Terms of Service.\n"
        "The author is not responsible for misuse of this software."
    )
    embed.add_field(name="Disclaimer", value=disclaimer_text, inline=False)

    embed.set_footer(text="Monica Music Bot v2.1 • By shio")
    await ctx.send(embed=embed)


@tree.command(name="help", description="Hiện help embed")
async def slash_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Monica Bot — Help", 
        color=0x5865F2, 
        description="Các lệnh chính:"
    )
    embed.add_field(name="/join  |  !join", value="Kêu bot vào kênh thoại của bạn", inline=False)
    embed.add_field(name="/play <query>  |  !play <query>", value="Thêm bài vào hàng đợi (link hoặc tên).", inline=False)
    embed.add_field(name="/pause / /resume / /skip / /stop", value="Dừng / tiếp tục / bỏ qua / dừng và xóa hàng đợi", inline=False)
    embed.add_field(name="/queue / /now / /volume", value="Xem hàng đợi (10 bài tiếp theo), hiển thị bài đang phát, đặt âm lượng", inline=False)

    disclaimer_text = (
        "⚠️ **Disclaimer**\n"
        "Monica-Music-Bot is for personal and educational use only.\n"
        "Using it with YouTube or copyrighted sources may violate their Terms of Service.\n"
        "The author is not responsible for misuse of this software."
    )
    embed.add_field(name="Disclaimer", value=disclaimer_text, inline=False)

    await interaction.response.send_message(embed=embed)


# error handlers
@bot.event
async def on_command_error(ctx, error):
    logger.exception("Command error: %s", error)
    try:
        await ctx.send("Đã có lỗi xảy ra. Mình đã ghi lại log để shio kiểm tra.")
    except Exception:
        pass

@bot.event
async def on_app_command_error(interaction, error):
    logger.exception("App command error: %s", error)
    try:
        await interaction.response.send_message("Đã có lỗi xảy ra. Mình đã ghi lại log để shio kiểm tra.", ephemeral=True)
    except Exception:
        pass

# Leave and Stop
@bot.command(name="leave")
async def text_leave(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        await ctx.send("Mình chưa kết nối kênh thoại nào cả :<")
        return
    try:
        await vc.disconnect()
    finally:
        p = players.pop(ctx.guild.id, None)
        if p:
            p.destroy()
    await ctx.send("Mình đã rời kênh thoại rùi, hẹn gặp lại :3")

@tree.command(name="leave", description="Bot rời kênh thoại")
async def slash_leave(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("Mình chưa kết nối kênh thoại nào cả :<", ephemeral=True)
        return
    try:
        await vc.disconnect()
    finally:
        p = players.pop(interaction.guild.id, None)
        if p:
            p.destroy()
    await interaction.response.send_message("Mình đã rời kênh thoại, hẹn gặp lại :3")

@bot.command(name="stop")
async def text_stop(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if vc:
        try:
            vc.stop()
        except Exception:
            pass
    player = players.pop(ctx.guild.id, None)
    if player:
        await player.clear_all()
        player.destroy()
    await ctx.send("⏹️ Đã dừng phát và xóa hàng đợi")

@tree.command(name="stop", description="Dừng phát nhạc và xóa hàng đợi")
async def slash_stop(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if vc:
        try:
            vc.stop()
        except Exception:
            pass
    player = players.pop(interaction.guild.id, None)
    if player:
        await player.clear_all()
        player.destroy()
    await interaction.response.send_message("⏹️ Đã dừng phát và xóa hàng đợi", ephemeral=True)

def _graceful_shutdown_sync():
    logger.info("Signal received: saving playlists and closing")
    try:
        save_playlists()
    except Exception:
        pass
    try:
        snap = {}
        for gid, p in list(players.items()):
            try:
                snap[str(gid)] = p.queue.snapshot()
            except Exception:
                pass
        with open("queues_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed snapshot during shutdown")

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.add_signal_handler(signal.SIGINT, _graceful_shutdown_sync)
        loop.add_signal_handler(signal.SIGTERM, _graceful_shutdown_sync)
    except Exception:
        pass

    if not TOKEN:
        logger.error("Token missing: update config.json or set DISCORD_TOKEN env var.")
    else:
        try:
            bot.run(TOKEN)
        except Exception as e:
            logger.exception("Bot terminated with exception: %s", e)