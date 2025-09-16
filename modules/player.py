import time
import asyncio
from collections import deque
from typing import Dict, Any, Optional
import discord
from modules.queue import AsyncDequeQueue
from modules.ytdl_track import YTDLTrack
from modules.metrics import metric_inc
from modules.utils import make_progress_bar, truncate, format_duration, _safe_create_task

# Các biến toàn cục sẽ được gán từ bot.py khi khởi tạo
MAX_QUEUE_SIZE = None
PREFETCH_NEXT = None
NOW_UPDATE_INTERVAL = None
IDLE_DISCONNECT_SECONDS = None
STREAM_PROFILE = None
logger = None
create_audio_source_wrapper = None
get_voice_client_cached = None

class MusicPlayer:
    """Music player for a Discord guild with queue management and playback control.

    This is a behavior-preserving extraction of the implementation in bot.py.
    All cross-module globals are injected from bot.py after import to avoid
    altering runtime logic. Additional safety metrics / latency hooks can be
    layered without removing existing semantics.
    """

    def __init__(self, guild: discord.Guild, text_channel: discord.TextChannel) -> None:
        from bot import bot as _bot  # lazy import to avoid circular
        self.bot = _bot
        self.guild = guild
        self.text_channel = text_channel
        self.queue: AsyncDequeQueue = AsyncDequeQueue()
        self.next_event: asyncio.Event = asyncio.Event()
        self.current: Optional[Dict[str, Any]] = None
        self.volume: float = 1.0
        self.loop_mode: bool = False
        self.loop_one: bool = False
        self.history: deque[Dict[str, Any]] = deque(maxlen=200)
        self._suppress_loop_requeue_once: bool = False
        # Capture the running event loop at construction time for thread-safe callbacks
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback: no running loop at construction (should be rare); will rely on bot internals later
            self._loop = None
        try:
            if self._loop:
                self._task = asyncio.get_running_loop().create_task(self._player_loop())
            else:
                self._task = asyncio.create_task(self._player_loop())
        except Exception:
            self._task = asyncio.create_task(self._player_loop())
        self._closing = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self.prefetch_task: Optional[asyncio.Task] = None
        self.vc: Optional[discord.VoiceClient] = None
        self.now_message: Optional[discord.Message] = None
        self.now_update_task: Optional[asyncio.Task] = None
        self._last_active = time.time()
        self._idle_warned = False
        # Watchdog restart tracking: attempts within window
        self._watchdog_attempts = 0
        self._watchdog_window_start = 0.0
        self._watchdog_max_attempts = 4
        self._watchdog_window_seconds = 60.0
        # Simple per-guild rate limiter for user-facing messages (token bucket)
        self._msg_tokens = 5
        self._msg_token_last = time.time()
        self._msg_token_rate = 1  # tokens per 10s
        # Prefetch task dedupe & global concurrency
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}
        self._prefetch_global_sem = asyncio.Semaphore(3)
        if PREFETCH_NEXT:
            try:
                # use safe task creation helper to ensure consistent scheduling
                self.prefetch_task = _safe_create_task(self._prefetch_worker(), loop=self._loop)
            except Exception:
                try:
                    self.prefetch_task = asyncio.create_task(self._prefetch_worker())
                except Exception:
                    self.prefetch_task = None

    async def _send_rate_limited(self, content=None, **kwargs):
        """Send a channel message but rate-limit frequent error/notify messages per guild.

        Uses a simple token-bucket stored on the player instance. Returns the message
        or None if suppressed.
        """
        try:
            now = time.time()
            # Refill tokens (rate tokens per 10s interval)
            elapsed = now - self._msg_token_last
            if elapsed > 0:
                # convert rate to tokens per second baseline (rate per 10s)
                refill = (self._msg_token_rate / 10.0) * elapsed
                self._msg_tokens = min(10, self._msg_tokens + refill)
                self._msg_token_last = now
            if self._msg_tokens < 1:
                # suppressed due to rate limit
                return None
            self._msg_tokens -= 1
            # Prefer using send with kwargs (embed/view/etc)
            try:
                return await self.text_channel.send(content, **kwargs)
            except discord.HTTPException as e:
                # Try to record a metric keyed by HTTP status when available
                try:
                    status = getattr(e, 'status', None) or getattr(e, 'status_code', None)
                    # some versions may expose response.status on the exception
                    if not status:
                        resp = getattr(e, 'response', None)
                        status = getattr(resp, 'status', None) if resp is not None else None
                    if status:
                        try:
                            metric_inc(f"http_send_status_{status}")
                        except Exception:
                            pass
                    else:
                        try:
                            metric_inc("http_send_status_unknown")
                        except Exception:
                            pass
                except Exception:
                    pass
                return None
            except Exception:
                # Non-HTTP errors: best-effort attempt failed
                try:
                    metric_inc("http_send_fail")
                except Exception:
                    pass
                return None
        except Exception:
            try:
                # best-effort: attempt regular send without token gating
                try:
                    return await self.text_channel.send(content, **kwargs)
                except Exception:
                    return None
            except Exception:
                return None

    @staticmethod
    def _tracks_equal(a: Any, b: Any) -> bool:
        try:
            if a is b:
                return True
            if isinstance(a, dict) and isinstance(b, dict):
                au = a.get("webpage_url") or a.get("url")
                bu = b.get("webpage_url") or b.get("url")
                if au and bu and au == bu:
                    return True
                at, bt = a.get("title"), b.get("title")
                ad, bd = a.get("duration"), b.get("duration")
                if at and bt and ad is not None and bd is not None and at == bt and ad == bd:
                    return True
        except Exception:
            pass
        return False

    def last_finished(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.history:
                return None
            for item in reversed(self.history):
                if not self.current or not self._tracks_equal(item, self.current):
                    return item
        except Exception:
            return None
        return None

    async def play_previous_now(self) -> Optional[Dict[str, Any]]:
        prev = self.last_finished()
        if not prev:
            return None
        async with self._lock:
            try:
                await self.queue.put_front(prev)
            except Exception:
                logger.debug("play_previous_now: failed to put prev track to front", exc_info=True)
                return None
            try:
                self._last_active = time.time()
            except Exception:
                logger.debug("play_previous_now: failed updating last_active", exc_info=True)
            try:
                if self.vc and (self.vc.is_playing() or self.vc.is_paused()):
                    self.vc.stop()
            except Exception:
                logger.debug("play_previous_now: failed stopping current playback", exc_info=True)
        return prev

    async def add_track(self, data: Dict[str, Any]) -> None:
        async with self._lock:
            size = self.queue.qsize()
            if size >= MAX_QUEUE_SIZE:
                raise RuntimeError("Hàng đợi đã đầy")
            try:
                data.setdefault("_enqueued_at", time.time())
            except Exception:
                pass
            await self.queue.put(data)
            try:
                self._last_active = time.time(); self._idle_warned = False
            except Exception:
                logger.debug("add_track: failed updating activity timestamps", exc_info=True)
        # After enqueue, refresh controls so buttons (Skip/Queue) are enabled
        try:
            if self.now_message is not None:
                from modules.ui_components import MusicControls as _Controls
                _view = _Controls(self.guild.id)
                _msg = await self.now_message.edit(view=_view)
                try:
                    _view.message = _msg
                except Exception:
                    pass
        except Exception:
            pass

    async def clear_all(self):
        async with self._lock:
            return await self.queue.clear()

    async def clear_by_title(self, title: str):
        lowered = title.lower()
        return await self.queue.remove_by_pred(lambda item: lowered in (item.get("title") or "").lower())

    async def enable_loop(self):
        async with self._lock:
            self.loop_mode = True; self.loop_one = False
            size = (1 if self.current else 0) + self.queue.qsize()
            logger.info("Loop-all enabled for guild=%s size=%s", self.guild.id, size)
            return size

    async def disable_loop(self):
        async with self._lock:
            self.loop_mode = False; self.loop_one = False
            logger.info("Loop-all disabled for guild=%s", self.guild.id)

    async def enable_loop_one(self):
        async with self._lock:
            self.loop_one = True; self.loop_mode = False
            logger.info("Loop-one enabled for guild=%s", self.guild.id)

    async def disable_loop_one(self):
        async with self._lock:
            self.loop_one = False
            logger.info("Loop-one disabled for guild=%s", self.guild.id)

    async def _prefetch_worker(self):
        try:
            idle_sleep = 0.5
            max_concurrent_prefetch = 2
            semaphore = asyncio.Semaphore(max_concurrent_prefetch)
            while True:
                if self._closing:
                    return
                if self.queue.empty():
                    metric_inc("prefetch_idle_cycles")
                    await asyncio.sleep(idle_sleep)
                    idle_sleep = min(idle_sleep * 1.5, 5.0)
                    continue
                idle_sleep = 0.5
                snap = self.queue.snapshot(limit=10)
                if not snap:
                    continue
                tasks = []
                window = min(5, max(1, max(len(snap)//2, 1)))
                for i, item in enumerate(snap[:window]):
                    if isinstance(item, dict) and not item.get("url"):
                        q = item.get("webpage_url") or item.get("title") or item.get("query")
                        if q:
                            # dedupe by query key (webpage_url)
                            key = q.strip()
                            if key in self._prefetch_tasks:
                                # already fetching this query; skip
                                continue
                            # create Task so it can be cancelled reliably later
                            try:
                                # acquire global prefetch semaphore to cap total concurrency
                                async def prefetch_wrapper(itm, pos, sem, k):
                                    async with self._prefetch_global_sem:
                                        try:
                                            return await self._prefetch_single_track(itm, pos, sem)
                                        finally:
                                            self._prefetch_tasks.pop(k, None)
                                t = _safe_create_task(prefetch_wrapper(item, i, semaphore, key), loop=self._loop)
                            except Exception:
                                try:
                                    t = asyncio.create_task(self._prefetch_single_track(item, i, semaphore))
                                except Exception:
                                    t = self._prefetch_single_track(item, i, semaphore)
                            self._prefetch_tasks[key] = t
                            tasks.append(t)
                if tasks:
                    # New optimized path using as_completed; fallback to legacy on error
                    try:
                        start = time.time()
                        for coro in asyncio.as_completed(tasks, timeout=10.0):
                            try:
                                await coro
                            except Exception:
                                pass
                            if time.time() - start > 10.0:
                                break
                    except Exception:
                        # Fallback to legacy FIRST_COMPLETED approach (preserve original logic path)
                        try:
                            done, pending = await asyncio.wait(tasks, timeout=10.0, return_when=asyncio.FIRST_COMPLETED)
                            for t in pending:
                                try:
                                    if hasattr(t, 'cancel') and not t.done():
                                        t.cancel()
                                except Exception:
                                    pass
                            for t in done:
                                try: await t
                                except Exception: pass
                        except Exception:
                            for t in tasks:
                                try:
                                    if hasattr(t, 'cancel') and not getattr(t, 'done', lambda: False)():
                                        t.cancel()
                                except Exception:
                                    pass
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Prefetch worker crashed")

    async def destroy(self):
        """Cancel background tasks and cleanup prefetched tasks when player is destroyed."""
        self._closing = True
        try:
            if self.prefetch_task and not self.prefetch_task.done():
                try: self.prefetch_task.cancel()
                except Exception: pass
        except Exception:
            pass
        try:
            for k, t in list(self._prefetch_tasks.items()):
                try:
                    if hasattr(t, 'cancel') and not t.done():
                        t.cancel()
                except Exception:
                    pass
        except Exception:
            pass

    async def _prefetch_single_track(self, head_item, position, semaphore):
        async with semaphore:
            query = head_item.get("webpage_url") or head_item.get("title") or head_item.get("query")
            if not query: return
            key = query.strip()
            try:
                resolved = await YTDLTrack.resolve(query, timeout=15.0)
                metric_inc("prefetch_resolved")
                async with self._lock:
                    cur = self.queue.snapshot()
                    if cur and len(cur) > position and cur[position] is head_item:
                        req_by = head_item.get("requested_by"); req_by_id = head_item.get("requested_by_id")
                        try:
                            head_item.clear(); head_item.update(resolved.data)
                        except Exception:
                            for k,v in resolved.data.items(): head_item[k]=v
                        if req_by: head_item["requested_by"] = req_by
                        if req_by_id: head_item["requested_by_id"] = req_by_id
                        metric_inc("prefetch_inplace_updates")
            except Exception:
                logger.debug("Prefetch resolve failed for query=%s", str(query)[:50])
                raise

    async def _start_now_update(self, started_at: float, duration: Optional[float]):
        from modules.ui_components import MusicControls as _Controls, create_now_playing_embed
        async def updater():
            try:
                while True:
                    if not self.now_message or self._closing:
                        return
                    try:
                        elapsed = time.time() - started_at
                        bar = make_progress_bar(elapsed, duration)
                        embed = create_now_playing_embed(self.current, extra_desc=bar, stream_profile=STREAM_PROFILE)
                        await self.now_message.edit(embed=embed, view=_Controls(self.guild.id))
                    except discord.NotFound:
                        # Tin nhắn đã bị xóa -> tạo lại
                        try:
                            self.now_message = await self._send_rate_limited(embed=embed, view=_Controls(self.guild.id))
                        except Exception:
                            pass
                    except discord.HTTPException:
                        # Lỗi tạm thời -> bỏ qua vòng này
                        pass
                    await asyncio.sleep(NOW_UPDATE_INTERVAL)
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

    async def _player_loop(self):
        from bot import players, GLOBAL_ALLOWED_DOMAINS
        from modules.audio_processor import validate_domain
        from modules.ui_components import MusicControls as _Controls, create_now_playing_embed
        from modules.voice_manager import get_voice_client_cached as _get_vc
        logger.info("Player start guild=%s", self.guild.id)
        try:
            while not self._closing:
                self.next_event.clear()
                try:
                    item = await self.queue.get(timeout=IDLE_DISCONNECT_SECONDS)
                except asyncio.TimeoutError:
                    try:
                        vc = self.vc or _get_vc(self.bot, self.guild)
                        if vc and vc.is_connected():
                            try: await self._send_rate_limited("Không ai phát nhạc nên mình đi nhaa. Hẹn gặp lại ✨")
                            except Exception: pass
                            await vc.disconnect()
                        logger.info("Idle queue timeout; disconnected voice (guild=%s)", self.guild.id)
                    except Exception:
                        pass
                    break
                track = None; data = None
                if isinstance(item, dict):
                    data = item
                    if not data.get("url"):
                        try:
                            resolved = await YTDLTrack.resolve(data.get("webpage_url") or data.get("title") or data.get("query"))
                            data = dict(resolved.data)
                            if item.get("requested_by"): data["requested_by"] = item.get("requested_by")
                        except Exception as e:
                            logger.exception("Failed to resolve queued dict: %s", e)
                            try: await self._send_rate_limited(f"Không thể phát mục đã xếp: {e}")
                            except Exception: pass
                            continue
                    track = YTDLTrack(data)
                elif isinstance(item, YTDLTrack):
                    track = item; data = track.data
                elif isinstance(item, str):
                    try:
                        track = await YTDLTrack.resolve(item); data = track.data
                    except Exception as e:
                        logger.exception("Failed to resolve queued string: %s", e)
                        try: await self._send_rate_limited(f"Không thể phát bài đã xếp: {e}")
                        except Exception: pass
                        continue
                else:
                    logger.error("Unknown queue item type: %s", type(item))
                    try: metric_inc("queue_unknown_type")
                    except Exception: pass
                    continue
                if not data or not data.get("url"):
                    try: await self._send_rate_limited("Không có stream URL cho bài này :<")
                    except Exception: pass
                    continue
                try:
                    try:
                        enq_ts = data.get("_enqueued_at")
                        if enq_ts:
                            import modules.metrics as _m; _m.metric_add_time("queue_wait_time", max(0.0, time.time() - enq_ts))
                    except Exception: pass
                    t0 = time.perf_counter()
                    try:
                        u = data.get("webpage_url") or data.get("url")
                        if u and not validate_domain(u, GLOBAL_ALLOWED_DOMAINS):
                            await self._send_rate_limited("Nguồn phát hiện không hợp lệ, bỏ qua.")
                            continue
                    except Exception: pass
                    src = create_audio_source_wrapper(data.get("url"), volume=self.volume)
                except Exception as e:
                    logger.exception("create_audio_source failed: %s", e)
                    try: await self._send_rate_limited("Lỗi khi tạo nguồn phát")
                    except Exception: pass
                    continue
                vc = self.vc or _get_vc(self.bot, self.guild)
                if not vc or not vc.is_connected():
                    try: await self._send_rate_limited("Mình chưa vô kênh thoại nào cả :<")
                    except Exception: pass
                    break
                played_at = time.time()
                logger.info("Start playback guild=%s title=%s dur=%s live=%s vol=%.2f profile=%s", self.guild.id, truncate(data.get("title"), 80), format_duration(data.get("duration")), bool(data.get("is_live")), self.volume, STREAM_PROFILE)
                def _after(err):
                    if err:
                        logger.exception("Playback error guild %s: %s", self.guild.id, err); metric_inc("playback_error")
                    else:
                        try:
                            elapsed = time.time() - played_at
                            logger.info("Finish playback guild=%s title=%s elapsed=%.2fs", self.guild.id, truncate(data.get("title"), 80), elapsed)
                        except Exception: pass
                        metric_inc("playback_finish")
                    # Thread-safe notify of next_event; use captured loop or Discord connection loop
                    try:
                        loop = self._loop
                        if loop:
                            loop.call_soon_threadsafe(self.next_event.set)
                            return
                    except Exception:
                        pass
                    # Fallback: try bot connection loop (discord.py internal)
                    try:
                        conn = getattr(self.bot, "_connection", None)
                        loop2 = getattr(conn, "loop", None)
                        if loop2:
                            loop2.call_soon_threadsafe(self.next_event.set)
                            return
                    except Exception:
                        pass
                    # Final fallback: attempt policy loop
                    try:
                        asyncio.get_event_loop_policy().get_event_loop().call_soon_threadsafe(self.next_event.set)
                    except Exception:
                        logger.exception("Failed to set next event (all fallbacks)")
                async with self._lock:
                    try:
                        # Optional tiny preroll: schedule play on next loop cycle to let FFmpeg warm up I/O buffers.
                        await asyncio.sleep(0)  # yield once
                        vc.play(src, after=_after)
                        try: vc.source._track_meta = {"title": data.get("title"), "url": data.get("webpage_url")}
                        except Exception: pass
                        self.current = data; self.history.append(data); metric_inc("playback_start")
                        try:
                            import modules.metrics as _m; _m.metric_add_time("play_start_delay", max(0.0, time.perf_counter() - t0))
                        except Exception: pass
                        async def _watchdog():  # enhanced connectivity & stutter mitigation
                            try:
                                await asyncio.sleep(5)
                                # Decide whether to attempt restart based on recent attempts (backoff)
                                do_restart = False
                                now = time.time()
                                # reset window if expired
                                if now - self._watchdog_window_start > self._watchdog_window_seconds:
                                    self._watchdog_window_start = now
                                    self._watchdog_attempts = 0
                                # If below threshold, allow restart
                                if self._watchdog_attempts < self._watchdog_max_attempts:
                                    do_restart = True
                                if (not vc.is_connected()) or (not vc.is_playing() and self.queue.qsize() > 0):
                                    if do_restart:
                                        self._watchdog_attempts += 1
                                        metric_inc("ffmpeg_restarts"); logger.warning("Watchdog restart attempt guild=%s attempt=%s", self.guild.id, self._watchdog_attempts)
                                        try:
                                            new_src = create_audio_source_wrapper(data.get("url"), volume=self.volume)
                                            if vc.is_connected():
                                                vc.play(new_src, after=_after)
                                        except Exception: logger.exception("FFmpeg restart failed")
                                    else:
                                        logger.warning("Watchdog suppression guild=%s attempts=%s", self.guild.id, self._watchdog_attempts)
                                # Light jitter guard: if playing but Discord reports not_paused and queue not empty, touch source ref (noop) to keep loop hot
                                else:
                                    try:
                                        _ = getattr(vc, 'source', None)
                                        if _ is not None:
                                            pass  # placeholder for future low-risk warm action
                                    except Exception:
                                        pass
                            except Exception: pass
                        try: asyncio.create_task(_watchdog())
                        except Exception: pass
                    except Exception as e:
                        logger.exception("vc.play failed: %s", e)
                        # Rate-limited message to avoid spamming the channel
                        try:
                            await self._send_rate_limited("Lỗi khi phát")
                        except Exception:
                            pass
                        continue
                try:
                    embed = create_now_playing_embed(data, stream_profile=STREAM_PROFILE)
                    if self.now_message:
                        try:
                            edit_fn = getattr(self.now_message, "edit", None)
                            if callable(edit_fn):
                                await edit_fn(embed=embed, view=_Controls(self.guild.id))
                            else:
                                # Use safe rate-limited send helper instead of direct send (checklist C4)
                                self.now_message = await self._send_rate_limited(embed=embed, view=_Controls(self.guild.id))
                        except Exception:
                            try:
                                self.now_message = await self._send_rate_limited(embed=embed, view=_Controls(self.guild.id))
                            except Exception:
                                logger.exception("Failed to send now-playing embed (both edit and send failed)")
                    else:
                        self.now_message = await self._send_rate_limited(embed=embed, view=_Controls(self.guild.id))
                    # Gắn tham chiếu message cho View nếu có để on_timeout có thể làm mới
                    try:
                        if self.now_message and getattr(self.now_message, 'components', None):
                            view = _Controls(self.guild.id)
                            view.message = self.now_message
                    except Exception:
                        pass
                    await self._start_now_update(played_at, data.get("duration"))
                except Exception:
                    logger.exception("Failed to send now-playing embed")
                await self.next_event.wait()
                try:
                    if self.now_update_task and not self.now_update_task.done():
                        self.now_update_task.cancel(); self.now_update_task = None
                except Exception: logger.debug("player_loop: failed cancelling now_update_task", exc_info=True)
                try:
                    if self.loop_one and isinstance(track, YTDLTrack) and track.data:
                        if self._suppress_loop_requeue_once:
                            logger.info("Loop-one: suppressed requeue after skip (guild=%s)", self.guild.id)
                        else:
                            await self.queue.put_front(track.data); logger.info("Loop-one repeat guild=%s title=%s", self.guild.id, truncate(track.data.get("title"), 80))
                        self._suppress_loop_requeue_once = False
                    elif self.loop_mode and isinstance(track, YTDLTrack) and track.data:
                        await self.queue.put(track.data); logger.info("Loop-all requeue guild=%s title=%s", self.guild.id, truncate(track.data.get("title"), 80))
                except Exception: logger.exception("Failed to requeue for loop mode (loop_one=%s loop_all=%s)", self.loop_one, self.loop_mode)
                vc = _get_vc(self.bot, self.guild)
                if self.queue.empty() and (not vc or not vc.is_playing()):
                    continue
        except asyncio.CancelledError:
            logger.info("Player loop cancelled guild=%s", self.guild.id)
        except Exception as e:
            logger.exception("Unhandled in player loop guild=%s: %s", self.guild.id, e)
        finally:
            try:
                from bot import players
                players.pop(self.guild.id, None)
            except Exception: pass
            try:
                if self.prefetch_task and not self.prefetch_task.done(): self.prefetch_task.cancel()
            except Exception: pass
            try:
                if self.now_update_task and not self.now_update_task.done(): self.now_update_task.cancel()
            except Exception: pass
            logger.info("Player stopped guild=%s", self.guild.id)

    def destroy(self):
        if getattr(self, '_destroying', False):
            return
        self._destroying = True; self._closing = True
        logger.debug("Destroying player for guild %s", self.guild.id)
        try:
            from bot import players; players.pop(self.guild.id, None)
        except Exception: pass
        tasks = [('prefetch_task', self.prefetch_task), ('now_update_task', self.now_update_task), ('_task', self._task)]
        for name, task in tasks:
            try:
                if task and not task.done(): task.cancel(); logger.debug("Cancelled %s for guild %s", name, self.guild.id)
            except Exception: pass
        try:
            if hasattr(self, 'queue') and self.queue:
                if hasattr(self.queue, '_dq'): self.queue._dq.clear()
                else:
                    try:
                        if self._loop and not self._loop.is_closed(): self._loop.create_task(self.queue.clear())
                        else: asyncio.create_task(self.queue.clear())
                    except Exception: pass
        except Exception: pass
        try:
            # Cancel any outstanding prefetch tasks
            for k, t in list(getattr(self, '_prefetch_tasks', {}).items()):
                try:
                    if hasattr(t, 'cancel') and not t.done():
                        t.cancel()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self.vc and self.vc.is_connected():
                try:
                    if self._loop and not self._loop.is_closed(): self._loop.create_task(self.vc.disconnect())
                    else: asyncio.create_task(self.vc.disconnect())
                except Exception: pass
        except Exception: pass
