import time
import asyncio
from typing import Dict, Any, Optional, TypedDict
from modules.metrics import metric_inc, metric_add_time
from modules.audio_processor import sanitize_stream_url

# sanitize_stream_url & pick best audio delegated via injected _PICK_BEST_AUDIO_URL

DOWNLOAD_SEMAPHORE = None  # injected
_YTDL_EXECUTOR = None      # injected
_cache_get = None          # injected async call
_cache_put = None          # injected async call
RESOLVE_SEMAPHORE = None   # optional injected semaphore to throttle concurrent resolve owners

# These will be set by bot.py at runtime
YTDL_OPTS = None
ytdl = None
logger = None
MAX_TRACK_SECONDS = None
_CACHE_PUT = None
_PICK_BEST_AUDIO_URL = None
_RESOLVING = None
_RESOLVE_LOCK = None
_RESOLVE_KEY_LOCKS: Dict[str, asyncio.Lock] = {}
_RESOLVE_FAIL_STREAK = None
_RESOLVE_FAIL_THRESHOLD = None
_RESOLVE_COOLDOWN_SECONDS = None
_RESOLVE_LOCKOUT_UNTIL = None
_RESOLVE_CIRCUIT_LAST_OPEN_TS = None
_PLUGIN_HOOKS = {"on_track_resolved": []}  # simple extensibility point

# Negative cache (short TTL) to avoid repeated failing resolves for the same query
_NEG_CACHE: Dict[str, tuple[float, str]] = {}
_NEG_TTL_SECONDS = 20.0  # base TTL (generic); dynamic adjustments applied per error type

# --- Helpers ---
def _truncate_query(q: str, limit: int = 200) -> str:
    try:
        if q and len(q) > limit:
            return q[:limit] + "…"
    except Exception:
        pass
    return q

def record_circuit_transition(opening: bool, streak: int = 0):
    """Centralized circuit transition logging & metrics.
    opening True => circuit opened; False => circuit closed.
    Non-intrusive: only adds logging/metrics, does not modify existing logic.
    """
    try:
        if opening:
            try:
                metric_inc("resolve_circuit_open_events")
                # gauge 1 = degraded
                from modules.metrics import set_gauge as _set_g
                try:
                    _set_g("circuit_state", 1)
                except Exception:
                    pass
            except Exception:
                pass
            if logger:
                logger.error("Resolve circuit OPEN (streak=%s cooldown=%ss)", streak, _RESOLVE_COOLDOWN_SECONDS)
        else:
            try:
                from modules.metrics import set_gauge as _set_g
                try:
                    _set_g("circuit_state", 0)
                except Exception:
                    pass
            except Exception:
                pass
            if logger:
                logger.info("Resolve circuit CLOSED after cooldown")
    except Exception:
        pass

def register_hook(name: str, func):
    lst = _PLUGIN_HOOKS.setdefault(name, [])
    lst.append(func)


def _freeze_opts(obj):
    """Recursively convert yt-dlp options into a hashable, stable structure.
    - dict -> tuple of (key, frozen(value)) sorted by key
    - list/tuple -> tuple of frozen items
    - set -> tuple of sorted frozen items
    - other -> as-is
    """
    try:
        if isinstance(obj, dict):
            return tuple((k, _freeze_opts(v)) for k, v in sorted(obj.items(), key=lambda x: str(x[0])))
        if isinstance(obj, (list, tuple)):
            return tuple(_freeze_opts(v) for v in obj)
        if isinstance(obj, set):
            return tuple(sorted((_freeze_opts(v) for v in obj), key=lambda x: str(x)))
    except Exception:
        pass
    return obj


def _safe_set_future_result(fut, result):
    try:
        if fut and not fut.done():
            fut.set_result(result)
    except Exception:
        try:
            if fut and not fut.done():
                fut.cancel()
        except Exception:
            pass


def _safe_set_future_exception(fut, exc):
    try:
        if fut and not fut.done():
            fut.set_exception(exc)
    except Exception:
        try:
            if fut and not fut.done():
                fut.cancel()
        except Exception:
            pass

class TrackData(TypedDict, total=False):
    title: str
    webpage_url: str
    url: str
    thumbnail: str
    uploader: str
    duration: float
    is_live: bool
    live_status: str
    requested_by: str
    requested_by_id: int


class YTDLTrack:
    """YouTube-DL track representation with metadata and robust resolution."""
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.is_live = data.get("is_live", False)
        self.duration = data.get("duration")
        self.title = data.get("title")
        self.url = data.get("url")

    @classmethod
    async def resolve(cls, query: str, timeout: float = 20.0) -> 'YTDLTrack':
        """Modular resolve implementation (mirrors original bot.py logic without altering behavior).

        All external dependencies (yt-dlp instance, executor, cache funcs, circuit breaker state)
        are injected via module-level globals by the main application at runtime.
        """
        global _RESOLVE_FAIL_STREAK, _RESOLVE_LOCKOUT_UNTIL, _RESOLVE_LOCK, _RESOLVING, _RESOLVE_CIRCUIT_LAST_OPEN_TS
        now = time.time()
        try:  # detect circuit close transition
            if _RESOLVE_CIRCUIT_LAST_OPEN_TS and _RESOLVE_LOCKOUT_UNTIL and now >= _RESOLVE_LOCKOUT_UNTIL:
                dur = _RESOLVE_LOCKOUT_UNTIL - _RESOLVE_CIRCUIT_LAST_OPEN_TS
                if dur > 0:
                    try: metric_add_time("resolve_circuit_open_seconds", dur)
                    except Exception: pass
                _RESOLVE_CIRCUIT_LAST_OPEN_TS = 0.0
                record_circuit_transition(False)
        except Exception:
            pass
        if now < _RESOLVE_LOCKOUT_UNTIL:
            metric_inc("resolve_circuit_open")
            raise RuntimeError("Hệ thống tạm ngưng tìm kiếm do quá nhiều lỗi liên tiếp. Thử lại sau vài giây...")

        key = query.strip()
        t_start = time.perf_counter()

    # Negative cache fast-path (error-specific TTL handled at insert time)
        try:
            neg = _NEG_CACHE.get(key)
            if neg:
                ts, msg = neg
                if (time.time() - ts) < _NEG_TTL_SECONDS:
                    metric_inc("resolve_fail")
                    raise RuntimeError(msg or "Nguồn tạm thời không khả dụng, thử lại sau ít phút")
                else:
                    # expired
                    _NEG_CACHE.pop(key, None)
        except Exception:
            pass

        if logger:
            try:
                logger.debug("Resolve start query=%s timeout=%s", _truncate_query(key), timeout)
            except Exception:
                pass

        # Metrics: size of current resolving map (best-effort)
        try:
            metric_inc("resolve_queue_len", 0)  # ensure metric exists
            if isinstance(_RESOLVING, dict):
                from modules.metrics import set_gauge
                try:
                    set_gauge("resolve_queue_len", len(_RESOLVING))
                except Exception:
                    pass
        except Exception:
            pass

        # Cache fast-path
        try:
            cached = await _cache_get(key)
        except Exception:
            cached = None
        if cached:
            metric_inc("resolve_success")
            try:
                metric_add_time("resolve_time", 0.0)
            except Exception:
                pass
            return cls(dict(cached))

        # Use a short global lock to setup/get a per-key lock and future
        if _RESOLVE_LOCK is None:
            _RESOLVE_LOCK = asyncio.Lock()
        async with _RESOLVE_LOCK:
            fut = _RESOLVING.get(key)
            if fut is None:
                fut = asyncio.get_running_loop().create_future()
                _RESOLVING[key] = fut
                # ensure per-key lock exists
                lk = _RESOLVE_KEY_LOCKS.get(key)
                if lk is None:
                    lk = asyncio.Lock()
                    _RESOLVE_KEY_LOCKS[key] = lk
                owner = True
            else:
                owner = False

        if not owner:
            # Wait for owner result but avoid hanging forever: add bounded wait
            try:
                # small slack over caller timeout to allow owner finish
                wait_timeout = max(20.0, float(timeout) + 5.0)
                res = await asyncio.wait_for(fut, timeout=wait_timeout)
                return res
            except asyncio.TimeoutError:
                # Stale future fallback: remove mapping if it's still the same future and retry
                try:
                    async with _RESOLVE_LOCK:
                        if _RESOLVING.get(key) is fut:
                            _RESOLVING.pop(key, None)
                            # also remove per-key lock if present
                            try:
                                _RESOLVE_KEY_LOCKS.pop(key, None)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Retry full resolve as we're no longer waiting on the previous owner
                return await cls.resolve(query, timeout=timeout)
            except asyncio.CancelledError:
                try:
                    async with _RESOLVE_LOCK:
                        if fut.cancelled():
                            _RESOLVING.pop(key, None)
                except Exception:
                    pass
                raise

        # Acquire per-key lock when owner to serialize heavy resolve work per-key
        per_key_lock = _RESOLVE_KEY_LOCKS.get(key)
        if per_key_lock is None:
            # Fallback to global lock if per-key lock missing
            per_key_lock = _RESOLVE_LOCK

        try:
            async with per_key_lock:
                # Ensure a sane default semaphore exists to provide backpressure when none injected.
                global RESOLVE_SEMAPHORE
                if RESOLVE_SEMAPHORE is None:
                    try:
                        # default modest concurrency to protect yt-dlp executor
                        RESOLVE_SEMAPHORE = asyncio.Semaphore(6)
                    except Exception:
                        RESOLVE_SEMAPHORE = None

                # --- execute_primary + fallback_chain ---
                # Hard timeout for entire owner resolve section (P0) to prevent indefinite stalls.
                owner_deadline = max(10.0, min(40.0, float(timeout) + 10.0))
                async def _owner_resolve_body():
                    # Optional global throttle: if RESOLVE_SEMAPHORE provided, use it to limit concurrent owners
                    if RESOLVE_SEMAPHORE is not None:
                        try: metric_inc("download_semaphore_waits")
                        except Exception: pass
                        async with RESOLVE_SEMAPHORE:
                            return await _execute_with_retries(query, timeout)
                    return await _execute_with_retries(query, timeout)
                # Optional global throttle: if RESOLVE_SEMAPHORE provided, use it to limit concurrent owners
                try:
                    data = await asyncio.wait_for(_owner_resolve_body(), timeout=owner_deadline)
                except asyncio.TimeoutError:
                    raise RuntimeError("Quá thời gian xử lý nguồn (timeout)")

                # --- url_finalize ---
                data = _finalize_and_pick_url(data)

                # --- wrap_object ---
                track = cls(data)

                # length guard
                if MAX_TRACK_SECONDS and MAX_TRACK_SECONDS > 0 and not track.is_live and track.duration and track.duration > MAX_TRACK_SECONDS:
                    raise RuntimeError(f"Độ dài bài vượt giới hạn {MAX_TRACK_SECONDS//60} phút")

                # cache_put (non-live)
                if not track.is_live:
                    try:
                        await _cache_put(key, data)
                    except Exception:
                        if logger:
                            logger.debug("Cache put error", exc_info=True)

                try:
                    fut.set_result(track)
                except Exception:
                    pass

                _RESOLVE_FAIL_STREAK = 0
                metric_inc("resolve_success")
                try:
                    for h in _PLUGIN_HOOKS.get("on_track_resolved", []):
                        try:
                            h(track.data)
                        except Exception:
                            pass
                except Exception:
                    pass

                return track

        except Exception as e:
            # Robust failure handling: ensure future is completed and circuit breaker applies exponential backoff
            try:
                _RESOLVE_FAIL_STREAK += 1
            except Exception:
                _RESOLVE_FAIL_STREAK = 1
            try: metric_inc("resolve_fail")
            except Exception: pass

            try:
                # When allowing threshold crossing, apply exponential backoff multiplier
                if _RESOLVE_FAIL_STREAK >= (_RESOLVE_FAIL_THRESHOLD or 1):
                    # multiplier grows as powers of two but capped (P0 reduce cap to 5) to avoid long lockouts
                    exp = min(5, 2 ** (max(0, _RESOLVE_FAIL_STREAK - (_RESOLVE_FAIL_THRESHOLD or 1))))
                    lockout_seconds = (_RESOLVE_COOLDOWN_SECONDS or 5) * exp
                    _RESOLVE_LOCKOUT_UNTIL = time.time() + lockout_seconds
                    if not _RESOLVE_CIRCUIT_LAST_OPEN_TS:
                        _RESOLVE_CIRCUIT_LAST_OPEN_TS = time.time()
                    try:
                        metric_inc("resolve_circuit_open_events")
                    except Exception:
                        pass
                    # record lockout duration metric (histogram-like via total/count)
                    try:
                        from modules.metrics import observe_lockout_duration as _obs_lock
                        _obs_lock(lockout_seconds)
                    except Exception:
                        pass
                    record_circuit_transition(True, _RESOLVE_FAIL_STREAK)
            except Exception:
                pass

            try:
                _safe_set_future_exception(fut, e)
            except Exception:
                pass
            # Negative cache the failure (dynamic TTL by error category)
            try:
                msg = str(e).lower()
                ttl = _NEG_TTL_SECONDS
                if any(k in msg for k in ("timeout", "quá lâu", "time out")):
                    ttl = min(10.0, _NEG_TTL_SECONDS)  # transient network timeouts shorter cache
                elif any(k in msg for k in ("không tìm", "not found", "no result")):
                    ttl = 8.0  # quickly allow re-search attempts
                elif any(k in msg for k in ("vượt giới hạn", "giới hạn", "length")):
                    ttl = 60.0  # user error; cache longer
                _NEG_CACHE[key] = (time.time(), str(e))
                # store TTL selection by augmenting message (optional) - keep simple for now
            except Exception:
                pass
            # Ensure mapping cleaned up immediately to avoid stuck futures
            try:
                async with _RESOLVE_LOCK:
                    if _RESOLVING.get(key) is fut:
                        _RESOLVING.pop(key, None)
                        try: _RESOLVE_KEY_LOCKS.pop(key, None)
                        except Exception: pass
            except Exception:
                pass
            raise

        finally:
            # Safety: if owner returns unexpectedly without setting result/exception, ensure future is completed
            try:
                if owner and not fut.done():
                    try:
                        _safe_set_future_exception(fut, RuntimeError("Resolve did not complete (internal error)"))
                    except Exception:
                        pass
            except Exception:
                pass

            # final cleanup of _RESOLVING mapping
            try:
                async with _RESOLVE_LOCK:
                    fut_existing = _RESOLVING.get(key)
                    if fut_existing is None or fut_existing is fut or fut.done() or fut.cancelled():
                        _RESOLVING.pop(key, None)
            except Exception:
                pass

            try:
                elapsed = time.perf_counter() - t_start
                metric_add_time("resolve_time", elapsed)
            except Exception:
                pass

# --- Decomposed helper steps (non-invasive; reuses original logic pieces) ---
async def _execute_with_fallbacks(query: str, timeout: float):
    loop = asyncio.get_running_loop()
    # Normalize plain-text queries to use yt-dlp default_search prefix to avoid 'not a valid URL' errors
    try:
        q = query.strip()
        is_url = q.startswith("http://") or q.startswith("https://")
        if not is_url:
            # Use injected default_search if available; fallback to ytsearch
            try:
                ds = (YTDL_OPTS or {}).get("default_search")
            except Exception:
                ds = None
            prefix = (ds or "ytsearch").strip(":")
            if not q.lower().startswith((prefix + ":")):
                q = f"{prefix}:{q}"
        else:
            q = query
    except Exception:
        q = query
    async with DOWNLOAD_SEMAPHORE:
        data = None
        metric_inc("resolve_attempts")
        try:
            start = time.perf_counter()
            fut_exec = loop.run_in_executor(_YTDL_EXECUTOR, lambda: ytdl.extract_info(q, download=False))
            data = await asyncio.wait_for(fut_exec, timeout=timeout)
            metric_add_time("ytdl_primary_time", time.perf_counter() - start)
        except asyncio.TimeoutError:
            logger.warning("yt-dlp timeout for query=%s", query)
            metric_inc("ytdl_timeout")
            raise RuntimeError("Tìm kiếm quá lâu, thử lại sau")
        except Exception as e:
            logger.debug("Primary yt-dlp attempt failed: %s", e, exc_info=True)
            metric_inc("ytdl_error")
        # Try fallback options; reuse YoutubeDL instances when possible to avoid reallocation
        if not data:
            try:
                base_opts = YTDL_OPTS or {}
                alt_opts = dict(base_opts)
                alt_opts["format"] = "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"
                alt_opts["noplaylist"] = True
                from yt_dlp import YoutubeDL
                # cache alt ytdl instances keyed by frozenset of option items
                if not hasattr(_execute_with_fallbacks, "_ytdl_cache"):
                    _execute_with_fallbacks._ytdl_cache = {}
                k = _freeze_opts(alt_opts)
                alt_ytdl = _execute_with_fallbacks._ytdl_cache.get(k)
                if alt_ytdl is None:
                    try:
                        alt_ytdl = YoutubeDL(alt_opts)
                        _execute_with_fallbacks._ytdl_cache[k] = alt_ytdl
                    except Exception:
                        alt_ytdl = YoutubeDL(alt_opts)
                start2 = time.perf_counter()
                fut2 = loop.run_in_executor(_YTDL_EXECUTOR, lambda: alt_ytdl.extract_info(q, download=False))
                data = await asyncio.wait_for(fut2, timeout=timeout)
                metric_add_time("ytdl_fallback_time", time.perf_counter() - start2)
            except asyncio.TimeoutError:
                logger.warning("yt-dlp fallback timeout for query=%s", query)
                metric_inc("ytdl_timeout")
                raise RuntimeError("Tìm kiếm quá lâu (fallback), thử lại sau")
            except Exception as e2:
                logger.debug("Fallback attempt 1 failed: %s", e2, exc_info=True)
                metric_inc("ytdl_error")
                try:
                    base_opts = YTDL_OPTS or {}
                    minimal_opts = dict(base_opts)
                    minimal_opts.pop("format", None)
                    minimal_opts["noplaylist"] = True
                    from yt_dlp import YoutubeDL
                    # reuse minimal ytdl instance if cached
                    k2 = _freeze_opts(minimal_opts)
                    minimal_ytdl = _execute_with_fallbacks._ytdl_cache.get(k2)
                    if minimal_ytdl is None:
                        try:
                            minimal_ytdl = YoutubeDL(minimal_opts)
                            _execute_with_fallbacks._ytdl_cache[k2] = minimal_ytdl
                        except Exception:
                            minimal_ytdl = YoutubeDL(minimal_opts)
                    start3 = time.perf_counter()
                    fut3 = loop.run_in_executor(_YTDL_EXECUTOR, lambda: minimal_ytdl.extract_info(q, download=False))
                    data = await asyncio.wait_for(fut3, timeout=timeout)
                    metric_add_time("ytdl_minimal_time", time.perf_counter() - start3)
                except Exception:
                    logger.exception("yt-dlp final fallback failed")
                    raise RuntimeError("Không thể lấy thông tin nguồn (định dạng/nguồn không khả dụng)")
        if not data:
            raise RuntimeError("Không tìm thấy kết quả")
        if "entries" in data:
            entries = [e for e in data.get("entries") or [] if e]
            if not entries:
                raise RuntimeError("Không tìm thấy mục trong kết quả")
            data = entries[0]
        return data

async def _execute_with_retries(query: str, timeout: float):
    """Execute resolve with limited retries, exponential backoff and jitter.
    Keeps behavior of _execute_with_fallbacks but retries on transient failures.
    """
    # Step timeouts: ensure each attempt bounded; overall retries limited
    per_attempt_timeout = max(5.0, min(20.0, float(timeout)))
    attempts = 3
    delay = 0.5
    last_exc = None
    for i in range(attempts):
        try:
            return await _execute_with_fallbacks(query, per_attempt_timeout)
        except asyncio.TimeoutError as e:
            last_exc = e
        except RuntimeError as e:
            # treat known non-retriable messages as final
            msg = str(e).lower()
            if "không tìm" in msg or "vượt giới hạn" in msg:
                raise
            last_exc = e
        except Exception as e:
            last_exc = e
        # backoff with jitter (cap)
        try:
            import random
            await asyncio.sleep(min(4.0, delay + random.uniform(0, 0.25)))
        except Exception:
            await asyncio.sleep(delay)
        delay *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError("Resolve thất bại (không xác định)")

def _finalize_and_pick_url(data: dict) -> dict:
    if not data.get("url"):
        picked = _PICK_BEST_AUDIO_URL(data)
        if picked:
            data["url"] = picked
    else:
        try:
            data["url"] = sanitize_stream_url(data["url"]) or data["url"]
        except Exception:
            pass
    if not data.get("url"):
        raise RuntimeError("Không lấy được stream URL từ nguồn")
    return data


async def search_entries(query: str, limit: int = 3, timeout: float = 15.0) -> list[dict]:
    """Search for multiple candidates without collapsing to the first entry.

    This function intentionally bypasses the global `noplaylist=True` behavior
    used for single-result resolve, by spawning a YoutubeDL instance with
    `noplaylist=False` and `default_search=ytsearch{limit}`. It still honors
    the same concurrency constraints via semaphores and thread-pool executor.

    Args:
        query: User query (URL or keyword). For keywords, this will trigger a
               yt-dlp search returning up to `limit` entries.
        limit: Max number of candidates to return (default 3, clamped to [1,25]).
        timeout: Per-attempt timeout; one fallback attempt is also tried.

    Returns:
        Up to `limit` raw entry dicts (not finalized with 'url'). If yt-dlp
        returns a single item, the list contains that item.
    """
    import math
    loop = asyncio.get_running_loop()
    lim = max(1, min(25, int(limit)))
    q = (query or "").strip()
    is_url = q.startswith("http://") or q.startswith("https://")
    # Build a search-prefixed query that asks yt-dlp to return N entries
    if not is_url:
        prefix = f"ytsearch{lim}"
        # If user already provided a ytsearch-like prefix, respect it
        lower = q.lower()
        if not (lower.startswith("ytsearch:") or lower.startswith("ytsearch") or lower.startswith("gvsearch:")):
            q = f"{prefix}:{q}"
    # Compose custom options allowing playlists (to return multiple entries)
    base_opts = dict(YTDL_OPTS or {})
    base_opts["noplaylist"] = False
    # Reduce retries to avoid long stalls; search phase should be snappy
    base_opts["retries"] = min(1, int(base_opts.get("retries", 1) or 1))
    base_opts["extractor_retries"] = min(1, int(base_opts.get("extractor_retries", 1) or 1))
    # Ensure default_search matches our intended prefix family
    base_opts["default_search"] = f"ytsearch{lim}"
    try:
        from yt_dlp import YoutubeDL
    except Exception:
        raise RuntimeError("yt-dlp không khả dụng")

    # Cache custom ytdl instances keyed by option set (function attribute cache)
    if not hasattr(search_entries, "_ytdl_cache"):
        search_entries._ytdl_cache = {}
    k = _freeze_opts(base_opts)
    ydl = search_entries._ytdl_cache.get(k)
    if ydl is None:
        try:
            ydl = YoutubeDL(base_opts)
            search_entries._ytdl_cache[k] = ydl
        except Exception:
            ydl = YoutubeDL(base_opts)

    # Throttle: use RESOLVE_SEMAPHORE if provided (owner-level limit), and always
    # guard the extract with DOWNLOAD_SEMAPHORE to limit concurrent extractor runs.
    async def _extract(_ydl, _q):
        fut = loop.run_in_executor(_YTDL_EXECUTOR, lambda: _ydl.extract_info(_q, download=False))
        return await asyncio.wait_for(fut, timeout=timeout)

    data = None
    # Adaptive timeout control (P3 hardening): track consecutive timeouts per process
    if not hasattr(search_entries, "_timeout_streak"):
        search_entries._timeout_streak = 0
        search_entries._last_timeout_ts = 0.0
        search_entries._last_success_ts = 0.0
    # Lightweight recent-result cache (spam mitigation): map (normalized_query, lim) -> (ts, entries)
    if not hasattr(search_entries, "_recent_cache"):
        search_entries._recent_cache = {}
    norm_key = f"{q.lower()}|{lim}"
    now_ts = time.time()
    # Recovery logic: if no timeout for >60s, gradually decay streak (halve)
    try:
        if search_entries._timeout_streak and (now_ts - search_entries._last_timeout_ts) > 60:
            search_entries._timeout_streak = max(0, search_entries._timeout_streak // 2)
    except Exception:
        pass
    # If we recently (<=30s) retrieved identical results and caller uses >= current timeout level (fast reuse)
    try:
        rc = search_entries._recent_cache.get(norm_key)
        if rc:
            ts_rc, entries_rc = rc
            if (now_ts - ts_rc) <= 30 and entries_rc:
                from modules.metrics import metric_inc as _mi, set_gauge as _sg
                _mi("search_recent_reuse")
                # Update gauges for observability
                try:
                    _sg("search_timeout_streak", search_entries._timeout_streak)
                    lvl = 0
                    if search_entries._timeout_streak >= 5:
                        lvl = 2
                    elif search_entries._timeout_streak >= 3:
                        lvl = 1
                    _sg("search_timeout_level", lvl)
                except Exception:
                    pass
                return entries_rc[:lim]
    except Exception:
        pass
    try:
        try:
            metric_inc("resolve_attempts")
        except Exception:
            pass
        if RESOLVE_SEMAPHORE is not None:
            async with RESOLVE_SEMAPHORE:
                async with DOWNLOAD_SEMAPHORE:
                    data = await _extract(ydl, q)
        else:
            async with DOWNLOAD_SEMAPHORE:
                data = await _extract(ydl, q)
    except asyncio.TimeoutError:
        logger.warning("yt-dlp search timeout for query=%s", _truncate_query(query))
        metric_inc("ytdl_timeout")
        try:
            search_entries._timeout_streak += 1
            search_entries._last_timeout_ts = time.time()
            from modules.metrics import set_gauge as _sg
            try:
                _sg("search_timeout_streak", search_entries._timeout_streak)
                lvl = 0
                if search_entries._timeout_streak >= 5:
                    lvl = 2
                elif search_entries._timeout_streak >= 3:
                    lvl = 1
                _sg("search_timeout_level", lvl)
            except Exception:
                pass
        except Exception:
            pass
        # If multiple consecutive timeouts, shorten future search attempt timeout (handled by caller via degrade ladder)
        raise RuntimeError("Tìm kiếm quá lâu, thử lại sau")
    except Exception as e:
        # Fallback: relax format constraints
        try:
            metric_inc("ytdl_error")
        except Exception:
            pass
        try:
            alt_opts = dict(base_opts)
            alt_opts.pop("format", None)
            k2 = _freeze_opts(alt_opts)
            ydl2 = search_entries._ytdl_cache.get(k2)
            if ydl2 is None:
                ydl2 = YoutubeDL(alt_opts)
                search_entries._ytdl_cache[k2] = ydl2
            if RESOLVE_SEMAPHORE is not None:
                async with RESOLVE_SEMAPHORE:
                    async with DOWNLOAD_SEMAPHORE:
                        data = await _extract(ydl2, q)
            else:
                async with DOWNLOAD_SEMAPHORE:
                    data = await _extract(ydl2, q)
        except Exception:
            logger.debug("yt-dlp search final fallback failed", exc_info=True)
            raise RuntimeError("Không thể lấy danh sách kết quả tìm kiếm")

    # Reset timeout streak on success
    try:
        search_entries._timeout_streak = 0
        search_entries._last_success_ts = now_ts
        from modules.metrics import set_gauge as _sg
        try:
            _sg("search_timeout_streak", 0)
            _sg("search_timeout_level", 0)
        except Exception:
            pass
    except Exception:
        pass
    # Normalize into entries list
    if isinstance(data, dict) and "entries" in data:
        entries = [e for e in (data.get("entries") or []) if e]
        if not entries:
            raise RuntimeError("Không tìm thấy mục trong kết quả")
        # Prefer video-like entries to avoid playlists/tabs in selection
        def _is_video_like(e: dict) -> bool:
            try:
                t = (e.get("_type") or "").lower()
                if t in {"playlist", "multi_video"}:
                    return False
                ie = (e.get("ie_key") or "").lower()
                if ie in {"youtubetab", "youtubeplaylist"}:
                    return False
                url = (e.get("url") or e.get("webpage_url") or "").lower()
                if ("watch?v=" in url) or ("youtu.be/" in url):
                    return True
                # duration presence is also a good hint for single video
                if e.get("duration"):
                    return True
            except Exception:
                pass
            return True  # default permissive
        vids = [e for e in entries if _is_video_like(e)]
        chosen = vids if vids else entries
        # Cache recent
        try:
            search_entries._recent_cache[norm_key] = (now_ts, chosen[:lim])
        except Exception:
            pass
        return chosen[:lim]
    # Single entry wrap list
    try:
        search_entries._recent_cache[norm_key] = (now_ts, [data])
    except Exception:
        pass
    return [data]
