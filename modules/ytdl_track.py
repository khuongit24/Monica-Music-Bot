import time
import asyncio
from typing import Dict, Any, Optional, TypedDict
from modules.metrics import metric_inc, metric_add_time, gauge_set
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
_RESOLVE_STATE = "closed"  # one of 'closed', 'open', 'half-open'
_RESOLVE_PROBE_ATTEMPTS = 0
_RESOLVE_PROBE_MAX = 2
_PLUGIN_HOOKS = {"on_track_resolved": []}  # simple extensibility point

# Negative cache (short TTL) to avoid repeated failing resolves for the same query
_NEG_CACHE: Dict[str, tuple[float, str]] = {}
_NEG_TTL_SECONDS = 20.0  # keep failures for a short time only

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
                metric_inc("resolve_circuit_open")
            except Exception:
                pass
            if logger:
                logger.error("Resolve circuit OPEN (streak=%s cooldown=%ss)", streak, _RESOLVE_COOLDOWN_SECONDS)
        else:
            if logger:
                logger.info("Resolve circuit CLOSED after cooldown")
            try:
                metric_inc("resolve_circuit_closed")
            except Exception:
                pass
    except Exception:
        pass

def register_hook(name: str, func):
    lst = _PLUGIN_HOOKS.setdefault(name, [])
    lst.append(func)


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
        # declare globals we may reassign
        global _RESOLVE_FAIL_STREAK, _RESOLVE_LOCKOUT_UNTIL, _RESOLVE_CIRCUIT_LAST_OPEN_TS
        global _RESOLVE_STATE, _RESOLVE_PROBE_ATTEMPTS, RESOLVE_SEMAPHORE
        # these are mutated/assigned in this function
        global _RESOLVE_LOCK, _RESOLVING

        now = time.time()

        # detect circuit close -> transition to half-open when cooldown expired
        try:
            if _RESOLVE_CIRCUIT_LAST_OPEN_TS and _RESOLVE_LOCKOUT_UNTIL and now >= _RESOLVE_LOCKOUT_UNTIL:
                dur = _RESOLVE_LOCKOUT_UNTIL - _RESOLVE_CIRCUIT_LAST_OPEN_TS
                if dur > 0:
                    try:
                        metric_add_time("resolve_circuit_open_seconds", dur)
                    except Exception:
                        pass
                _RESOLVE_CIRCUIT_LAST_OPEN_TS = 0.0
                # Transition to half-open and allow a small number of probes
                try:
                    _RESOLVE_STATE = "half-open"
                    _RESOLVE_PROBE_ATTEMPTS = 0
                except Exception:
                    pass
                record_circuit_transition(False)
        except Exception:
            pass

        if _RESOLVE_LOCKOUT_UNTIL and now < _RESOLVE_LOCKOUT_UNTIL:
            try:
                metric_inc("resolve_circuit_open")
            except Exception:
                pass
            raise RuntimeError("Hệ thống tạm ngưng tìm kiếm do quá nhiều lỗi liên tiếp. Thử lại sau vài giây...")

        key = query.strip()
        t_start = time.perf_counter()

        # Negative cache fast-path
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
                # Optional global throttle: if RESOLVE_SEMAPHORE provided, use it to limit concurrent owners
                if RESOLVE_SEMAPHORE is not None:
                    # record waiting on semaphore for metrics
                    try: metric_inc("download_semaphore_waits")
                    except Exception: pass
                    async with RESOLVE_SEMAPHORE:
                        data = await _execute_with_retries(query, timeout)
                else:
                    data = await _execute_with_retries(query, timeout)

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
                try:
                    if _RESOLVE_STATE == "half-open":
                        _RESOLVE_STATE = "closed"
                        _RESOLVE_PROBE_ATTEMPTS = 0
                        try:
                            metric_inc("resolve_circuit_closed")
                        except Exception:
                            pass
                except Exception:
                    pass
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
                # If we're in half-open mode and a probe failed, increment probe counter and re-open if needed
                if _RESOLVE_STATE == "half-open":
                    _RESOLVE_PROBE_ATTEMPTS += 1
                    # if probe attempts exceed allowed, treat as open again
                    if _RESOLVE_PROBE_ATTEMPTS >= (_RESOLVE_PROBE_MAX or 2):
                        _RESOLVE_STATE = "open"
                        # apply cooldown multiplier for repeated failures
                        exp = min(8, 2 ** (max(0, _RESOLVE_FAIL_STREAK - (_RESOLVE_FAIL_THRESHOLD or 1))))
                        _RESOLVE_LOCKOUT_UNTIL = time.time() + ((_RESOLVE_COOLDOWN_SECONDS or 5) * exp)
                        if not _RESOLVE_CIRCUIT_LAST_OPEN_TS:
                            _RESOLVE_CIRCUIT_LAST_OPEN_TS = time.time()
                        try:
                            metric_inc("resolve_circuit_open_events")
                        except Exception:
                            pass
                        record_circuit_transition(True, _RESOLVE_FAIL_STREAK)
                else:
                    # When allowing threshold crossing, apply exponential backoff multiplier
                    if _RESOLVE_FAIL_STREAK >= (_RESOLVE_FAIL_THRESHOLD or 1):
                        _RESOLVE_STATE = "open"
                        exp = min(8, 2 ** (max(0, _RESOLVE_FAIL_STREAK - (_RESOLVE_FAIL_THRESHOLD or 1))))
                        _RESOLVE_LOCKOUT_UNTIL = time.time() + ((_RESOLVE_COOLDOWN_SECONDS or 5) * exp)
                        if not _RESOLVE_CIRCUIT_LAST_OPEN_TS:
                            _RESOLVE_CIRCUIT_LAST_OPEN_TS = time.time()
                        try:
                            metric_inc("resolve_circuit_open_events")
                        except Exception:
                            pass
                        record_circuit_transition(True, _RESOLVE_FAIL_STREAK)
            except Exception:
                pass

            try:
                _safe_set_future_exception(fut, e)
            except Exception:
                pass
            # Negative cache the failure (short TTL)
            try:
                _NEG_CACHE[key] = (time.time(), str(e))
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
            # reset consecutive timeout counter on success
            try:
                gauge_set("resolve_consecutive_timeouts", 0)
            except Exception:
                pass
        except asyncio.TimeoutError:
            logger.warning("yt-dlp timeout for query=%s", query)
            metric_inc("ytdl_timeout")
            try:
                # increment consecutive timeout gauge by 1
                # use metric_inc as a simple counter fallback if gauge_set not available
                metric_inc("resolve_consecutive_timeouts", 1)
            except Exception:
                pass
            raise RuntimeError("Tìm kiếm quá lâu, thử lại sau")
        except Exception as e:
            logger.debug("Primary yt-dlp attempt failed: %s", e, exc_info=True)
            metric_inc("ytdl_error")
            # Check if it's HTTP error
            if "http" in str(e).lower() or "connection" in str(e).lower():
                metric_inc("http_errors")
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
                k = tuple(sorted(alt_opts.items()))
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
                try:
                    metric_inc("resolve_consecutive_timeouts", 1)
                except Exception:
                    pass
                raise RuntimeError("Tìm kiếm quá lâu (fallback), thử lại sau")
            except Exception as e2:
                logger.debug("Fallback attempt 1 failed: %s", e2, exc_info=True)
                metric_inc("ytdl_error")
                # Check if it's HTTP error
                if "http" in str(e2).lower() or "connection" in str(e2).lower():
                    metric_inc("http_errors")
                try:
                    base_opts = YTDL_OPTS or {}
                    minimal_opts = dict(base_opts)
                    minimal_opts.pop("format", None)
                    minimal_opts["noplaylist"] = True
                    from yt_dlp import YoutubeDL
                    # reuse minimal ytdl instance if cached
                    k2 = tuple(sorted(minimal_opts.items()))
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
