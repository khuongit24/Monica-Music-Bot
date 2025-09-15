"""
Metrics tracking for Monica Bot
"""
from typing import Dict

# --- Metrics ---
_METRICS = {
    "resolve_attempts": 0,
    "resolve_success": 0,
    "resolve_fail": 0,
    "resolve_circuit_open": 0,
    "resolve_queue_len": 0,
    "download_semaphore_waits": 0,
    # number of times circuit transitioned to open (distinct events)
    "resolve_circuit_open_events": 0,
    "cache_hits": 0,
    "cache_miss": 0,
    "queue_add": 0,
    "playback_start": 0,
    "playback_finish": 0,
    "playback_error": 0,
    "resolve_time_total_seconds": 0.0,
    "resolve_time_count": 0,
    "ffmpeg_restarts": 0,
    "ffmpeg_restart_count": 0,
    # prefetch related
    "prefetch_resolved": 0,
    "prefetch_idle_cycles": 0,
    "prefetch_inplace_updates": 0,
    # observability / rare conditions
    "queue_unknown_type": 0,
    # latency metrics (average derived from *_total_seconds / *_count)
    "play_start_delay_total_seconds": 0.0,
    "play_start_delay_count": 0,
    "queue_wait_time_total_seconds": 0.0,
    "queue_wait_time_count": 0,
    # audio source prepare metrics
    "audio_prepare_time_total_seconds": 0.0,
    "audio_prepare_time_count": 0,
    # counters for whether we copied stream or transcoded
    "audio_prepare_copy": 0,
    "audio_prepare_transcode": 0,
    # HTTP errors
    "http_errors": 0,
    "http_4xx_errors": 0,
    "http_5xx_errors": 0,
    # Gauges
    "queue_size": 0,
    # Cache metrics
    "cache_evicted": 0,
    "cache_size": 0,
}

def metric_inc(name: str, delta: int = 1):
    """Increment a metric by delta."""
    try:
        _METRICS[name] = _METRICS.get(name, 0) + delta
    except Exception:
        pass

def metric_add_time(name: str, seconds: float):
    """Add time to resolve timing metrics."""
    try:
        _METRICS[f"{name}_total_seconds"] = _METRICS.get(f"{name}_total_seconds", 0.0) + seconds
        _METRICS[f"{name}_count"] = _METRICS.get(f"{name}_count", 0) + 1
    except Exception:
        pass

def metrics_snapshot() -> Dict[str, int]:
    """Get a snapshot of current metrics."""
    return dict(_METRICS)

def get_average_resolve_time() -> float:
    """Get average resolve time in seconds."""
    total = _METRICS.get("resolve_time_total_seconds", 0.0)
    count = _METRICS.get("resolve_time_count", 0) or 0
    return (total / count) if count else 0.0


def set_gauge(name: str, value):
    """Set a gauge-like metric to a specific value."""
    try:
        _METRICS[name] = value
    except Exception:
        pass


def hist_observe_time(name: str, seconds: float):
    """Convenience to add timing observations (keeps *_total_seconds and *_count)."""
    try:
        _METRICS.setdefault(f"{name}_total_seconds", 0.0)
        _METRICS.setdefault(f"{name}_count", 0)
        _METRICS[f"{name}_total_seconds"] += float(seconds)
        _METRICS[f"{name}_count"] += 1
    except Exception:
        pass


def gauge_set(name: str, value):
    """Set a simple gauge value."""
    try:
        _METRICS[name] = value
    except Exception:
        pass
