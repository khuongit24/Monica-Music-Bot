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
    "cache_hits": 0,
    "cache_miss": 0,
    "queue_add": 0,
    "playback_start": 0,
    "playback_finish": 0,
    "playback_error": 0,
    "resolve_time_total_seconds": 0.0,
    "resolve_time_count": 0,
    "ffmpeg_restarts": 0,
    # prefetch related
    "prefetch_resolved": 0,
    "prefetch_idle_cycles": 0,
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
