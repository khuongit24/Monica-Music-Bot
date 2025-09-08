"""
Configuration management for Monica Bot
"""
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("Monica.Config")

# Load environment variables from .env file if exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key and not os.getenv(key):
                            os.environ[key] = value
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")

CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    # Token should NEVER be in config file - use environment variables only
    "prefix": "!",
    "owner_id": 1019539342621949983,
    "max_queue_size": 200,
    "download_concurrency": 2,
    "cache_size_limit": 200,
    "ffmpeg_bitrate": "128k",
    "ffmpeg_threads": 1,
    "prefetch_next": True,
    # streaming profile — 'stable' (default) or 'low-latency'
    "stream_profile": "stable",
    # how often to update the now-playing progress (seconds)
    "now_update_interval_seconds": 12,
    "idle_disconnect_seconds": 300,  # auto-disconnect after 5 minutes idle (was 900)
    "max_track_seconds": 0,
    # new optional features (non-breaking)
    "trace_logging": False,
    "structured_logging": False,
    "language": "vi",
    "enable_prometheus": False,
    "prometheus_port": 9109,
    "queue_persistence_enabled": False,
}

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json, merged with defaults."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_conf = json.load(f)
            # Remove token from user config if it exists (security measure)
            if "token" in user_conf:
                logger.warning("Token found in config.json - this is insecure. Please use DISCORD_TOKEN environment variable instead.")
                del user_conf["token"]
            config = {**DEFAULT_CONFIG, **user_conf}
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            config = DEFAULT_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()
    
    return validate_config(config)

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize configuration values (non-destructive fallback).

    Does not remove existing keys; only adjusts clearly invalid numeric ranges.
    Logic of application remains unchanged – this merely ensures safe bounds.
    """
    changed = False
    def clamp_int(key, minimum, default):
        nonlocal changed
        try:
            if int(cfg.get(key)) < minimum:
                logger.warning("Config '%s'=%s < %s; fallback to %s", key, cfg.get(key), minimum, default)
                cfg[key] = default; changed = True
        except Exception:
            logger.warning("Config '%s' invalid (%s); fallback to %s", key, cfg.get(key), default)
            cfg[key] = default; changed = True
    clamp_int("max_queue_size", 1, DEFAULT_CONFIG["max_queue_size"])
    clamp_int("download_concurrency", 1, DEFAULT_CONFIG["download_concurrency"])
    clamp_int("cache_size_limit", 1, DEFAULT_CONFIG["cache_size_limit"])
    clamp_int("ffmpeg_threads", 1, DEFAULT_CONFIG["ffmpeg_threads"])
    clamp_int("now_update_interval_seconds", 3, DEFAULT_CONFIG["now_update_interval_seconds"])
    clamp_int("idle_disconnect_seconds", 30, DEFAULT_CONFIG["idle_disconnect_seconds"])
    clamp_int("max_track_seconds", 0, DEFAULT_CONFIG["max_track_seconds"])  # 0 means unlimited
    profile = cfg.get("stream_profile")
    if profile not in ("stable", "low-latency", "super-low-latency"):
        logger.warning("Unknown stream_profile=%s; fallback to 'stable'", profile)
        cfg["stream_profile"] = "stable"; changed = True
    if changed:
        try:
            persist_config(cfg)
        except Exception:
            logger.debug("Persist after validation failed", exc_info=True)
    return cfg

def get_token() -> str:
    """Get Discord token from environment variables only."""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is required!")
    return token

def persist_config(config: Dict[str, Any]) -> None:
    """Persist configuration to config.json atomically."""
    try:
        # Atomic write to avoid truncation corruption
        tmp_path = CONFIG_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            os.replace(tmp_path, CONFIG_PATH)
        except Exception:
            try:
                os.remove(CONFIG_PATH)
            except Exception:
                pass
            try:
                os.rename(tmp_path, CONFIG_PATH)
            except Exception:
                raise
    except Exception:
        logger.exception("Failed to persist config")
