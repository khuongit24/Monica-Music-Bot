"""
Voice connection management and caching module.
Handles voice client caching and connection optimizations.
"""

import time
import asyncio
import logging
from typing import Dict, Tuple, Optional
import discord

logger = logging.getLogger("Monica.VoiceManager")

# Voice client connection optimization
_VOICE_CLIENT_CACHE: Dict[int, Tuple[discord.VoiceClient, float]] = {}  # guild_id -> (voice_client, cache_time)
_VOICE_CACHE_TIMEOUT = 12.0  # Cache voice clients for 12 seconds

# Background cleanup task handle (created by bot.py via start_voice_cleanup_task)
_VOICE_CLEANUP_TASK: Optional[asyncio.Task] = None

_VOICE_CONNECT_MAX_RETRIES = 3
_VOICE_CONNECT_BASE_BACKOFF = 0.5
_VOICE_CONNECT_JITTER = 0.5

try:
    from modules.metrics import metric_inc
except Exception:
    # metrics import may be unavailable in some test contexts
    def metric_inc(name: str, delta: int = 1):
        return


def get_voice_client_cached(bot, guild) -> Optional[discord.VoiceClient]:
    """Get voice client with caching to reduce repeated lookups."""
    if not guild:
        return None
    
    guild_id = guild.id
    current_time = time.time()
    
    # Check cache first
    cached_entry = _VOICE_CLIENT_CACHE.get(guild_id)
    if cached_entry:
        vc, cache_time = cached_entry
        if current_time - cache_time < _VOICE_CACHE_TIMEOUT:
            # Verify the cached client is still valid
            if vc and vc.is_connected() and vc.guild.id == guild_id:
                return vc
            else:
                # Invalid cached entry, remove it
                _VOICE_CLIENT_CACHE.pop(guild_id, None)
    
    # Cache miss or invalid, perform lookup
    vc = discord.utils.get(bot.voice_clients, guild=guild)
    if vc and vc.is_connected():
        _VOICE_CLIENT_CACHE[guild_id] = (vc, current_time)
    
    return vc


def invalidate_voice_cache(guild_id: int):
    """Invalidate voice client cache for a specific guild."""
    _VOICE_CLIENT_CACHE.pop(guild_id, None)


def start_voice_cleanup_task(loop: Optional[asyncio.AbstractEventLoop] = None):
    """Start the periodic cleanup task and store a reference for controlled shutdown."""
    global _VOICE_CLEANUP_TASK
    if _VOICE_CLEANUP_TASK is not None and not _VOICE_CLEANUP_TASK.done():
        return _VOICE_CLEANUP_TASK
    if loop is None:
        loop = asyncio.get_event_loop()
    _VOICE_CLEANUP_TASK = loop.create_task(cleanup_voice_cache())
    return _VOICE_CLEANUP_TASK


async def stop_voice_cleanup_task():
    """Cancel and await the cleanup task if running. Safe to call from shutdown."""
    global _VOICE_CLEANUP_TASK
    t = _VOICE_CLEANUP_TASK
    _VOICE_CLEANUP_TASK = None
    if t is None:
        return
    try:
        t.cancel()
        await t
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Exception while stopping voice cleanup task", exc_info=True)


async def cleanup_voice_cache():
    """Clean up expired voice client cache entries.

    This function is cancellable and will clear the cache on shutdown for a clean exit.
    """
    try:
        while True:
            try:
                current_time = time.time()
                expired_guilds = []
                for guild_id, (vc, cache_time) in list(_VOICE_CLIENT_CACHE.items()):
                    if current_time - cache_time > _VOICE_CACHE_TIMEOUT * 2:  # Double timeout for cleanup
                        expired_guilds.append(guild_id)

                for guild_id in expired_guilds:
                    _VOICE_CLIENT_CACHE.pop(guild_id, None)

            except Exception as e:
                logger.debug("Voice cache cleanup error: %s", e)
                await asyncio.sleep(30)  # Fallback sleep on error
            await asyncio.sleep(max(1.0, _VOICE_CACHE_TIMEOUT * 2))
    except asyncio.CancelledError:
        # Best-effort cleanup on cancellation
        try:
            _VOICE_CLIENT_CACHE.clear()
        except Exception:
            pass
        raise


async def ensure_connected_for_user(ctx_or_interaction, bot) -> Optional[discord.VoiceClient]:
    """Ensure voice connection when user requests join - extracted from bot.py for modularity."""
    user = getattr(ctx_or_interaction, 'author', None) or getattr(ctx_or_interaction, 'user', None)
    guild = getattr(ctx_or_interaction, 'guild', None)
    
    if not user or not getattr(user, 'voice', None) or not user.voice.channel:
        try:
            if hasattr(ctx_or_interaction, 'response'):
                await ctx_or_interaction.response.send_message("❌ Bạn cần vào voice channel trước.", ephemeral=True)
            else:
                await ctx_or_interaction.send("❌ Bạn cần vào voice channel trước.")
        except Exception:
            pass
        return None
    
    ch = user.voice.channel
    vc = get_voice_client_cached(bot, guild)
    
    try:
        if vc and vc.is_connected():
            if vc.channel.id != ch.id:
                try:
                    metric_inc("voice_connect_reconnects")
                except Exception:
                    pass
                await vc.move_to(ch)
        else:
            # Perform connect with retry, jitter and backoff
            last_exc = None
            for attempt in range(1, _VOICE_CONNECT_MAX_RETRIES + 1):
                try:
                    try:
                        metric_inc("voice_connect_attempts")
                    except Exception:
                        pass
                    vc = await ch.connect()
                    logger.info("Connected to voice channel: %s (guild: %s)", ch.name, guild.id)
                    try:
                        metric_inc("voice_connect_success")
                    except Exception:
                        pass
                    break
                except Exception as e:
                    last_exc = e
                    try:
                        metric_inc("voice_connect_failures")
                        code = getattr(e, 'code', None) or getattr(e, 'status', None) or None
                        if code is not None:
                            try:
                                metric_inc(f"voice_connect_failure_code_{code}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if attempt >= _VOICE_CONNECT_MAX_RETRIES:
                        vc = None
                        break
                    backoff = _VOICE_CONNECT_BASE_BACKOFF * (2 ** (attempt - 1))
                    backoff += float(__import__('random').uniform(0, _VOICE_CONNECT_JITTER))
                    await asyncio.sleep(backoff)
    except Exception:
        logger.exception("Connect failed")
        try:
            if hasattr(ctx_or_interaction, 'response'):
                await ctx_or_interaction.response.send_message("❌ Không thể kết nối voice channel.", ephemeral=True)
            else:
                await ctx_or_interaction.send("❌ Không thể kết nối voice channel.")
        except Exception:
            pass
        return None
    
    return vc
