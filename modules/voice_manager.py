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
_VOICE_CACHE_TIMEOUT = 5.0  # Cache voice clients for 5 seconds


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


async def cleanup_voice_cache():
    """Clean up expired voice client cache entries."""
    while True:
        try:
            current_time = time.time()
            expired_guilds = []
            for guild_id, (vc, cache_time) in _VOICE_CLIENT_CACHE.items():
                if current_time - cache_time > _VOICE_CACHE_TIMEOUT * 2:  # Double timeout for cleanup
                    expired_guilds.append(guild_id)
            
            for guild_id in expired_guilds:
                _VOICE_CLIENT_CACHE.pop(guild_id, None)
            
            await asyncio.sleep(_VOICE_CACHE_TIMEOUT * 2)  # Clean up every 10 seconds
        except Exception as e:
            logger.debug("Voice cache cleanup error: %s", e)
            await asyncio.sleep(30)  # Fallback sleep on error


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
                await vc.move_to(ch)
        else:
            vc = await ch.connect()
            logger.info("Connected to voice channel: %s (guild: %s)", ch.name, guild.id)
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
