"""
Runtime helpers cho UI: Now Playing embed và updater.

- build_now_embed(data, extra_desc, stream_profile) -> discord.Embed
- start_now_updater(player, started_at, duration, interval) -> asyncio.Task

Chỉ sử dụng API công khai của player: now_message, _loop, _closing, current, guild.
Giữ tương thích với logic cũ trong bot.py.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Any

import discord

from modules.utils import THEME_COLOR, truncate, make_progress_bar


def build_now_embed(data: dict, extra_desc: Optional[str] = None, stream_profile: str = "stable") -> discord.Embed:
    try:
        from modules.ui_components import create_now_playing_embed as _np
        return _np(data, extra_desc=extra_desc, stream_profile=stream_profile)
    except Exception:
        # Fallback giống cũ
        title = truncate(data.get("title", "Now Playing"), 80)
        embed = discord.Embed(
            title=title,
            url=data.get("webpage_url"),
            color=THEME_COLOR,
            timestamp=discord.utils.utcnow(),
            description=(f"{'🔴 LIVE' if data.get('is_live') else '🎧 Now Playing'}\n"
                         f"{extra_desc if extra_desc else ''}")
        )
        uploader = data.get("uploader") or data.get("channel")
        if uploader:
            embed.add_field(name="Kênh", value=truncate(str(uploader), 60), inline=True)
        dur = data.get("duration")
        if dur and isinstance(dur, (int, float)):
            m = int(dur // 60); s = int(dur % 60)
            embed.add_field(name="Thời lượng", value=f"{m}:{s:02d}", inline=True)
        return embed


def start_now_updater(player: Any, started_at: float, duration: Optional[float], interval: int, stream_profile: str) -> asyncio.Task:
    """Tạo task cập nhật Now Playing định kỳ và trả về task.
    player cần có: now_message, _closing, guild, current, _loop (optional).
    """
    from modules.ui_components import MusicControls as _Controls

    async def updater():
        try:
            while True:
                if not getattr(player, "now_message", None) or getattr(player, "_closing", False):
                    return
                try:
                    elapsed = time.time() - started_at
                    bar = make_progress_bar(elapsed, duration)
                    embed = build_now_embed(player.current, extra_desc=bar, stream_profile=stream_profile)
                    view = _Controls(player.guild.id)
                    msg = await player.now_message.edit(embed=embed, view=view)
                    try:
                        # Bind message back so view can self-renew on timeout
                        view.message = msg
                    except Exception:
                        pass
                except discord.HTTPException:
                    pass
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception:
            import logging
            logging.getLogger("Monica").exception("Now update task failed")

    # Tạo task bằng event loop của player nếu có, nếu không dùng asyncio.create_task
    try:
        return player._loop.create_task(updater())
    except Exception:
        return asyncio.create_task(updater())
