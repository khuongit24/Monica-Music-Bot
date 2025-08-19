import asyncio
import json
import os
import logging
import time
import signal
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
import types

import discord
from discord.ext import commands
from discord import ui
from yt_dlp import YoutubeDL
import yt_dlp

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
    "prefetch_next": False
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
MAX_QUEUE_SIZE = int(CONFIG.get("max_queue_size", 200))
DOWNLOAD_CONCURRENCY = max(1, int(CONFIG.get("download_concurrency", 1)))
CACHE_TTL_SECONDS = int(CONFIG.get("cache_ttl_seconds", 900))
CACHE_SIZE_LIMIT = int(CONFIG.get("cache_size_limit", 200))
FFMPEG_BITRATE = str(CONFIG.get("ffmpeg_bitrate", "96k"))
FFMPEG_THREADS = int(CONFIG.get("ffmpeg_threads", 1))
PREFETCH_NEXT = bool(CONFIG.get("prefetch_next", False))

THEME_COLOR = 0x9155FD  
OK_COLOR = 0x2ECC71
ERR_COLOR = 0xE74C3C

def format_duration(sec: Optional[int]) -> str:
    if not sec:
        return "LIVE" if sec == 0 else "??:??"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def truncate(text: str, n: int = 60) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[:n-1].rstrip() + "…"

logger = logging.getLogger("Monica")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = RotatingFileHandler("Monica.log", maxBytes=5_000_000, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)
tree = bot.tree

YTDL_OPTS = {
    "format": "bestaudio[ext=webm]/bestaudio/best",
    "quiet": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
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

DOWNLOAD_SEMAPHORE = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

_TRACK_CACHE = OrderedDict()

def _cache_get(key: str):
    entry = _TRACK_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        _TRACK_CACHE.pop(key, None)
        return None
    _TRACK_CACHE.move_to_end(key)
    return entry["data"]

def _cache_put(key: str, data: dict):
    if key in _TRACK_CACHE:
        _TRACK_CACHE.move_to_end(key)
    _TRACK_CACHE[key] = {"data": data, "ts": time.time()}
    while len(_TRACK_CACHE) > CACHE_SIZE_LIMIT:
        _TRACK_CACHE.popitem(last=False)

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
        await asyncio.sleep(60 * 10)

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
    async def resolve(cls, query: str):
        key = query.strip()
        cached = _cache_get(key)
        if cached:
            return cls(cached)

        loop = asyncio.get_running_loop()
        async with DOWNLOAD_SEMAPHORE:
            try:
                data = await loop.run_in_executor(None, lambda: ytdl.extract_info(query, download=False))
            except yt_dlp.utils.DownloadError as e:
                logger.error("yt-dlp download error: %s", e)
                raise RuntimeError(f"yt-dlp error: {e}")
            except Exception as e:
                logger.exception("yt-dlp extract_info failed: %s", e)
                raise RuntimeError("Không thể lấy thông tin nguồn")

        if not data:
            raise RuntimeError("Không tìm thấy kết quả")
        if "entries" in data:
            data = data["entries"][0]
            if data is None:
                raise RuntimeError("Không tìm thấy mục trong kết quả")

        if not data.get("url"):
            raise RuntimeError("Không lấy được stream URL từ nguồn")

        track = cls(data)
        try:
            if not track.is_live:
                _cache_put(key, data)
        except Exception:
            logger.exception("Cache put error (ignored)")

        return track

def create_audio_source(stream_url: str, volume: float = 1.0):
    vol = max(0.0, min(float(volume), 4.0))
    options = f'-vn -af "volume={vol}" -b:a {FFMPEG_BITRATE} -ar 48000 -threads {FFMPEG_THREADS}'
    kwargs = {"before_options": FFMPEG_BEFORE, "options": options}
    try:
        return discord.FFmpegOpusAudio(stream_url, **kwargs)
    except Exception as e:
        logger.warning("FFmpegOpusAudio failed (%s); fallback to PCM", e)
        return discord.FFmpegPCMAudio(stream_url, **kwargs)

class MusicPlayer:
    def __init__(self, guild: discord.Guild, text_channel: discord.TextChannel):
        self.bot = bot
        self.guild = guild
        self.text_channel = text_channel
        self.queue: asyncio.Queue = asyncio.Queue()
        self.next = asyncio.Event()
        self.current: Optional[dict] = None
        self.volume: float = 1.0
        self.loop_mode: bool = False  
        self._task = self.bot.loop.create_task(self._player_loop())
        self._closing = False
        self._lock = asyncio.Lock()

    async def _player_loop(self):
        logger.info("Player start guild=%s", self.guild.id)
        try:
            while not self._closing:
                self.next.clear()
                try:
                    self.current = await asyncio.wait_for(self.queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    try:
                        await self.text_channel.send("Không ai phát nhạc à? Mình đi đây hẹ hẹ hẹ")
                    except Exception:
                        pass
                    break

                if isinstance(self.current, dict):
                    track = YTDLTrack(self.current)
                elif isinstance(self.current, YTDLTrack):
                    track = self.current
                else:
                    if isinstance(self.current, str):
                        try:
                            track = await YTDLTrack.resolve(self.current)
                        except Exception as e:
                            logger.exception("Failed to resolve queued string: %s", e)
                            try:
                                await self.text_channel.send(f"Không thể phát bài đã xếp: {e}")
                            except Exception:
                                pass
                            continue
                    else:
                        logger.error("Unknown queue item type: %s", type(self.current))
                        continue

                if not track.stream_url:
                    logger.error("Track has no stream URL: %s", track.title)
                    try:
                        await self.text_channel.send("Không có stream URL cho bài này :<")
                    except Exception:
                        pass
                    continue

                try:
                    src = create_audio_source(track.stream_url, volume=self.volume)
                except Exception as e:
                    logger.exception("create_audio_source failed: %s", e)
                    try:
                        await self.text_channel.send(f"Lỗi khi tạo nguồn phát: {e}")
                    except Exception:
                        pass
                    continue

                vc = discord.utils.get(self.bot.voice_clients, guild=self.guild)
                if not vc or not vc.is_connected():
                    try:
                        await self.text_channel.send("Mình chưa vô kênh thoại nào cả :<")
                    except Exception:
                        pass
                    continue

                def _after(err):
                    if err:
                        logger.exception("Playback error guild %s: %s", self.guild.id, err)
                    try:
                        self.bot.loop.call_soon_threadsafe(self.next.set)
                    except Exception:
                        logger.exception("Failed to set next event")

                async with self._lock:
                    try:
                        vc.play(src, after=_after)
                        try:
                            vc.source._track_meta = {"title": track.title, "url": track.webpage_url}
                        except Exception:
                            pass
                    except Exception as e:
                        logger.exception("vc.play failed: %s", e)
                        try:
                            await self.text_channel.send(f"Lỗi khi phát: {e}")
                        except Exception:
                            pass
                        continue

                try:
                    requested_by = (track.data.get("requested_by") if isinstance(track.data, dict) else None)
                    desc = f"{'🔴 LIVE —' if track.is_live else '🎧 Now playing —'} {truncate(track.title or 'Unknown', 80)}"
                    embed = discord.Embed(description=desc, color=THEME_COLOR, timestamp=discord.utils.utcnow())
                    embed.set_author(name=track.uploader or "Unknown artist")
                    if track.thumbnail:
                        embed.set_thumbnail(url=track.thumbnail)
                    embed.add_field(name="⏱️ Thời lượng", value=format_duration(track.duration), inline=True)
                    if requested_by:
                        embed.add_field(name="Yêu cầu bởi", value=truncate(requested_by, 30), inline=True)
                    embed.set_footer(text="Monica • Discord Music Bot ✨")
                    await self.text_channel.send(embed=embed, view=MusicControls(self.guild.id))
                except Exception:
                    logger.exception("Failed to send now-playing embed")

                await self.next.wait()

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
            players.pop(self.guild.id, None)
            logger.info("Player stopped guild=%s", self.guild.id)

    def destroy(self):
        self._closing = True
        players.pop(self.guild.id, None)
        try:
            if not self._task.done():
                self._task.cancel()
        except Exception:
            logger.exception("Error cancelling player task")
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass

# global structures
players: Dict[int, MusicPlayer] = {}
guild_locks: Dict[int, asyncio.Lock] = {}

def get_player_for_ctx(ctx):
    guild = getattr(ctx, "guild", None)
    text_channel = getattr(ctx, "channel", None) or getattr(ctx, "text_channel", None)
    if guild is None:
        raise RuntimeError("No guild in context")
    player = players.get(guild.id)
    if not player:
        player = MusicPlayer(guild=guild, text_channel=text_channel)
        players[guild.id] = player
    return player

def _get_guild_lock(guild_id: int) -> asyncio.Lock:
    lk = guild_locks.get(guild_id)
    if not lk:
        lk = asyncio.Lock()
        guild_locks[guild_id] = lk
    return lk

class MusicControls(ui.View):
    def __init__(self, guild_id: int, *, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.guild_id = guild_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message("Mình đang không ở trong kênh thoại nào để điều chỉnh nhạc cả", ephemeral=True)
            return False
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc or not vc.is_connected():
            await interaction.response.send_message("Mình chưa vô kênh thoại nào cả :<", ephemeral=True)
            return False
        if interaction.user.voice.channel.id != vc.channel.id:
            await interaction.response.send_message("Bạn phải ở cùng kênh thoại với mình để điều khiển", ephemeral=True)
            return False
        return True

    @ui.button(emoji="⏯️", style=discord.ButtonStyle.primary, row=0)
    async def pause_resume(self, inter: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=inter.guild)
        if not vc or not getattr(vc, "source", None):
            await inter.response.send_message("Không có bài nào đang phát", ephemeral=True); return
        if vc.is_paused():
            vc.resume(); await inter.response.send_message("▶️ Tiếp tục phát nhạc", ephemeral=True)
        elif vc.is_playing():
            vc.pause(); await inter.response.send_message("⏸️ Đã tạm dừng nhạc", ephemeral=True)
        else:
            await inter.response.send_message("Lỗi khó nói, không thể điều chỉnh", ephemeral=True)

    @ui.button(emoji="⏭️", style=discord.ButtonStyle.secondary, row=0)
    async def skip(self, inter: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=inter.guild)
        if not vc or not vc.is_playing():
            await inter.response.send_message("Không có bài nhạc nào để bỏ qua", ephemeral=True); return
        vc.stop(); await inter.response.send_message("⏭️ Đã bỏ qua bài nhạc này", ephemeral=True)

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
            await inter.response.send_message("Hàng đợi trống trơn", ephemeral=True); return
        upcoming = list(player.queue._queue)[:10]
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

@bot.event
async def on_ready():
    logger.info("Bot ready: %s (ID: %s)", bot.user, bot.user.id)
    try:
        await tree.sync()
        logger.info("Synced application commands.")
    except Exception:
        logger.exception("Failed to sync commands")
    bot.loop.create_task(_cache_cleanup_loop())
    try:
        await bot.change_presence(activity=discord.Game(name="vibing with 300 bài code thiếu nhi ✨"))
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

@tree.command(name="join", description="Kêu bot vào kênh thoại")
async def slash_join(interaction: discord.Interaction):
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Bạn chưa ở trong kênh thoại nào", ephemeral=True); return
    ch = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    try:
        if vc and vc.is_connected():
            await vc.move_to(ch)
        else:
            await ch.connect()
        await interaction.response.send_message(f"✅ Đã kết nối tới **{ch.name}**")
    except Exception:
        logger.exception("join failed")
        await interaction.response.send_message("Không thể kết nối kênh thoại", ephemeral=True)

@bot.command(name="join")
async def text_join(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn chưa ở trong kênh thoại nào"); return
    ch = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    try:
        if vc and vc.is_connected():
            await vc.move_to(ch)
        else:
            await ch.connect()
        await ctx.send(f"✅ Đã kết nối tới **{ch.name}**")
    except Exception:
        logger.exception("join failed (text)")
        await ctx.send("Không thể kết nối kênh thoại")

@tree.command(name="leave", description="Bot rời kênh thoại")
async def slash_leave(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("Bot chưa kết nối kênh thoại nào", ephemeral=True); return
    try:
        await vc.disconnect()
    finally:
        p = players.pop(interaction.guild.id, None)
        if p:
            p.destroy()
    await interaction.response.send_message("Mình đã rời kênh thoại, hẹn gặp lại :3")

@bot.command(name="leave")
async def text_leave(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        await ctx.send("Bot chưa kết nối kênh thoại nào"); return
    try:
        await vc.disconnect()
    finally:
        p = players.pop(ctx.guild.id, None)
        if p:
            p.destroy()
    await ctx.send("Mình đã rời kênh thoại, hẹn gặp lại :3")

@tree.command(name="play", description="Phát nhạc từ URL hoặc tên bài nhạc (YouTube)")
async def slash_play(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.followup.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc", ephemeral=True); return
    ch = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        try:
            vc = await ch.connect()
        except Exception:
            logger.exception("Connect failed")
            await interaction.followup.send("Không thể kết nối vào kênh thoại", ephemeral=True); return

    player = get_player_for_ctx(types.SimpleNamespace(bot=bot, guild=interaction.guild, channel=interaction.channel))
    if player.queue.qsize() >= MAX_QUEUE_SIZE:
        await interaction.followup.send("Hàng đợi đã đầy", ephemeral=True); return

    try:
        track = await YTDLTrack.resolve(query)
    except Exception as e:
        logger.exception("Resolve failed: %s", e)
        await interaction.followup.send(f"Lỗi khi tìm kiếm: {e}", ephemeral=True); return

    # attach requester metadata
    data = dict(track.data)
    data["requested_by"] = interaction.user.display_name
    await player.queue.put(data)
    embed = discord.Embed(description=f"✅ **Đã thêm vào hàng đợi**\n{truncate(track.title, 80)}", color=OK_COLOR)
    embed.set_footer(text="Monica • Đã thêm vào hàng đợi ✨")
    await interaction.followup.send(embed=embed, view=MusicControls(interaction.guild.id))

@bot.command(name="play")
async def text_play(ctx, *, query: str):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc"); return
    ch = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        try:
            vc = await ch.connect()
        except Exception:
            logger.exception("Connect failed (text)")
            await ctx.send("Không thể kết nối kênh thoại."); return

    player = get_player_for_ctx(ctx)
    if player.queue.qsize() >= MAX_QUEUE_SIZE:
        await ctx.send("Hàng đợi đã đầy"); return

    try:
        track = await YTDLTrack.resolve(query)
    except Exception as e:
        logger.exception("Resolve failed (text): %s", e)
        await ctx.send(f"Lỗi khi tìm kiếm: {e}"); return

    data = dict(track.data)
    data["requested_by"] = ctx.author.display_name
    await player.queue.put(data)
    await ctx.send(f"✅ Đã thêm vào hàng đợi: **{truncate(track.title, 80)}**")

@tree.command(name="pause", description="Tạm dừng nhạc")
async def slash_pause(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True); return
    vc.pause(); await interaction.response.send_message("⏸️ Đã tạm dừng.", ephemeral=True)

@bot.command(name="pause")
async def text_pause(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát"); return
    vc.pause(); await ctx.send("⏸️ Đã tạm dừng")

@tree.command(name="resume", description="Tiếp tục phát")
async def slash_resume(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_paused():
        await interaction.response.send_message("Không có bài nhạc nào bị tạm dừng", ephemeral=True); return
    vc.resume(); await interaction.response.send_message("▶️ Tiếp tục phát", ephemeral=True)

@bot.command(name="resume")
async def text_resume(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_paused():
        await ctx.send("Không có bài nhạc nào bị tạm dừng"); return
    vc.resume(); await ctx.send("▶️ Đã tiếp tục phát")

@tree.command(name="skip", description="Bỏ qua bài đang phát")
async def slash_skip(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có nhạc đang phát để bỏ qua", ephemeral=True); return
    vc.stop(); await interaction.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)

@bot.command(name="skip")
async def text_skip(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát để bỏ qua"); return
    vc.stop(); await ctx.send("⏭️ Đã skip bài hiện tại")

@tree.command(name="queue", description="Hiện 10 bài nhạc tiếp theo")
async def slash_queue(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or player.queue.empty():
        await interaction.response.send_message("Hàng đợi trống", ephemeral=True); return
    upcoming = list(player.queue._queue)[:10]
    text = "\n".join(
        f"{idx+1}. {truncate(item.get('title') if isinstance(item, dict) else str(item), 45)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
        for idx, item in enumerate(upcoming)
    )
    embed = discord.Embed(title="Queue (next up)", description=text, color=0x2F3136)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.command(name="queue")
async def text_queue(ctx):
    player = players.get(ctx.guild.id)
    if not player or player.queue.empty():
        await ctx.send("Hàng đợi trống"); return
    upcoming = list(player.queue._queue)[:10]
    text = "\n".join(
        f"{idx+1}. {truncate(item.get('title') if isinstance(item, dict) else str(item), 45)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
        for idx, item in enumerate(upcoming)
    )
    await ctx.send(embed=discord.Embed(title="Queue (next up)", description=text, color=0x2F3136))

@tree.command(name="now", description="Hiện bài đang phát")
async def slash_now(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not getattr(vc, "source", None):
        await interaction.response.send_message("Không có bài nào đang phát", ephemeral=True); return
    player = players.get(interaction.guild.id)
    if player and player.current:
        data = player.current
        embed = discord.Embed(title=truncate(data.get("title", "Now Playing"), 80), url=data.get("webpage_url"), color=THEME_COLOR, timestamp=discord.utils.utcnow())
        if data.get("thumbnail"):
            embed.set_thumbnail(url=data.get("thumbnail"))
        embed.add_field(name="⏱️ Thời lượng", value=format_duration(data.get("duration")), inline=True)
        if data.get("requested_by"):
            embed.add_field(name="🙋 Yêu cầu bởi", value=truncate(data.get("requested_by"), 30), inline=True)
        await interaction.response.send_message(embed=embed)
    else:
        meta = getattr(vc.source, "_track_meta", None)
        if meta:
            embed = discord.Embed(title=truncate(meta.get("title", "Now Playing"), 80), url=meta.get("url"), color=THEME_COLOR, timestamp=discord.utils.utcnow())
            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message("Không có metadata hiện tại.", ephemeral=True)

@bot.command(name="now")
async def text_now(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not getattr(vc, "source", None):
        await ctx.send("Không có bài nào đang phát"); return
    player = players.get(ctx.guild.id)
    if player and player.current:
        data = player.current
        await ctx.send(f"Now playing: {data.get('title')}")
    else:
        meta = getattr(vc.source, "_track_meta", None)
        if meta:
            await ctx.send(f"Now playing: {meta.get('title')}")
        else:
            await ctx.send("Không có metadata hiện tại.")

@tree.command(name="volume", description="Đặt âm lượng (áp dụng cho bài tiếp theo)")
async def slash_volume(interaction: discord.Interaction, vol: float):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có phiên chơi nhạc đang hoạt động", ephemeral=True); return
    player.volume = max(0.0, min(vol, 4.0))
    await interaction.response.send_message(f"🔊 Đã đặt âm lượng (áp dụng cho bài tiếp theo): {player.volume}", ephemeral=True)

@bot.command(name="volume")
async def text_volume(ctx, vol: float):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có phiên chơi nhạc đang hoạt động"); return
    player.volume = max(0.0, min(vol, 4.0))
    await ctx.send(f"🔊 Đã đặt âm lượng (áp dụng cho bài tiếp theo): {player.volume}")

@tree.command(name="list_playlists", description="Liệt kê các playlist đã lưu")
async def slash_list_playlists(interaction: discord.Interaction):
    if not PLAYLISTS:
        await interaction.response.send_message("Chưa có playlist nào.", ephemeral=True); return
    keys = sorted(PLAYLISTS.keys())
    await interaction.response.send_message("Playlist đã lưu:\n" + "\n".join(keys), ephemeral=True)

@bot.command(name="list_playlists")
async def text_list_playlists(ctx):
    if not PLAYLISTS:
        await ctx.send("Chưa có playlist nào."); return
    keys = sorted(PLAYLISTS.keys())
    await ctx.send("Playlist đã lưu:\n" + "\n".join(keys))

@tree.command(name="save_playlist", description="Lưu playlist hiện tại")
@commands.check(lambda interaction: True if OWNER_ID is None else interaction.user.id == int(OWNER_ID))
async def slash_save_playlist(interaction: discord.Interaction, name: str):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có playlist để lưu.", ephemeral=True); return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await interaction.response.send_message(f"✅ Đã lưu playlist `{name}`.", ephemeral=True)

@bot.command(name="save_playlist")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_save_playlist(ctx, name: str):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có playlist để lưu."); return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await ctx.send(f"✅ Đã lưu playlist `{name}`.")

@tree.command(name="play_playlist", description="Phát playlist đã lưu theo tên")
async def slash_play_playlist(interaction: discord.Interaction, name: str):
    if name not in PLAYLISTS:
        await interaction.response.send_message("Không tìm thấy playlist", ephemeral=True); return
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Bạn cần vào kênh thoại để yêu cầu phát nhạc", ephemeral=True); return
    ch = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await ch.connect()
    player = get_player_for_ctx(types.SimpleNamespace(bot=bot, guild=interaction.guild, channel=interaction.channel))
    for item in PLAYLISTS[name]:
        player.queue.put_nowait(item)
    await interaction.response.send_message(f"✅ Đã thêm playlist `{name}` vào hàng đợi.", ephemeral=True)

@bot.command(name="play_playlist")
async def text_play_playlist(ctx, name: str):
    if name not in PLAYLISTS:
        await ctx.send("Không tìm thấy playlist."); return
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc"); return
    ch = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        await ch.connect()
    player = get_player_for_ctx(ctx)
    for item in PLAYLISTS[name]:
        player.queue.put_nowait(item)
    await ctx.send(f"✅ Đã thêm playlist `{name}` vào hàng đợi.")

@tree.command(name="shutdown", description="Tắt bot")
@commands.check(lambda interaction: True if OWNER_ID is None else interaction.user.id == int(OWNER_ID))
async def slash_shutdown(interaction: discord.Interaction):
    await interaction.response.send_message("⚠️ Đang tắt bot...")
    save_playlists()
    try:
        snap = {}
        for gid, p in players.items():
            try:
                snap[str(gid)] = list(p.queue._queue)
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

@bot.command(name="shutdown")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_shutdown(ctx):
    await ctx.send("⚠️ Đang tắt bot...")
    save_playlists()
    try:
        snap = {}
        for gid, p in players.items():
            try:
                snap[str(gid)] = list(p.queue._queue)
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

@bot.event
async def on_command_error(ctx, error):
    logger.exception("Command error: %s", error)
    try:
        await ctx.send(f"Error: {error}")
    except Exception:
        pass

@bot.event
async def on_app_command_error(interaction, error):
    logger.exception("App command error: %s", error)
    try:
        await interaction.response.send_message(f"Error: {error}", ephemeral=True)
    except Exception:
        pass

def _graceful_shutdown():
    logger.info("Signal received: saving playlists and closing")
    try:
        save_playlists()
    except Exception:
        pass
    try:
        snap = {}
        for gid, p in players.items():
            try:
                snap[str(gid)] = list(p.queue._queue)
            except Exception:
                pass
        with open("queues_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed snapshot during shutdown")
    try:
        loop = asyncio.get_event_loop()
        for vc in list(bot.voice_clients):
            try:
                loop.create_task(vc.disconnect())
            except Exception:
                pass
        loop.create_task(bot.close())
    except Exception:
        pass

try:
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, _graceful_shutdown)
    loop.add_signal_handler(signal.SIGTERM, _graceful_shutdown)
except Exception:
    pass

if __name__ == "__main__":
    if not TOKEN:
        logger.error("Token missing: update config.json or set DISCORD_TOKEN env var.")
    else:
        try:
            bot.run(TOKEN)
        except Exception as e:
            logger.exception("Bot terminated with exception: %s", e)
