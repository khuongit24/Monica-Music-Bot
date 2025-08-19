# Discord Music Bot - ver_1.0_release
# Production-ready-ish for a private server (educational / personal use)
# Features:
#  - Slash commands + prefix fallback (configurable)
#  - Queue with asyncio.Queue per guild
#  - Persistent playlists saved in playlists.json
#  - Robust error handling and logging
#  - Docker-friendly: reads token from CONFIG or env var
import asyncio
import json
import logging
import os
from typing import Dict, Optional

import discord
from discord import app_commands
from discord.ext import commands
from yt_dlp import YoutubeDL

# --- Configuration ---
CONFIG_PATH = "config.json"
DEFAULT_PREFIX = "!"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"token": os.getenv("DISCORD_TOKEN", ""), "prefix": DEFAULT_PREFIX, "owner_id": None}

TOKEN = CONFIG.get("token") or os.getenv("DISCORD_TOKEN")
PREFIX = CONFIG.get("prefix", DEFAULT_PREFIX)
OWNER_ID = CONFIG.get("owner_id")  # optional, used for admin-only commands

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("discord_bot")

# --- Discord bot setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)
tree = bot.tree

# --- ytdl and ffmpeg options ---
YTDL_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
}

FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn"
}

ytdl = YoutubeDL(YTDL_OPTS)

# --- Persistent storage for playlists ---
PLAYLISTS_PATH = "playlists.json"
try:
    if os.path.exists(PLAYLISTS_PATH):
        with open(PLAYLISTS_PATH, "r", encoding="utf-8") as f:
            PLAYLISTS = json.load(f)
    else:
        PLAYLISTS = {}
except Exception as e:
    logger.exception("Không thể đọc playlists.json: %s", e)
    PLAYLISTS = {}

def save_playlists():
    try:
        with open(PLAYLISTS_PATH, "w", encoding="utf-8") as f:
            json.dump(PLAYLISTS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Lỗi khi lưu playlists.json: %s", e)

# --- Music player per guild ---
class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get("title")
        self.webpage_url = data.get("webpage_url")

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))
        if data is None:
            raise RuntimeError("Không thể lấy thông tin từ nguồn.")
        if "entries" in data:
            data = data["entries"][0]
        filename = data["url"] if stream else ytdl.prepare_filename(data)
        source = discord.FFmpegPCMAudio(filename, **FFMPEG_OPTIONS)
        return cls(source, data=data)

class MusicPlayer:
    def __init__(self, ctx):
        self.bot = ctx.bot
        self._ctx = ctx
        self.queue = asyncio.Queue()
        self.next = asyncio.Event()
        self.current = None
        self.volume = 0.5
        self._task = self.bot.loop.create_task(self.player_loop())

    async def player_loop(self):
        try:
            while True:
                self.next.clear()
                # Wait for the next track
                self.current = await self.queue.get()
                try:
                    source = await YTDLSource.from_url(self.current["webpage_url"], loop=self.bot.loop, stream=True)
                except Exception as e:
                    await self._ctx.send(f"Không thể phát bài: {e}")
                    continue

                vc: discord.VoiceClient = discord.utils.get(self.bot.voice_clients, guild=self._ctx.guild)
                if not vc or not vc.is_connected():
                    await self._ctx.send("Bot chưa kết nối vào kênh thoại.")
                    continue

                vc.play(source, after=lambda e: self.bot.loop.call_soon_threadsafe(self.next.set))
                vc.source = source
                vc.source.volume = self.volume
                await self._ctx.send(f"🎶 Đang phát: **{source.title}**")
                await self.next.wait()
        except asyncio.CancelledError:
            logger.info("Music player task cancelled for guild %s", getattr(self._ctx.guild, "id", None))
        except Exception as e:
            logger.exception("Lỗi không mong muốn trong player_loop: %s", e)

    def destroy(self):
        self._task.cancel()

players: Dict[int, MusicPlayer] = {}

def get_player(ctx) -> MusicPlayer:
    player = players.get(ctx.guild.id)
    if not player:
        player = MusicPlayer(ctx)
        players[ctx.guild.id] = player
    return player

# --- Helper: is_owner decorator ---
def is_owner():
    def predicate(interaction: discord.Interaction):
        if OWNER_ID is None:
            return True
        return int(OWNER_ID) == interaction.user.id
    return app_commands.check(predicate)

# --- Events ---
@bot.event
async def on_ready():
    logger.info("Bot ready: %s (ID: %s)", bot.user, bot.user.id)
    try:
        await tree.sync()
        logger.info("Synced application commands.")
    except Exception as e:
        logger.exception("Không thể sync commands: %s", e)

# --- Voice control commands (slash + fallback) ---
@tree.command(name="join", description="Kêu bot vào kênh thoại của bạn")
async def slash_join(interaction: discord.Interaction):
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Bạn chưa ở trong kênh thoại.", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        await channel.connect()
    await interaction.response.send_message(f"Đã kết nối tới **{channel.name}**")

@bot.command(name="join")
async def text_join(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn chưa ở trong kênh thoại.")
        return
    channel = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        await channel.connect()
    await ctx.send(f"Đã kết nối tới **{channel.name}**")

@tree.command(name="leave", description="Bot rời kênh thoại")
async def slash_leave(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("Bot chưa kết nối kênh thoại.", ephemeral=True)
        return
    await vc.disconnect()
    player = players.pop(interaction.guild.id, None)
    if player:
        player.destroy()
    await interaction.response.send_message("Đã rời kênh thoại.")

@bot.command(name="leave")
async def text_leave(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        await ctx.send("Bot chưa kết nối kênh thoại.")
        return
    await vc.disconnect()
    player = players.pop(ctx.guild.id, None)
    if player:
        player.destroy()
    await ctx.send("Đã rời kênh thoại.")

# --- Playback commands ---
async def fetch_info(search: str):
    loop = bot.loop
    data = await loop.run_in_executor(None, lambda: ytdl.extract_info(search, download=False))
    if data is None:
        raise RuntimeError("Không tìm thấy kết quả.")
    if "entries" in data:
        data = data["entries"][0]
    return {"webpage_url": data.get("webpage_url"), "title": data.get("title")}

@tree.command(name="play", description="Phát nhạc từ URL hoặc từ khóa (YouTube)")
async def slash_play(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.followup.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc.", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    try:
        track = await fetch_info(query)
    except Exception as e:
        await interaction.followup.send(f"Lỗi khi tìm kiếm: {e}", ephemeral=True)
        return

    guild_ctx = types.SimpleNamespace(bot=bot, guild=interaction.guild)
    player = get_player(guild_ctx)
    await player.queue.put(track)
    await interaction.followup.send(f"Đã thêm vào hàng đợi: **{track.get('title')}**")

@bot.command(name="play", help="play <url hoặc từ khóa>")
async def text_play(ctx, *, query: str):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc.")
        return
    channel = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    try:
        track = await fetch_info(query)
    except Exception as e:
        await ctx.send(f"Lỗi khi tìm kiếm: {e}")
        return

    player = get_player(ctx)
    await player.queue.put(track)
    await ctx.send(f"Đã thêm vào hàng đợi: **{track.get('title')}**")

@tree.command(name="pause", description="Tạm dừng nhạc")
async def slash_pause(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có nhạc đang phát.", ephemeral=True)
        return
    vc.pause()
    await interaction.response.send_message("Đã tạm dừng.")

@bot.command(name="pause")
async def text_pause(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có nhạc đang phát.")
        return
    vc.pause()
    await ctx.send("Đã tạm dừng.")

@tree.command(name="resume", description="Tiếp tục phát")
async def slash_resume(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected() or not vc.is_paused():
        await interaction.response.send_message("Không có nhạc bị tạm dừng.", ephemeral=True)
        return
    vc.resume()
    await interaction.response.send_message("Đã tiếp tục phát.")

@bot.command(name="resume")
async def text_resume(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected() or not vc.is_paused():
        await ctx.send("Không có nhạc bị tạm dừng.")
        return
    vc.resume()
    await ctx.send("Đã tiếp tục phát.")

@tree.command(name="skip", description="Bỏ bài đang phát")
async def slash_skip(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có nhạc đang phát để bỏ.", ephemeral=True)
        return
    vc.stop()
    await interaction.response.send_message("Bài hiện tại đã bị bỏ.")

@bot.command(name="skip")
async def text_skip(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có nhạc đang phát để bỏ.")
        return
    vc.stop()
    await ctx.send("Bài hiện tại đã bị bỏ.")

@tree.command(name="queue", description="Hiện 10 bài tiếp theo trong hàng đợi")
async def slash_queue(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or player.queue.empty():
        await interaction.response.send_message("Hàng đợi trống.", ephemeral=True)
        return
    upcoming = list(player.queue._queue)[:10]
    text = "\\n".join(f"{idx+1}. {item.get('title')}" for idx, item in enumerate(upcoming))
    await interaction.response.send_message(f"Hàng đợi tiếp theo:\\n{text}")

@bot.command(name="queue")
async def text_queue(ctx):
    player = players.get(ctx.guild.id)
    if not player or player.queue.empty():
        await ctx.send("Hàng đợi trống.")
        return
    upcoming = list(player.queue._queue)[:10]
    text = "\\n".join(f"{idx+1}. {item.get('title')}" for idx, item in enumerate(upcoming))
    await ctx.send(f"Hàng đợi tiếp theo:\\n{text}")

@tree.command(name="now", description="Hiện bài đang phát")
async def slash_now(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not getattr(vc, "source", None):
        await interaction.response.send_message("Không có bài nào đang phát.", ephemeral=True)
        return
    await interaction.response.send_message(f"Đang phát: **{vc.source.title}**")

@bot.command(name="now")
async def text_now(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not getattr(vc, "source", None):
        await ctx.send("Không có bài nào đang phát.")
        return
    await ctx.send(f"Đang phát: **{vc.source.title}**")

@tree.command(name="volume", description="Đặt âm lượng (0.0 - 2.0)")
async def slash_volume(interaction: discord.Interaction, vol: float):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có phiên chơi nhạc nào đang hoạt động.", ephemeral=True)
        return
    player.volume = max(0.0, min(vol, 2.0))
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if vc and getattr(vc, "source", None):
        vc.source.volume = player.volume
    await interaction.response.send_message(f"Đã đặt âm lượng: {player.volume}")

@bot.command(name="volume")
async def text_volume(ctx, vol: float):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có phiên chơi nhạc nào đang hoạt động.")
        return
    player.volume = max(0.0, min(vol, 2.0))
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if vc and getattr(vc, "source", None):
        vc.source.volume = player.volume
    await ctx.send(f"Đã đặt âm lượng: {player.volume}")

# --- Playlist management (persistent) ---
import types

@tree.command(name="list_playlists", description="Liệt kê playlist đã lưu (server global)")
async def slash_list_playlists(interaction: discord.Interaction):
    if not PLAYLISTS:
        await interaction.response.send_message("Chưa có playlist nào.", ephemeral=True)
        return
    keys = sorted(PLAYLISTS.keys())
    await interaction.response.send_message("Playlist đã lưu:\n" + "\\n".join(keys))

@bot.command(name="list_playlists")
async def text_list_playlists(ctx):
    if not PLAYLISTS:
        await ctx.send("Chưa có playlist nào.")
        return
    keys = sorted(PLAYLISTS.keys())
    await ctx.send("Playlist đã lưu:\n" + "\\n".join(keys))

@tree.command(name="save_playlist", description="Lưu playlist hiện tại thành tên được chỉ định")
@is_owner()
async def slash_save_playlist(interaction: discord.Interaction, name: str):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có playlist/phiên để lưu.", ephemeral=True)
        return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await interaction.response.send_message(f"Đã lưu playlist `{name}`.")

@bot.command(name="save_playlist")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_save_playlist(ctx, name: str):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có playlist/phiên để lưu.")
        return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await ctx.send(f"Đã lưu playlist `{name}`.")

@tree.command(name="play_playlist", description="Phát playlist đã lưu theo tên")
async def slash_play_playlist(interaction: discord.Interaction, name: str):
    if name not in PLAYLISTS:
        await interaction.response.send_message("Không tìm thấy playlist.", ephemeral=True)
        return
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Bạn cần vào kênh thoại để yêu cầu phát nhạc.", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    player = get_player(types.SimpleNamespace(bot=bot, guild=interaction.guild))
    for item in PLAYLISTS[name]:
        await player.queue.put(item)
    await interaction.response.send_message(f"Đã thêm playlist `{name}` vào hàng đợi.")

@bot.command(name="play_playlist")
async def text_play_playlist(ctx, name: str):
    if name not in PLAYLISTS:
        await ctx.send("Không tìm thấy playlist.")
        return
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn cần vào kênh thoại để yêu cầu phát nhạc.")
        return
    channel = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    player = get_player(ctx)
    for item in PLAYLISTS[name]:
        await player.queue.put(item)
    await ctx.send(f"Đã thêm playlist `{name}` vào hàng đợi.")

# --- Shutdown command (owner only) ---
@tree.command(name="shutdown", description="Tắt bot (chỉ owner nếu được cấu hình)")
@is_owner()
async def slash_shutdown(interaction: discord.Interaction):
    await interaction.response.send_message("Đang tắt bot...")
    await bot.close()

@bot.command(name="shutdown")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_shutdown(ctx):
    await ctx.send("Đang tắt bot...")
    await bot.close()

# --- Error handlers ---
@bot.event
async def on_command_error(ctx, error):
    logger.exception("Lỗi command: %s", error)
    try:
        await ctx.send(f"Lỗi: {error}")
    except Exception:
        pass

@bot.event
async def on_app_command_error(interaction, error):
    logger.exception("Lỗi app command: %s", error)
    try:
        await interaction.response.send_message(f"Lỗi: {error}", ephemeral=True)
    except Exception:
        pass

# --- Entry point ---
if __name__ == "__main__":
    if not TOKEN:
        logger.error("Không tìm thấy token bot. Thiết lập trong config.json hoặc biến môi trường DISCORD_TOKEN.")
    else:
        bot.run(TOKEN)