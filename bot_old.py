import asyncio, json, os, logging, typing, time
from typing import Dict, Optional
import discord
from discord.ext import commands
from discord import app_commands, ui
from yt_dlp import YoutubeDL

# --- Config ---
CONFIG_PATH = "config.json"
DEFAULT_PREFIX = "!"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"token": os.getenv("DISCORD_TOKEN", ""), "prefix": DEFAULT_PREFIX, "owner_id": None}

TOKEN = CONFIG.get("token") or os.getenv("DISCORD_TOKEN")
PREFIX = CONFIG.get("prefix", DEFAULT_PREFIX)
OWNER_ID = CONFIG.get("owner_id")

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("discord_bot")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)
tree = bot.tree

#yt-dlp / ffmpeg options
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

PLAYLISTS_PATH = "playlists.json"
try:
    if os.path.exists(PLAYLISTS_PATH):
        with open(PLAYLISTS_PATH, "r", encoding="utf-8") as f:
            PLAYLISTS = json.load(f)
    else:
        PLAYLISTS = {}
except Exception as e:
    logger.exception("Cannot read playlists.json: %s", e)
    PLAYLISTS = {}

def save_playlists():
    try:
        with open(PLAYLISTS_PATH, "w", encoding="utf-8") as f:
            json.dump(PLAYLISTS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Error saving playlists.json: %s", e)

# --- Music player ---
class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get("title")
        self.webpage_url = data.get("webpage_url")
        self.thumbnail = data.get("thumbnail")
        self.uploader = data.get("uploader")
        self.duration = data.get("duration")  # seconds

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))
        if data is None:
            raise RuntimeError("Không thể lấy thông tin từ nguồn")
        if "entries" in data:
            data = data["entries"][0]
        filename = data["url"] if stream else ytdl.prepare_filename(data)
        source = discord.FFmpegPCMAudio(filename, **FFMPEG_OPTIONS)
        return cls(source, data=data)

class MusicPlayer:
    def __init__(self, ctx):
        self.bot = ctx.bot
        self.guild = ctx.guild
        self.text_channel = ctx.channel
        self.queue = asyncio.Queue()
        self.next = asyncio.Event()
        self.current = None
        self.volume = 0.5
        self.started_at = None 
        self._task = self.bot.loop.create_task(self.player_loop())

    async def player_loop(self):
        try:
            while True:
                self.next.clear()
                self.current = await self.queue.get()
                try:
                    source = await YTDLSource.from_url(self.current["webpage_url"], loop=self.bot.loop, stream=True)
                except Exception as e:
                    await self.text_channel.send(f"Không thể phát bài: {e}")
                    continue

                vc: discord.VoiceClient = discord.utils.get(self.bot.voice_clients, guild=self.guild)
                if not vc or not vc.is_connected():
                    await self.text_channel.send("Chưa vào voice sao mà tui phát nhạc được?")
                    continue

                vc.play(source, after=lambda e: self.bot.loop.call_soon_threadsafe(self.next.set))
                vc.source = source
                vc.source.volume = self.volume
                self.started_at = time.time()
                try:
                    embed = make_now_playing_embed(source)
                    view = MusicControls(self.guild.id)
                    await self.text_channel.send(embed=embed, view=view)
                except Exception as e:
                    logger.exception("Không thể gửi tin nhắn Now-Playing: %s", e)

                await self.next.wait()
        except asyncio.CancelledError:
            logger.info("Trình phát nhạc đã bị dừng %s", self.guild.id)
        except Exception as e:
            logger.exception("Gặp lỗi khó nói: %s", e)

    def destroy(self):
        self._task.cancel()

players: Dict[int, MusicPlayer] = {}

def get_player(ctx) -> MusicPlayer:
    player = players.get(ctx.guild.id)
    if not player:
        player = MusicPlayer(ctx)
        players[ctx.guild.id] = player
    return player

# --- Helper: create embed cards ---
def format_duration(sec: typing.Optional[int]) -> str:
    if not sec:
        return "Unknown"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def make_now_playing_embed(source: YTDLSource) -> discord.Embed:
    title = source.title or "Unknown title"
    url = source.webpage_url or ""
    thumb = getattr(source, "thumbnail", None)
    uploader = getattr(source, "uploader", None)
    duration = getattr(source, "duration", None)
    embed = discord.Embed(title=title, url=url, description=f"By {uploader}" if uploader else None, color=0x1DB954)
    if thumb:
        embed.set_thumbnail(url=thumb)
    embed.add_field(name="Thời lượng", value=format_duration(duration), inline=True)
    embed.add_field(name="Điều chỉnh", value="Dùng các nút dưới đây để điều chỉnh nhạc", inline=True)
    embed.set_footer(text="Monica Bot • By shio")
    return embed

# --- Interactive controls (buttons) ---
class MusicControls(ui.View):
    def __init__(self, guild_id: int, *, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.guild_id = guild_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message("Chưa vào voice sao mà tui điều chỉnh được?", ephemeral=True)
            return False
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc or not vc.is_connected():
            await interaction.response.send_message("Bot chưa kết nối kênh thoại nào", ephemeral=True)
            return False
        return True

    @ui.button(label="⏯️ Tạm dừng / Tiếp tục", style=discord.ButtonStyle.primary, custom_id="btn_pause_resume")
    async def pause_resume(self, interaction: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc or not getattr(vc, "source", None):
            await interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True)
            return
        if vc.is_paused():
            vc.resume()
            await interaction.response.send_message("Tiếp tục phát", ephemeral=True)
        elif vc.is_playing():
            vc.pause()
            await interaction.response.send_message("Tạm dừng", ephemeral=True)
        else:
            await interaction.response.send_message("Gặp lỗi khó nói, không thể điều chỉnh", ephemeral=True)

    @ui.button(label="⏭️ Skip", style=discord.ButtonStyle.secondary, custom_id="btn_skip")
    async def skip(self, interaction: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc or not vc.is_playing():
            await interaction.response.send_message("Không có bài nhạc nào đang phát để skip", ephemeral=True)
            return
        vc.stop()
        await interaction.response.send_message("Bài hiện tại đã bị skip", ephemeral=True)

    @ui.button(label="⏹️ Stop", style=discord.ButtonStyle.danger, custom_id="btn_stop")
    async def stop(self, interaction: discord.Interaction, button: ui.Button):
        vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
        if not vc:
            await interaction.response.send_message("Chưa vào voice sao mà tui dừng nhạc được?", ephemeral=True)
            return
        vc.stop()
        player = players.pop(interaction.guild.id, None)
        if player:
            player.destroy()
        await interaction.response.send_message("Đã dừng phát nhạc và xóa hàng đợi", ephemeral=True)

    @ui.button(label="📜 Queue", style=discord.ButtonStyle.secondary, custom_id="btn_queue")
    async def show_queue(self, interaction: discord.Interaction, button: ui.Button):
        player = players.get(interaction.guild.id)
        if not player or player.queue.empty():
            await interaction.response.send_message("Hàng đợi trống(chưa có bài nhạc nào á)", ephemeral=True)
            return
        upcoming = list(player.queue._queue)[:10]
        text = "\n".join(f"{idx+1}. {item.get('title')}" for idx, item in enumerate(upcoming))
        embed = discord.Embed(title="Queue (next up)", description=text, color=0x2F3136)
        await interaction.response.send_message(embed=embed, ephemeral=True)

# --- Utility command fetcher ---
async def fetch_info(search: str):
    loop = bot.loop
    data = await loop.run_in_executor(None, lambda: ytdl.extract_info(search, download=False))
    if data is None:
        raise RuntimeError("Không tìm thấy kết quả.")
    if "entries" in data:
        data = data["entries"][0]
    return {"webpage_url": data.get("webpage_url"), "title": data.get("title"), "thumbnail": data.get("thumbnail"), "uploader": data.get("uploader"), "duration": data.get("duration")}

# --- Events ---
@bot.event
async def on_ready():
    logger.info("Bot ready: %s (ID: %s)", bot.user, bot.user.id)
    try:
        await tree.sync()
        logger.info("Synced application commands.")
    except Exception as e:
        logger.exception("Failed to sync commands: %s", e)
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="300 bài code thiếu nhi"))

@tree.command(name="join", description="Để gọi bot vào kênh thoại của bạn")
async def slash_join(interaction: discord.Interaction):
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Hong có voice chat nào để vào theo bạn hết :<", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        await channel.connect()
    await interaction.response.send_message(f"✅ Đã kết nối tới **{channel.name}**")

@bot.command(name="join")
async def text_join(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Bạn chưa tạo voice sao mà tui vào được?")
        return
    channel = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        await channel.connect()
    await ctx.send(f"✅ Đã kết nối tới **{channel.name}**")

@tree.command(name="leave", description="Đá Bot rời kênh thoại")
async def slash_leave(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("Mình không ở trong voice mà sao lại đá mình :<", ephemeral=True)
        return
    await vc.disconnect()
    player = players.pop(interaction.guild.id, None)
    if player:
        player.destroy()
    await interaction.response.send_message("👋 Mình đã rời kênh thoại roài")

@bot.command(name="leave")
async def text_leave(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        await ctx.send("Mình không có trong voice để mà bị đá đâu")
        return
    await vc.disconnect()
    player = players.pop(ctx.guild.id, None)
    if player:
        player.destroy()
    await ctx.send("👋 Đã rời kênh thoại roài")

@tree.command(name="play", description="Phát nhạc từ URL (link youtube) hoặc tên bài nhạc")
async def slash_play(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.followup.send("Chưa vào voice sao mà tui phát nhạc được?", ephemeral=True)
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

    player = get_player(types.SimpleNamespace(bot=bot, guild=interaction.guild, channel=interaction.channel))
    await player.queue.put(track)
    embed = discord.Embed(description=f"✅ Đã thêm vào hàng đợi: **{track.get('title')}**", color=0x2ECC71)
    view = MusicControls(interaction.guild.id)
    await interaction.followup.send(embed=embed, view=view)

@bot.command(name="play", help="play <url hoặc từ khóa>")
async def text_play(ctx, *, query: str):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Chưa vào voice sao mà tui phát nhạc được?")
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
    embed = discord.Embed(description=f"✅ Đã thêm vào hàng đợi: **{track.get('title')}**", color=0x2ECC71)
    view = MusicControls(ctx.guild.id)
    await ctx.send(embed=embed, view=view)

@tree.command(name="pause", description="Tạm dừng nhạc")
async def slash_pause(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True)
        return
    vc.pause()
    await interaction.response.send_message("⏸️ Đã tạm dừng", ephemeral=True)

@bot.command(name="pause")
async def text_pause(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát")
        return
    vc.pause()
    await ctx.send("⏸️ Đã tạm dừng.")

@tree.command(name="resume", description="Tiếp tục phát")
async def slash_resume(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected() or not vc.is_paused():
        await interaction.response.send_message("Không có bài nhạc nào để tạm dừng", ephemeral=True)
        return
    vc.resume()
    await interaction.response.send_message("▶️ Tiếp tục phát nhạc", ephemeral=True)

@bot.command(name="resume")
async def text_resume(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected() or not vc.is_paused():
        await ctx.send("Không có bài nhạc nào bị tạm dừng.")
        return
    vc.resume()
    await ctx.send("▶️ Đã tiếp tục phát.")

@tree.command(name="skip", description="Bỏ qua bài đang phát")
async def slash_skip(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Không có nhạc đang phát để bỏ qua", ephemeral=True)
        return
    vc.stop()
    await interaction.response.send_message("⏭️ Bài hiện tại đã bị skip", ephemeral=True)

@bot.command(name="skip")
async def text_skip(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_playing():
        await ctx.send("Không có bài nhạc nào đang phát để skip")
        return
    vc.stop()
    await ctx.send("⏭️ Bài hiện tại đã bị skip")

@tree.command(name="queue", description="Hiện 10 bài nhạc tiếp theo trong hàng đợi")
async def slash_queue(interaction: discord.Interaction):
    player = players.get(interaction.guild.id)
    if not player or player.queue.empty():
        await interaction.response.send_message("Hàng đợi trống", ephemeral=True)
        return
    upcoming = list(player.queue._queue)[:10]
    text = "\n".join(f"{idx+1}. {item.get('title')}" for idx, item in enumerate(upcoming))
    embed = discord.Embed(title="Queue (next up)", description=text, color=0x2F3136)
    await interaction.response.send_message(embed=embed)

@bot.command(name="queue")
async def text_queue(ctx):
    player = players.get(ctx.guild.id)
    if not player or player.queue.empty():
        await ctx.send("Hàng đợi trống")
        return
    upcoming = list(player.queue._queue)[:10]
    text = "\n".join(f"{idx+1}. {item.get('title')}" for idx, item in enumerate(upcoming))
    await ctx.send(embed=discord.Embed(title="Queue (next up)", description=text, color=0x2F3136))

@tree.command(name="now", description="Hiện bài nhạc đang phát")
async def slash_now(interaction: discord.Interaction):
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not getattr(vc, "source", None):
        await interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True)
        return
    embed = make_now_playing_embed(vc.source)
    await interaction.response.send_message(embed=embed)

@bot.command(name="now")
async def text_now(ctx):
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not getattr(vc, "source", None):
        await ctx.send("Không có bài nhạc nào đang phát")
        return
    await ctx.send(embed=make_now_playing_embed(vc.source))

@tree.command(name="volume", description="Đặt âm lượng (0.0 - 2.0)")
async def slash_volume(interaction: discord.Interaction, vol: float):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có phiên chơi nhạc nào đang hoạt động", ephemeral=True)
        return
    player.volume = max(0.0, min(vol, 2.0))
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if vc and getattr(vc, "source", None):
        vc.source.volume = player.volume
    await interaction.response.send_message(f"🔊 Đã đặt âm lượng: {player.volume}", ephemeral=True)

@bot.command(name="volume")
async def text_volume(ctx, vol: float):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có phiên chơi nhạc nào đang hoạt động")
        return
    player.volume = max(0.0, min(vol, 2.0))
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if vc and getattr(vc, "source", None):
        vc.source.volume = player.volume
    await ctx.send(f"🔊 Đã đặt âm lượng: {player.volume}")

import types

@tree.command(name="list_playlists", description="Liệt kê các playlist đã lưu")
async def slash_list_playlists(interaction: discord.Interaction):
    if not PLAYLISTS:
        await interaction.response.send_message("Chưa có playlist nào.", ephemeral=True)
        return
    keys = sorted(PLAYLISTS.keys())
    await interaction.response.send_message("Playlist đã lưu:\n" + "\n".join(keys))

@bot.command(name="list_playlists")
async def text_list_playlists(ctx):
    if not PLAYLISTS:
        await ctx.send("Chưa có playlist nào.")
        return
    keys = sorted(PLAYLISTS.keys())
    await ctx.send("Playlist đã lưu:\n" + "\n".join(keys))

@tree.command(name="save_playlist", description="Lưu playlist hiện tại")
@app_commands.check(lambda interaction: True if OWNER_ID is None else interaction.user.id == int(OWNER_ID))
async def slash_save_playlist(interaction: discord.Interaction, name: str):
    player = players.get(interaction.guild.id)
    if not player:
        await interaction.response.send_message("Không có playlist nào để lưu.", ephemeral=True)
        return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await interaction.response.send_message(f"✅ Đã lưu playlist `{name}`.")

@bot.command(name="save_playlist")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_save_playlist(ctx, name: str):
    player = players.get(ctx.guild.id)
    if not player:
        await ctx.send("Không có playlist để lưu.")
        return
    items = list(player.queue._queue)
    PLAYLISTS[name] = items
    save_playlists()
    await ctx.send(f"✅ Đã lưu playlist `{name}`.")

@tree.command(name="play_playlist", description="Phát playlist đã lưu theo tên")
async def slash_play_playlist(interaction: discord.Interaction, name: str):
    if name not in PLAYLISTS:
        await interaction.response.send_message("Không tìm thấy playlist", ephemeral=True)
        return
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Chưa vào voice sao mà tui phát nhạc được?", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    player = get_player(types.SimpleNamespace(bot=bot, guild=interaction.guild, channel=interaction.channel))
    for item in PLAYLISTS[name]:
        await player.queue.put(item)
    await interaction.response.send_message(f"✅ Đã thêm playlist `{name}` vào hàng đợi.")

@bot.command(name="play_playlist")
async def text_play_playlist(ctx, name: str):
    if name not in PLAYLISTS:
        await ctx.send("Không tìm thấy playlist.")
        return
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Chưa vào voice sao mà tui phát nhạc được?")
        return
    channel = ctx.author.voice.channel
    vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if not vc or not vc.is_connected():
        vc = await channel.connect()

    player = get_player(ctx)
    for item in PLAYLISTS[name]:
        await player.queue.put(item)
    await ctx.send(f"✅ Đã thêm playlist `{name}` vào hàng đợi.")

@tree.command(name="shutdown", description="Tắt bot")
@app_commands.check(lambda interaction: True if OWNER_ID is None else interaction.user.id == int(OWNER_ID))
async def slash_shutdown(interaction: discord.Interaction):
    await interaction.response.send_message("⚠️ Đang tắt bot...")
    await bot.close()

@bot.command(name="shutdown")
@commands.check(lambda ctx: True if OWNER_ID is None else ctx.author.id == int(OWNER_ID))
async def text_shutdown(ctx):
    await ctx.send("⚠️ Đang tắt bot...")
    await bot.close()

@bot.command(name="help")
async def text_help(ctx):
    embed = discord.Embed(title="Trợ giúp — Câu lệnh", color=0x5865F2, description="Danh sách lệnh của bot")
    embed.add_field(name="/join or !join", value="Kêu bot vào kênh thoại của bạn", inline=False)
    embed.add_field(name="/play <query> or !play <query>", value="Thêm bài nhạc vào hàng đợi (Link hoặc tên bài nhạc)", inline=False)
    embed.add_field(name="/pause / /resume / /skip / /stop", value="Dừng/tạm dừng/tiếp tục/bỏ qua bài nhạc", inline=False)
    embed.add_field(name="/queue / /now / /volume", value="Xem hàng đợi, hiển thị bài đang phát, đặt âm lượng", inline=False)
    embed.add_field(name="/list_playlists / /save_playlist / /play_playlist", value="Quản lý playlist", inline=False)
    embed.set_footer(text="Monica Music Bot | By shio")
    await ctx.send(embed=embed)

@tree.command(name="help", description="Hiện help embed")
async def slash_help(interaction: discord.Interaction):
    embed = discord.Embed(title="Help — Commands", color=0x5865F2, description="Danh sách lệnh của bot")
    embed.add_field(name="/join or !join", value="Kêu bot vào kênh thoại của bạn", inline=False)
    embed.add_field(name="/play <query> or !play <query>", value="Thêm bài vào hàng đợi (Link hoặc tên bài nhạc)", inline=False)
    embed.add_field(name="/pause / /resume / /skip / /stop", value="Dừng/tạm dừng/tiếp tục/bỏ bài", inline=False)
    embed.add_field(name="/queue / /now / /volume", value="Xem hàng đợi, hiển thị bài đang phát, đặt âm lượng", inline=False)
    embed.add_field(name="/list_playlists / /save_playlist / /play_playlist", value="Quản lý playlist", inline=False)
    await interaction.response.send_message(embed=embed)

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

if __name__ == "__main__":
    if not TOKEN:
        logger.error("Không tìm thấy Token. Bạn kiểm tra lại file config nhé")
    else:
        bot.run(TOKEN)