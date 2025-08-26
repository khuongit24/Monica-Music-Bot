"""
UI Components and Discord interaction handlers.
Centralizes Discord UI elements like buttons, modals, and embeds.
"""

import logging
import discord
from discord import ui
from typing import Optional
import time

from modules.utils import truncate, format_duration, THEME_COLOR, ERR_COLOR
from modules.metrics import metric_inc

logger = logging.getLogger("Monica.UIComponents")


class MusicControls(ui.View):
    """Enhanced music control buttons with better interaction handling."""
    
    def __init__(self, guild_id: int, *, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.guild_id = guild_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Validate user permissions and voice state for interactions."""
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message(
                "❌ Bạn cần vào voice channel để sử dụng.", ephemeral=True
            )
            return False
        
        # Import here to avoid circular imports
        from modules.voice_manager import get_voice_client_cached
        vc = get_voice_client_cached(interaction.client, interaction.guild)
        
        if not vc or not vc.is_connected():
            await interaction.response.send_message(
                "❌ Bot chưa kết nối voice channel.", ephemeral=True
            )
            return False
        
        if interaction.user.voice.channel.id != vc.channel.id:
            await interaction.response.send_message(
                "❌ Bạn phải ở cùng voice channel với bot.", ephemeral=True
            )
            return False
        
        # Refresh player's activity to avoid idle disconnect during interaction
        try:
            from bot import players  # Import when needed
            player = players.get(interaction.guild.id)
            if player:
                player._last_active = time.time()
        except Exception:
            pass
        
        return True

    @ui.button(emoji="⏯️", label="Tạm dừng/Tiếp tục", style=discord.ButtonStyle.primary, row=0)
    async def pause_resume(self, inter: discord.Interaction, button: ui.Button):
        """Toggle pause/resume playback."""
        from modules.voice_manager import get_voice_client_cached
        vc = get_voice_client_cached(inter.client, inter.guild)
        
        if vc.is_playing():
            vc.pause()
            await inter.response.send_message("⏸️ Đã tạm dừng", ephemeral=True)
        elif vc.is_paused():
            vc.resume()
            await inter.response.send_message("▶️ Đã tiếp tục", ephemeral=True)
        else:
            await inter.response.send_message("❌ Không có gì để pause/resume", ephemeral=True)

    @ui.button(emoji="⏭️", label="Bỏ qua", style=discord.ButtonStyle.secondary, row=0)
    async def skip(self, inter: discord.Interaction, button: ui.Button):
        """Skip current track."""
        from modules.voice_manager import get_voice_client_cached
        from bot import players  # Import when needed
        
        vc = get_voice_client_cached(inter.client, inter.guild)
        player = players.get(inter.guild.id)
        
        if not vc.is_playing():
            await inter.response.send_message("❌ Không có gì để skip", ephemeral=True)
            return
        
        if player and player.queue.empty():
            await inter.response.send_message("❌ Hàng đợi trống", ephemeral=True)
            return
        
        # Handle loop_one mode
        if player and player.loop_one:
            player._suppress_loop_requeue_once = True
        
        vc.stop()
        await inter.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)

    @ui.button(emoji="⏹️", label="Dừng phát", style=discord.ButtonStyle.danger, row=0)
    async def stop(self, inter: discord.Interaction, button: ui.Button):
        """Stop playback and clear queue."""
        from modules.voice_manager import get_voice_client_cached
        from bot import players  # Import when needed
        
        vc = get_voice_client_cached(inter.client, inter.guild)
        player = players.get(inter.guild.id)
        
        if vc:
            vc.stop()
        if player:
            await player.clear_all()
        
        await inter.response.send_message("⏹️ Đã dừng phát và xóa hàng đợi", ephemeral=True)

    @ui.button(emoji="📜", label="Hàng đợi", style=discord.ButtonStyle.secondary, row=1)
    async def show_queue(self, inter: discord.Interaction, button: ui.Button):
        """Display current queue."""
        from bot import players  # Import when needed
        
        player = players.get(inter.guild.id)
        if not player or player.queue.empty():
            await inter.response.send_message("📜 Hàng đợi trống", ephemeral=True)
            return
        
        upcoming = player.queue.snapshot()[:10]
        text = "\n".join(
            f"{idx+1}. {truncate(item.get('title') if isinstance(item, dict) else str(item), 45)} — {format_duration(item.get('duration') if isinstance(item, dict) else None)}"
            for idx, item in enumerate(upcoming)
        )
        
        embed = discord.Embed(
            title="📜 Hàng đợi (10 bài tiếp theo)", 
            description=text, 
            color=THEME_COLOR
        )
        await inter.response.send_message(embed=embed, ephemeral=True)

    @ui.button(emoji="↩️", label="Quay lại", style=discord.ButtonStyle.secondary, row=1)
    async def reverse(self, inter: discord.Interaction, button: ui.Button):
        """Play previous track."""
        from bot import players  # Import when needed
        
        player = players.get(inter.guild.id)
        if not player or not player.history:
            await inter.response.send_message("❌ Không có bài nào trong lịch sử", ephemeral=True)
            return
        
        try:
            last = await player.play_previous_now()
            if last:
                title = truncate(last.get('title', 'Unknown'), 80)
                await inter.response.send_message(f"↩️ Đang chuyển về: {title}", ephemeral=True)
            else:
                await inter.response.send_message("❌ Không thể quay lại bài trước", ephemeral=True)
        except Exception:
            await inter.response.send_message("❌ Lỗi khi quay lại bài trước", ephemeral=True)


class ReportModal(ui.Modal, title="Báo cáo lỗi gặp phải"):
    """Enhanced bug report modal with better validation."""
    
    ten_loi = ui.TextInput(
        label="Tên lỗi bạn gặp", 
        placeholder="VD: Bị giật, delay, không phát được…", 
        required=True, 
        max_length=120
    )
    chuc_nang = ui.TextInput(
        label="Chức năng liên quan đến lỗi", 
        placeholder="VD: play, skip, reverse, queue…", 
        required=True, 
        max_length=80
    )
    mo_ta = ui.TextInput(
        label="Mô tả chi tiết tình trạng gặp lỗi", 
        style=discord.TextStyle.paragraph, 
        required=True, 
        max_length=1500
    )

    def __init__(self, user: discord.abc.User, guild: Optional[discord.Guild]):
        super().__init__()
        self._user = user
        self._guild = guild

    async def on_submit(self, interaction: discord.Interaction):
        """Handle bug report submission with proper encoding."""
        try:
            report_content = (
                f"=== BUG REPORT [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n"
                f"User: {self._user} (ID: {self._user.id})\n"
                f"Guild: {self._guild.name if self._guild else 'DM'} (ID: {self._guild.id if self._guild else 'N/A'})\n"
                f"Error: {self.ten_loi.value}\n"
                f"Function: {self.chuc_nang.value}\n"
                f"Description: {self.mo_ta.value}\n"
                "=" * 50 + "\n\n"
            )
            
            # Append to report_bug.log with UTF-8 encoding
            with open("report_bug.log", "a", encoding="utf-8") as f:
                f.write(report_content)
            
            logger.info("Bug report submitted by user %s", self._user.id)
            
        except Exception as e:
            logger.error("Failed to write bug report: %s", e)
        
        await interaction.response.send_message(
            "Cảm ơn bạn đã đóng góp! Báo cáo đã được ghi lại ❤️", 
            ephemeral=True
        )


def create_now_playing_embed(data: dict, extra_desc: Optional[str] = None) -> discord.Embed:
    """Create standardized now-playing embed."""
    title = truncate(data.get("title", "Now Playing"), 80)
    embed = discord.Embed(
        title=title,
        url=data.get("webpage_url"),
        color=THEME_COLOR,
        timestamp=discord.utils.utcnow(),
        description=(f"{'🔴 LIVE' if data.get('is_live') else '🎧 Now Playing'}\n"
                     f"{extra_desc if extra_desc else ''}")
    )
    
    if data.get("thumbnail"):
        try:
            embed.set_thumbnail(url=data.get("thumbnail"))
        except Exception:
            pass
    
    embed.add_field(
        name="👤 Nghệ sĩ", 
        value=truncate(data.get("uploader") or "Unknown", 64), 
        inline=True
    )
    embed.add_field(
        name="⏱️ Thời lượng", 
        value=format_duration(data.get("duration")), 
        inline=True
    )
    
    if data.get("requested_by"):
        embed.add_field(
            name="🙋 Yêu cầu", 
            value=truncate(data.get("requested_by"), 30), 
            inline=True
        )
    
    try:
        embed.set_footer(text="Profile: {STREAM_PROFILE} • Sẽ mất thêm vài giây để mình xử lý yêu cầu. Bạn chịu khó đợi thêm chút nha 💕")
    except Exception:
        pass
    
    return embed


def create_queue_add_embed(track_data: dict) -> discord.Embed:
    """Create standardized queue-add embed."""
    desc_title = truncate(track_data.get("title", "Đã thêm vào hàng đợi"), 80)
    embed = discord.Embed(
        title="✅ Đã thêm vào hàng đợi",
        url=track_data.get("webpage_url"),
        description=desc_title,
        color=discord.Color.green(),
    )
    
    if track_data.get("thumbnail"):
        try:
            embed.set_thumbnail(url=track_data.get("thumbnail"))
        except Exception:
            pass
    
    if track_data.get("uploader"):
        embed.add_field(
            name="👤 Nghệ sĩ", 
            value=truncate(track_data.get("uploader"), 64), 
            inline=True
        )
    
    if track_data.get("duration"):
        embed.add_field(
            name="⏱️ Thời lượng", 
            value=format_duration(track_data.get("duration")), 
            inline=True
        )
    
    embed.set_footer(text="Nếu gặp bạn gặp phải lỗi gì thì dùng /report để được hỗ trợ sửa lỗi nhanh chóng nhé ✨")
    return embed


def create_error_embed(message: str, title: str = "❌ Lỗi") -> discord.Embed:
    """Create standardized error embed."""
    return discord.Embed(
        title=title,
        description=message,
        color=ERR_COLOR
    )
