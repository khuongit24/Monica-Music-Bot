"""
UI Components and Discord interaction handlers.
Centralizes Discord UI elements like buttons, modals, and embeds.
"""

import logging
import discord
from discord import ui
from typing import Optional
import time
import sys

from modules.utils import truncate, format_duration, THEME_COLOR, ERR_COLOR
from modules.metrics import metric_inc

logger = logging.getLogger("Monica.UIComponents")


def _get_runtime():
    """Lấy players và bot đang chạy từ module thực thi.

    Ưu tiên __main__ (khi chạy `python bot.py`), sau đó mới đến 'bot'.
    Tránh tình trạng import lại `bot` tạo module song song khiến state bị lệch.
    Trả về (players_dict, bot_obj). Nếu không có thì trả về ({}, None).
    """
    try:
        mod = sys.modules.get('__main__') or sys.modules.get('bot')
        players = getattr(mod, 'players', {}) if mod else {}
        bot_obj = getattr(mod, 'bot', None) if mod else None
        return players, bot_obj
    except Exception:
        return {}, None


class MusicControls(ui.View):
    """Enhanced music control buttons với xử lý trạng thái theo ngữ cảnh."""
    
    def __init__(self, guild_id: int, *, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.guild_id = guild_id
        # Áp dụng trạng thái nút theo bối cảnh ngay khi tạo
        try:
            self._apply_state()
            # Nếu view đã được gắn vào message sẵn (khi edit), re-apply sau 1 tick để chắc chắn
            async def _reapply_later():
                try:
                    await discord.utils.sleep_until(discord.utils.utcnow())  # schedule next loop tick
                    self._apply_state()
                    if getattr(self, 'message', None):
                        try:
                            await self.message.edit(view=self)
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                import asyncio as _aio
                _aio.create_task(_reapply_later())
            except Exception:
                pass
        except Exception:
            # Không làm gián đoạn nếu Discord internals thay đổi
            logger.debug("MusicControls: apply_state failed on init", exc_info=True)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Validate user permissions and voice state for interactions."""
        # Gia hạn thời gian sống của View khi có tương tác
        try:
            self.timeout = max((self.timeout or 0), 120)
            self.restart()
        except Exception:
            pass
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
            players, _ = _get_runtime()
            player = players.get(interaction.guild.id)
            if player:
                player._last_active = time.time()
        except Exception:
            pass
        
        return True

    async def on_timeout(self) -> None:
        """Khi View hết hạn, cố gắng giữ UI không chết nếu còn đang phát."""
        try:
            _players, _ = _get_runtime()
            player = _players.get(self.guild_id) if _players else None
            vc = getattr(player, 'vc', None)
            still_active = bool(vc and (getattr(vc, 'is_playing', lambda: False)() or getattr(vc, 'is_paused', lambda: False)()))
            if still_active and getattr(self, 'message', None):
                # Tạo View mới với timeout mới và thay thế
                new_view = MusicControls(self.guild_id, timeout=300)
                try:
                    await self.message.edit(view=new_view)
                    # Gắn message cho view mới để vòng đời nối tiếp
                    new_view.message = self.message
                except Exception:
                    pass
        except Exception:
            logger.debug("MusicControls.on_timeout: failed to recreate view", exc_info=True)

    def _apply_state(self) -> None:
        """Bật/tắt các nút dựa trên trạng thái hiện tại của player/voice.

        - Disable Skip/Stop nếu không đang phát.
        - Disable Reverse nếu không có lịch sử.
        - Disable Queue nếu hàng đợi trống.
        """
        try:
            # Lấy player/voice hiện tại (best-effort)
            try:
                _players, _bot = _get_runtime()
                from modules.voice_manager import get_voice_client_cached as _get_vc
                g = getattr(_bot, 'get_guild', lambda _gid: None)(self.guild_id) if _bot else None
                player = _players.get(self.guild_id) if _players else None
                vc = _get_vc(_bot, g) if _bot and g else None
            except Exception:
                player = None
                vc = None

            # Xác định trạng thái phát theo 2 nguồn: VoiceClient và player.current để tránh trạng thái trễ
            vc_play = bool(vc and (getattr(vc, 'is_playing', lambda: False)() or getattr(vc, 'is_paused', lambda: False)()))
            # consider "playing" if we have a current track or voice is playing/paused
            pl_play = bool(player and getattr(player, 'current', None))
            is_playing = bool(vc_play or pl_play)
            # Tránh false-negative: ưu tiên qsize() nếu có
            try:
                has_queue = bool(player and getattr(player, 'queue', None) and player.queue.qsize() > 0)
            except Exception:
                has_queue = bool(getattr(player, 'queue', None) and not player.queue.empty())
            # history viewable if deque has at least 1 item
            try:
                has_history = bool(player and getattr(player, 'history', None) and len(player.history) > 0)
            except Exception:
                has_history = bool(player and getattr(player, 'history', None))

            # Tìm các nút theo label hoặc emoji
            btn_skip = None; btn_stop = None; btn_rev = None; btn_queue = None
            for item in self.children:
                if isinstance(item, ui.Button):
                    if item.label == "Bỏ qua" or item.emoji == "⏭️":
                        btn_skip = item
                    elif item.label == "Dừng phát" or item.emoji == "⏹️":
                        btn_stop = item
                    elif item.label == "Quay lại" or item.emoji == "↩️":
                        btn_rev = item
                    elif item.label == "Hàng đợi" or item.emoji == "📜":
                        btn_queue = item

            # Skip: enable if something is playing and there is a next item
            if btn_skip:
                btn_skip.disabled = not (is_playing and has_queue)
            # Stop: enable if voice is playing or paused, or player claims current
            if btn_stop:
                btn_stop.disabled = not is_playing
            # Reverse: enable if we have any history
            if btn_rev:
                btn_rev.disabled = not has_history
            # Queue: enable if queue has items (even if not playing yet)
            if btn_queue:
                btn_queue.disabled = not has_queue
        except Exception:
            logger.debug("MusicControls: _apply_state error", exc_info=True)

    @ui.button(emoji="⏯️", label="Tạm dừng/Tiếp tục", style=discord.ButtonStyle.primary, row=0)
    async def pause_resume(self, inter: discord.Interaction, button: ui.Button):
        """Toggle pause/resume playback."""
        from modules.voice_manager import get_voice_client_cached
        vc = get_voice_client_cached(inter.client, inter.guild)
        try:
            metric_inc("ui_click_pause_resume")
        except Exception:
            pass
        
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
        players, _ = _get_runtime()
        try:
            metric_inc("ui_click_skip")
        except Exception:
            pass
        
        vc = get_voice_client_cached(inter.client, inter.guild)
        player = players.get(inter.guild.id)
        
        if not vc or not vc.is_connected() or (not vc.is_playing() and not vc.is_paused()):
            await inter.response.send_message("❌ Không có gì để skip", ephemeral=True)
            return
        
        if player and player.queue.empty():
            await inter.response.send_message("❌ Hàng đợi trống", ephemeral=True)
            return
        
        # Handle loop_one mode
        if player and getattr(player, 'loop_one', False):
            player._suppress_loop_requeue_once = True
        
        # Dừng để chuyển sang bài kế tiếp trong vòng lặp player
        vc.stop()
        await inter.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)

    @ui.button(emoji="⏹️", label="Dừng phát", style=discord.ButtonStyle.danger, row=0)
    async def stop(self, inter: discord.Interaction, button: ui.Button):
        """Stop playback and clear queue."""
        from modules.voice_manager import get_voice_client_cached
        players, _ = _get_runtime()
        try:
            metric_inc("ui_click_stop")
        except Exception:
            pass
        
        vc = get_voice_client_cached(inter.client, inter.guild)
        player = players.get(inter.guild.id)
        
        if vc:
            try:
                vc.stop()
            except Exception:
                pass
        if player:
            try:
                await player.clear_all()
            except Exception:
                pass
        
        await inter.response.send_message("⏹️ Đã dừng phát và xóa hàng đợi", ephemeral=True)

    @ui.button(emoji="📜", label="Hàng đợi", style=discord.ButtonStyle.secondary, row=1)
    async def show_queue(self, inter: discord.Interaction, button: ui.Button):
        """Display current queue."""
        players, _ = _get_runtime()
        try:
            metric_inc("ui_click_queue")
        except Exception:
            pass
        
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
        players, _ = _get_runtime()
        try:
            metric_inc("ui_click_reverse")
        except Exception:
            pass
        
        player = players.get(inter.guild.id)
        if not player or not getattr(player, 'history', None):
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


# =============================
# Help UI (nâng cấp giao diện trợ giúp)
# =============================

def _fmt_cmd_list(lines: list[str]) -> str:
    """Ghép danh sách lệnh thành chuỗi thoáng mát và dễ đọc."""
    return "\n\n".join(f"▸ {line}" for line in lines)


def create_help_embed(page: str, *, prefix: str, version: str, stream_profile: str | None = None) -> discord.Embed:
    """Tạo embed trợ giúp với nhiều trang (nhóm lệnh) đẹp mắt và thân thiện.

    Args:
        page: Tên trang hiện tại (overview, playback, queue, loop, info, config, report, sources)
        prefix: Prefix cho lệnh text (VD: !)
        version: Phiên bản bot để hiển thị footer
        stream_profile: Profile stream (stable/low-latency/super-low-latency)
    """
    page = (page or "overview").lower()

    footer = f"✨ Monica Music Bot {version}"
    if stream_profile:
        footer += f" • 🎚️ {stream_profile.title()}"

    if page == "playback":
        embed = discord.Embed(
            title="🎵 Phát nhạc",
            description="Điều khiển phát nhạc đơn giản và nhanh chóng",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🚀 Lệnh chính",
            value=_fmt_cmd_list([
                "`/play <tên bài>` — Phát nhạc",
                "`/pause` • `/resume` — Tạm dừng/tiếp tục", 
                "`/skip` • `/stop` — Bỏ qua/dừng"
            ]),
            inline=False,
        )
        embed.add_field(
            name="💡 Mẹo",
            value="Sau khi phát nhạc, dùng **nút bấm** dưới tin nhắn để điều khiển nhanh hơn",
            inline=False,
        )
    elif page == "queue":
        embed = discord.Embed(
            title="📜 Hàng đợi",
            description="Quản lý danh sách phát của bạn",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🎛️ Lệnh chính",
            value=_fmt_cmd_list([
                "`/queue` — Xem danh sách phát",
                "`/clear <tên bài>` — Xóa bài theo tên",
                "`/clear_all` — Xóa tất cả"
            ]),
            inline=False,
        )
    elif page == "loop":
        embed = discord.Embed(
            title="🔁 Loop & Lịch sử",
            description="Lặp lại những bài hát yêu thích",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🎵 Chế độ lặp",
            value=_fmt_cmd_list([
                "`/loop` — Lặp bài hiện tại",
                "`/loop_all` — Lặp toàn bộ danh sách",
                "`/unloop` — Tắt chế độ lặp"
            ]),
            inline=False,
        )
    elif page == "info":
        embed = discord.Embed(
            title="ℹ️ Thông tin & Giám sát",
            description="Theo dõi trạng thái hoạt động của Monica",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🔍 Lệnh thông tin",
            value=_fmt_cmd_list([
                "`/now` — Bài đang phát",
                "`/stats` — Thống kê hoạt động",
                "`/health` — Kiểm tra tình trạng bot"
            ]),
            inline=False,
        )
    elif page == "config":
        embed = discord.Embed(
            title="⚙️ Cấu hình & Debug",
            description="Tùy chỉnh Monica theo sở thích của bạn",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🎚️ Lệnh cấu hình",
            value=_fmt_cmd_list([
                "`/profile <mode>` — Thay đổi chất lượng",
                "`/volume <số>` — Điều chỉnh âm lượng",
                "`/debug_track <bài>` — Kiểm tra metadata"
            ]),
            inline=False,
        )
    elif page == "report":
        embed = discord.Embed(
            title="💌 Báo cáo & Góp ý",
            description="Giúp Monica ngày càng hoàn thiện hơn",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="📝 Cách báo cáo",
            value=_fmt_cmd_list([
                "`/report` — Mở form báo cáo",
                "Mô tả chi tiết vấn đề gặp phải",
                "Đội ngũ sẽ xem xét và phản hồi"
            ]),
            inline=False,
        )
    elif page == "sources":
        embed = discord.Embed(
            title="✨ Nguồn nhạc",
            description="Monica hỗ trợ nhiều nền tảng âm nhạc",
            color=THEME_COLOR,
        )
        embed.add_field(
            name="🎶 Nền tảng được hỗ trợ",
            value=_fmt_cmd_list([
                "YouTube — Video và nhạc",
                "SoundCloud — Âm nhạc độc lập", 
                "Bandcamp — Nghệ sĩ indie"
            ]),
            inline=False,
        )
        embed.add_field(
            name="👋 Mẹo tìm nhạc",
            value="Thử cả tên bài và tên nghệ sĩ để có kết quả tốt nhất",
            inline=False,
        )
    else:
        # overview
        embed = discord.Embed(
            title="🎵 Monica Bot — Trợ giúp",
            color=THEME_COLOR,
            description="Chào mừng bạn đến với Monica! Bắt đầu với `/play <tên bài>` để phát nhạc",
        )
        embed.add_field(
            name="🎯 Các nhóm lệnh",
            value=_fmt_cmd_list([
                "🎵 **Phát nhạc** — Điều khiển phát nhạc",
                "📜 **Hàng đợi** — Quản lý playlist",
                "🔁 **Loop & Lịch sử** — Lặp lại bài hát",
                "ℹ️ **Thông tin** — Theo dõi hoạt động"
            ]),
            inline=False,
        )

    # Footer đơn giản và gọn gàng
    embed.set_footer(text=f"Monica Music Bot {version} • Sử dụng dropdown để khám phá thêm")
    return embed


class HelpSelect(ui.Select):
    """Select chuyển trang trợ giúp với giao diện thân thiện."""
    def __init__(self, *, default_page: str = "overview"):
        options = [
            discord.SelectOption(label="🏠 Tổng quan", value="overview", emoji="🎵", description="Bắt đầu với Monica"),
            discord.SelectOption(label="🎶 Phát nhạc", value="playback", emoji="🎶", description="Phát và điều khiển nhạc"),
            discord.SelectOption(label="📜 Hàng đợi", value="queue", emoji="📜", description="Quản lý danh sách phát"),
            discord.SelectOption(label="🔁 Loop", value="loop", emoji="🔁", description="Lặp lại bài hát"),
            discord.SelectOption(label="ℹ️ Thông tin", value="info", emoji="ℹ️", description="Trạng thái bot"),
            discord.SelectOption(label="⚙️ Cấu hình", value="config", emoji="⚙️", description="Tùy chỉnh Monica"),
            discord.SelectOption(label="💌 Báo cáo", value="report", emoji="💌", description="Góp ý và báo lỗi"),
            discord.SelectOption(label="🌍 Nguồn nhạc", value="sources", emoji="🌍", description="Nền tảng hỗ trợ"),
        ]
        super().__init__(
            placeholder="Chọn nhóm lệnh để xem chi tiết",
            min_values=1,
            max_values=1,
            options=options,
            row=0,
        )
        self.current_page = default_page
        self._prefix = "!"
        self._version = ""
        self._profile = None

    def bind_meta(self, *, prefix: str, version: str, profile: str | None):
        self._prefix = prefix
        self._version = version
        self._profile = profile
        return self

    async def callback(self, interaction: discord.Interaction):  # type: ignore[override]
        try:
            metric_inc("ui_help_change_page")
        except Exception:
            pass
        self.current_page = self.values[0]
        embed = create_help_embed(
            self.current_page, prefix=self._prefix, version=self._version, stream_profile=self._profile
        )
        # Gia hạn view khi chuyển trang
        try:
            if self.view:
                self.view.timeout = max((self.view.timeout or 0), 120)
                self.view.restart()
        except Exception:
            pass
        await interaction.response.edit_message(embed=embed, view=self.view)


class HelpView(ui.View):
    """View chứa Select chuyển trang và nút mở Report."""
    def __init__(self, *, prefix: str, version: str, stream_profile: str | None = None, timeout: float = 300):
        super().__init__(timeout=timeout)
        sel = HelpSelect().bind_meta(prefix=prefix, version=version, profile=stream_profile)
        self.add_item(sel)

    @ui.button(emoji="💌", label="Gửi góp ý cho Monica", style=discord.ButtonStyle.secondary, row=1)
    async def open_report(self, interaction: discord.Interaction, button: ui.Button):
        """Mở form báo cáo thân thiện."""
        try:
            metric_inc("ui_click_open_report")
        except Exception:
            pass
        try:
            modal = ReportModal(interaction.user, interaction.guild)
            await interaction.response.send_modal(modal)
        except Exception:
            await interaction.response.send_message(
                "🚫 Oops! Không thể mở form báo cáo lúc này. Thử lại sau nhé! 😅", 
                ephemeral=True
            )


# =============================
# Queue paginator (Prev/Next)
# =============================

class QueuePaginatorView(ui.View):
    """View phân trang danh sách hàng đợi với nút Prev/Next.

    - Mỗi trang tối đa 10 bài
    - Không thay đổi logic queue, chỉ đọc snapshot tại thời điểm mở
    - Dùng chung cho slash (ephemeral) và text (công khai)
    """

    def __init__(self, items: list, *, title: str = "Queue (next up)", page_size: int = 10, timeout: float = 180.0):
        super().__init__(timeout=timeout)
        self._items = items or []
        self._page_size = max(1, page_size)
        self._title = title
        self._page = 0

    def _total_pages(self) -> int:
        import math
        return max(1, math.ceil(len(self._items) / self._page_size))

    def _page_slice(self):
        start = self._page * self._page_size
        end = start + self._page_size
        return self._items[start:end]

    def build_embed(self) -> discord.Embed:
        cur = self._page_slice()
        if not cur:
            desc = "Trống"
        else:
            lines = []
            base_index = self._page * self._page_size
            for i, item in enumerate(cur):
                idx = base_index + i + 1
                if isinstance(item, dict):
                    title = truncate(item.get('title') or str(item), 50)
                    dur = format_duration(item.get('duration'))
                else:
                    title = truncate(str(item), 50)
                    dur = format_duration(None)
                lines.append(f"{idx}. {title} — {dur}")
            desc = "\n".join(lines)
        embed = discord.Embed(title=self._title, description=desc, color=THEME_COLOR)
        embed.set_footer(text=f"Trang {self._page+1}/{self._total_pages()}")
        return embed

    async def refresh(self, interaction: discord.Interaction):
        # Cập nhật trạng thái nút và embed
        total = self._total_pages()
        prev_btn: ui.Button = self.children[0]  # type: ignore[assignment]
        next_btn: ui.Button = self.children[1]  # type: ignore[assignment]
        prev_btn.disabled = (self._page <= 0)
        next_btn.disabled = (self._page >= total - 1)
        # Gia hạn khi có tương tác
        try:
            self.timeout = max((self.timeout or 0), 120)
            self.restart()
        except Exception:
            pass
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    @ui.button(emoji="⬅️", label="Prev", style=discord.ButtonStyle.secondary)
    async def prev_page(self, interaction: discord.Interaction, button: ui.Button):
        try:
            metric_inc("ui_queue_prev")
        except Exception:
            pass
        if self._page > 0:
            self._page -= 1
        await self.refresh(interaction)

    @ui.button(emoji="➡️", label="Next", style=discord.ButtonStyle.secondary)
    async def next_page(self, interaction: discord.Interaction, button: ui.Button):
        try:
            metric_inc("ui_queue_next")
        except Exception:
            pass
        total = self._total_pages()
        if self._page < total - 1:
            self._page += 1
        await self.refresh(interaction)


def create_now_playing_embed(data: dict, extra_desc: Optional[str] = None, *, stream_profile: Optional[str] = None) -> discord.Embed:
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
        if not stream_profile:
            # Best-effort: lấy từ config nếu không truyền vào
            try:
                from modules.config import load_config as _load_cfg
                stream_profile = _load_cfg().get("stream_profile")
            except Exception:
                stream_profile = None
        footer_txt = (
            f"Profile: {stream_profile} • Sẽ mất thêm vài giây để mình xử lý yêu cầu. Bạn chịu khó đợi thêm chút nha 💕"
            if stream_profile else
            "Sẽ mất thêm vài giây để mình xử lý yêu cầu. Bạn chịu khó đợi thêm chút nha 💕"
        )
        embed.set_footer(text=footer_txt)
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
