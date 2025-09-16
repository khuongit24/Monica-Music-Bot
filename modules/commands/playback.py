import discord
from typing import Any


# Các hàm dưới đây không đăng ký lệnh trực tiếp; bot.py sẽ gọi tới.

async def handle_play(runtime: Any, ctx_or_interaction: Any, query: str) -> None:
    """Chia sẻ logic xử lý phát nhạc (play) cho text/slash.
    runtime: đối tượng giữ bot, get_player_for_ctx, STREAM_PROFILE.
    """
    await runtime.handle_play_request(ctx_or_interaction, query)

async def handle_pause(bot: discord.Client, ctx_or_interaction: Any) -> None:
    vc = discord.utils.get(bot.voice_clients, guild=ctx_or_interaction.guild)
    if not vc or not vc.is_playing():
        # gửi theo kiểu context
        if hasattr(ctx_or_interaction, "response"):
            await ctx_or_interaction.response.send_message("Không có bài nhạc nào đang phát", ephemeral=True)
        else:
            await ctx_or_interaction.send("Không có bài nhạc nào đang phát")
        return
    vc.pause()
    if hasattr(ctx_or_interaction, "response"):
        await ctx_or_interaction.response.send_message("⏸️ Đã tạm dừng.", ephemeral=True)
    else:
        await ctx_or_interaction.send("⏸️ Đã tạm dừng")

async def handle_resume(bot: discord.Client, ctx_or_interaction: Any) -> None:
    vc = discord.utils.get(bot.voice_clients, guild=ctx_or_interaction.guild)
    if not vc or not vc.is_paused():
        if hasattr(ctx_or_interaction, "response"):
            await ctx_or_interaction.response.send_message("Không có bài nhạc nào bị tạm dừng", ephemeral=True)
        else:
            await ctx_or_interaction.send("Không có bài nhạc nào bị tạm dừng")
        return
    vc.resume()
    if hasattr(ctx_or_interaction, "response"):
        await ctx_or_interaction.response.send_message("▶️ Tiếp tục phát", ephemeral=True)
    else:
        await ctx_or_interaction.send("▶️ Đã tiếp tục phát")

async def handle_skip(bot: discord.Client, players: dict[int, Any], ctx_or_interaction: Any) -> None:
    vc = discord.utils.get(bot.voice_clients, guild=ctx_or_interaction.guild)
    if not vc or not vc.is_playing():
        if hasattr(ctx_or_interaction, "response"):
            await ctx_or_interaction.response.send_message("Không có nhạc đang phát để bỏ qua", ephemeral=True)
        else:
            await ctx_or_interaction.send("Không có bài nhạc nào đang phát để bỏ qua")
        return
    player = players.get(ctx_or_interaction.guild.id)
    if not player:
        vc.stop()
        if hasattr(ctx_or_interaction, "response"):
            await ctx_or_interaction.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)
        else:
            await ctx_or_interaction.send("⏭️ Đã skip bài hiện tại")
        return
    if player.queue.empty():
        if hasattr(ctx_or_interaction, "response"):
            await ctx_or_interaction.response.send_message(
                "Không có bài nhạc nào kế tiếp để mình chuyển qua, bạn thêm bài hát mới vào nhé 😋",
                ephemeral=True,
            )
        else:
            await ctx_or_interaction.send(
                "Không có bài nhạc nào kế tiếp để mình chuyển qua, bạn thêm bài hát mới vào nhé 😋"
            )
        return
    if getattr(player, "loop_one", False):
        player._suppress_loop_requeue_once = True
    vc.stop()
    if hasattr(ctx_or_interaction, "response"):
        await ctx_or_interaction.response.send_message("⏭️ Đã skip bài hiện tại", ephemeral=True)
    else:
        await ctx_or_interaction.send("⏭️ Đã skip bài hiện tại")

async def handle_stop(bot: discord.Client, players: dict[int, Any], ctx_or_interaction: Any) -> None:
    vc = discord.utils.get(bot.voice_clients, guild=ctx_or_interaction.guild)
    if vc:
        try:
            vc.stop()
        except Exception:
            pass
    player = players.get(ctx_or_interaction.guild.id)
    if player:
        try:
            await player.disable_loop()
        except Exception:
            pass
        try:
            await player.clear_all()
        except Exception:
            pass
        try:
            if player.vc and getattr(player.vc, "is_playing", lambda: False)():
                player.vc.stop()
        except Exception:
            pass
        try:
            player.current = None
            if player.now_update_task and not player.now_update_task.done():
                player.now_update_task.cancel()
            player.now_message = None
        except Exception:
            pass
    if hasattr(ctx_or_interaction, "response"):
        await ctx_or_interaction.response.send_message("⏹️ Đã dừng phát và xóa hàng đợi", ephemeral=True)
    else:
        await ctx_or_interaction.send("⏹️ Đã dừng phát và xóa hàng đợi")
