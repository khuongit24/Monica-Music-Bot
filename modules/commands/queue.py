import discord
from typing import Any

from modules.ui_components import QueuePaginatorView

async def show_queue_ephemeral(players: dict[int, Any], interaction: discord.Interaction) -> None:
    player = players.get(interaction.guild.id)
    if not player or player.queue.empty():
        await interaction.response.send_message("Hàng đợi đang trống, bạn thêm nhạc vào nhé ✨", ephemeral=True)
        return
    upcoming = player.queue.snapshot(limit=50)
    view = QueuePaginatorView(upcoming, title="Queue (next up)")
    await interaction.response.send_message(embed=view.build_embed(), view=view, ephemeral=True)

async def show_queue_public(players: dict[int, Any], ctx) -> None:
    player = players.get(ctx.guild.id)
    if not player or player.queue.empty():
        await ctx.send("Hàng đợi đang trống, bạn thêm nhạc vào nhé ✨")
        return
    upcoming = player.queue.snapshot(limit=50)
    view = QueuePaginatorView(upcoming, title="Queue (next up)")
    msg = await ctx.send(embed=view.build_embed(), view=view)
    view.message = msg
