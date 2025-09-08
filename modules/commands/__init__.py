"""Command groups for Monica Bot.

Các module con triển khai logic cho lệnh theo nhóm, để bot.py chỉ còn mỏng.
Mỗi hàm handler đều nhận runtime tham chiếu tới bot/players và helper cần thiết.
"""

__all__ = [
    "playback",
    "queue",
]
