import time
import discord
from discord import ui
from typing import Optional

# Các biến toàn cục sẽ được gán từ bot.py khi khởi tạo
bot = None
players = None
logger = None
BUG_REPORT_LOG_PATH = None
truncate = None
format_duration = None

class MusicControls(ui.View):
    # ... (phần thân class sẽ được copy từ bot.py, giữ nguyên logic, chỉ cập nhật import)
    pass

class ReportModal(ui.Modal, title="Báo cáo lỗi gặp phải"):
    # ... (phần thân class sẽ được copy từ bot.py, giữ nguyên logic, chỉ cập nhật import)
    pass
