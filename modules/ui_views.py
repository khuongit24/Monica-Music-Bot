"""Shim UI views

Tệp này chỉ đóng vai trò chuyển tiếp để giữ tương thích ngược, tránh trùng lặp.
Tất cả UI components chuẩn nằm ở modules.ui_components.
"""

from modules.ui_components import (
    MusicControls,
    ReportModal,
    HelpView,
    QueuePaginatorView,
    create_help_embed,
    create_now_playing_embed,
)

__all__ = [
    "MusicControls",
    "ReportModal",
    "HelpView",
    "QueuePaginatorView",
    "create_help_embed",
    "create_now_playing_embed",
]
