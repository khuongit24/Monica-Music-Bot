"""Centralized message/i18n system.

Primary language: VI. Optional EN toggle via config.language ('vi'|'en').
Non-invasive: code referencing old constants still works (constants map to dynamic lookup for active language).
"""

_VI = {
	"NOWAIT_JOIN_REQUIRED": "Bạn cần vào kênh thoại để yêu cầu phát nhạc",
	"VOICE_CONNECT_FAIL": "Không thể kết nối vào kênh thoại",
	"QUEUE_FULL": "Hàng đợi đã đầy",
	"RESOLVE_ERROR_PREFIX": "Lỗi khi tìm kiếm:",
	"NO_STREAM_URL": "Không có stream URL cho bài này :<",
	"INVALID_SOURCE": "Nguồn này không được hỗ trợ. Hỗ trợ: YouTube, SoundCloud, Bandcamp, Mixcloud, Audius.",
	"IDLE_GOODBYE": "Không ai phát nhạc nên mình đi nhaa. Hẹn gặp lại ✨",
	"LOOP_ONE_SUPPRESS": "Loop-one: suppressed requeue after skip",
	"SOURCE_CREATION_ERROR": "Lỗi khi tạo nguồn phát",
	"PLAY_ERROR": "Lỗi khi phát",
}

_EN = {
	"NOWAIT_JOIN_REQUIRED": "You must join a voice channel to request playback",
	"VOICE_CONNECT_FAIL": "Failed to connect to voice channel",
	"QUEUE_FULL": "Queue is full",
	"RESOLVE_ERROR_PREFIX": "Resolve error:",
	"NO_STREAM_URL": "No stream URL for this track :<",
	"INVALID_SOURCE": "This source is not allowed. Supported: YouTube, SoundCloud, Bandcamp, Mixcloud, Audius.",
	"IDLE_GOODBYE": "No activity, I'm leaving the channel. See you ✨",
	"LOOP_ONE_SUPPRESS": "Loop-one: suppressed requeue after skip",
	"SOURCE_CREATION_ERROR": "Error creating audio source",
	"PLAY_ERROR": "Error during playback",
}

_ACTIVE = _VI

def set_language(lang: str):
	global _ACTIVE
	if lang and lang.lower().startswith("en"):
		_ACTIVE = _EN
	else:
		_ACTIVE = _VI

def msg(key: str) -> str:
	return _ACTIVE.get(key, _VI.get(key, key))

# Backward-compatible constant names (evaluated dynamically via msg())
class _ConstProxy:
	def __getattr__(self, name):
		return msg(name)

_proxy = _ConstProxy()
NOWAIT_JOIN_REQUIRED = _proxy.NOWAIT_JOIN_REQUIRED
VOICE_CONNECT_FAIL = _proxy.VOICE_CONNECT_FAIL
QUEUE_FULL = _proxy.QUEUE_FULL
RESOLVE_ERROR_PREFIX = _proxy.RESOLVE_ERROR_PREFIX
NO_STREAM_URL = _proxy.NO_STREAM_URL
INVALID_SOURCE = _proxy.INVALID_SOURCE
IDLE_GOODBYE = _proxy.IDLE_GOODBYE
LOOP_ONE_SUPPRESS = _proxy.LOOP_ONE_SUPPRESS
SOURCE_CREATION_ERROR = _proxy.SOURCE_CREATION_ERROR
PLAY_ERROR = _proxy.PLAY_ERROR
