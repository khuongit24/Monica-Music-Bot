"""
Audio processing and FFmpeg configuration module.
Handles stream profiles, audio source creation, and URL sanitization.
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any
from modules.metrics import hist_observe_time, metric_inc
from modules.config import load_config
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from functools import lru_cache
import discord

logger = logging.getLogger("Monica.AudioProcessor")

# Expose heuristic tuning options (can be adjusted at runtime by bot if desired)
PICKER_CONFIG = {
    "hls_penalty": 1200,
    "opus_bonus": 1200,
    "aac_bonus": 800,
    "container_bonus": {"webm": 600, "opus": 600, "ogg": 600, "m4a": 550, "mp3": 300},
}


@lru_cache(maxsize=256)
def sanitize_stream_url(url: Optional[str]) -> Optional[str]:
    """Strip problematic query params (like range=) that can cause mid-track starts.

    Many YouTube formats sometimes include a pre-filled byte range which makes FFmpeg
    start decoding somewhere in the middle of the file. Removing these params forces
    FFmpeg to fetch from the beginning when possible.
    """
    if not url:
        return url
    try:
        pr = urlparse(url)
        q = parse_qsl(pr.query, keep_blank_values=True)
        # Remove range- and offset-related hints; keep stable params
        bad_keys = {"range", "rn", "rbuf", "start", "st", "begin", "sq", "dur", "t", "offset"}
        filtered = [(k, v) for (k, v) in q if k.lower() not in bad_keys]
        new_q = urlencode(filtered)
        return urlunparse((pr.scheme, pr.netloc, pr.path, pr.params, new_q, pr.fragment))
    except Exception:
        return url


def pick_best_audio_url(info: dict) -> Optional[str]:
    """Select the best quality audio URL with a more aggressive quality heuristic.

    Updated strategy (quality-first, low rebuffer risk):
    1. Filter to audio-capable formats (ignore those with acodec 'none').
    2. Score primarily by audio bitrate (abr).
    3. Strong bonus for Opus codec (already optimized for Discord) then AAC.
    4. Container preference: webm/opus > m4a > webm > others.
    5. Penalize HLS/m3u8 (higher latency / instability) unless only option.
    6. Penalize large positive start_time (partial segment).
    7. Fallback to single direct 'url' if formats list unusable.
    """
    direct = info.get("url")
    formats = info.get("formats") or []
    if not formats and direct:
        return sanitize_stream_url(direct)
    if not formats:
        return None

    candidates = []
    for f in formats:
        try:
            acodec = f.get("acodec")
            if acodec and acodec != "none":
                candidates.append(f)
        except Exception:
            continue
    if not candidates:
        candidates = formats

    def score(f):
        score = 0
        # Primary: audio bitrate (abr)
        abr = f.get("abr")
        try:
            if abr:
                score += int(abr) * 12  # slightly higher weight than before
        except Exception:
            pass
        # Codec preference
        acodec = (f.get("acodec") or "").lower()
        if "opus" in acodec:
            score += PICKER_CONFIG.get("opus_bonus", 1200)
        elif "aac" in acodec or "mp4a" in acodec:
            score += PICKER_CONFIG.get("aac_bonus", 800)
        elif acodec and acodec != "none":
            score += 200
        # Container preference
        ext = (f.get("ext") or "").lower()
        score += PICKER_CONFIG.get("container_bonus", {}).get(ext, 0)
        # Penalize HLS/m3u8 unless only choice
        proto = (f.get("protocol") or "").lower()
        if ("m3u8" in proto) or ("hls" in proto):
            score -= PICKER_CONFIG.get("hls_penalty", 900)
        # Prefer audio-only
        if f.get("vcodec") in (None, "none"):
            score += 150
        # Penalize offset starts
        try:
            st = f.get("start_time")
            if st and float(st) > 0.25:
                score -= 1200
        except Exception:
            pass
        return score

    try:
        best = max(candidates, key=score)
        try:
            # Store picked format metadata for downstream optimizations (non-breaking extra keys)
            info.setdefault("_picked_fmt_ext", (best.get("ext") or "").lower())
            info.setdefault("_picked_fmt_acodec", (best.get("acodec") or "").lower())
        except Exception:
            pass
        return sanitize_stream_url(best.get("url"))
    except Exception:
        return sanitize_stream_url(direct)


def get_ffmpeg_options_for_profile(stream_profile: str, volume: float, ffmpeg_bitrate: str, ffmpeg_threads: int, http_ua: str) -> Tuple[str, str]:
    """Generate FFmpeg options based on stream profile and settings."""
    vol = max(0.0, min(float(volume), 4.0))
    
    # Base reconnect options shared by profiles; include safe start-at-0 and headers
    ffmpeg_before_base = (
        "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -reconnect_at_eof 1 "
        "-rw_timeout 15000000 -nostdin -http_persistent 1 -seekable 1 -thread_queue_size 1024 "
        "-ss 0 "
        f"-headers \"User-Agent: {http_ua}\\r\\n\""
    )
    
    if stream_profile == "super-low-latency":
        # Extreme low-latency: tiniest probe/analyze, no input buffering. Requires strong CPU/network.
        # Notes: may stutter on unstable links. Best when bot is close to Discord region.
        before = ffmpeg_before_base
        opts = (
            f'-vn -af "volume={vol}" -b:a {ffmpeg_bitrate} -ar 48000 -threads {ffmpeg_threads} '
            f'-nostats -loglevel error -probesize 16k -analyzeduration 0 -bufsize 256k -rtbufsize 256k '
            f'-fflags nobuffer -flags low_delay -max_delay 0 -reorder_queue_size 0 -flush_packets 1'
        )
    elif stream_profile == "low-latency":
        # Lower analyzeduration/probesize to start faster, keep reasonable buffers to reduce stutter
        before = ffmpeg_before_base
        opts = (
            f'-vn -af "volume={vol}" -b:a {ffmpeg_bitrate} -ar 48000 -threads {ffmpeg_threads} '
            f'-nostats -loglevel error -probesize 64k -analyzeduration 100000 -bufsize 512k -rtbufsize 512k'
        )
    else:  # stable
        # Stable: be conservative; normalize timestamps and throttle input with -re (on before_options) to avoid fast playback
        before = ffmpeg_before_base + " -re"
        opts = (
            f'-vn -af "volume={vol}" -b:a {ffmpeg_bitrate} -ar 48000 -threads {ffmpeg_threads} '
            f'-fflags +genpts -avoid_negative_ts make_zero -muxpreload 0 -muxdelay 0 '
            f'-nostats -loglevel error -probesize 512k -analyzeduration 1500000 -bufsize 1M -rtbufsize 1M'
        )
    return before, opts


def create_audio_source(stream_url: str, volume: float, stream_profile: str, ffmpeg_bitrate: str, ffmpeg_threads: int, http_ua: str):
    """Create Discord audio source optimizing for maximum audio quality.

    Quality improvements:
    - If volume is 1.0 and source appears to be Opus (webm/opus/ogg), attempt stream copy (-c:a copy) to avoid re-encode.
    - Falls back to previous transcode path for other codecs or when volume != 1.0 (need volume filter).
    """
    # Delegate to new builder which can take optional metadata from ytdl
    from modules.config import load_config as _load_config
    cfg = _load_config()
    safe_threads = int(cfg.get('ffmpeg_threads', ffmpeg_threads or 1) or 1)
    start_ts = time.time()
    try:
        before, options, used_copy = build_ffmpeg_args(stream_url, volume, stream_profile, ffmpeg_bitrate, safe_threads, http_ua, metadata=None)
        hist_observe_time('audio_prepare_time', time.time() - start_ts)
        if used_copy:
            metric_inc('audio_prepare_copy', 1)
        else:
            metric_inc('audio_prepare_transcode', 1)
    except Exception:
        # Fallback to previous behavior if builder fails
        before, options = get_ffmpeg_options_for_profile(stream_profile, volume, ffmpeg_bitrate, safe_threads, http_ua)
        used_copy = False
    kwargs = {"before_options": before, "options": options}
    try:
        logger.info("FFmpeg profile=%s copy=%s options=%s", stream_profile, used_copy, options)
        return discord.FFmpegOpusAudio(stream_url, **kwargs)
    except Exception as e:
        logger.warning("FFmpegOpusAudio failed (%s); fallback to PCM", e)
        return discord.FFmpegPCMAudio(stream_url, **kwargs)


def build_ffmpeg_args(stream_url: str, volume: float, stream_profile: str, ffmpeg_bitrate: str, ffmpeg_threads: int, http_ua: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str, bool]:
    """Build before_options and options for ffmpeg. Returns (before, options, used_copy).

    If `metadata` (the yt-dlp info dict) is provided, prefer using its format acodec/ext fields to
    decide whether stream copy (-c:a copy) is safe. Otherwise fall back to heuristics based on URL.
    """
    stream_url = sanitize_stream_url(stream_url) or stream_url
    lower_url = (stream_url or "").lower()
    used_copy = False
    # simple metadata-aware decision
    try:
        if metadata:
            fmt_acodec = (metadata.get('_picked_fmt_acodec') or metadata.get('acodec') or '').lower()
            fmt_ext = (metadata.get('_picked_fmt_ext') or metadata.get('ext') or '').lower()
            # If codec is opus or container indicates opus/m4a and volume ==1, copy is safe
            if volume == 1.0 and ('opus' in fmt_acodec or fmt_ext in ('webm','opus','ogg','m4a')):
                used_copy = True
        else:
            # URL heuristic
            if volume == 1.0 and any(tok in lower_url for tok in ('.webm', '.opus', '.ogg', 'mime=audio%2Fwebm', 'mime=audio/webm', '.m4a')):
                used_copy = True
    except Exception:
        used_copy = False

    before, options = get_ffmpeg_options_for_profile(stream_profile, volume, ffmpeg_bitrate, ffmpeg_threads, http_ua)
    if used_copy:
        try:
            if stream_profile == 'stable':
                copy_opts = '-vn -c:a copy -ar 48000 -ac 2 -fflags +genpts -avoid_negative_ts make_zero -nostats -loglevel error'
            else:
                copy_opts = '-vn -c:a copy -ar 48000 -ac 2 -nostats -loglevel error'
            options = copy_opts
        except Exception:
            used_copy = False

    # adjust buffers for high bitrate when not copying
    try:
        br_num = int(str(ffmpeg_bitrate).lower().replace('k','').strip() or 0)
    except Exception:
        br_num = 0
    if br_num >= 192 and not used_copy:
        if '-bufsize' not in options:
            options += ' -bufsize 2M'
        if '-rtbufsize' not in options:
            options += ' -rtbufsize 2M'
        if '-thread_queue_size' in before:
            before = before.replace('-thread_queue_size 1024', '-thread_queue_size 2048')

    return before, options, used_copy


def validate_domain(url: str, allowed_domains: set) -> bool:
    """Validate if URL domain is in allowed list."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.endswith("bandcamp.com"):
            return True
        return netloc in allowed_domains
    except Exception:
        return False
