"""
Audio processing and FFmpeg configuration module.
Handles stream profiles, audio source creation, and URL sanitization.
"""

import logging
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import discord

logger = logging.getLogger("Monica.AudioProcessor")


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
    """Choose the highest quality reasonable audio format.

    Strategy (quality-first):
    1. Filter to audio-only (no video) formats with a valid abr (audio bitrate).
    2. Prefer highest abr, then preferred container (m4a > webm > others).
    3. Penalize HLS/m3u8 unless it's the only option.
    4. Avoid formats with large non-zero start_time.
    5. Fallback to original 'url' if present and no formats scored.
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
        # Primary: audio bitrate
        abr = f.get("abr")
        try:
            if abr:
                score += int(abr) * 10  # weight bitrate strongly
        except Exception:
            pass
        # Container preference
        ext = (f.get("ext") or "").lower()
        if ext == "m4a":
            score += 500
        elif ext == "webm":
            score += 400
        # Penalize HLS/m3u8 unless only choice
        proto = (f.get("protocol") or "").lower()
        if ("m3u8" in proto) or ("hls" in proto):
            score -= 800
        # Prefer audio-only
        if f.get("vcodec") in (None, "none"):
            score += 100
        # Penalize offset starts
        try:
            st = f.get("start_time")
            if st and float(st) > 0.5:
                score -= 1000
        except Exception:
            pass
        return score

    try:
        best = max(candidates, key=score)
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
    """Create Discord audio source with profile-specific FFmpeg options."""
    stream_url = sanitize_stream_url(stream_url) or stream_url
    before, options = get_ffmpeg_options_for_profile(stream_profile, volume, ffmpeg_bitrate, ffmpeg_threads, http_ua)
    kwargs = {"before_options": before, "options": options}
    
    try:
        logger.info("FFmpeg profile=%s options=%s", stream_profile, options)
        return discord.FFmpegOpusAudio(stream_url, **kwargs)
    except Exception as e:
        logger.warning("FFmpegOpusAudio failed (%s); fallback to PCM", e)
        return discord.FFmpegPCMAudio(stream_url, **kwargs)


def validate_domain(url: str, allowed_domains: set) -> bool:
    """Validate if URL domain is in allowed list."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.endswith("bandcamp.com"):
            return True
        return netloc in allowed_domains
    except Exception:
        return False
