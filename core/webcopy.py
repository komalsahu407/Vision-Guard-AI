"""Make a browser-playable H.264 copy of a video (browsers cannot play .avi, mpeg4, xvid, etc.).

Detection always runs on the ORIGINAL file; this copy is only used by the <video> player.
Needs ffmpeg: uses the one bundled with `pip install imageio-ffmpeg`, or a system ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def web_path(cache_dir: Path) -> Path:
    return cache_dir / "web.mp4"


def needs_copy(video: Path) -> bool:
    if video.suffix.lower() not in (".mp4", ".m4v", ".webm"):
        return True
    if video.suffix.lower() == ".webm":
        return False
    cap = cv2.VideoCapture(str(video))
    code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()
    fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).lower()
    return fourcc not in ("avc1", "h264", "x264")


def ensure(video: Path, cache_dir: Path) -> str:
    """Return 'original' (plays as is), 'ready' (web.mp4 exists) or 'failed:<reason>'."""
    out = web_path(cache_dir)
    if out.exists() and out.stat().st_size > 1000:
        return "ready"
    if not needs_copy(video):
        return "original"
    exe = ffmpeg_exe()
    if not exe:
        return "failed:ffmpeg missing (run: pip install imageio-ffmpeg)"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "web.tmp.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(video), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "28", "-pix_fmt", "yuv420p", "-vf", "scale='min(960,iw)':-2", "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        tmp.replace(out)
        return "ready"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"failed:{type(e).__name__}"
