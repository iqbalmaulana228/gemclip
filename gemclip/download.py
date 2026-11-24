from __future__ import annotations

from pathlib import Path

import yt_dlp

from .models import VideoInfo


def download_video(url: str, out_dir: Path, max_height: int) -> VideoInfo:
    """Unduh video YouTube (satu video saja) ke out_dir/source.mp4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<={max_height}][ext=mp4]/"
        f"b[height<={max_height}]"
    )
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    path = out_dir / "source.mp4"
    if not path.exists():
        matches = sorted(out_dir.glob("source.*"))
        if not matches:
            raise RuntimeError("Video tidak berhasil diunduh. Cek link YouTube-nya.")
        path = matches[0]

    return VideoInfo(
        path=path,
        title=str(info.get("title") or "video"),
        duration=float(info.get("duration") or 0.0),
    )
