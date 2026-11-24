from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .models import Segment, VideoInfo, Word

CACHE_DIRNAME = "cache"
INFO_FILE = "info.json"

# ID video YouTube selalu 11 karakter. Bentuk URL yang umum: watch?v=, youtu.be/, shorts/, embed/, live/.
_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str | None:
    """Ambil ID video dari link YouTube. Mengembalikan None jika tidak dikenali."""
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def video_cache_dir(cache_root: Path, video_id: str) -> Path:
    return cache_root / video_id


def _write_atomic(path: Path, text: str) -> None:
    """Tulis ke berkas sementara dulu, supaya berkas cache tidak pernah setengah jadi."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_download(cache_dir: Path) -> VideoInfo | None:
    """Kembalikan video yang sudah diunduh sebelumnya, atau None jika belum ada atau tidak lengkap."""
    info_path = cache_dir / INFO_FILE
    if not info_path.is_file():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        source = cache_dir / str(data["file"])
        if not source.is_file():
            return None
        return VideoInfo(path=source, title=str(data["title"]), duration=float(data["duration"]))
    except (ValueError, KeyError, TypeError, OSError):
        return None


def save_download(cache_dir: Path, info: VideoInfo) -> None:
    """Catat bahwa unduhan sudah lengkap. Dipanggil hanya setelah unduhan berhasil."""
    payload = {"file": info.path.name, "title": info.title, "duration": info.duration}
    _write_atomic(cache_dir / INFO_FILE, json.dumps(payload, ensure_ascii=False))


def transcript_path(cache_dir: Path, model: str, lo: float, hi: float) -> Path:
    """Berkas transkrip untuk satu model dan satu rentang. Ganti model atau rentang berarti transkrip ulang."""
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cache_dir / f"transcript_{safe_model}_{lo:.1f}_{hi:.1f}.json"


def load_transcript(path: Path) -> tuple[list[Segment], list[Word]] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [Segment(**s) for s in data["segments"]]
        words = [Word(**w) for w in data["words"]]
    except (ValueError, KeyError, TypeError, OSError):
        return None
    return segments, words


def save_transcript(path: Path, segments: list[Segment], words: list[Word]) -> None:
    payload = {
        "segments": [asdict(s) for s in segments],
        "words": [asdict(w) for w in words],
    }
    _write_atomic(path, json.dumps(payload, ensure_ascii=False))
