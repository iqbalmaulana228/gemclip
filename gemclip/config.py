from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    work_dir: Path
    clip_min_sec: float
    clip_max_sec: float
    default_clip_count: int
    max_height: int
    use_youtube_captions: bool = True  # caption YouTube dipakai lebih dulu sebelum transkripsi Gemini


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
        work_dir=Path(os.getenv("WORK_DIR", "output")).resolve(),
        clip_min_sec=float(os.getenv("CLIP_MIN_SEC", "30")),
        clip_max_sec=float(os.getenv("CLIP_MAX_SEC", "60")),
        default_clip_count=int(os.getenv("DEFAULT_CLIP_COUNT", "10")),
        max_height=int(os.getenv("MAX_VIDEO_HEIGHT", "720")),
        use_youtube_captions=_flag("USE_YOUTUBE_CAPTIONS", "1"),
    )
