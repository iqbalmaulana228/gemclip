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


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
        work_dir=Path(os.getenv("WORK_DIR", "output")).resolve(),
        clip_min_sec=float(os.getenv("CLIP_MIN_SEC", "30")),
        clip_max_sec=float(os.getenv("CLIP_MAX_SEC", "60")),
        default_clip_count=int(os.getenv("DEFAULT_CLIP_COUNT", "10")),
        max_height=int(os.getenv("MAX_VIDEO_HEIGHT", "720")),
    )
