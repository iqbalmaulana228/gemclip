from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Moment:
    start: float
    end: float
    score: int
    title: str
    hook: str
    caption: str
    hashtags: tuple[str, ...]


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    title: str
    duration: float
