from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .cache import (
    CACHE_DIRNAME,
    load_download,
    load_transcript,
    save_download,
    save_transcript,
    transcript_path,
    video_cache_dir,
    youtube_id,
)
from .captions import STYLES, build_srt, clip_text
from .config import Settings
from .download import download_video
from .models import VideoInfo
from .moments import find_moments
from .render import check_ffmpeg, render_clip
from .reframe import RATIOS
from .transcribe import transcribe

Progress = Callable[[str, float], None]


def _post_text(title: str, caption: str, hashtags: tuple[str, ...]) -> str:
    return f"{title}\n\n{caption}\n\n{' '.join(hashtags)}\n"


def resolve_range(start_sec: float | None, end_sec: float | None, duration: float) -> tuple[float, float]:
    """Hitung rentang analisis dan pastikan masih berada di dalam durasi video."""
    lo = max(0.0, start_sec or 0.0)
    hi = min(duration, end_sec) if end_sec is not None else duration
    if lo >= duration:
        raise ValueError("Waktu mulai melewati durasi video.")
    if hi <= lo:
        raise ValueError("Waktu selesai harus lebih besar dari waktu mulai.")
    return lo, hi


def _fetch_video(url: str, job_dir: Path, settings: Settings) -> tuple[VideoInfo, Path | None]:
    """
    Unduh video, atau pakai salinan yang sudah ada dari cache.

    Mengembalikan (info, folder_cache). folder_cache None jika link tidak dikenali sebagai video YouTube,
    dan dalam kasus itu unduhan disimpan di folder job seperti biasa.
    """
    video_id = youtube_id(url)
    if video_id is None:
        return download_video(url, job_dir, settings.max_height), None

    cache_dir = video_cache_dir(settings.work_dir / CACHE_DIRNAME, video_id)
    cached = load_download(cache_dir)
    if cached is not None:
        return cached, cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    info = download_video(url, cache_dir, settings.max_height)
    save_download(cache_dir, info)
    return info, cache_dir


def run_job(
    url: str,
    clip_count: int,
    settings: Settings,
    job_dir: Path,
    progress: Progress,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    instruction: str = "",
    analysis: str = "transcript",
    ratio: str = "9:16",
    caption_style: str = "hormozi",
) -> dict:
    """
    Jalankan seluruh pipeline: unduh, transkrip, pilih momen dengan Gemini, lalu render.

    Unduhan dan transkrip disimpan per ID video, jadi job berikutnya untuk video yang sama
    tidak mengulang langkah itu. Mengubah jumlah klip, durasi, atau instruksi tidak memicu ulang.
    """
    if ratio not in RATIOS:
        raise ValueError(f"Rasio tidak dikenal: {ratio}")
    if caption_style not in STYLES:
        raise ValueError(f"Gaya caption tidak dikenal: {caption_style}")

    check_ffmpeg()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY belum diisi di file .env.")
    job_dir.mkdir(parents=True, exist_ok=True)

    min_sec = clip_min if clip_min is not None else settings.clip_min_sec
    max_sec = clip_max if clip_max is not None else settings.clip_max_sec

    progress("Mengunduh video", 0.02)
    info, cache_dir = _fetch_video(url, job_dir, settings)
    duration = info.duration
    if not duration or duration <= 0:
        raise RuntimeError("Durasi video tidak terbaca. Coba link YouTube lain.")

    lo, hi = resolve_range(start_sec, end_sec, duration)
    full_video = lo <= 0.0 and hi >= duration - 0.5
    clip_range = None if full_video else (lo, hi)

    cache_path = transcript_path(cache_dir, settings.gemini_model, lo, hi) if cache_dir else None
    cached = load_transcript(cache_path) if cache_path else None
    if cached is not None:
        segments, words = cached
        progress("Memakai transkrip tersimpan", 0.3)
    else:
        progress("Membuat transkrip dengan Gemini", 0.12)
        segments, words = transcribe(
            video_path=info.path,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            work_dir=job_dir,
            duration=duration,
            clip_range=clip_range,
            progress=lambda msg, frac: progress(msg, 0.12 + 0.2 * frac),
        )
        # Transkrip kosong tidak disimpan, supaya kesalahan sesaat tidak terus terulang.
        if cache_path and words:
            save_transcript(cache_path, segments, words)
    if not words:
        raise RuntimeError("Tidak ada ucapan yang terdeteksi di rentang video ini.")

    progress("Memilih momen dengan Gemini", 0.35)
    moments = find_moments(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        video_path=info.path,
        segments=segments,
        words=words,
        duration=duration,
        count=clip_count,
        min_sec=min_sec,
        max_sec=max_sec,
        range_start=lo,
        range_end=hi,
        instruction=instruction,
        work_dir=job_dir,
        progress=lambda msg, frac: progress(msg, 0.35 + 0.15 * frac),
        analysis=analysis,
    )
    if not moments:
        raise RuntimeError(
            "Gemini tidak menemukan momen yang cocok. Coba ubah durasi klip, perluas rentang waktu, "
            "atau longgarkan instruksi."
        )

    clips: list[dict] = []
    total = len(moments)
    for i, m in enumerate(moments, 1):
        stem = f"clip_{i:02d}"
        progress(f"Merender klip {i} dari {total}", 0.5 + 0.5 * (i - 1) / total)
        render_clip(
            source=info.path,
            start=m.start,
            end=m.end,
            words=words,
            out_path=job_dir / f"{stem}.mp4",
            work_dir=job_dir,
            stem=stem,
            hook=m.hook,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            ratio=ratio,
            style=caption_style,
        )
        (job_dir / f"{stem}.txt").write_text(_post_text(m.title, m.caption, m.hashtags), encoding="utf-8")
        (job_dir / f"{stem}.srt").write_text(build_srt(words, m.start, m.end - m.start), encoding="utf-8")
        clips.append(
            {
                "index": i,
                "file": f"{stem}.mp4",
                "srt": f"{stem}.srt",
                "start": m.start,
                "end": m.end,
                "duration": round(m.end - m.start, 1),
                "score": m.score,
                "title": m.title,
                "hook": m.hook,
                "caption": m.caption,
                "hashtags": list(m.hashtags),
                "transcript": clip_text(words, m.start, m.end),
            }
        )

    result = {"title": info.title, "clips": clips}
    (job_dir / "clips.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("Selesai", 1.0)
    return result
