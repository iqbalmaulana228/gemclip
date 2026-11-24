from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .pipeline import run_job

log = logging.getLogger("gemclip")

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID_RE = re.compile(r"[a-f0-9]{12}")
FILE_RE = re.compile(r"[A-Za-z0-9_\-]+\.(mp4|txt|srt)")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
MAX_CLIP_SEC = 180

settings = load_settings()
settings.work_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gemclip")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # satu job berjalan sekaligus agar CPU tidak kebanjiran


class JobRequest(BaseModel):
    url: str
    clips: int = Field(default=10, ge=1, le=20)
    clip_min_sec: float | None = Field(default=None, ge=5, le=MAX_CLIP_SEC)
    clip_max_sec: float | None = Field(default=None, ge=5, le=MAX_CLIP_SEC)
    start: str | None = None  # "mm:ss", "hh:mm:ss", atau detik. Kosong berarti dari awal video
    end: str | None = None  # kosong berarti sampai akhir video
    instruction: str = Field(default="", max_length=500)
    analysis: Literal["transcript", "video"] = "transcript"
    ratio: Literal["9:16", "1:1", "16:9"] = "9:16"
    caption_style: Literal["hormozi", "neon", "minimal"] = "hormozi"


def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in YOUTUBE_HOSTS


def parse_timecode(value: str | None) -> float | None:
    """Ubah "90", "1:30", atau "1:02:03" menjadi detik. Mengembalikan None jika kosong."""
    if value is None or not value.strip():
        return None
    parts = value.strip().split(":")
    if len(parts) > 3:
        raise ValueError(f"Format waktu tidak dikenali: {value}")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Format waktu tidak dikenali: {value}") from exc
    if any(n < 0 for n in nums):
        raise ValueError(f"Waktu tidak boleh negatif: {value}")
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def _update(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _worker(job_id: str, url: str, opts: dict) -> None:
    job_dir = settings.work_dir / job_id

    def progress(step: str, fraction: float) -> None:
        _update(job_id, step=step, progress=round(fraction, 3))

    try:
        with _run_lock:
            _update(job_id, step="Mulai", progress=0.0)
            result = run_job(url, progress=progress, job_dir=job_dir, settings=settings, **opts)
        _update(
            job_id,
            status="done",
            title=result["title"],
            clips=result["clips"],
            progress=1.0,
        )
    except Exception as exc:  # tampilkan pesan ke pengguna, detail tetap di log
        log.exception("Job %s gagal", job_id)
        _update(job_id, status="error", error=str(exc) or "Terjadi kesalahan yang tidak diketahui.")


@app.get("/api/config")
def get_config() -> dict:
    return {
        "model": settings.gemini_model,
        "default_clip_count": settings.default_clip_count,
        "max_clips": 20,
        "clip_min_sec": settings.clip_min_sec,
        "clip_max_sec": settings.clip_max_sec,
        "max_clip_sec": MAX_CLIP_SEC,
        "has_api_key": bool(settings.gemini_api_key),
    }


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if not is_youtube_url(req.url):
        raise HTTPException(400, "Masukkan link YouTube yang valid.")
    if not settings.gemini_api_key:
        raise HTTPException(400, "GEMINI_API_KEY belum diisi di file .env. Isi lalu jalankan ulang aplikasi.")

    clip_min = req.clip_min_sec if req.clip_min_sec is not None else settings.clip_min_sec
    clip_max = req.clip_max_sec if req.clip_max_sec is not None else settings.clip_max_sec
    if clip_min > clip_max:
        raise HTTPException(400, "Durasi minimum tidak boleh lebih besar dari durasi maksimum.")

    try:
        start_sec = parse_timecode(req.start)
        end_sec = parse_timecode(req.end)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise HTTPException(400, "Waktu selesai harus lebih besar dari waktu mulai.")

    opts = {
        "clip_count": req.clips,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "instruction": req.instruction.strip(),
        "analysis": req.analysis,
        "ratio": req.ratio,
        "caption_style": req.caption_style,
    }
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "step": "Menunggu giliran",
            "progress": 0.0,
            "title": None,
            "clips": [],
            "error": None,
            "ratio": req.ratio,
            "caption_style": req.caption_style,
        }
    threading.Thread(target=_worker, args=(job_id, req.url.strip(), opts), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(404, "Job tidak ditemukan.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job tidak ditemukan. Mungkin aplikasi sudah dimulai ulang.")
        return dict(job)


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> FileResponse:
    if not JOB_ID_RE.fullmatch(job_id) or not FILE_RE.fullmatch(filename):
        raise HTTPException(404)
    path = settings.work_dir / job_id / filename
    if not path.is_file():
        raise HTTPException(404)
    if filename.endswith(".mp4"):
        media = "video/mp4"
    else:
        media = "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media, filename=filename)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
