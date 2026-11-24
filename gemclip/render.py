from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .captions import build_ass
from .models import Word
from .reframe import (
    DEADZONE_RATIO,
    MAX_STEP_RATIO,
    crop_centers,
    crop_width,
    output_size,
    read_video_info,
    track_hits,
)


class RenderError(RuntimeError):
    pass


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RenderError("ffmpeg tidak ditemukan. Install ffmpeg dan pastikan ada di PATH.")


def _cut_segment(source: Path, start: float, duration: float, dest: Path) -> None:
    """Potong segmen dengan akurat (re-encode), agar waktu kata sama dengan waktu segmen."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"Gagal memotong segmen: {proc.stderr[-500:]}")


def render_clip(
    source: Path,
    start: float,
    end: float,
    words: list[Word],
    out_path: Path,
    work_dir: Path,
    stem: str,
    hook: str = "",
    *,
    api_key: str,
    model: str,
    ratio: str = "9:16",
    style: str = "hormozi",
) -> Path:
    """
    Render satu klip dengan rasio, caption, dan crop yang mengikuti wajah.

    Posisi wajah dilacak lewat Gemini. Jika pelacakan gagal, crop memakai bagian tengah frame.
    Frame dibaca dari segmen, dipotong sesuai posisi wajah, lalu dikirim ke ffmpeg lewat pipe.
    Audio diambil dari segmen yang sama dan caption dibakar dengan file ASS.
    """
    check_ffmpeg()
    out_w, out_h = output_size(ratio)
    duration = end - start

    segment = work_dir / f"{stem}_seg.mp4"
    _cut_segment(source, start, duration, segment)

    fps, src_w, src_h, n_frames = read_video_info(segment)
    if n_frames <= 0:
        raise RenderError("Segmen video kosong.")
    hits = track_hits(api_key, model, segment, work_dir, stem, src_w, fps)
    cw = crop_width(src_w, src_h, ratio)
    centers = crop_centers(
        hits,
        n_frames,
        default_x=src_w / 2,
        deadzone_px=DEADZONE_RATIO * cw,
        max_step_px=MAX_STEP_RATIO * cw,
    )

    ass_name = f"{stem}.ass"
    (work_dir / ass_name).write_text(
        build_ass(words, start, duration, hook, size=(out_w, out_h), style=style),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.3f}",
        "-i", "pipe:0",
        "-i", str(segment),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", f"subtitles={ass_name}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]

    log_path = work_dir / f"{stem}_ffmpeg.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=work_dir, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
        )
        try:
            cap = cv2.VideoCapture(str(segment))
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cx = float(centers[min(i, len(centers) - 1)])
                x0 = int(np.clip(round(cx - cw / 2), 0, src_w - cw))
                crop = np.ascontiguousarray(frame[:, x0:x0 + cw])
                interp = cv2.INTER_AREA if cw > out_w else cv2.INTER_CUBIC
                out = cv2.resize(crop, (out_w, out_h), interpolation=interp)
                proc.stdin.write(out.tobytes())
                i += 1
            cap.release()
        except BrokenPipeError:
            pass
        finally:
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait()

    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-500:]
        raise RenderError(f"ffmpeg gagal merender klip: {tail}")

    segment.unlink(missing_ok=True)
    return out_path
