from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel, TypeAdapter, ValidationError

log = logging.getLogger(__name__)

RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),  # Shorts, TikTok, Reels
    "1:1": (1080, 1080),  # Instagram, LinkedIn
    "16:9": (1920, 1080),  # YouTube, X
}
TRACK_EVERY_SEC = 2.0  # jarak antar frame yang dikirim ke Gemini untuk melacak wajah
TRACK_WIDTH = 480  # lebar frame yang dikirim: cukup untuk melihat wajah, dan hemat token

# Penghalusan crop, semuanya dinyatakan sebagai bagian dari lebar crop agar tidak bergantung resolusi.
DEADZONE_RATIO = 0.15  # kamera diam selama wajah masih dalam jarak ini dari pusat
MAX_STEP_RATIO = 0.05  # gerakan kamera maksimum per frame
CATCH_UP = 0.3  # bagian dari selisih yang dikejar kamera per frame


class FaceHint(BaseModel):
    frame: int
    x: float


TRACK_PROMPT = """Kamu melihat beberapa frame berurutan dari sebuah video pendek. Setiap frame diberi label nomor dan waktunya.

Untuk setiap frame, tentukan posisi horizontal pusat wajah pembicara utama, yaitu orang yang paling dominan di layar.
- x adalah pecahan dari lebar frame: 0 untuk tepi kiri, 0.5 untuk tengah, 1 untuk tepi kanan.
- Jika tidak ada wajah yang terlihat, isi x dengan -1.
- Kembalikan satu entri untuk setiap frame, berisi field frame (nomor frame) dan x.
"""


def output_size(ratio: str) -> tuple[int, int]:
    if ratio not in RATIOS:
        raise ValueError(f"Rasio tidak dikenal: {ratio}")
    return RATIOS[ratio]


def crop_width(src_w: int, src_h: int, ratio: str = "9:16") -> int:
    """Lebar potongan dengan rasio keluaran dari sumber, dengan tinggi penuh."""
    out_w, out_h = output_size(ratio)
    cw = min(src_w, int(round(src_h * out_w / out_h)))
    return cw - (cw % 2)


def read_video_info(video_path: Path) -> tuple[float, int, int, int]:
    """Kembalikan (fps, lebar, tinggi, jumlah_frame) dari sebuah video."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, src_w, src_h, n_frames


def extract_frames(segment: Path, out_dir: Path) -> list[Path]:
    """Ambil satu frame setiap TRACK_EVERY_SEC detik sebagai JPEG kecil."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(segment),
        "-vf", f"fps={1.0 / TRACK_EVERY_SEC:.4f},scale={TRACK_WIDTH}:-2",
        "-q:v", "5",
        str(out_dir / "frame_%04d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Gagal mengambil frame untuk pelacakan wajah: {proc.stderr[-300:]}")
    return sorted(out_dir.glob("frame_*.jpg"))


def _request_face_hints(api_key: str, model: str, frames: list[Path]) -> list[FaceHint]:
    contents: list = []
    for i, path in enumerate(frames):
        contents.append(f"Frame {i}, waktu {i * TRACK_EVERY_SEC:.1f} detik:")
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))
    contents.append(TRACK_PROMPT)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[FaceHint],
            temperature=0.1,
        ),
    )
    try:
        return TypeAdapter(list[FaceHint]).validate_json(response.text or "[]")
    except ValidationError as exc:
        raise RuntimeError("Jawaban pelacakan wajah dari Gemini tidak valid.") from exc


def track_hits(
    api_key: str,
    model: str,
    segment: Path,
    work_dir: Path,
    stem: str,
    src_w: int,
    fps: float,
) -> np.ndarray:
    """
    Minta Gemini menentukan posisi wajah pembicara pada beberapa frame di segmen.

    Mengembalikan array [[indeks_frame, pusat_x_dalam_piksel], ...] yang sudah diurutkan.
    Jika gagal, mengembalikan array kosong sehingga crop jatuh ke tengah frame.
    """
    frames_dir = work_dir / f"{stem}_frames"
    try:
        frames = extract_frames(segment, frames_dir)
        hints = _request_face_hints(api_key, model, frames) if frames else []
    except Exception as exc:  # pelacakan hanya penyempurna, jadi kegagalannya tidak boleh menggagalkan klip
        log.warning("Pelacakan wajah dengan Gemini gagal, memakai crop tengah: %s", exc)
        return np.zeros((0, 2))
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return _hints_to_hits(hints, len(frames), src_w, fps)


def _hints_to_hits(hints: list[FaceHint], n_frames: int, src_w: int, fps: float) -> np.ndarray:
    by_frame: dict[int, float] = {}
    for h in hints:
        if 0 <= h.frame < n_frames and 0.0 <= h.x <= 1.0:
            by_frame[h.frame] = h.x * src_w
    if not by_frame:
        return np.zeros((0, 2))
    rows = sorted((round(f * TRACK_EVERY_SEC * fps), x) for f, x in by_frame.items())
    return np.array(rows, dtype=float)


def crop_centers(
    hits: np.ndarray,
    n_frames: int,
    default_x: float,
    *,
    deadzone_px: float = 0.0,
    max_step_px: float = float("inf"),
) -> np.ndarray:
    """
    Ubah deteksi wajah yang jarang menjadi posisi pusat crop untuk setiap frame.

    Target (posisi wajah) diisi dengan interpolasi antar-deteksi. Kamera virtual mengikuti
    target dengan aturan berikut:
    - Selama target masih di dalam deadzone_px dari pusat kamera, kamera diam. Ini
      meredam getaran dari deteksi yang sedikit berubah.
    - Di luar zona itu, kamera mengejar sebagian selisihnya tiap frame (CATCH_UP),
      dengan langkah maksimum max_step_px per frame. Gerakannya jadi halus, tidak meloncat.
    """
    if n_frames <= 0:
        return np.zeros(0)
    if len(hits) == 0:
        return np.full(n_frames, default_x)

    target = np.interp(np.arange(n_frames), hits[:, 0], hits[:, 1])
    cam = np.empty(n_frames)
    pos = float(target[0])
    cam[0] = pos
    for i in range(1, n_frames):
        diff = float(target[i]) - pos
        if abs(diff) > deadzone_px:
            excess = diff - float(np.sign(diff)) * deadzone_px
            pos += float(np.clip(CATCH_UP * excess, -max_step_px, max_step_px))
        cam[i] = pos
    return cam
