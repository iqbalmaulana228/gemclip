from __future__ import annotations

import subprocess
from pathlib import Path


def plan_windows(
    start: float,
    end: float,
    chunk_sec: float = 900.0,
    overlap_sec: float = 60.0,
) -> list[tuple[float, float]]:
    """
    Bagi rentang [start, end] menjadi jendela-jendela berurutan.

    Jendela bersebelahan saling tumpang tindih sebesar overlap_sec agar momen yang
    kebetulan berada di batas potongan tidak terlewat.
    """
    if end <= start:
        return []
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap harus lebih kecil dari panjang potongan")

    windows: list[tuple[float, float]] = []
    a = start
    while True:
        b = min(a + chunk_sec, end)
        windows.append((round(a, 3), round(b, 3)))
        if b >= end:
            break
        a = b - overlap_sec
    return windows


def cut_chunk(source: Path, start: float, duration: float, dest: Path) -> Path:
    """
    Potong satu bagian video untuk dianalisis Gemini.

    Di-encode ulang dengan kualitas rendah agar cepat dan ukuran unggahan kecil.
    Re-encode dipilih agar waktu di dalam potongan tepat dimulai dari 0.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Gagal memotong bagian video untuk analisis: {proc.stderr[-300:]}")
    return dest


def extract_audio(source: Path, start: float, duration: float, dest: Path) -> Path:
    """
    Ambil audio satu rentang sebagai mp3 mono 16 kHz yang kecil, untuk transkripsi Gemini.

    Tanpa video, sehingga unggahan jauh lebih kecil daripada video potongan.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "48k",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Gagal mengambil audio untuk transkripsi: {proc.stderr[-300:]}")
    return dest
