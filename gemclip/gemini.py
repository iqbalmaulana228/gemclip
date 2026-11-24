from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# Batas waktu satu permintaan. Audio 15 menit atau potongan video bisa butuh beberapa menit, jadi dibuat longgar.
GEMINI_TIMEOUT_MS = 300_000
RETRY_DELAY_SEC = 5.0
TRANSIENT_MARKERS = ("429", "503", "504", "unavailable", "resource_exhausted", "deadline", "timed out", "timeout")
PERMANENT_MARKERS = ("400", "401", "403")


def make_client(api_key: str) -> genai.Client:
    """
    Klien Gemini dengan batas waktu HTTP. Retry dilakukan di generate_with_retry supaya jumlahnya terkendali,
    bukan oleh retry bawaan SDK.
    """
    http_options = types.HttpOptions(
        timeout=GEMINI_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def is_transient(exc: Exception) -> bool:
    """Error sementara (kuota, server sibuk, timeout) layak dicoba lagi. Error 4xx lainnya tidak."""
    text = str(exc).lower()
    if any(marker in text for marker in PERMANENT_MARKERS):
        return False
    return any(marker in text for marker in TRANSIENT_MARKERS)


def generate_with_retry(client, *, model: str, contents, config, retries: int = 1, delay: float = RETRY_DELAY_SEC):
    """Panggil generate_content. Error sementara dicoba ulang maksimal `retries` kali dengan jeda bertambah."""
    attempt = 0
    while True:
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            attempt += 1
            log.warning("Gemini sementara gagal (%s), mencoba lagi (%d/%d)", exc, attempt, retries)
            time.sleep(delay * attempt)


def upload_and_wait(client: genai.Client, path: Path, mime_type: str, timeout_sec: float = 900) -> types.File:
    """Unggah berkas ke Gemini, lalu tunggu sampai statusnya ACTIVE."""
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type=mime_type),
    )
    waited = 0.0
    while uploaded.state is not None and uploaded.state.name == "PROCESSING":
        if waited >= timeout_sec:
            raise RuntimeError("Gemini terlalu lama memproses berkas.")
        time.sleep(5)
        waited += 5
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state is None or uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini gagal memproses berkas (status: {uploaded.state}).")
    return uploaded


def delete_quietly(client: genai.Client, name: str) -> None:
    """Hapus berkas sementara di Gemini. Gagal menghapus tidak fatal karena berkas akan kedaluwarsa sendiri."""
    try:
        client.files.delete(name=name)
    except Exception:
        log.warning("Gagal menghapus berkas sementara di Gemini: %s", name)
