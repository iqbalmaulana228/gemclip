from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

log = logging.getLogger(__name__)


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
