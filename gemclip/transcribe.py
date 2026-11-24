from __future__ import annotations

from pathlib import Path
from typing import Callable

from google.genai import types
from pydantic import BaseModel, TypeAdapter, ValidationError

from .chunks import extract_audio, plan_windows
from .gemini import delete_quietly, generate_with_retry, make_client, upload_and_wait
from .models import Segment, Word

CHUNK_SEC = 900.0  # audio dipecah per sekitar 15 menit agar unggahan dan jawaban tetap kecil
Progress = Callable[[str, float], None]


class SpeechPiece(BaseModel):
    start_sec: float
    end_sec: float
    text: str


TRANSCRIBE_PROMPT = """Transkripsikan semua ucapan dalam audio ini.

Aturan:
- Tulis ucapan persis seperti yang didengar, dalam bahasa aslinya. Jangan diterjemahkan.
- Pecah menjadi kalimat atau frasa pendek. Setiap potongan biasanya berdurasi 2 sampai 15 detik.
- start_sec dan end_sec adalah detik dari awal audio ini, yang dimulai dari 0. Gunakan angka desimal.
- Lewati musik, efek suara, dan bagian tanpa ucapan. Jangan menulis keterangan seperti [musik].
- Kembalikan daftar kosong jika tidak ada ucapan.
"""


def parse_pieces(raw: str) -> list[SpeechPiece]:
    try:
        return TypeAdapter(list[SpeechPiece]).validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"Format transkrip dari Gemini tidak valid: {raw[:300]}") from exc


def split_words(text: str, start: float, end: float) -> list[Word]:
    """
    Bagi waktu sebuah kalimat ke tiap katanya, sebanding dengan panjang kata.

    Gemini memberi waktu per kalimat, bukan per kata, jadi ini perkiraan. Caption karaoke
    yang dihasilkan cukup rapi, tetapi bisa sedikit meleset dibanding transkripsi per kata.
    """
    tokens = text.split()
    if not tokens:
        return []
    if end <= start:
        end = start + 0.3 * len(tokens)

    total = sum(len(t) for t in tokens)
    span = end - start
    words: list[Word] = []
    cursor = start
    for token in tokens:
        dur = span * len(token) / total
        words.append(Word(start=round(cursor, 3), end=round(cursor + dur, 3), text=token))
        cursor += dur
    return words


def transcribe(
    video_path: Path,
    api_key: str,
    model: str,
    work_dir: Path,
    duration: float,
    clip_range: tuple[float, float] | None = None,
    progress: Progress | None = None,
) -> tuple[list[Segment], list[Word]]:
    """
    Transkrip ucapan dengan Gemini. Audio diambil per potongan sekitar 15 menit, lalu tiap potongan dikirim.

    Jika clip_range diberikan, hanya rentang itu yang ditranskrip. Waktu yang dikembalikan
    tetap absolut terhadap video asli.
    """
    lo, hi = clip_range if clip_range is not None else (0.0, duration)
    windows = plan_windows(lo, hi, CHUNK_SEC, overlap_sec=0.0)
    client = make_client(api_key)

    segments: list[Segment] = []
    words: list[Word] = []
    for idx, (a, b) in enumerate(windows, 1):
        if progress:
            progress(f"Membuat transkrip bagian {idx} dari {len(windows)}", (idx - 1) / len(windows))
        span = b - a
        audio_path = work_dir / f"audio_part_{idx:02d}.mp3"
        extract_audio(video_path, a, span, audio_path)
        try:
            audio = upload_and_wait(client, audio_path, "audio/mpeg")
            try:
                response = generate_with_retry(
                    client,
                    model=model,
                    contents=[audio, TRANSCRIBE_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[SpeechPiece],
                        temperature=0.1,
                    ),
                )
            finally:
                delete_quietly(client, audio.name)
        finally:
            audio_path.unlink(missing_ok=True)

        for piece in parse_pieces(response.text or "[]"):
            text = piece.text.strip()
            if not text:
                continue
            # Jawaban Gemini bersifat relatif terhadap potongan, jadi dibatasi lalu digeser ke waktu video.
            s = min(max(piece.start_sec, 0.0), span)
            e = min(max(piece.end_sec, s), span)
            start, end = round(a + s, 3), round(a + e, 3)
            segments.append(Segment(start=start, end=end, text=text))
            words.extend(split_words(text, start, end))

    if progress:
        progress("Transkrip selesai", 1.0)
    segments.sort(key=lambda seg: seg.start)
    words.sort(key=lambda w: w.start)
    return segments, words
