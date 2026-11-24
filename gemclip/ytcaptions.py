from __future__ import annotations

import logging
import math
import re
import shutil
from pathlib import Path

import yt_dlp

from .models import Segment, Word
from .transcribe import split_words

log = logging.getLogger(__name__)

CAPTION_LANGS = ("id", "en")  # urutan prioritas bahasa
_TS = r"(?:\d+:)?\d{2}:\d{2}\.\d{3}"
_CUE_LINE = re.compile(rf"^\s*({_TS})\s+-->\s+({_TS})")
_INLINE = re.compile(rf"<({_TS})>")  # penanda waktu per kata pada caption otomatis YouTube
_TAG = re.compile(r"<[^>]+>")


def _to_sec(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_cues(text: str) -> list[tuple[float, float, str]]:
    """Ambil setiap cue (waktu mulai, waktu selesai, isi mentah) dari berkas VTT."""
    lines = text.replace("\r", "").split("\n")
    cues: list[tuple[float, float, str]] = []
    i = 0
    while i < len(lines):
        match = _CUE_LINE.match(lines[i])
        i += 1
        if not match:
            continue
        body: list[str] = []
        while i < len(lines) and not _CUE_LINE.match(lines[i]):
            if lines[i].strip():
                body.append(lines[i])
            i += 1
        cues.append((_to_sec(match.group(1)), _to_sec(match.group(2)), "\n".join(body)))
    return cues


def _tagged_words(start: float, end: float, body: str) -> list[Word]:
    """
    Kata dari cue otomatis. Waktu tiap kata dibaca dari penanda <hh:mm:ss.mmm>.

    Baris pertama cue adalah teks lama yang masih tampil, jadi hanya baris terakhir sebelum penanda pertama yang dipakai.
    """
    matches = list(_INLINE.finditer(body))
    lead = _TAG.sub("", body[: matches[0].start()]).split("\n")[-1]
    pieces: list[tuple[float, str]] = []
    if lead.strip():
        pieces.append((start, lead))
    for idx, match in enumerate(matches):
        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = _TAG.sub("", body[match.end(): seg_end])
        if text.strip():
            pieces.append((_to_sec(match.group(1)), text))

    words: list[Word] = []
    for idx, (t, text) in enumerate(pieces):
        t_end = pieces[idx + 1][0] if idx + 1 < len(pieces) else end
        words.extend(split_words(" ".join(text.split()), t, max(t_end, t)))
    return words


def build_caption_track(
    cues: list[tuple[float, float, str]], lo: float = 0.0, hi: float = math.inf
) -> tuple[list[Segment], list[Word]] | None:
    """
    Ubah cue caption menjadi kalimat dan kata, hanya untuk rentang [lo, hi).

    Jika ada cue dengan penanda per kata (caption otomatis), hanya cue itu yang dipakai dan cue salinannya dibuang.
    Caption manual tidak punya penanda, jadi waktunya dibagi per panjang kata seperti transkrip Gemini.
    """
    tagged = any(_INLINE.search(body) for _, _, body in cues)
    segments: list[Segment] = []
    words: list[Word] = []
    for start, end, body in cues:
        if tagged:
            if not _INLINE.search(body):
                continue  # cue salinan: baris yang sudah pernah tampil
            cue_words = _tagged_words(start, end, body)
        else:
            text = " ".join(_TAG.sub("", body).split())
            if not text or (segments and segments[-1].text == text):
                continue
            cue_words = split_words(text, start, end)

        cue_words = [w for w in cue_words if lo <= w.start < hi]
        if not cue_words:
            continue
        segments.append(
            Segment(
                start=cue_words[0].start,
                end=cue_words[-1].end,
                text=" ".join(w.text for w in cue_words),
            )
        )
        words.extend(cue_words)

    if not segments:
        return None
    words.sort(key=lambda w: w.start)
    return segments, words


def parse_caption_text(text: str, lo: float = 0.0, hi: float = math.inf) -> tuple[list[Segment], list[Word]] | None:
    """Baca isi berkas VTT. Mengembalikan None jika tidak ada kata di dalam rentang."""
    return build_caption_track(_parse_cues(text), lo, hi)


def _first_track(subs_dir: Path, lo: float, hi: float) -> tuple[list[Segment], list[Word]] | None:
    for lang in CAPTION_LANGS:
        for path in sorted(subs_dir.glob(f"subs.{lang}*.vtt")):
            result = parse_caption_text(path.read_text(encoding="utf-8", errors="ignore"), lo, hi)
            if result is not None:
                return result
    return None


def fetch_youtube_captions(
    url: str, subs_dir: Path, lo: float = 0.0, hi: float = math.inf
) -> tuple[list[Segment], list[Word]] | None:
    """
    Ambil caption YouTube (manual, atau otomatis jika tidak ada manual) lewat yt-dlp.

    Tidak mengunduh video. Mengembalikan None jika caption tidak ada, tidak bisa diambil, atau kosong,
    sehingga pemanggil bisa mundur ke transkripsi Gemini.
    """
    subs_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(CAPTION_LANGS),
        "subtitlesformat": "vtt",
        "outtmpl": str(subs_dir / "subs.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return _first_track(subs_dir, lo, hi)
    except Exception as exc:
        log.warning("Caption YouTube tidak bisa dipakai, akan memakai Gemini: %s", exc)
        return None
    finally:
        shutil.rmtree(subs_dir, ignore_errors=True)
