from __future__ import annotations

from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types
from pydantic import BaseModel, TypeAdapter, ValidationError

from .chunks import cut_chunk, plan_windows
from .gemini import delete_quietly, upload_and_wait
from .models import Moment, Segment, Word

SNAP_TOLERANCE = 1.5  # detik: jarak maksimum untuk menempelkan batas klip ke batas kata
MAX_INSTRUCTION_CHARS = 500

Progress = Callable[[str, float], None]


class MomentDraft(BaseModel):
    start_sec: float
    end_sec: float
    viral_score: int
    title: str
    hook: str
    caption: str
    hashtags: list[str]


PROMPT_TEMPLATE = """Kamu adalah editor video pendek profesional untuk YouTube Shorts, TikTok, dan Instagram Reels.

Video terlampir adalah potongan dari video yang lebih panjang. Potongan ini berdurasi {window_len:.0f} detik, dan waktunya dihitung mulai dari 0 di awal potongan. Transkrip di bawah memakai waktu yang sama sebagai referensi, tetapi nilai momen dengan melihat dan mendengar video secara langsung.

Pilih hingga {count} momen terbaik di dalam potongan ini. Aturan:
- Setiap klip berdurasi antara {min_sec} dan {max_sec} detik.
- Klip dimulai di awal sebuah ide dan berakhir ketika ide itu selesai. Jangan memotong di tengah kalimat.
- Klip harus bisa dipahami tanpa konteks dari bagian video lain.
- Klip tidak boleh saling tumpang tindih.
- start_sec dan end_sec adalah detik dari awal potongan ini, berupa angka desimal, dan harus berada antara 0 dan {window_len:.0f}.
- viral_score adalah bilangan 0 sampai 100 yang menilai seberapa kuat momen itu untuk menahan penonton.
- title adalah judul singkat dan menarik, dalam bahasa yang sama dengan video.
- hook adalah satu kalimat pembuka yang ditampilkan di awal klip.
- caption adalah satu hingga dua kalimat untuk deskripsi postingan.
- hashtags berisi 3 sampai 5 hashtag tanpa spasi.
- Jangan menambahkan fakta yang tidak ada di video.
- Urutkan hasil dari viral_score tertinggi ke terendah.
{instruction_block}
Transkrip:
{transcript}
"""

INSTRUCTION_TEMPLATE = """
Instruksi tambahan dari pengguna. Ikuti selama tidak melanggar aturan di atas. Jika tidak ada momen yang sesuai instruksi, kembalikan daftar kosong.
<<<
{instruction}
>>>
"""


PROMPT_TRANSCRIPT_TEMPLATE = """Kamu adalah editor video pendek profesional untuk YouTube Shorts, TikTok, dan Instagram Reels.

Di bawah ini adalah transkrip sebuah video panjang. Waktu dalam transkrip adalah detik dari awal video asli. Kamu hanya bisa membaca transkrip, jadi nilai momen dari isi dan cara penyampaian kalimatnya.

Pilih hingga {count} momen terbaik dari rentang {range_start:.0f} sampai {range_end:.0f} detik. Aturan:
- Setiap klip berdurasi antara {min_sec} dan {max_sec} detik.
- Klip dimulai di awal sebuah ide dan berakhir ketika ide itu selesai. Jangan memotong di tengah kalimat.
- Klip harus bisa dipahami tanpa konteks dari bagian lain.
- Klip tidak boleh saling tumpang tindih.
- start_sec dan end_sec adalah detik dari awal video asli, berupa angka desimal, dan harus berada di dalam rentang di atas.
- viral_score adalah bilangan 0 sampai 100 yang menilai seberapa kuat momen itu untuk menahan penonton.
- title adalah judul singkat dan menarik, dalam bahasa yang sama dengan video.
- hook adalah satu kalimat pembuka yang ditampilkan di awal klip.
- caption adalah satu hingga dua kalimat untuk deskripsi postingan.
- hashtags berisi 3 sampai 5 hashtag tanpa spasi.
- Jangan menambahkan fakta yang tidak ada di transkrip.
- Urutkan hasil dari viral_score tertinggi ke terendah.
{instruction_block}
Transkrip:
{transcript}
"""


def build_transcript_prompt(
    segments: list[Segment],
    count: int,
    min_sec: float,
    max_sec: float,
    range_start: float,
    range_end: float,
    instruction: str = "",
) -> str:
    instruction = instruction.strip()[:MAX_INSTRUCTION_CHARS]
    instruction_block = INSTRUCTION_TEMPLATE.format(instruction=instruction) if instruction else ""
    return PROMPT_TRANSCRIPT_TEMPLATE.format(
        count=count,
        min_sec=int(min_sec),
        max_sec=int(max_sec),
        range_start=range_start,
        range_end=range_end,
        instruction_block=instruction_block,
        transcript=format_transcript(segments),
    )


def format_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments)


def build_prompt(
    segments: list[Segment],
    count: int,
    min_sec: float,
    max_sec: float,
    window_len: float,
    instruction: str = "",
) -> str:
    instruction = instruction.strip()[:MAX_INSTRUCTION_CHARS]
    instruction_block = INSTRUCTION_TEMPLATE.format(instruction=instruction) if instruction else ""
    return PROMPT_TEMPLATE.format(
        window_len=window_len,
        count=count,
        min_sec=int(min_sec),
        max_sec=int(max_sec),
        instruction_block=instruction_block,
        transcript=format_transcript(segments),
    )


def _window_segments(segments: list[Segment], a: float, b: float) -> list[Segment]:
    """Ambil kalimat yang berada di dalam jendela [a, b], dengan waktu relatif terhadap a."""
    out: list[Segment] = []
    for s in segments:
        if s.end <= a or s.start >= b:
            continue
        out.append(Segment(start=max(0.0, s.start - a), end=min(b - a, s.end - a), text=s.text))
    return out


def parse_drafts(raw: str) -> list[MomentDraft]:
    try:
        return TypeAdapter(list[MomentDraft]).validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"Format jawaban Gemini tidak valid: {raw[:300]}") from exc


def _shift(draft: MomentDraft, offset: float, window_len: float) -> MomentDraft:
    """Batasi draft ke panjang potongan, lalu ubah waktunya ke waktu video asli."""
    start = max(0.0, min(draft.start_sec, window_len))
    end = max(0.0, min(draft.end_sec, window_len))
    return draft.model_copy(update={"start_sec": start + offset, "end_sec": end + offset})


def _snap(t: float, points: list[float]) -> float:
    """Tempelkan waktu t ke titik kata terdekat jika jaraknya masih wajar."""
    if not points:
        return t
    best = min(points, key=lambda p: abs(p - t))
    return best if abs(best - t) <= SNAP_TOLERANCE else t


def _trim_end(start: float, end: float, words: list[Word], min_sec: float, max_sec: float) -> float:
    """Potong klip yang terlalu panjang di akhir kata, bukan di tengah kata."""
    limit = start + max_sec
    candidates = [w.end for w in words if start + min_sec <= w.end <= limit]
    return max(candidates) if candidates else limit


def _clean_hashtags(raw: list[str]) -> tuple[str, ...]:
    tags: list[str] = []
    for item in raw:
        tag = "#" + item.strip().lstrip("#").replace(" ", "")
        if tag != "#" and tag not in tags:
            tags.append(tag)
    return tuple(tags[:5])


def clean_moments(
    drafts: list[MomentDraft],
    words: list[Word],
    duration: float,
    count: int,
    min_sec: float,
    max_sec: float,
    *,
    range_start: float = 0.0,
    range_end: float | None = None,
) -> list[Moment]:
    """
    Validasi, sesuaikan ke batas kata, buang yang tumpang tindih, dan ambil yang terbaik.

    Klip dibatasi ke rentang [range_start, range_end] jika diberikan.
    """
    lo = max(0.0, range_start)
    hi = min(duration, range_end) if range_end is not None else duration
    starts = [w.start for w in words]
    ends = [w.end for w in words]

    candidates: list[Moment] = []
    for d in drafts:
        start = max(lo, min(d.start_sec, hi))
        end = max(lo, min(d.end_sec, hi))
        if end <= start:
            continue

        start = _snap(start, starts)
        end = _snap(end, ends)
        start = max(lo, start - 0.15)
        end = min(hi, end + 0.25)

        if end - start > max_sec:
            end = _trim_end(start, end, words, min_sec, max_sec)
        if end - start < min_sec:
            continue

        candidates.append(
            Moment(
                start=round(start, 2),
                end=round(end, 2),
                score=max(0, min(100, d.viral_score)),
                title=d.title.strip(),
                hook=d.hook.strip(),
                caption=d.caption.strip(),
                hashtags=_clean_hashtags(d.hashtags),
            )
        )

    candidates.sort(key=lambda m: m.score, reverse=True)
    kept: list[Moment] = []
    for m in candidates:
        if all(m.end <= k.start or m.start >= k.end for k in kept):
            kept.append(m)
        if len(kept) >= count:
            break
    return kept


def _analyze_transcript(
    client: genai.Client,
    model: str,
    segments: list[Segment],
    words: list[Word],
    duration: float,
    count: int,
    min_sec: float,
    max_sec: float,
    lo: float,
    hi: float,
    instruction: str,
    progress: Progress | None,
) -> list[Moment]:
    """Analisis hanya dari transkrip. Satu permintaan saja, tanpa unggah video."""
    if progress:
        progress("Menganalisis transkrip dengan Gemini", 0.3)
    in_range = [s for s in segments if s.end > lo and s.start < hi]
    if not in_range:
        raise RuntimeError("Transkrip kosong pada rentang yang dipilih.")

    response = client.models.generate_content(
        model=model,
        contents=build_transcript_prompt(in_range, count, min_sec, max_sec, lo, hi, instruction),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[MomentDraft],
            temperature=0.4,
        ),
    )
    drafts = parse_drafts(response.text or "")
    if progress:
        progress("Menyusun pilihan terbaik", 1.0)
    return clean_moments(drafts, words, duration, count, min_sec, max_sec, range_start=lo, range_end=hi)


def find_moments(
    api_key: str,
    model: str,
    video_path: Path,
    segments: list[Segment],
    words: list[Word],
    duration: float,
    count: int,
    min_sec: float,
    max_sec: float,
    *,
    range_start: float = 0.0,
    range_end: float | None = None,
    instruction: str = "",
    work_dir: Path | None = None,
    chunk_sec: float = 900.0,
    overlap_sec: float = 60.0,
    progress: Progress | None = None,
    analysis: str = "video",
) -> list[Moment]:
    """
    Cari momen terbaik di dalam rentang video.

    analysis="transcript": Gemini hanya membaca transkrip. Murah dan cepat, tetapi tidak melihat video.
    analysis="video": Gemini menonton video. Video dipecah menjadi potongan (default 15 menit)
    agar muat di context window. Setiap potongan dianalisis terpisah, lalu kandidat digabung.
    """
    if analysis not in ("video", "transcript"):
        raise ValueError("analysis harus 'video' atau 'transcript'.")

    lo = max(0.0, range_start)
    hi = min(duration, range_end) if range_end is not None else duration
    if hi <= lo:
        raise RuntimeError("Rentang waktu tidak valid: akhir harus lebih besar dari awal.")

    client = genai.Client(api_key=api_key)
    if analysis == "transcript":
        return _analyze_transcript(
            client, model, segments, words, duration, count, min_sec, max_sec, lo, hi, instruction, progress
        )

    windows = plan_windows(lo, hi, chunk_sec, overlap_sec)
    work = work_dir or Path(video_path).parent
    all_drafts: list[MomentDraft] = []

    for idx, (a, b) in enumerate(windows, 1):
        if progress:
            progress(f"Menganalisis bagian {idx} dari {len(windows)} dengan Gemini", (idx - 1) / len(windows))
        window_len = b - a
        chunk_path = work / f"gemini_part_{idx:02d}.mp4"
        cut_chunk(Path(video_path), a, window_len, chunk_path)
        try:
            video = upload_and_wait(client, chunk_path, "video/mp4")
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        video,
                        build_prompt(
                            _window_segments(segments, a, b),
                            count,
                            min_sec,
                            max_sec,
                            window_len,
                            instruction,
                        ),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[MomentDraft],
                        temperature=0.4,
                    ),
                )
            finally:
                delete_quietly(client, video.name)
        finally:
            chunk_path.unlink(missing_ok=True)

        for draft in parse_drafts(response.text or ""):
            all_drafts.append(_shift(draft, a, window_len))

    if progress:
        progress("Menyusun pilihan terbaik", 1.0)
    return clean_moments(
        all_drafts,
        words,
        duration,
        count,
        min_sec,
        max_sec,
        range_start=lo,
        range_end=hi,
    )
