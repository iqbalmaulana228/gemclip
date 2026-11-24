from __future__ import annotations

from .models import Word

MAX_WORDS = 4  # jumlah kata per baris caption
MAX_CHARS = 22  # jumlah karakter per baris caption
MAX_SECONDS = 2.5  # durasi maksimum satu baris caption
MAX_GAP = 0.6  # jeda diam yang memisahkan dua baris caption

STYLES = ("hormozi", "neon", "minimal")

# Warna ASS memakai format &HAABBGGRR. Setiap gaya mengatur warna kata yang sudah dan belum diucapkan.
# Untuk BorderStyle 3 (kotak buram), OutlineColour adalah warna kotaknya.
_STYLE_DEFS: dict[str, dict] = {
    "hormozi": {  # tebal, kuning saat diucapkan, tepi hitam tebal
        "bold": -1,
        "caption_primary": "&H0000FFFF",
        "caption_secondary": "&H00FFFFFF",
        "caption_outline": "&H00000000",
        "caption_back": "&H80000000",
        "caption_border": 1,
        "caption_outline_w": 6,
        "caption_shadow": 1,
        "hook_primary": "&H00FFFFFF",
        "hook_box": "&H00000000",
    },
    "neon": {  # cyan dengan pendar, untuk gaya futuristik
        "bold": -1,
        "caption_primary": "&H00FFFF00",
        "caption_secondary": "&H00E0E0E0",
        "caption_outline": "&H00402000",
        "caption_back": "&HA0FFC000",
        "caption_border": 1,
        "caption_outline_w": 4,
        "caption_shadow": 4,
        "hook_primary": "&H00FFFF00",
        "hook_box": "&HA0000000",
    },
    "minimal": {  # kotak gelap semi transparan, tidak tebal, bersih
        "bold": 0,
        "caption_primary": "&H00FFFFFF",
        "caption_secondary": "&H00B0B0B0",
        "caption_outline": "&HB0000000",
        "caption_back": "&H00000000",
        "caption_border": 3,
        "caption_outline_w": 8,
        "caption_shadow": 0,
        "hook_primary": "&H00FFFFFF",
        "hook_box": "&HB0000000",
    },
}

POP_EFFECT = "{\\fscx88\\fscy88\\t(0,120,\\fscx100\\fscy100)}"


def _header(width: int, height: int, style: str) -> str:
    """Header ASS untuk resolusi dan gaya tertentu. Ukuran dan margin mengikuti resolusi keluaran."""
    d = _STYLE_DEFS[style]
    scale = min(width, height) / 1080
    vertical = height > width
    caption_margin = round(height * (560 / 1920 if vertical else 0.12))
    hook_margin = round(height * (260 / 1920 if vertical else 0.07))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{round(84 * scale)},{d["caption_primary"]},{d["caption_secondary"]},{d["caption_outline"]},{d["caption_back"]},{d["bold"]},0,0,0,100,100,0,0,{d["caption_border"]},{d["caption_outline_w"]},{d["caption_shadow"]},2,60,60,{caption_margin},1
Style: Hook,Arial,{round(64 * scale)},{d["hook_primary"]},{d["hook_primary"]},{d["hook_box"]},&H80000000,-1,0,0,0,100,100,0,0,3,18,0,8,70,70,{hook_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _clean(text: str) -> str:
    """Hapus karakter yang punya arti khusus di format ASS."""
    for ch in "{}\\\r\n":
        text = text.replace(ch, "")
    return text.strip()


def _relative_words(words: list[Word], clip_start: float, clip_duration: float) -> list[Word]:
    """Ambil kata yang berada di dalam klip, lalu ubah waktunya menjadi relatif terhadap awal klip."""
    relative: list[Word] = []
    for w in words:
        s = w.start - clip_start
        e = w.end - clip_start
        if e <= 0 or s >= clip_duration:
            continue
        token = _clean(w.text)
        if not token:
            continue
        relative.append(Word(start=max(0.0, s), end=min(clip_duration, e), text=token))
    return relative


def group_words(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current:
            gap = w.start - current[-1].end
            chars = sum(len(x.text) for x in current) + len(w.text) + len(current)
            span = w.end - current[0].start
            if len(current) >= MAX_WORDS or chars > MAX_CHARS or span > MAX_SECONDS or gap > MAX_GAP:
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


def build_ass(
    words: list[Word],
    clip_start: float,
    clip_duration: float,
    hook: str = "",
    *,
    size: tuple[int, int] = (1080, 1920),
    style: str = "hormozi",
) -> str:
    """
    Buat isi file ASS untuk satu klip.

    Kata diberi efek karaoke (\\k) sehingga setiap kata berubah warna saat diucapkan.
    Waktu kata diubah menjadi relatif terhadap awal klip.
    """
    if style not in _STYLE_DEFS:
        raise ValueError(f"Gaya caption tidak dikenal: {style}")
    width, height = size
    relative = _relative_words(words, clip_start, clip_duration)

    lines = [_header(width, height, style)]

    hook_text = _clean(hook)
    if hook_text:
        hook_end = min(3.0, clip_duration)
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(hook_end)},Hook,,0,0,0,,{hook_text}\n")

    for group in group_words(relative):
        start = group[0].start
        end = min(clip_duration, group[-1].end + 0.2)
        parts = []
        for i, w in enumerate(group):
            centis = max(1, round((w.end - w.start) * 100))
            sep = " " if i < len(group) - 1 else ""
            parts.append(f"{{\\k{centis}}}{w.text}{sep}")
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{POP_EFFECT}{''.join(parts)}\n"
        )

    return "".join(lines)


def build_srt(words: list[Word], clip_start: float, clip_duration: float) -> str:
    """Buat berkas SRT untuk satu klip, dengan baris yang sama seperti caption di video."""
    relative = _relative_words(words, clip_start, clip_duration)
    blocks = []
    for i, group in enumerate(group_words(relative), 1):
        start = group[0].start
        end = min(clip_duration, group[-1].end + 0.2)
        text = " ".join(w.text for w in group)
        blocks.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    return "\n".join(blocks)


def clip_text(words: list[Word], start: float, end: float) -> str:
    """Teks transkrip yang berada di dalam rentang klip, untuk ditampilkan dan disalin."""
    tokens = [_clean(w.text) for w in words if w.end > start and w.start < end]
    return " ".join(t for t in tokens if t)
