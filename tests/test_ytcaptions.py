from pathlib import Path

import pytest

from gemclip import ytcaptions
from gemclip.ytcaptions import fetch_youtube_captions, parse_caption_text

PLAIN_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
Halo semua

00:00:03.500 --> 00:00:06.000
<c.colorE5E5E5>selamat datang</c> di kelas
"""

# Bentuk caption otomatis YouTube: cue pertama berisi penanda waktu per kata, cue kedua adalah salinan baris lama.
AUTO_VTT = """WEBVTT
Kind: captions
Language: id

00:00:00.000 --> 00:00:02.000 align:start position:0%

hello<00:00:00.500><c> there</c><00:00:01.000><c> friends</c>

00:00:02.000 --> 00:00:02.010 align:start position:0%
hello there friends


00:00:02.010 --> 00:00:04.000 align:start position:0%
hello there friends
how<00:00:02.500><c> are</c><00:00:03.000><c> you</c>
"""


def test_plain_captions_keep_cue_times_and_strip_markup():
    segments, words = parse_caption_text(PLAIN_VTT)
    assert [(s.start, s.end, s.text) for s in segments] == [
        (1.0, 3.0, "Halo semua"),
        (3.5, 6.0, "selamat datang di kelas"),
    ]
    assert [w.text for w in words] == ["Halo", "semua", "selamat", "datang", "di", "kelas"]


def test_auto_captions_use_word_timing_and_skip_rolling_copies():
    segments, words = parse_caption_text(AUTO_VTT)
    assert [s.text for s in segments] == ["hello there friends", "how are you"]
    assert [w.text for w in words] == ["hello", "there", "friends", "how", "are", "you"]
    assert [w.start for w in words[:3]] == pytest.approx([0.0, 0.5, 1.0])
    assert words[3].start == pytest.approx(2.01)
    assert words[5].end == pytest.approx(4.0)


def test_range_filter_keeps_only_words_inside_window():
    segments, words = parse_caption_text(PLAIN_VTT, lo=3.0, hi=10.0)
    assert [s.text for s in segments] == ["selamat datang di kelas"]
    assert all(w.start >= 3.0 for w in words)


def test_empty_caption_file_gives_none():
    assert parse_caption_text("WEBVTT\n\n") is None


class _FakeYDL:
    """Pengganti yt-dlp: menulis berkas VTT ke folder tujuan, tanpa jaringan."""

    files: dict[str, str] = {}
    error: Exception | None = None

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        assert self.opts["skip_download"] is True, "caption saja, video tidak boleh diunduh"
        assert self.opts["subtitlesformat"] == "vtt"
        if self.error is not None:
            raise self.error
        for lang, text in self.files.items():
            Path(self.opts["outtmpl"].replace("%(ext)s", f"{lang}.vtt")).write_text(text, encoding="utf-8")


@pytest.fixture
def fake_ydl(monkeypatch):
    _FakeYDL.files = {}
    _FakeYDL.error = None
    monkeypatch.setattr(ytcaptions.yt_dlp, "YoutubeDL", _FakeYDL)
    return _FakeYDL


def test_fetch_reads_english_captions_and_cleans_up(tmp_path, fake_ydl):
    fake_ydl.files = {"en": PLAIN_VTT}
    subs = tmp_path / "subs"

    result = fetch_youtube_captions("https://youtu.be/dQw4w9WgXcQ", subs)

    assert result is not None and len(result[0]) == 2
    assert not subs.exists(), "folder caption sementara harus dihapus"


def test_fetch_prefers_indonesian_over_english(tmp_path, fake_ydl):
    fake_ydl.files = {
        "en": PLAIN_VTT,
        "id": "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nselamat pagi\n",
    }
    result = fetch_youtube_captions("https://youtu.be/dQw4w9WgXcQ", tmp_path / "subs")
    assert result is not None
    assert [s.text for s in result[0]] == ["selamat pagi"]


def test_fetch_returns_none_when_downloader_fails(tmp_path, fake_ydl):
    fake_ydl.error = RuntimeError("tidak ada subtitle")
    assert fetch_youtube_captions("https://youtu.be/dQw4w9WgXcQ", tmp_path / "subs") is None
    assert not (tmp_path / "subs").exists()


def test_fetch_returns_none_when_no_caption_file(tmp_path, fake_ydl):
    assert fetch_youtube_captions("https://youtu.be/dQw4w9WgXcQ", tmp_path / "subs") is None
