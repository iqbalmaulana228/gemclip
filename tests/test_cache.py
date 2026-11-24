from dataclasses import replace
from pathlib import Path

import pytest

from gemclip import pipeline
from gemclip.cache import (
    load_download,
    load_transcript,
    save_download,
    save_transcript,
    transcript_path,
    youtube_id,
)
from gemclip.config import Settings
from gemclip.models import Segment, VideoInfo, Word

URL = "https://youtu.be/dQw4w9WgXcQ"  # ID 11 karakter, jadi dikenali dan di-cache


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/abc123", None),  # ID terlalu pendek
        ("https://example.com/video", None),
    ],
)
def test_youtube_id(url, expected):
    assert youtube_id(url) == expected


def test_transcript_path_depends_on_model_and_range(tmp_path):
    a = transcript_path(tmp_path, "gemini-3.6-flash", 0.0, 120.0)
    b = transcript_path(tmp_path, "gemini-3.6-flash", 10.0, 120.0)
    c = transcript_path(tmp_path, "gemini-other", 0.0, 120.0)
    assert len({a, b, c}) == 3
    assert "/" not in transcript_path(tmp_path, "models/x", 0, 1).name


def test_transcript_round_trip(tmp_path):
    path = tmp_path / "t.json"
    segments = [Segment(1.0, 2.5, "halo dunia")]
    words = [Word(1.0, 1.5, "halo"), Word(1.5, 2.5, "dunia")]
    save_transcript(path, segments, words)
    assert load_transcript(path) == (segments, words)


def test_load_transcript_ignores_broken_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("bukan json", encoding="utf-8")
    assert load_transcript(path) is None
    assert load_transcript(tmp_path / "tidak_ada.json") is None


def test_download_cache_requires_info_and_source(tmp_path):
    cache_dir = tmp_path / "abc"
    cache_dir.mkdir()
    assert load_download(cache_dir) is None

    source = cache_dir / "source.mp4"
    source.write_bytes(b"video")
    info = VideoInfo(path=source, title="Judul", duration=90.0)
    save_download(cache_dir, info)

    loaded = load_download(cache_dir)
    assert loaded == info

    source.unlink()
    assert load_download(cache_dir) is None, "berkas video yang hilang harus dianggap belum ada"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-test",
        work_dir=tmp_path,
        clip_min_sec=5,
        clip_max_sec=60,
        default_clip_count=2,
        max_height=720,
    )


@pytest.fixture
def stubs(monkeypatch):
    calls = {"download": 0, "transcribe": 0, "moments": 0}

    def fake_download(url, out_dir, max_height):
        calls["download"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "source.mp4"
        target.write_bytes(b"fake video")
        return VideoInfo(path=target, title="Video cache", duration=120.0)

    def fake_transcribe(**kwargs):
        calls["transcribe"] += 1
        return [Segment(0.0, 5.0, "halo")], [Word(0.0, 0.5, "halo")]

    def fake_find_moments(**kwargs):
        calls["moments"] += 1
        return []  # sengaja kosong, supaya job berhenti setelah tahap cache

    monkeypatch.setattr(pipeline, "download_video", fake_download)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "find_moments", fake_find_moments)
    monkeypatch.setattr(pipeline, "check_ffmpeg", lambda: None)
    return calls


def _run(tmp_path: Path, name: str, settings: Settings | None = None, **opts):
    return pipeline.run_job(
        URL,
        clip_count=opts.pop("clip_count", 2),
        settings=settings or _settings(tmp_path),
        job_dir=tmp_path / name,
        progress=lambda s, f: None,
        **opts,
    )


def test_second_job_reuses_download_and_transcript(tmp_path, stubs):
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job1", clip_count=2)
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job2", clip_count=5, instruction="fokus pada tips")

    assert stubs["download"] == 1
    assert stubs["transcribe"] == 1
    assert stubs["moments"] == 2
    assert (tmp_path / "cache" / "dQw4w9WgXcQ" / "source.mp4").exists()


def test_different_range_transcribes_again(tmp_path, stubs):
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job1")
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job2", start_sec=10, end_sec=60)

    assert stubs["download"] == 1, "unduhan tetap dipakai ulang"
    assert stubs["transcribe"] == 2, "rentang berbeda butuh transkrip sendiri"


def test_different_model_transcribes_again(tmp_path, stubs):
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job1")
    other = replace(_settings(tmp_path), gemini_model="gemini-other")
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(tmp_path, "job2", settings=other)

    assert stubs["transcribe"] == 2


def test_empty_transcript_is_not_cached(tmp_path, stubs, monkeypatch):
    def silent_transcribe(**kwargs):
        stubs["transcribe"] += 1
        return [], []

    monkeypatch.setattr(pipeline, "transcribe", silent_transcribe)
    for name in ("job1", "job2"):
        with pytest.raises(RuntimeError, match="Tidak ada ucapan"):
            _run(tmp_path, name)

    assert stubs["transcribe"] == 2
    assert not list((tmp_path / "cache" / "dQw4w9WgXcQ").glob("transcript_*.json"))


def test_unrecognized_link_is_downloaded_into_job_folder(tmp_path, stubs):
    short_url = "https://youtu.be/abc123"  # ID tidak dikenali, jadi tidak ada cache
    for name in ("job1", "job2"):
        with pytest.raises(RuntimeError, match="tidak menemukan momen"):
            pipeline.run_job(
                short_url,
                clip_count=2,
                settings=_settings(tmp_path),
                job_dir=tmp_path / name,
                progress=lambda s, f: None,
            )
    assert stubs["download"] == 2
    assert (tmp_path / "job1" / "source.mp4").exists()
    assert not (tmp_path / "cache").exists()
