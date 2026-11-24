import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gemclip import pipeline
from gemclip import render as render_mod
from gemclip.config import Settings
from gemclip.models import Moment, Segment, VideoInfo, Word

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak tersedia",
)


def _settings(tmp_path: Path, api_key: str = "test-key") -> Settings:
    return Settings(
        gemini_api_key=api_key,
        gemini_model="gemini-test",
        work_dir=tmp_path,
        clip_min_sec=5,
        clip_max_sec=60,
        default_clip_count=2,
        max_height=720,
    )


def _make_source(path: Path, seconds: int = 20) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )


def _stub_download(source: Path, duration: float):
    def fake_download(url, out_dir, max_height):
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "source.mp4"
        shutil.copy(source, target)
        return VideoInfo(path=target, title="Video uji", duration=duration)

    return fake_download


def _no_face_tracking(monkeypatch):
    """Pelacakan wajah lewat Gemini dimatikan di tes ini. Tanpa jaringan, crop jatuh ke tengah."""
    monkeypatch.setattr(render_mod, "track_hits", lambda *args, **kwargs: np.zeros((0, 2)))


@needs_ffmpeg
def test_run_job_end_to_end_with_stubbed_network(tmp_path, monkeypatch):
    source = tmp_path / "_source_fixture.mp4"
    _make_source(source, 20)

    words = [Word(start=i * 0.8, end=i * 0.8 + 0.6, text=f"kata{i}") for i in range(25)]

    def fake_transcribe(**kwargs):
        return [Segment(0.0, 20.0, "tes")], words

    def fake_find_moments(**kwargs):
        return [
            Moment(2.0, 10.0, 90, "Judul pertama", "Hook pertama", "Caption pertama", ("#satu",)),
            Moment(11.0, 19.0, 70, "Judul kedua", "Hook kedua", "Caption kedua", ("#dua", "#tiga")),
        ]

    monkeypatch.setattr(pipeline, "download_video", _stub_download(source, 20.0))
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "find_moments", fake_find_moments)
    _no_face_tracking(monkeypatch)

    steps = []
    job_dir = tmp_path / "job1"
    result = pipeline.run_job(
        "https://youtu.be/abc123",
        clip_count=2,
        settings=_settings(tmp_path),
        job_dir=job_dir,
        progress=lambda step, frac: steps.append((step, frac)),
    )

    assert result["title"] == "Video uji"
    assert len(result["clips"]) == 2
    assert steps[-1] == ("Selesai", 1.0)
    assert all(0.0 <= frac <= 1.0 for _, frac in steps)

    for clip in result["clips"]:
        mp4 = job_dir / clip["file"]
        assert mp4.exists()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, check=True,
        )
        assert probe.stdout.strip() == "1080,1920"

    post_text = (job_dir / "clip_01.txt").read_text(encoding="utf-8")
    assert "Judul pertama" in post_text and "#satu" in post_text

    srt = (job_dir / "clip_01.srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,000 --> ")

    saved = json.loads((job_dir / "clips.json").read_text(encoding="utf-8"))
    assert saved["clips"][1]["hashtags"] == ["#dua", "#tiga"]
    tokens = saved["clips"][0]["transcript"].split()
    assert tokens[0] == "kata2" and tokens[-1] == "kata12", "transkrip hanya berisi kata di dalam klip 2 s/d 10 s"
    assert saved["clips"][0]["srt"] == "clip_01.srt"


def test_run_job_rejects_unknown_ratio_or_style(tmp_path):
    with pytest.raises(ValueError, match="Rasio"):
        pipeline.run_job(
            "https://youtu.be/abc123",
            clip_count=1,
            settings=_settings(tmp_path),
            job_dir=tmp_path / "job5",
            progress=lambda s, f: None,
            ratio="4:3",
        )
    with pytest.raises(ValueError, match="Gaya"):
        pipeline.run_job(
            "https://youtu.be/abc123",
            clip_count=1,
            settings=_settings(tmp_path),
            job_dir=tmp_path / "job6",
            progress=lambda s, f: None,
            caption_style="pelangi",
        )


@needs_ffmpeg
def test_run_job_passes_range_duration_and_instruction(tmp_path, monkeypatch):
    source = tmp_path / "_source_fixture.mp4"
    _make_source(source, 20)
    captured = {}

    def fake_transcribe(**kwargs):
        captured["transcribe"] = kwargs
        return [Segment(5.0, 15.0, "tes")], [Word(5.0, 5.5, "tes")]

    def fake_find_moments(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "download_video", _stub_download(source, 20.0))
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "find_moments", fake_find_moments)

    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        pipeline.run_job(
            "https://youtu.be/abc123",
            clip_count=3,
            settings=_settings(tmp_path),
            job_dir=tmp_path / "job3",
            progress=lambda s, f: None,
            clip_min=10,
            clip_max=25,
            start_sec=5,
            end_sec=15,
            instruction="fokus pada tips",
        )

    assert captured["transcribe"]["clip_range"] == (5.0, 15.0)
    assert captured["transcribe"]["api_key"] == "test-key"
    assert captured["transcribe"]["model"] == "gemini-test"
    assert captured["transcribe"]["duration"] == 20.0
    assert captured["range_start"] == 5.0
    assert captured["range_end"] == 15.0
    assert captured["min_sec"] == 10
    assert captured["max_sec"] == 25
    assert captured["instruction"] == "fokus pada tips"
    assert captured["analysis"] == "transcript"


def test_run_job_full_video_does_not_limit_transcription(tmp_path, monkeypatch):
    captured = {}

    def fake_download(url, out_dir, max_height):
        return VideoInfo(path=tmp_path / "x.mp4", title="t", duration=600.0)

    def fake_transcribe(**kwargs):
        captured["clip_range"] = kwargs["clip_range"]
        return [], []

    monkeypatch.setattr(pipeline, "download_video", fake_download)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "check_ffmpeg", lambda: None)

    with pytest.raises(RuntimeError, match="Tidak ada ucapan"):
        pipeline.run_job(
            "https://youtu.be/abc123",
            clip_count=1,
            settings=_settings(tmp_path),
            job_dir=tmp_path / "job4",
            progress=lambda s, f: None,
        )
    assert captured["clip_range"] is None


def test_run_job_requires_api_key(tmp_path):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        pipeline.run_job(
            "https://youtu.be/abc123",
            clip_count=1,
            settings=_settings(tmp_path, api_key=""),
            job_dir=tmp_path / "job2",
            progress=lambda s, f: None,
        )


def test_resolve_range_validates_bounds():
    assert pipeline.resolve_range(None, None, 100) == (0.0, 100.0)
    assert pipeline.resolve_range(10, 50, 100) == (10.0, 50.0)
    assert pipeline.resolve_range(10, 500, 100) == (10.0, 100.0)  # akhir dipotong ke durasi
    with pytest.raises(ValueError):
        pipeline.resolve_range(120, None, 100)
    with pytest.raises(ValueError):
        pipeline.resolve_range(50, 40, 100)
