import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gemclip import render as render_mod
from gemclip.models import Word
from gemclip.render import render_clip

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak tersedia",
)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "stream=codec_type,width,height:format=duration",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _make_source(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )


@needs_ffmpeg
def test_render_clip_produces_vertical_video_with_audio(tmp_path, monkeypatch):
    captured = {}

    def fake_track(api_key, model, segment, work_dir, stem, src_w, fps):
        captured.update(api_key=api_key, model=model, src_w=src_w)
        return np.zeros((0, 2))  # tanpa deteksi, crop jatuh ke tengah

    monkeypatch.setattr(render_mod, "track_hits", fake_track)

    source = tmp_path / "source.mp4"
    _make_source(source)

    words = [Word(start=1.0 + i * 0.5, end=1.4 + i * 0.5, text=f"kata{i}") for i in range(8)]
    work = tmp_path / "work"
    work.mkdir()
    out = work / "clip_01.mp4"

    render_clip(
        source=source,
        start=1.0,
        end=5.0,
        words=words,
        out_path=out,
        work_dir=work,
        stem="clip_01",
        hook="Tes hook",
        api_key="test-key",
        model="gemini-test",
    )

    assert captured == {"api_key": "test-key", "model": "gemini-test", "src_w": 1280}
    assert out.exists()
    info = _probe(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert 3.5 <= float(info["format"]["duration"]) <= 4.5
    assert (work / "clip_01.ass").exists()
    assert not (work / "clip_01_seg.mp4").exists()
    assert not (work / "clip_01_frames").exists()


@needs_ffmpeg
@pytest.mark.parametrize(
    "ratio, expected, style",
    [
        ("9:16", (1080, 1920), "hormozi"),
        ("1:1", (1080, 1080), "neon"),
        ("16:9", (1920, 1080), "minimal"),
    ],
)
def test_render_clip_respects_ratio_and_style(tmp_path, monkeypatch, ratio, expected, style):
    monkeypatch.setattr(render_mod, "track_hits", lambda *a, **k: np.zeros((0, 2)))
    source = tmp_path / "source.mp4"
    _make_source(source)
    words = [Word(start=1.0 + i * 0.5, end=1.4 + i * 0.5, text=f"kata{i}") for i in range(6)]
    out = tmp_path / f"clip_{ratio.replace(':', 'x')}.mp4"

    render_clip(
        source=source,
        start=1.0,
        end=4.0,
        words=words,
        out_path=out,
        work_dir=tmp_path,
        stem="clip_01",
        hook="Tes",
        api_key="test-key",
        model="gemini-test",
        ratio=ratio,
        style=style,
    )

    info = _probe(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == expected
    assert f"PlayResX: {expected[0]}" in (tmp_path / "clip_01.ass").read_text(encoding="utf-8")


def test_render_clip_rejects_unknown_ratio(tmp_path):
    with pytest.raises(ValueError):
        render_clip(
            source=tmp_path / "tidak_ada.mp4",
            start=0.0,
            end=1.0,
            words=[],
            out_path=tmp_path / "x.mp4",
            work_dir=tmp_path,
            stem="x",
            api_key="test-key",
            model="gemini-test",
            ratio="4:3",
        )
