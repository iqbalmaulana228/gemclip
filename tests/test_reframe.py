import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

from gemclip import gemini as gemini_mod
from gemclip import reframe as reframe_mod
from gemclip.reframe import crop_centers, crop_width, track_hits

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg tidak tersedia",
)


def test_crop_width_is_9_by_16_and_even():
    cw = crop_width(1280, 720)
    assert cw % 2 == 0
    assert abs(cw - 720 * 9 / 16) <= 2
    assert cw <= 1280


def test_crop_width_never_exceeds_source():
    assert crop_width(400, 720) <= 400


@pytest.mark.parametrize(
    "ratio, src, expected",
    [
        ("9:16", (1280, 720), 404),
        ("1:1", (1280, 720), 720),
        ("16:9", (1280, 720), 1280),
    ],
)
def test_crop_width_follows_output_ratio(ratio, src, expected):
    assert crop_width(*src, ratio) == expected


def test_output_size_rejects_unknown_ratio():
    with pytest.raises(ValueError):
        reframe_mod.output_size("4:3")


def test_crop_centers_without_faces_uses_default():
    centers = crop_centers(np.zeros((0, 2)), n_frames=90, default_x=640)
    assert len(centers) == 90
    assert np.allclose(centers, 640)


def test_crop_centers_follows_face_and_stays_in_range():
    hits = np.array([[0, 200.0], [60, 800.0]])
    centers = crop_centers(hits, n_frames=120, default_x=640)
    assert len(centers) == 120
    assert centers.min() >= 199
    assert centers.max() <= 801
    # kamera bergerak dari kiri ke kanan, jadi pusat di tengah video harus berada di antara keduanya
    assert 300 < centers[30] < 700


def test_crop_centers_step_is_limited():
    hits = np.array([[0, 100.0], [5, 900.0]])  # lompatan besar dalam waktu singkat
    centers = crop_centers(hits, n_frames=150, default_x=640, max_step_px=20)
    assert np.max(np.abs(np.diff(centers))) <= 20 + 1e-9


def test_crop_centers_ignores_jitter_inside_deadzone():
    hits = np.array([[0, 500.0], [30, 520.0], [60, 490.0]])  # getaran kecil di sekitar 500
    centers = crop_centers(hits, n_frames=90, default_x=640, deadzone_px=61)
    assert np.allclose(centers, 500.0)


def test_crop_centers_moves_once_face_leaves_deadzone():
    hits = np.array([[0, 500.0], [60, 800.0]])  # wajah pindah jauh ke kanan dan berhenti
    centers = crop_centers(hits, n_frames=121, default_x=640, deadzone_px=61)
    # kamera berhenti di tepi zona toleransi, yaitu 61 px di kiri posisi wajah
    assert centers[-1] == pytest.approx(800 - 61, abs=1.0)
    assert np.all(np.diff(centers) >= -1e-9)  # hanya bergerak ke satu arah


def _make_video(path, seconds: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=25:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _patch_gemini(monkeypatch, payload: str = "[]", error: Exception | None = None) -> list[dict]:
    """Ganti klien Gemini dengan klien tiruan. Mengembalikan daftar permintaan yang diterima."""
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            if error is not None:
                raise error
            return SimpleNamespace(text=payload)

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            assert api_key == "test-key"
            self.models = FakeModels()

    monkeypatch.setattr(gemini_mod.genai, "Client", FakeClient)
    return calls


@needs_ffmpeg
def test_track_hits_maps_gemini_answers_to_pixels(tmp_path, monkeypatch):
    segment = tmp_path / "seg.mp4"
    _make_video(segment, seconds=5)  # frame dikirim tiap 2 detik: t = 0, 2, 4
    payload = json.dumps(
        [
            {"frame": 0, "x": 0.25},
            {"frame": 1, "x": -1},  # tidak ada wajah, diabaikan
            {"frame": 2, "x": 0.75},
        ]
    )
    calls = _patch_gemini(monkeypatch, payload)

    hits = track_hits("test-key", "gemini-test", segment, tmp_path, "clip_01", src_w=1280, fps=25.0)

    assert hits.shape == (2, 2)
    assert hits[0] == pytest.approx([0, 320.0])
    assert hits[1] == pytest.approx([100, 960.0])  # frame 2 = detik ke-4 = frame video ke-100
    images = [c for c in calls[0]["contents"] if isinstance(c, types.Part)]
    assert len(images) == 3
    assert calls[0]["config"].response_mime_type == "application/json"
    assert not (tmp_path / "clip_01_frames").exists(), "frame sementara harus dihapus"


@needs_ffmpeg
def test_track_hits_falls_back_to_center_when_gemini_fails(tmp_path, monkeypatch):
    segment = tmp_path / "seg.mp4"
    _make_video(segment, seconds=4)
    _patch_gemini(monkeypatch, error=RuntimeError("jaringan terputus"))

    hits = track_hits("test-key", "gemini-test", segment, tmp_path, "clip_02", src_w=1280, fps=25.0)

    assert hits.shape == (0, 2)
    centers = crop_centers(hits, n_frames=100, default_x=640)
    assert np.allclose(centers, 640)
    assert not (tmp_path / "clip_02_frames").exists()


@needs_ffmpeg
def test_track_hits_without_any_face_returns_empty(tmp_path, monkeypatch):
    segment = tmp_path / "seg.mp4"
    _make_video(segment, seconds=3)
    _patch_gemini(monkeypatch, json.dumps([{"frame": 0, "x": -1}, {"frame": 1, "x": -1}]))

    hits = track_hits("test-key", "gemini-test", segment, tmp_path, "clip_03", src_w=1280, fps=25.0)
    assert hits.shape == (0, 2)
